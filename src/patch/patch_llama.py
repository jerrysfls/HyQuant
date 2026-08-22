"""In-place attention patch for Llama-3 models."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch

from src.patch._common import (
    D_STAGE_MODES as _D_STAGE_MODES,
    compute_vertical_indices,
    handle_d_stage_cache,
    infer_kv_repeat,
    is_d_stage_cache,
    repeat_interleave_manual,
    resolve_attention_interface,
    validate_patch_args,
)


def _is_d_stage_cache(past_key_values) -> bool:
    """Check if past_key_values is a DStageKVCache (duck-typing)."""
    return hasattr(past_key_values, "get_layer") and hasattr(past_key_values, "_layers")


def build_llama3_attention_forward(
    *,
    mode: str,
    window_size: int,
    top_ratio: float,
    use_vertical_indices: bool = True,
):
    """Build a patched Llama3 attention forward function."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward

    def patched_llama3_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = bool(kwargs.pop("output_attentions", False))
        kwargs.pop("use_cache", None)
        kwargs.pop("position_ids", None)

        if past_key_values is None and "past_key_values" in kwargs:
            past_key_values = kwargs.pop("past_key_values")
        if past_key_values is None and "past_key_value" in kwargs:
            past_key_values = kwargs.pop("past_key_value")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            raise ValueError("position_embeddings are required for patched Llama attention.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len = int(query_states.shape[2])
        _has_d_stage_cache = (
            mode in _D_STAGE_MODES
            and past_key_values is not None
            and is_d_stage_cache(past_key_values)
        )

        if _has_d_stage_cache:
            key_states, value_states = handle_d_stage_cache(
                self.layer_idx, past_key_values,
                query_states, key_states, value_states,
                self, sin, cos, cache_position,
            )
        elif past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
            )

        if not (_has_d_stage_cache and q_len == 1):
            if attention_mask is not None and attention_mask.ndim == 4:
                attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        attention_interface = resolve_attention_interface(
            self.config,
            eager_attention_forward,
            output_attentions=output_attentions,
        )
        attn_weights = None

        if mode == "dense" or (q_len == 1 and mode not in _D_STAGE_MODES):
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else float(getattr(self, "attention_dropout", 0.0)),
                scaling=getattr(self, "scaling", None),
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        elif q_len == 1 and mode in _D_STAGE_MODES and _has_d_stage_cache:
            layer_cache = past_key_values.get_layer(self.layer_idx)
            kv_repeat = infer_kv_repeat(self)
            attn_output = layer_cache.decode_attention(query_states, kv_repeat)
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        # no DStageKVCache attached: quantize-dequantize round trip as fallback
        elif q_len == 1 and mode in _D_STAGE_MODES:
            from src.ops.decode_d_stage import _quantize_by_bits, _dequantize_by_bits
            d_cfg = _D_STAGE_MODES[mode]
            layer_idx = getattr(self, "layer_idx", 1)

            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            value_states_full = repeat_interleave_manual(value_states, kv_repeat, 1)

            B, H, N, D = key_states_full.shape
            BH = B * H

            k_bits = d_cfg["k_bits"]
            v_bits = d_cfg["v_bits"]

            k_3d = key_states_full.reshape(BH, N, D)
            v_3d = value_states_full.reshape(BH, N, D)

            k_packed, k_scale = _quantize_by_bits(k_3d, k_bits)
            v_packed, v_scale = _quantize_by_bits(v_3d, v_bits)
            k_deq = _dequantize_by_bits(k_packed, k_scale, D, k_bits).reshape(B, H, N, D).to(key_states_full.dtype)
            v_deq = _dequantize_by_bits(v_packed, v_scale, D, v_bits).reshape(B, H, N, D).to(value_states_full.dtype)

            # Preserve window + topk in bf16
            if N > window_size:
                k_deq[:, :, -window_size:, :] = key_states_full[:, :, -window_size:, :]
                v_deq[:, :, -window_size:, :] = value_states_full[:, :, -window_size:, :]
                if d_cfg["vert"]:
                    prefix_len = N - window_size
                    vidx = compute_vertical_indices(query_states, key_states_full, window_size, top_ratio)
                    k_deq[:, :, :prefix_len, :].scatter_(
                        2, vidx.unsqueeze(-1).expand(-1, -1, -1, D),
                        key_states_full[:, :, :prefix_len, :].gather(2, vidx.unsqueeze(-1).expand(-1, -1, -1, D)))
                    v_deq[:, :, :prefix_len, :].scatter_(
                        2, vidx.unsqueeze(-1).expand(-1, -1, -1, D),
                        value_states_full[:, :, :prefix_len, :].gather(2, vidx.unsqueeze(-1).expand(-1, -1, -1, D)))

            sm_scale = query_states.shape[-1] ** -0.5
            scores = (query_states @ k_deq.transpose(-1, -2)) * sm_scale
            attn_output = torch.softmax(scores, dim=-1) @ v_deq
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode in _D_STAGE_MODES and q_len > 1:
            d_cfg = _D_STAGE_MODES[mode]
            kv_repeat = infer_kv_repeat(self)

            if d_cfg["prefill_kernel"] == "vquant_vert":
                # HYBRID_ATTN_PREFILL_KERNEL=fp8 switches QK to FP8 E4M3 on H100
                import os as _os
                _prefill_kernel = _os.environ.get(
                    "HYBRID_ATTN_PREFILL_KERNEL", "int8"
                ).lower()
                vertical_indices = compute_vertical_indices(
                    query_states=query_states, key_states=key_states,
                    window_size=window_size, top_ratio=top_ratio,
                    kv_repeat=kv_repeat,
                )
                if _prefill_kernel == "fp8":
                    from src.ops.sage_unified_fp8 import (
                        sage_vertical_attention_v2_fp8, FP8_SUPPORTED,
                    )
                    if not FP8_SUPPORTED:
                        from src.ops.sage_unified import sage_vertical_attention_v2
                        attn_output = sage_vertical_attention_v2(
                            query_states, key_states, value_states,
                            vertical_indices, window_size=window_size, smooth_k=True,
                        )
                    else:
                        attn_output = sage_vertical_attention_v2_fp8(
                            query_states, key_states, value_states,
                            vertical_indices, window_size=window_size, smooth_k=True,
                        )
                else:
                    from src.ops.sage_unified import sage_vertical_attention_v2
                    attn_output = sage_vertical_attention_v2(
                        query_states, key_states, value_states,
                        vertical_indices, window_size=window_size, smooth_k=True,
                    )
            else:
                key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
                value_states_full = repeat_interleave_manual(value_states, kv_repeat, 1)
                from src.ops.v_sageattention import full_attention_fp16_triton as _fa
                attn_output = _fa(query_states, key_states_full, value_states_full,
                                  tensor_layout="HND", is_causal=True)
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "full_attention":
            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            value_states_full = repeat_interleave_manual(value_states, kv_repeat, 1)
            from src.ops.v_sageattention import full_attention_fp16_triton

            attn_output = full_attention_fp16_triton(
                query_states, key_states_full, value_states_full,
                tensor_layout="HND", is_causal=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode in ("sage_w", "sage_w0"):
            kv_repeat = infer_kv_repeat(self)
            from src.ops.sage_unified import sage_unified_attention

            w = window_size if mode == "sage_w" else 0
            attn_output = sage_unified_attention(
                query_states, key_states, value_states,
                window_size=w, smooth_k=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "vquant_vert":
            kv_repeat = infer_kv_repeat(self)
            from src.ops.sage_unified import sage_vertical_attention_v2

            vertical_indices = compute_vertical_indices(
                query_states=query_states, key_states=key_states,
                window_size=window_size, top_ratio=top_ratio,
                kv_repeat=kv_repeat,
            )
            attn_output = sage_vertical_attention_v2(
                query_states, key_states, value_states,
                vertical_indices, window_size=window_size, smooth_k=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "sparse_only":
            # MInference-style baseline: same vert+window protection as vquant_vert
            # but the rest of the prefix is masked to -inf instead of INT8-quantised.
            kv_repeat = infer_kv_repeat(self)
            from src.ops.sparse_baseline import sparse_only_attention

            vertical_indices = compute_vertical_indices(
                query_states=query_states, key_states=key_states,
                window_size=window_size, top_ratio=top_ratio,
                kv_repeat=kv_repeat,
            )
            attn_output = sparse_only_attention(
                query_states, key_states, value_states,
                vertical_indices, window_size=window_size,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        else:
            raise ValueError(f"Unsupported mode '{mode}' in patched Llama forward.")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    return patched_llama3_attention_forward


def patch_llama3_attention_forward_model(
    model: torch.nn.Module,
    *,
    mode: str = "sage_w",
    window_size: int = 256,
    top_ratio: float = 0.05,
    k_bits: Optional[int] = None,
    v_bits: Optional[int] = None,
    report_stats: bool = False,
    use_vertical_indices: bool = True,
) -> torch.nn.Module:
    """Patch all LlamaAttention modules in-place."""
    mode = validate_patch_args(mode=mode, window_size=window_size, top_ratio=top_ratio,
                               k_bits=k_bits, v_bits=v_bits)
    del report_stats
    from transformers.models.llama.modeling_llama import LlamaAttention

    llama3_forward = build_llama3_attention_forward(
        mode=mode, window_size=window_size, top_ratio=top_ratio,
        use_vertical_indices=use_vertical_indices,
    )

    patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward
            module.forward = MethodType(llama3_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError("No LlamaAttention modules were found to patch.")
    return model
