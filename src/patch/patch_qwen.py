"""In-place attention patch for Qwen3 models."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch

from src.patch._common import (
    D_STAGE_MODES as _D_STAGE_MODES,
    compute_vertical_indices,
    infer_kv_repeat,
    pop_past_key_value_aliases,
    repeat_interleave_manual,
    resolve_attention_interface,
    trim_4d_attention_mask,
    validate_patch_args,
)


def _is_d_stage_cache(past_key_values) -> bool:
    """Duck-type check for DStageKVCache."""
    return hasattr(past_key_values, "get_layer") and hasattr(past_key_values, "_layers")


def build_qwen3_attention_forward(
    *,
    mode: str,
    window_size: int,
    top_ratio: float,
    use_vertical_indices: bool = True,
):
    """Build a patched Qwen3 attention forward function."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

    def patched_qwen3_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        output_attentions = bool(kwargs.get("output_attentions", False))
        kwargs.pop("use_cache", None)
        kwargs.pop("position_ids", None)
        past_key_value = pop_past_key_value_aliases(past_key_value, kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        q_len = int(query_states.shape[2])
        _has_d_stage_cache = (
            mode in _D_STAGE_MODES
            and past_key_value is not None
            and _is_d_stage_cache(past_key_value)
        )

        if _has_d_stage_cache:
            layer_cache = past_key_value.get_layer(self.layer_idx)
            if q_len > 1:
                # prefill: save query tail for topk, then standard bf16 accumulation
                layer_cache.save_query_tail(query_states)
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position,
                     "query_states": query_states},
                )
            else:
                # decode: freeze on first step, then append staging
                kv_repeat = infer_kv_repeat(self)
                if layer_cache.phase == "prefill":
                    layer_cache.freeze(kv_repeat=kv_repeat)
                layer_cache.append_staging(key_states, value_states)
        else:
            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                    "query_states": query_states,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs,
                )

        if not (_has_d_stage_cache and q_len == 1):
            attention_mask = trim_4d_attention_mask(attention_mask, key_states)
        attention_interface = resolve_attention_interface(
            self.config,
            eager_attention_forward,
            output_attentions=output_attentions,
        )
        attn_weights = None

        if mode == "dense" or (q_len == 1 and mode not in _D_STAGE_MODES):
            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else float(self.attention_dropout),
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        elif q_len == 1 and mode in _D_STAGE_MODES and _has_d_stage_cache:
            layer_cache = past_key_value.get_layer(self.layer_idx)
            kv_repeat = infer_kv_repeat(self)
            attn_output = layer_cache.decode_attention(query_states, kv_repeat)
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        # no DStageKVCache attached: quantize-dequantize round trip as fallback
        elif q_len == 1 and mode in _D_STAGE_MODES:
            from src.ops.decode_d_stage import _quantize_by_bits, _dequantize_by_bits
            d_cfg = _D_STAGE_MODES[mode]
            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            value_states_full = repeat_interleave_manual(value_states, kv_repeat, 1)

            B, H, N, D = key_states_full.shape
            BH = B * H
            k_bits, v_bits = d_cfg["k_bits"], d_cfg["v_bits"]
            k_3d = key_states_full.reshape(BH, N, D)
            v_3d = value_states_full.reshape(BH, N, D)

            k_packed, k_scale = _quantize_by_bits(k_3d, k_bits)
            v_packed, v_scale = _quantize_by_bits(v_3d, v_bits)
            k_deq = _dequantize_by_bits(k_packed, k_scale, D, k_bits).reshape(B, H, N, D).to(key_states_full.dtype)
            v_deq = _dequantize_by_bits(v_packed, v_scale, D, v_bits).reshape(B, H, N, D).to(value_states_full.dtype)

            if N > window_size:
                k_deq[:, :, -window_size:, :] = key_states_full[:, :, -window_size:, :]
                v_deq[:, :, -window_size:, :] = value_states_full[:, :, -window_size:, :]
                if d_cfg["vert"]:
                    prefix_len = N - window_size
                    vidx = compute_vertical_indices(query_states, key_states_full, window_size, top_ratio)
                    gather_idx = vidx.unsqueeze(-1).expand(-1, -1, -1, D)
                    k_deq[:, :, :prefix_len, :].scatter_(
                        2, gather_idx,
                        key_states_full[:, :, :prefix_len, :].gather(2, gather_idx))
                    v_deq[:, :, :prefix_len, :].scatter_(
                        2, gather_idx,
                        value_states_full[:, :, :prefix_len, :].gather(2, gather_idx))

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

        elif mode in ("sage_w", "sage_w0"):
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
            # same vert+window protection as vquant_vert, but the rest of the
            # prefix is masked to -inf instead of INT8-quantised
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

        elif mode == "minference_sparse":
            # MInference-style binary mask (Eq. 1 of jiang2024minference):
            # vert+window kept in bf16, rest of the prefix masked to -inf
            kv_repeat = infer_kv_repeat(self)
            from src.ops.sparse_baseline import minference_sparse_attention

            vertical_indices = compute_vertical_indices(
                query_states=query_states, key_states=key_states,
                window_size=window_size, top_ratio=top_ratio,
                kv_repeat=kv_repeat,
            )
            attn_output = minference_sparse_attention(
                query_states, key_states, value_states,
                vertical_indices, window_size=window_size,
            )
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

        else:
            raise ValueError(f"Unsupported mode '{mode}' in patched Qwen3 forward.")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    return patched_qwen3_attention_forward


def patch_qwen3_attention_forward_model(
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
    """Patch all Qwen3Attention modules in-place."""
    mode = validate_patch_args(mode=mode, window_size=window_size, top_ratio=top_ratio,
                               k_bits=k_bits, v_bits=v_bits)
    del report_stats
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    qwen3_forward = build_qwen3_attention_forward(
        mode=mode, window_size=window_size, top_ratio=top_ratio,
        use_vertical_indices=use_vertical_indices,
    )

    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward
            module.forward = MethodType(qwen3_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError("No Qwen3Attention modules were found to patch.")
    return model
