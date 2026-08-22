"""GLM4 attention patch with dense/full_attention/sage_w/sage_w0/vquant_vert modes."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch

from src.patch._common import (
    PatchMode,
    compute_vertical_indices,
    infer_kv_repeat,
    pop_past_key_value_aliases,
    repeat_interleave_manual,
    resolve_attention_interface,
    trim_4d_attention_mask,
    validate_patch_args,
)


def build_glm4_attention_forward(
    *,
    mode: PatchMode,
    window_size: int,
    top_ratio: float,
    use_vertical_indices: bool = True,
):
    """Build a patched GLM4 attention forward function."""
    from transformers.models.glm4.modeling_glm4 import apply_rotary_pos_emb, eager_attention_forward

    def patched_glm4_attention_forward(
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "query_states": query_states,
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        q_len = int(query_states.shape[2])
        attention_mask = trim_4d_attention_mask(attention_mask, key_states)
        attention_interface = resolve_attention_interface(
            self.config,
            eager_attention_forward,
            output_attentions=output_attentions,
        )
        attn_weights = None

        if mode == "dense" or q_len == 1:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else float(getattr(self, "attention_dropout", 0.0)),
                scaling=getattr(self, "scaling", None),
                sliding_window=getattr(self, "sliding_window", None),
                **kwargs,
            )

        elif mode in ("sage_w", "sage_w0"):
            from src.ops.sage_unified import sage_unified_attention

            w = window_size if mode == "sage_w" else 0
            attn_output = sage_unified_attention(
                query_states,
                key_states,
                value_states,
                window_size=w,
                smooth_k=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "vquant_vert":
            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            from src.ops.sage_unified import sage_vertical_attention_v2

            vertical_indices = compute_vertical_indices(
                query_states=query_states,
                key_states=key_states_full,
                window_size=window_size,
                top_ratio=top_ratio,
            )
            attn_output = sage_vertical_attention_v2(
                query_states,
                key_states,
                value_states,
                vertical_indices,
                window_size=window_size,
                smooth_k=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "sparse_only":
            # MInference-style baseline: same vert+window protection as vquant_vert
            # but the rest of the prefix is masked to -inf instead of INT8-quantised.
            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            from src.ops.sparse_baseline import sparse_only_attention

            vertical_indices = compute_vertical_indices(
                query_states=query_states,
                key_states=key_states_full,
                window_size=window_size,
                top_ratio=top_ratio,
            )
            attn_output = sparse_only_attention(
                query_states,
                key_states,
                value_states,
                vertical_indices,
                window_size=window_size,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        elif mode == "full_attention":
            kv_repeat = infer_kv_repeat(self)
            key_states_full = repeat_interleave_manual(key_states, kv_repeat, 1)
            value_states_full = repeat_interleave_manual(value_states, kv_repeat, 1)
            from src.ops.v_sageattention import full_attention_fp16_triton

            attn_output = full_attention_fp16_triton(
                query_states,
                key_states_full,
                value_states_full,
                tensor_layout="HND",
                is_causal=True,
            )
            attn_output = attn_output.transpose(1, 2).to(hidden_states.dtype).contiguous()

        else:
            raise ValueError(f"Unsupported mode '{mode}' in patched GLM4 forward.")

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

    return patched_glm4_attention_forward


def patch_glm4_attention_forward_model(
    model: torch.nn.Module,
    *,
    mode: str = "sage_w",
    window_size: int = 256,
    top_ratio: float = 0.05,
    report_stats: bool = False,
    use_vertical_indices: bool = True,
) -> torch.nn.Module:
    """Patch all Glm4Attention modules in-place."""
    mode = validate_patch_args(mode=mode, window_size=window_size, top_ratio=top_ratio)
    del report_stats
    from transformers.models.glm4.modeling_glm4 import Glm4Attention

    glm4_forward = build_glm4_attention_forward(
        mode=mode,
        window_size=window_size,
        top_ratio=top_ratio,
        use_vertical_indices=use_vertical_indices,
    )

    patched = 0
    for module in model.modules():
        if isinstance(module, Glm4Attention):
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward
            module.forward = MethodType(glm4_forward, module)
            patched += 1

    if patched == 0:
        raise RuntimeError("No Glm4Attention modules were found to patch.")
    return model
