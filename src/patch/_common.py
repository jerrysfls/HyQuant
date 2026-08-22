"""Shared helpers for attention patch modules."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

PatchMode = Literal[
    "dense", "full_attention",
    "sage_w", "sage_w0", "vquant_vert", "minference_sparse", "sparse_only",
    "d_k4v4", "d_k4v4_vert", "d_k8v8",
    "d_k4v2", "d_k4v2_vert",
    "d_k2v4", "d_k2v4_vert",
    "hybrid_full", "hybrid_k4v2", "hybrid_k2v4",
]

_VALID_MODES = {
    "dense", "full_attention",
    "sage_w", "sage_w0", "vquant_vert", "minference_sparse", "sparse_only",
    "d_k4v4", "d_k4v4_vert", "d_k8v8",
    "d_k4v2", "d_k4v2_vert",
    "d_k2v4", "d_k2v4_vert",
    "hybrid_full", "hybrid_k4v2", "hybrid_k2v4",
}

_VALID_K_BITS = {2, 4, 8}
_VALID_V_BITS = {2, 4, 8}

# D-stage decode modes: KV bit widths, vertical-line retention, prefill kernel.
# prefill_kernel: "full" (bf16 baseline) or "vquant_vert" (P-stage hybrid kernel)
D_STAGE_MODES = {
    "d_k4v4":       {"k_bits": 4, "v_bits": 4, "vert": False, "prefill_kernel": "full"},
    "d_k4v4_vert":  {"k_bits": 4, "v_bits": 4, "vert": True,  "prefill_kernel": "full"},
    "d_k8v8":       {"k_bits": 8, "v_bits": 8, "vert": False, "prefill_kernel": "full"},
    "d_k4v2":       {"k_bits": 4, "v_bits": 2, "vert": False, "prefill_kernel": "full"},
    "d_k4v2_vert":  {"k_bits": 4, "v_bits": 2, "vert": True,  "prefill_kernel": "full"},
    "d_k2v4":       {"k_bits": 2, "v_bits": 4, "vert": False, "prefill_kernel": "full"},
    "d_k2v4_vert":  {"k_bits": 2, "v_bits": 4, "vert": True,  "prefill_kernel": "full"},
    "hybrid_full":  {"k_bits": 4, "v_bits": 4, "vert": True,  "prefill_kernel": "vquant_vert"},
    "hybrid_k4v2":  {"k_bits": 4, "v_bits": 2, "vert": True,  "prefill_kernel": "vquant_vert"},
    "hybrid_k2v4":  {"k_bits": 2, "v_bits": 4, "vert": True,  "prefill_kernel": "vquant_vert"},
}


def is_d_stage_mode(mode: str) -> bool:
    """True for modes that decode against a quantized D-stage KV cache."""
    return mode == "d_stage" or mode in D_STAGE_MODES


def validate_patch_args(
    mode: str,
    window_size: int,
    top_ratio: float,
    k_bits: Optional[int] = None,
    v_bits: Optional[int] = None,
) -> PatchMode:
    """Validate and normalize public patch arguments."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {sorted(_VALID_MODES)}.")
    # window_size=0 means "no sliding-window protection" (used by ablation studies
    # for sage_w0 baseline and vquant_vert with vertical-only mode).
    if window_size < 0:
        raise ValueError(f"window_size must be >= 0, got {window_size}.")
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}.")
    if mode == "d_stage":
        if k_bits not in _VALID_K_BITS:
            raise ValueError(f"k_bits must be in {sorted(_VALID_K_BITS)}, got {k_bits}.")
        if v_bits not in _VALID_V_BITS:
            raise ValueError(f"v_bits must be in {sorted(_VALID_V_BITS)}, got {v_bits}.")
    return mode  # type: ignore[return-value]


def repeat_interleave_manual(x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
    """Repeat tensor values along one axis without relying on PyTorch head helpers."""
    if repeats == 1:
        return x
    x_exp = x.unsqueeze(dim + 1)
    expanded = x_exp.expand(*x.shape[:dim], x.shape[dim], repeats, *x.shape[dim + 1 :])
    out = expanded.reshape(*x.shape[:dim], x.shape[dim] * repeats, *x.shape[dim + 1 :])
    return out.contiguous()


def infer_kv_repeat(attn_module: torch.nn.Module) -> int:
    """Infer the KV-to-Q head expansion ratio for GQA/MQA attention modules."""
    if hasattr(attn_module, "num_key_value_groups") and attn_module.num_key_value_groups is not None:
        return max(int(attn_module.num_key_value_groups), 1)
    if hasattr(attn_module, "num_heads") and hasattr(attn_module, "num_key_value_heads"):
        num_heads = int(attn_module.num_heads)
        num_key_value_heads = int(attn_module.num_key_value_heads)
        return max(num_heads // max(num_key_value_heads, 1), 1)
    return 1


def resolve_attention_interface(
    config: Any,
    eager_attention_forward: Callable[..., Any],
    *,
    output_attentions: bool = False,
) -> Callable[..., Any]:
    """Resolve HF attention backend while keeping eager as fallback."""
    implementation = getattr(config, "_attn_implementation", "eager")
    if implementation == "eager":
        return eager_attention_forward

    # Some transformers versions cannot return attention weights for sdpa.
    if implementation == "sdpa" and output_attentions:
        return eager_attention_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return eager_attention_forward

    return ALL_ATTENTION_FUNCTIONS.get(implementation, eager_attention_forward)


def pop_past_key_value_aliases(
    past_key_value: Any,
    kwargs: dict,
) -> Any:
    """Normalize cache aliases used by different model forward signatures."""
    if past_key_value is None and "past_key_values" in kwargs:
        past_key_value = kwargs.pop("past_key_values")
    if past_key_value is None and "past_key_value" in kwargs:
        past_key_value = kwargs.pop("past_key_value")
    return past_key_value


def trim_4d_attention_mask(attention_mask: Any, key_states: Any) -> Any:
    """Align 4D mask length to effective KV length after cache updates."""
    if attention_mask is not None and getattr(attention_mask, "ndim", None) == 4:
        return attention_mask[:, :, :, : key_states.shape[-2]]
    return attention_mask


def compute_vertical_indices(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    window_size: int,
    top_ratio: float,
    kv_repeat: int = 1,
) -> torch.Tensor:
    """Compute per-head important key indices for vertical attention.

    Args:
        query_states: [B, H_Q, Q, D]
        key_states:   [B, H, K, D] where H == H_Q (kv_repeat=1) OR H == H_KV (kv_repeat>1, GQA-native)
        kv_repeat:    if >1, key_states is at KV-head granularity and we compute the
                      score without materializing the kv_repeat-expanded K.
    Returns:
        Tensor [B, H_Q, top_k] with int64 key positions selected from the prefix segment.
    """
    batch, num_heads, query_len, head_dim = query_states.shape
    key_len = int(key_states.shape[2])
    prefix_len = max(0, key_len - window_size)
    if prefix_len == 0:
        return torch.empty((batch, num_heads, 0), device=query_states.device, dtype=torch.int64)

    tail_len = min(window_size, query_len)
    query_tail = query_states[:, :, -tail_len:, :]
    q_mean = query_tail.mean(dim=2, keepdim=True)        # [B, H_Q, 1, D]

    if kv_repeat <= 1 or key_states.shape[1] == num_heads:
        # non-GQA, or K already expanded to H_Q heads
        key_prefix = key_states[:, :, :prefix_len, :]
        scores_by_key = torch.matmul(q_mean, key_prefix.transpose(-1, -2)).squeeze(2)  # [B, H_Q, P]
    else:
        # GQA-native: key_states is [B, H_KV, K, D]; avoid 8x expand.
        H_KV = key_states.shape[1]
        assert num_heads == H_KV * kv_repeat, (
            f"H_Q ({num_heads}) != H_KV ({H_KV}) * kv_repeat ({kv_repeat})"
        )
        key_prefix = key_states[:, :, :prefix_len, :]                       # [B, H_KV, P, D]
        q_g = q_mean.view(batch, H_KV, kv_repeat, head_dim)                 # [B, H_KV, R, D]
        # einsum dispatches to bmm, no materialised expanded K.
        scores_by_key = torch.einsum("bhrd,bhpd->bhrp", q_g, key_prefix)    # [B, H_KV, R, P]
        scores_by_key = scores_by_key.reshape(batch, num_heads, prefix_len) # [B, H_Q, P]

    top_k = max(1, int(math.ceil(prefix_len * top_ratio)))
    top_k = min(top_k, prefix_len)
    _, topk_indices = torch.topk(scores_by_key, k=top_k, dim=-1, largest=True, sorted=True)
    return topk_indices.to(torch.int64)


def is_d_stage_cache(past_key_values: Any) -> bool:
    """Check if past_key_values is a D-stage cache (duck-typing)."""
    return hasattr(past_key_values, "get_layer") and hasattr(past_key_values, "_layers")


def handle_d_stage_cache(
    layer_idx: int,
    past_key_values: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_module: torch.nn.Module,
    sin: torch.Tensor,
    cos: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handle D-stage cache update for both prefill and decode phases.

    Returns:
        (key_states, value_states) — updated KV for prefill, or original for decode.
    """
    layer_cache = past_key_values.get_layer(layer_idx)
    q_len = query_states.shape[2]

    if q_len > 1:
        layer_cache.save_query_tail(query_states)
        return past_key_values.update(
            key_states, value_states, layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    else:
        kv_repeat = infer_kv_repeat(attn_module)
        if layer_cache.phase == "prefill":
            layer_cache.freeze(kv_repeat=kv_repeat)
        layer_cache.append_staging(key_states, value_states)
        return key_states, value_states
