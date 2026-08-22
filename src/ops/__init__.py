"""Public operator entrypoints for hybrid attention.

P-stage: sage_unified_attention (int8_pv_fp16/int8_pv_fp16_window), sage_vertical_attention_v2 (int8_pv_fp16_vertical)
         sage_fp8_attention (int8_pv_fp8), full_attention_fp16_triton (full_attention)
D-stage: DStageKVCache (per-channel quantized KV cache)
"""

from src.ops.sage_unified import (
    sage_unified_attention,
    sage_vertical_attention_v2,
)
from src.ops.sage_fp8_attention import sage_fp8_attention
from src.ops.d_stage_kv_cache import DStageKVCache
from src.ops.v_sageattention import full_attention_fp16_triton
from src.ops.decode_d_stage import fused_d_stage_decode

__all__ = [
    "sage_unified_attention",
    "sage_vertical_attention_v2",
    "sage_fp8_attention",
    "full_attention_fp16_triton",
    "fused_d_stage_decode",
    "DStageKVCache",
    "create_cache_for_mode",
]


def create_cache_for_mode(mode: str, num_layers: int, head_dim: int = 128,
                          k_bits: int = 4, v_bits: int = 4, **kwargs):
    """Create the appropriate KV cache for a given patch mode.

    Args:
        mode: Patch mode string (e.g. "d_stage", "d_k4v4").
        num_layers: Number of transformer layers.
        head_dim: Attention head dimension.
        k_bits: Key quantization bits (for d_stage mode).
        v_bits: Value quantization bits (for d_stage mode).
        **kwargs: Forwarded to cache constructor (window_size, top_ratio, etc.).

    Returns:
        Cache instance, or None for non-D-stage modes.
    """
    from src.patch._common import D_STAGE_MODES, is_d_stage_mode

    if not is_d_stage_mode(mode):
        return None

    if mode == "d_stage":
        return DStageKVCache(
            num_layers=num_layers,
            k_bits=k_bits,
            v_bits_fn=lambda i, _vb=v_bits: _vb,
            window_size=kwargs.get("window_size", 256),
            top_ratio=kwargs.get("top_ratio", 0.05),
            use_vert=True,
        )

    cfg = D_STAGE_MODES[mode]

    return DStageKVCache(
        num_layers=num_layers,
        k_bits=cfg["k_bits"],
        v_bits_fn=lambda i, _vb=cfg["v_bits"]: _vb,
        window_size=kwargs.get("window_size", 256),
        top_ratio=kwargs.get("top_ratio", 0.05),
        use_vert=cfg["vert"],
    )
