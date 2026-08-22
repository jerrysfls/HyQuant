"""Full attention baseline (fp16 Triton).

Provides `full_attention_fp16_triton` — bf16/fp16 causal attention with no quantization.
Delegates to `sage_unified_attention` with WINDOW=seq_len (entire sequence as fp16 window).
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def full_attention_fp16_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = True,
    sm_scale: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Full causal attention in fp16/bf16 (Triton). No quantization. Precision baseline.

    Args:
        q, k, v: [B, H, N, D] (HND) or [B, N, H, D] (NHD)
        tensor_layout: "HND" or "NHD"
        is_causal: Must be True
        sm_scale: Attention scale (default: D^-0.5)
    Returns:
        Output tensor with same layout as input.
    """
    del kwargs
    from src.ops.sage_unified import sage_unified_attention

    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    seq_len = int(q.shape[2])
    out = sage_unified_attention(
        q, k, v,
        window_size=seq_len,
        sm_scale=sm_scale,
        smooth_k=False,
    )

    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out


__all__ = ["full_attention_fp16_triton"]
