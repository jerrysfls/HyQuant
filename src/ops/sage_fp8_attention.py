"""SageAttention INT8 QK + FP8 PV attention (CUDA SM90).

Wraps the vendored ``sageattn_qk_int8_pv_fp8_cuda_sm90`` with an interface
consistent with the other operators in this package.  Requires an SM90 GPU
(e.g. H100) and CUDA 12.3+.  The CUDA kernels are JIT-compiled on first use.
"""

from __future__ import annotations

from typing import Optional

import torch


def sage_fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """INT8 QK + FP8 PV attention via SageAttention CUDA SM90 kernel.

    Args:
        q: Query tensor ``[B, H_q, N, D]`` (HND layout).
        k: Key tensor ``[B, H_kv, N, D]`` (HND layout).
        v: Value tensor ``[B, H_kv, N, D]`` (HND layout).
        is_causal: Apply causal mask.
        sm_scale: Softmax scale; defaults to ``1/sqrt(D)``.
        smooth_k: Centre keys before quantisation (reduces outlier impact).

    Returns:
        Output tensor ``[B, H_q, N, D]`` (HND layout), same dtype as *q*.
    """
    from src.ops.vendor.sageattention.core_sm90 import sageattn_qk_int8_pv_fp8_cuda_sm90

    return sageattn_qk_int8_pv_fp8_cuda_sm90(
        q, k, v,
        tensor_layout="HND",
        is_causal=is_causal,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
    )
