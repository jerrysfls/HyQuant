"""FP8 (E4M3) variant of ``sage_vertical_attention_v2``.

Mirrors :mod:`src.ops.sage_unified.sage_vertical_attention_v2` but quantises
Q and the rest-segment K to FP8 E4M3 per-block instead of INT8. E4M3's wider
dynamic range (max 448 vs int8's 127) is friendlier to attention outliers,
and Hopper fp8 mma doubles QK throughput; the per-block scale design is
identical to the int8 path, only the mma dtype changes.

Loaded lazily: when the running Triton/PyTorch build has no FP8 dtypes,
FP8_SUPPORTED is False and the patch layer falls back to the int8 path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Capability detection (so import never crashes on older stacks)
# ---------------------------------------------------------------------------

def _fp8_dtypes():
    """Return (triton_fp8_dtype, torch_fp8_dtype) or (None, None) if unavailable."""
    tl_fp8 = None
    for name in ("float8e4nv", "float8_e4m3fn", "float8e4b15"):
        if hasattr(tl, name):
            tl_fp8 = getattr(tl, name)
            break
    pt_fp8 = None
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        if hasattr(torch, name):
            pt_fp8 = getattr(torch, name)
            break
    return tl_fp8, pt_fp8


_TL_FP8, _PT_FP8 = _fp8_dtypes()
FP8_SUPPORTED: bool = (_TL_FP8 is not None) and (_PT_FP8 is not None)
FP8_MAX = 448.0   # E4M3 max representable value
_LOG2_E = 1.44269504


# ---------------------------------------------------------------------------
# Per-block FP8 quantisation
# ---------------------------------------------------------------------------

@triton.jit
def _quant_per_block_fp8_kernel(
    Input, Output, Scale, L,
    stride_iz, stride_ih, stride_in,
    stride_oz, stride_oh, stride_on,
    stride_sz, stride_sh,
    sm_scale,
    FP8_MAX_VAL: tl.constexpr,
    C: tl.constexpr, BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = (Input + off_b * stride_iz + off_h * stride_ih
                  + offs_n[:, None] * stride_in + offs_k[None, :])
    output_ptrs = (Output + off_b * stride_oz + off_h * stride_oh
                   + offs_n[:, None] * stride_on + offs_k[None, :])
    scale_ptr = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L).to(tl.float32)
    x *= sm_scale
    absmax = tl.max(tl.abs(x))
    scale = absmax / FP8_MAX_VAL
    # Guard against divide-by-zero on all-zero blocks
    scale = tl.where(scale > 0, scale, 1.0)
    x_q = x / scale
    x_q = tl.where(x_q > FP8_MAX_VAL, FP8_MAX_VAL, x_q)
    x_q = tl.where(x_q < -FP8_MAX_VAL, -FP8_MAX_VAL, x_q)

    tl.store(output_ptrs, x_q.to(Output.type.element_ty),
             mask=offs_n[:, None] < L)
    tl.store(scale_ptr, scale)


def per_block_fp8_e4m3(q, k, km=None, BLKQ=128, BLKK=64,
                       sm_scale: Optional[float] = None,
                       tensor_layout: str = "HND"):
    """Quantise Q and K to FP8 E4M3 with one fp32 scale per block.

    Mirrors ``vendor.sageattention.triton.quant_per_block.per_block_int8``.
    """
    assert FP8_SUPPORTED, "FP8 not supported by current Triton/torch build"
    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        s_qi = (q.stride(0), q.stride(1), q.stride(2))
        s_ki = (k.stride(0), k.stride(1), k.stride(2))
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        s_qi = (q.stride(0), q.stride(2), q.stride(1))
        s_ki = (k.stride(0), k.stride(2), k.stride(1))
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_fp8 = torch.empty(q.shape, dtype=_PT_FP8, device=q.device)
    k_fp8 = torch.empty(k.shape, dtype=_PT_FP8, device=k.device)
    s_qo = (q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2)) if tensor_layout == "HND" \
        else (q_fp8.stride(0), q_fp8.stride(2), q_fp8.stride(1))
    s_ko = (k_fp8.stride(0), k_fp8.stride(1), k_fp8.stride(2)) if tensor_layout == "HND" \
        else (k_fp8.stride(0), k_fp8.stride(2), k_fp8.stride(1))

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ),
                          device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK),
                          device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    # Bake sm_scale * log2(e) into Q (same as sage int8 path)
    q_sm_scale = sm_scale * _LOG2_E

    grid_q = ((qo_len + BLKQ - 1) // BLKQ, h_qo, b)
    _quant_per_block_fp8_kernel[grid_q](
        q, q_fp8, q_scale, qo_len,
        s_qi[0], s_qi[1], s_qi[2],
        s_qo[0], s_qo[1], s_qo[2],
        q_scale.stride(0), q_scale.stride(1),
        sm_scale=q_sm_scale,
        FP8_MAX_VAL=FP8_MAX,
        C=head_dim, BLK=BLKQ,
    )
    grid_k = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    _quant_per_block_fp8_kernel[grid_k](
        k, k_fp8, k_scale, kv_len,
        s_ki[0], s_ki[1], s_ki[2],
        s_ko[0], s_ko[1], s_ko[2],
        k_scale.stride(0), k_scale.stride(1),
        sm_scale=1.0,
        FP8_MAX_VAL=FP8_MAX,
        C=head_dim, BLK=BLKK,
    )
    return q_fp8, q_scale, k_fp8, k_scale


# ---------------------------------------------------------------------------
# FP8 segment (mirror of _seg_int8_counted with tl.dot dispatching to FP8 mma)
# ---------------------------------------------------------------------------

@triton.jit
def _seg_fp8_counted(
    acc, l_i, m_i,
    q, q_scale,
    K_ptrs, K_scale_ptr, V_ptrs,
    stride_kn, stride_vn,
    n_valid,
    total_len,
    lo, hi,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    offs_n: tl.constexpr,
):
    """FP8 E4M3 K-tile loop. q/k are fp8e4nv tiles; mma output is fp32."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        abs_n = start_n + offs_n
        k_mask = abs_n[None, :] < total_len

        k = tl.load(K_ptrs, mask=k_mask)   # fp8 [HEAD_DIM, BLOCK_N]
        k_scale = tl.load(K_scale_ptr)

        # tl.dot with both fp8 inputs → fp32 acc (Hopper FP8 Tensor Core)
        qk = tl.dot(q, k, out_dtype=tl.float32) * (q_scale * k_scale)

        mask = k_mask & (abs_n[None, :] < n_valid[:, None])
        qk += tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=abs_n[:, None] < total_len)
        p_fp16 = p.to(tl.float16)
        acc += tl.dot(p_fp16, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


# Re-use the fp16 segments from the int8 sibling — they don't touch FP8.
from src.ops.sage_unified import _seg_fp16, _seg_fp16_counted, reorder_kv_for_vertical


@triton.jit
def _attn_fwd_vertical_v2_fp8(
    Q_fp8, K_rest_fp8, V_rest, Q_scale, K_rest_scale,
    Q_raw, K_vert, V_vert, K_win, V_win,
    N_rest_valid, N_vert_valid,
    Out,
    stride_qz, stride_qh, stride_qn,
    stride_krz, stride_krn,
    stride_vrz, stride_vrn,
    stride_qrz, stride_qrh, stride_qrn,
    stride_kvz, stride_kvn,
    stride_vvz, stride_vvn,
    stride_kwz, stride_kwh, stride_kwn,
    stride_vwz, stride_vwh, stride_vwn,
    stride_nrz, stride_nrn,
    stride_nvz, stride_nvn,
    stride_oz, stride_oh, stride_on,
    qo_len, kv_len, R_len, V_len,
    H: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    WINDOW: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    bh = off_z * H + off_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)

    q_fp8 = tl.load(
        Q_fp8 + (off_z * stride_qz + off_h * stride_qh)
        + offs_m[:, None] * stride_qn + offs_k[None, :],
        mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale + q_scale_offset + start_m)
    q_raw = tl.load(
        Q_raw + (off_z * stride_qrz + off_h * stride_qrh)
        + offs_m[:, None] * stride_qrn + offs_k[None, :],
        mask=offs_m[:, None] < qo_len)

    Kr_ptrs = K_rest_fp8 + bh * stride_krz + offs_n[None, :] * stride_krn + offs_k[:, None]
    Kr_scale_ptr = K_rest_scale + bh * tl.cdiv(R_len, BLOCK_N)
    Vr_ptrs = V_rest + bh * stride_vrz + offs_n[:, None] * stride_vrn + offs_k[None, :]

    Kv_ptrs = K_vert + bh * stride_kvz + offs_n[None, :] * stride_kvn + offs_k[:, None]
    Vv_ptrs = V_vert + bh * stride_vvz + offs_n[:, None] * stride_vvn + offs_k[None, :]

    Kw_ptrs = K_win + (off_z * stride_kwz + off_h * stride_kwh) + offs_n[None, :] * stride_kwn + offs_k[:, None]
    Vw_ptrs = V_win + (off_z * stride_vwz + off_h * stride_vwh) + offs_n[:, None] * stride_vwn + offs_k[None, :]

    nr_valid = tl.load(N_rest_valid + bh * stride_nrz + offs_m * stride_nrn,
                       mask=offs_m < qo_len, other=0).to(tl.int32)
    nv_valid = tl.load(N_vert_valid + bh * stride_nvz + offs_m * stride_nvn,
                       mask=offs_m < qo_len, other=0).to(tl.int32)

    O_ptrs = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.full([BLOCK_M], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    nr_max = tl.max(nr_valid, axis=0)
    nv_max = tl.max(nv_valid, axis=0)
    nr_hi = ((nr_max + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    nv_hi = ((nv_max + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    # Seg 1: fp8 rest
    acc, l_i, m_i = _seg_fp8_counted(
        acc, l_i, m_i, q_fp8, q_scale,
        Kr_ptrs, Kr_scale_ptr, Vr_ptrs, stride_krn, stride_vrn,
        nr_valid, R_len, 0, nr_hi,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_n,
    )

    # Seg 2: fp16 vert
    acc, l_i, m_i = _seg_fp16_counted(
        acc, l_i, m_i, q_raw,
        Kv_ptrs, Vv_ptrs, stride_kvn, stride_vvn,
        nv_valid, V_len, 0, nv_hi,
        BLOCK_M, HEAD_DIM, BLOCK_N, offs_n,
    )

    # Seg 3: fp16 window (causal)
    diag_end = (start_m + 1) * BLOCK_M
    window_lo = tl.maximum(start_m * BLOCK_M - (WINDOW - 1), 0)
    window_lo_aligned = (window_lo // BLOCK_N) * BLOCK_N
    Kw_s3 = Kw_ptrs + window_lo_aligned * stride_kwn
    Vw_s3 = Vw_ptrs + window_lo_aligned * stride_vwn
    acc, l_i, m_i = _seg_fp16(
        acc, l_i, m_i, q_raw, kv_len,
        Kw_s3, Vw_s3, stride_kwn, stride_vwn,
        window_lo_aligned, diag_end,
        BLOCK_M, HEAD_DIM, BLOCK_N, True, offs_m, offs_n,
    )

    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def sage_vertical_attention_v2_fp8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    vertical_indices: torch.Tensor,
    *, window_size: int = 256,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """FP8-QK version of ``sage_vertical_attention_v2``.

    Same algorithm: KV reorder + count-based causal + 3 segments
    (rest fp8 → vert fp16 → window fp16) under one online softmax.
    """
    if not FP8_SUPPORTED:
        raise RuntimeError(
            "FP8 (E4M3) dtype not available in this Triton/torch build."
        )

    dtype = q.dtype
    B, H_q, N, D = q.shape
    _, H_kv, _, _ = k.shape
    num_kv_groups = H_q // H_kv

    if sm_scale is None:
        sm_scale = D ** -0.5

    km = k.mean(dim=2, keepdim=True) if smooth_k else None
    q_fp8, q_scale, _, _ = per_block_fp8_e4m3(
        q, k, km=km, sm_scale=sm_scale, tensor_layout="HND",
    )

    k_smoothed = (k - km) if km is not None else k
    sm_scale_log2e = sm_scale * _LOG2_E
    q_scaled = (q * sm_scale_log2e).to(torch.float16) if dtype != torch.float16 else q * sm_scale_log2e

    if num_kv_groups > 1:
        v = v.repeat_interleave(num_kv_groups, dim=1)
        k_smoothed = k_smoothed.repeat_interleave(num_kv_groups, dim=1)

    k_raw = k_smoothed.to(torch.float16) if dtype != torch.float16 else k_smoothed
    v_fp16 = v.to(torch.float16) if dtype != torch.float16 else v

    BH = B * H_q
    k_3d = k_raw.reshape(BH, N, D).contiguous()
    v_3d = v_fp16.reshape(BH, N, D).contiguous()

    k_rest, v_rest, k_vert, v_vert, n_rest_valid, n_vert_valid, R, V = \
        reorder_kv_for_vertical(k_3d, v_3d, vertical_indices, window_size)

    # Quantise rest K to FP8
    if R > 0:
        k_rest_4d = k_rest.reshape(B, H_q, R, D)
        q_dummy = torch.zeros(B, H_q, 1, D, device=q.device, dtype=q.dtype)
        _, _, k_rest_fp8, k_rest_scale = per_block_fp8_e4m3(
            q_dummy, k_rest_4d, km=None, sm_scale=1.0, tensor_layout="HND",
        )
        k_rest_fp8 = k_rest_fp8.reshape(BH, R, D).contiguous()
        k_rest_scale = k_rest_scale.reshape(BH, -1).contiguous()
    else:
        k_rest_fp8 = torch.zeros(BH, 0, D, device=q.device, dtype=_PT_FP8)
        k_rest_scale = torch.zeros(BH, 0, device=q.device, dtype=torch.float32)

    out = torch.empty(q_fp8.shape, dtype=dtype, device=q.device)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N, BLOCK_M), H_q, B)

    _attn_fwd_vertical_v2_fp8[grid](
        q_fp8, k_rest_fp8, v_rest, q_scale, k_rest_scale,
        q_scaled, k_vert, v_vert, k_raw, v_fp16,
        n_rest_valid, n_vert_valid,
        out,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_rest_fp8.stride(0), k_rest_fp8.stride(1) if R > 0 else 1,
        v_rest.stride(0), v_rest.stride(1) if R > 0 else 1,
        q_scaled.stride(0), q_scaled.stride(1), q_scaled.stride(2),
        k_vert.stride(0), k_vert.stride(1) if V > 0 else 1,
        v_vert.stride(0), v_vert.stride(1) if V > 0 else 1,
        k_raw.stride(0), k_raw.stride(1), k_raw.stride(2),
        v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2),
        n_rest_valid.stride(0), n_rest_valid.stride(1),
        n_vert_valid.stride(0), n_vert_valid.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        N, N, R, V,
        H=H_q, HEAD_DIM=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, WINDOW=window_size,
        num_warps=8, num_stages=4,
    )
    return out


__all__ = [
    "sage_vertical_attention_v2_fp8",
    "per_block_fp8_e4m3",
    "FP8_SUPPORTED",
]
