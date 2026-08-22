"""Unified int8_pv_fp16 / int8_pv_fp16_window kernel.

Structure: 3 segment calls, no runtime if/else inside loops.

int8_pv_fp16:        call_1(int8, prefix) + call_2(int8, diagonal)
int8_pv_fp16_window: call_1(int8, prefix_before_window) + call_2(fp16, window) + call_3(int8, diagonal_after_window)

When WINDOW=0, call_2 and call_3 collapse to zero iterations, giving pure int8_pv_fp16.
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from src.ops.vendor.sageattention.triton.quant_per_block import (
    per_block_int8 as _per_block_int8_triton,
)

_LOG2_E = 1.44269504


# --- Segment A: int8 QK (identical to sage inner) ---
@triton.jit
def _seg_int8(
    acc, l_i, m_i,
    q, q_scale,
    kv_len,
    K_ptrs, K_scale_ptr, V_ptrs,
    stride_kn, stride_vn,
    lo, hi,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

        mask = k_mask
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk += tl.where(mask, 0, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij

        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


# --- Segment B: fp16 QK (window region, no quantization) ---
@triton.jit
def _seg_fp16(
    acc, l_i, m_i,
    q_raw,
    kv_len,
    K_raw_ptrs, V_ptrs,
    stride_kn_raw, stride_vn,
    lo, hi,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    offs_m: tl.constexpr, offs_n: tl.constexpr,
):
    # q_raw is already pre-scaled by sm_scale * LOG2_E, no extra multiply needed
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_raw_ptrs, mask=k_mask)
        qk = tl.dot(q_raw, k).to(tl.float32)

        mask = k_mask
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        qk += tl.where(mask, 0, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij

        K_raw_ptrs += BLOCK_N * stride_kn_raw
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_unified(
    Q_int8, K_int8, V, Q_scale, K_scale,
    Q_raw, K_raw,
    Out, Lse,
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_on,
    stride_qrz, stride_qrh, stride_qrn,
    stride_krz, stride_krh, stride_krn,
    qo_len, kv_len,
    H: tl.constexpr, num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE_LOG2E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # int8 pointers
    Q_ptrs = Q_int8 + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K_int8 + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_ptrs = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    # fp16 pointers
    Q_raw_ptrs = Q_raw + (off_z * stride_qrz + off_h * stride_qrh) + offs_m[:, None] * stride_qrn + offs_k[None, :]
    K_raw_ptrs = K_raw + (off_z * stride_krz + (off_h // num_kv_groups) * stride_krh) + offs_n[None, :] * stride_krn + offs_k[:, None]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_int8 = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    q_raw = tl.load(Q_raw_ptrs, mask=offs_m[:, None] < qo_len)

    # Compute segment boundaries (all compile-time when WINDOW is constexpr)
    diag_start = start_m * BLOCK_M
    diag_end = (start_m + 1) * BLOCK_M

    if WINDOW > 0:
        # Window covers [window_lo, diag_end), aligned to BLOCK_N
        window_lo_raw = diag_start - (WINDOW - 1)
        window_lo = tl.maximum(window_lo_raw, 0)
        window_lo_aligned = (window_lo // BLOCK_N) * BLOCK_N
    else:
        window_lo_aligned = diag_start

    # === Segment 1: int8 prefix [0, window_lo_aligned) — no causal mask ===
    prefix_end = window_lo_aligned
    K_ptrs_s1 = K_ptrs
    K_scale_s1 = K_scale_ptr
    V_ptrs_s1 = V_ptrs
    acc, l_i, m_i = _seg_int8(
        acc, l_i, m_i,
        q_int8, q_scale,
        kv_len,
        K_ptrs_s1, K_scale_s1, V_ptrs_s1,
        stride_kn, stride_vn,
        0, prefix_end,
        BLOCK_M, HEAD_DIM, BLOCK_N,
        False,  # no causal mask in prefix
        offs_m, offs_n,
    )

    if WINDOW > 0:
        # === Segment 2: fp16 window [window_lo_aligned, diag_end) — with causal mask ===
        K_raw_s2 = K_raw_ptrs + window_lo_aligned * stride_krn
        V_ptrs_s2 = V_ptrs + window_lo_aligned * stride_vn
        acc, l_i, m_i = _seg_fp16(
            acc, l_i, m_i,
            q_raw,
            kv_len,
            K_raw_s2, V_ptrs_s2,
            stride_krn, stride_vn,
            window_lo_aligned, diag_end,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            True,  # causal mask in window+diagonal
            offs_m, offs_n,
        )
    else:
        # === Pure sage: int8 diagonal [diag_start, diag_end) — with causal mask ===
        K_ptrs_s2 = K_ptrs + diag_start * stride_kn
        K_scale_s2 = K_scale_ptr + diag_start // BLOCK_N
        V_ptrs_s2 = V_ptrs + diag_start * stride_vn
        acc, l_i, m_i = _seg_int8(
            acc, l_i, m_i,
            q_int8, q_scale,
            kv_len,
            K_ptrs_s2, K_scale_s2, V_ptrs_s2,
            stride_kn, stride_vn,
            diag_start, diag_end,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            True,  # causal mask
            offs_m, offs_n,
        )

    acc = acc / l_i[:, None]
    tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def sage_unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int = 0,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
) -> torch.Tensor:
    """Unified int8_pv_fp16 / int8_pv_fp16_window attention.

    Args:
        window_size: 0 = int8_pv_fp16 (all int8 QK), >0 = int8_pv_fp16_window
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    dtype = q.dtype
    B, H_q, N, D = q.shape
    _, H_kv, _, _ = k.shape
    num_kv_groups = H_q // H_kv

    if sm_scale is None:
        sm_scale = D**-0.5
    sm_scale_log2e = sm_scale * _LOG2_E

    # Key smoothing
    km = k.mean(dim=2, keepdim=True) if smooth_k else None

    # Quantize Q/K (with smoothing applied to K internally)
    q_int8, q_scale, k_int8, k_scale = _per_block_int8_triton(
        q, k, km=km, sm_scale=sm_scale, tensor_layout="HND",
    )

    # Raw Q/K for window path (with smoothing applied to K)
    k_smoothed = k - km if km is not None else k
    q_scaled = (q * sm_scale_log2e).to(torch.float16) if dtype != torch.float16 else q * sm_scale_log2e
    k_raw = k_smoothed.to(torch.float16) if dtype != torch.float16 else k_smoothed

    # Expand K for GQA
    if num_kv_groups > 1:
        k_int8 = k_int8.repeat_interleave(num_kv_groups, dim=1)
        k_scale_expanded = q_scale.new_zeros(B, H_q, k_scale.shape[2])
        for g in range(num_kv_groups):
            k_scale_expanded[:, g::num_kv_groups, :] = k_scale
        k_scale = k_scale_expanded
        v = v.repeat_interleave(num_kv_groups, dim=1)
        k_raw = k_raw.repeat_interleave(num_kv_groups, dim=1)

    v_fp16 = v.to(torch.float16) if dtype != torch.float16 else v

    out = torch.empty(q_int8.shape, dtype=dtype, device=q.device)

    BLOCK_M = 128
    BLOCK_N = 64
    stage = 3

    lse = torch.empty((0,), dtype=torch.float32, device="cpu")
    grid = (triton.cdiv(N, BLOCK_M), H_q, B)

    _attn_fwd_unified[grid](
        q_int8, k_int8, v_fp16, q_scale, k_scale,
        q_scaled, k_raw,
        out, lse,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2),
        v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        q_scaled.stride(0), q_scaled.stride(1), q_scaled.stride(2),
        k_raw.stride(0), k_raw.stride(1), k_raw.stride(2),
        N, N,
        H=H_q, num_kv_groups=1,  # Already expanded
        HEAD_DIM=D,
        SM_SCALE_LOG2E=sm_scale_log2e,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        STAGE=stage,
        RETURN_LSE=False,
        WINDOW=window_size,
        num_warps=4 if D == 64 else 8,
        num_stages=4,
    )
    return out


# ============================================================
# Vertical attention: KV reordered into [rest | vertical | window]
# ============================================================

@triton.jit
def _seg_int8_counted(
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
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        abs_n = start_n + offs_n
        k_mask = abs_n[None, :] < total_len
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

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
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _seg_fp16_counted(
    acc, l_i, m_i,
    q_raw,
    K_ptrs, V_ptrs,
    stride_kn, stride_vn,
    n_valid,
    total_len,
    lo, hi,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    offs_n: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        abs_n = start_n + offs_n
        k_mask = abs_n[None, :] < total_len
        k = tl.load(K_ptrs, mask=k_mask)
        qk = tl.dot(q_raw, k).to(tl.float32)

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
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_vertical_v2(
    Q_int8, K_rest_int8, V_rest, Q_scale, K_rest_scale,
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

    q_int8 = tl.load(
        Q_int8 + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :],
        mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale + q_scale_offset + start_m)
    q_raw = tl.load(
        Q_raw + (off_z * stride_qrz + off_h * stride_qrh) + offs_m[:, None] * stride_qrn + offs_k[None, :],
        mask=offs_m[:, None] < qo_len)

    # rest int8 [BH, R, D]
    Kr_ptrs = K_rest_int8 + bh * stride_krz + offs_n[None, :] * stride_krn + offs_k[:, None]
    Kr_scale_ptr = K_rest_scale + bh * tl.cdiv(R_len, BLOCK_N)
    Vr_ptrs = V_rest + bh * stride_vrz + offs_n[:, None] * stride_vrn + offs_k[None, :]

    # vert fp16 [BH, V, D]
    Kv_ptrs = K_vert + bh * stride_kvz + offs_n[None, :] * stride_kvn + offs_k[:, None]
    Vv_ptrs = V_vert + bh * stride_vvz + offs_n[:, None] * stride_vvn + offs_k[None, :]

    # window fp16 [B, H_q, N, D]
    Kw_ptrs = K_win + (off_z * stride_kwz + off_h * stride_kwh) + offs_n[None, :] * stride_kwn + offs_k[:, None]
    Vw_ptrs = V_win + (off_z * stride_vwz + off_h * stride_vwh) + offs_n[:, None] * stride_vwn + offs_k[None, :]

    nr_valid = tl.load(N_rest_valid + bh * stride_nrz + offs_m * stride_nrn, mask=offs_m < qo_len, other=0).to(tl.int32)
    nv_valid = tl.load(N_vert_valid + bh * stride_nvz + offs_m * stride_nvn, mask=offs_m < qo_len, other=0).to(tl.int32)

    O_ptrs = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.full([BLOCK_M], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    nr_max = tl.max(nr_valid, axis=0)
    nv_max = tl.max(nv_valid, axis=0)
    nr_hi = ((nr_max + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    nv_hi = ((nv_max + BLOCK_N - 1) // BLOCK_N) * BLOCK_N

    # Seg 1: int8 rest
    acc, l_i, m_i = _seg_int8_counted(
        acc, l_i, m_i, q_int8, q_scale,
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


def reorder_kv_for_vertical(k, v, vertical_indices, window_size):
    """Reorder prefix KV into [rest | vertical], sorted by original position."""
    BH, N, D = k.shape
    B, H, VLEN = vertical_indices.shape
    device = k.device
    prefix_len = max(0, N - window_size)
    if prefix_len == 0:
        empty_kv = torch.zeros(BH, 0, D, device=device, dtype=k.dtype)
        empty_n = torch.zeros(BH, N, device=device, dtype=torch.int32)
        return empty_kv, empty_kv, empty_kv, empty_kv, empty_n, empty_n, 0, 0

    vid = vertical_indices.reshape(BH, VLEN).to(torch.int64).clamp(0, prefix_len - 1)
    vid_sorted, _ = torch.sort(vid, dim=1)

    # Build rest mask
    vert_mask = torch.zeros(BH, prefix_len, device=device, dtype=torch.bool)
    vert_mask.scatter_(1, vid_sorted.clamp(0, prefix_len - 1), True)
    rest_mask = ~vert_mask

    rest_counts = rest_mask.sum(dim=1)
    R = int(rest_counts.max().item())
    V = VLEN

    # stable argsort of -mask puts True positions first, giving compacted indices
    sorted_indices = rest_mask.to(torch.int32).mul(-1).argsort(dim=1, stable=True)
    rest_indices = sorted_indices[:, :R].to(torch.int64)
    # Re-sort rest_indices by position value (so they're in original order)
    rest_indices, _ = rest_indices.sort(dim=1)

    # Gather KV
    g_rest = rest_indices.unsqueeze(-1).expand(-1, -1, D).clamp(0, N - 1)
    k_rest = k.gather(1, g_rest)
    v_rest = v.gather(1, g_rest)

    g_vert = vid_sorted.unsqueeze(-1).expand(-1, -1, D).clamp(0, N - 1)
    k_vert = k.gather(1, g_vert)
    v_vert = v.gather(1, g_vert)

    # Per-query valid counts — aligned to kernel block boundaries to avoid overlap with window seg
    BLOCK_M = 128
    BLOCK_N = 64
    pos = torch.arange(N, device=device, dtype=torch.int64)
    block_starts = (pos // BLOCK_M) * BLOCK_M
    window_lo_aligned = ((block_starts - (window_size - 1)).clamp(0) // BLOCK_N) * BLOCK_N
    n_rest_valid = torch.searchsorted(rest_indices, window_lo_aligned.unsqueeze(0).expand(BH, -1).contiguous()).to(torch.int32)
    n_vert_valid = torch.searchsorted(vid_sorted, window_lo_aligned.unsqueeze(0).expand(BH, -1).contiguous()).to(torch.int32)

    return k_rest.contiguous(), v_rest.contiguous(), k_vert.contiguous(), v_vert.contiguous(), \
           n_rest_valid.contiguous(), n_vert_valid.contiguous(), R, V


def sage_vertical_attention_v2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    vertical_indices: torch.Tensor,
    *, window_size: int = 256, sm_scale: Optional[float] = None, smooth_k: bool = True,
) -> torch.Tensor:
    """int8_pv_fp16_vertical: KV reorder + count-based causal."""
    dtype = q.dtype
    B, H_q, N, D = q.shape
    _, H_kv, _, _ = k.shape
    num_kv_groups = H_q // H_kv

    if sm_scale is None:
        sm_scale = D**-0.5
    sm_scale_log2e = sm_scale * _LOG2_E

    km = k.mean(dim=2, keepdim=True) if smooth_k else None
    q_int8, q_scale, _, _ = _per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout="HND")

    k_smoothed = k - km if km is not None else k
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

    # Quantize rest K
    if R > 0:
        k_rest_4d = k_rest.reshape(B, H_q, R, D)
        q_dummy = torch.zeros(B, H_q, 1, D, device=q.device, dtype=q.dtype)
        _, _, k_rest_int8, k_rest_scale = _per_block_int8_triton(
            q_dummy, k_rest_4d, km=None, sm_scale=1.0, tensor_layout="HND")
        k_rest_int8 = k_rest_int8.reshape(BH, R, D).contiguous()
        k_rest_scale = k_rest_scale.reshape(BH, -1).contiguous()
    else:
        k_rest_int8 = torch.zeros(BH, 0, D, device=q.device, dtype=torch.int8)
        k_rest_scale = torch.zeros(BH, 0, device=q.device, dtype=torch.float32)

    out = torch.empty(q_int8.shape, dtype=dtype, device=q.device)
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(N, BLOCK_M), H_q, B)

    _attn_fwd_vertical_v2[grid](
        q_int8, k_rest_int8, v_rest, q_scale, k_rest_scale,
        q_scaled, k_vert, v_vert, k_raw, v_fp16,
        n_rest_valid, n_vert_valid,
        out,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2),
        k_rest_int8.stride(0), k_rest_int8.stride(1) if R > 0 else 1,
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

