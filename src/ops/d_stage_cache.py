"""D-stage KV Cache: mixed-precision decode attention (optimised fused kernel).

Two kernel paths:

A) `d_stage_decode_attention` (legacy)
   - Dequantizes prefix in Python (PyTorch ops, hundreds of MB temp HBM)
   - Calls 4-segment bf16 Triton kernel

B) `fused_decode_attention_k4v4` (optimised)
   - On-chip int4 dequant inside the kernel — no HBM materialisation of bf16 K/V
   - GQA-aware: K/V stored at H_KV granularity, kernel reads via h_kv = h_q // kv_repeat
   - **Even/odd channel split**: packed bytes loaded contiguously [N, D//2]
     (no duplicate-byte loads); even channels come from the high nibble, odd
     channels from the low nibble. Two accumulators acc_even / acc_odd.
   - 4 segments share one online-softmax state (m_i, l_i, acc_*)
   - `triton.autotune` over BLOCK_KV / num_warps / num_stages
   - Software pipelining via `num_stages>=3` to hide HBM latency
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from src.ops.decode_d_stage import (
    GROUP_SIZE,
    quantize_k_int4_channel,
    quantize_v_int2_channel,
    dequantize_k_int4_channel,
    dequantize_v_int2_channel,
)

_LOG2_E = 1.44269504


# ============================================================
# Legacy: 4-segment kernel (Python-side dequant for the rest segment)
# Kept for K_BITS != 4 fallback in DStageLayerCache.decode_attention.
# ============================================================

@triton.jit
def _bf16_segment(
    acc_even, acc_odd, l_i, m_i,
    q_even, q_odd,
    K_base, V_base,
    stride_kn, stride_vn,
    stride_kd, stride_vd,
    N_seg,
    sm_log2e,
    D_HALF: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Process a bf16 segment (topk / staging / window). Stride-2 loads on
    the D dimension so we share q_even/q_odd state with the quantised path."""
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_d_half = tl.arange(0, D_HALF)
    offs_d_even = 2 * offs_d_half
    offs_d_odd = 2 * offs_d_half + 1

    for start in range(0, N_seg, BLOCK_KV):
        ns = start + offs_kv
        mask_n = ns < N_seg

        k_even = tl.load(
            K_base + ns[:, None] * stride_kn + offs_d_even[None, :] * stride_kd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        k_odd = tl.load(
            K_base + ns[:, None] * stride_kn + offs_d_odd[None, :] * stride_kd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        qk = (tl.sum(q_even[None, :] * k_even, axis=1)
              + tl.sum(q_odd[None, :] * k_odd, axis=1)) * sm_log2e
        qk = tl.where(mask_n, qk, -1.0e6)

        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        p = tl.math.exp2(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        v_even = tl.load(
            V_base + ns[:, None] * stride_vn + offs_d_even[None, :] * stride_vd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        v_odd = tl.load(
            V_base + ns[:, None] * stride_vn + offs_d_odd[None, :] * stride_vd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        acc_even += tl.sum(p[:, None] * v_even, axis=0)
        acc_odd += tl.sum(p[:, None] * v_odd, axis=0)
        m_i = m_ij

    return acc_even, acc_odd, l_i, m_i


@triton.jit
def _bf16_segment_full(
    acc, l_i, m_i,
    q,
    K_ptr, V_ptr,
    stride_kn, stride_vn,
    N_seg,
    sm_log2e,
    D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """[Legacy] bf16 segment with full-D accumulator (used by legacy kernel only)."""
    offs_d = tl.arange(0, D)
    offs_kv = tl.arange(0, BLOCK_KV)

    for start in range(0, N_seg, BLOCK_KV):
        mask_n = (start + offs_kv) < N_seg
        k = tl.load(K_ptr + (start + offs_kv[:, None]) * stride_kn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_log2e
        qk = tl.where(mask_n, qk, -1.0e6)
        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        p = tl.math.exp2(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha
        v = tl.load(V_ptr + (start + offs_kv[:, None]) * stride_vn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _d_stage_decode_kernel(
    Q,
    K_quant, V_quant, K_scale, V_scale,
    K_topk, V_topk,
    K_staging, V_staging,
    K_window, V_window,
    Out,
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kpz, stride_kpn,
    stride_vpz, stride_vpn,
    stride_ksz,
    stride_vsz,
    stride_ktz, stride_ktn,
    stride_vtz, stride_vtn,
    stride_kstz, stride_kstn,
    stride_vstz, stride_vstn,
    stride_kwz, stride_kwh, stride_kwn,
    stride_vwz, stride_vwh, stride_vwn,
    stride_oz, stride_oh, stride_od,
    N_quant, N_topk, N_staging, WINDOW,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GROUP_SZ: tl.constexpr,
):
    """[Legacy] 4-segment bf16 kernel — used as fallback for K_BITS != 4."""
    off_z = tl.program_id(1).to(tl.int64)
    off_h = tl.program_id(0).to(tl.int64)
    bh = off_z * H + off_h

    offs_d = tl.arange(0, D)
    q = tl.load(Q + off_z * stride_qz + off_h * stride_qh + offs_d * stride_qd).to(tl.float32)
    sm_log2e = sm_scale * 1.44269504

    m_i = tl.full([1], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    if N_quant > 0:
        acc, l_i, m_i = _bf16_segment_full(
            acc, l_i, m_i, q,
            K_quant + bh * stride_kpz, V_quant + bh * stride_vpz,
            stride_kpn, stride_vpn,
            N_quant, sm_log2e, D, BLOCK_KV,
        )
    if N_topk > 0:
        acc, l_i, m_i = _bf16_segment_full(
            acc, l_i, m_i, q,
            K_topk + bh * stride_ktz, V_topk + bh * stride_vtz,
            stride_ktn, stride_vtn,
            N_topk, sm_log2e, D, BLOCK_KV,
        )
    if N_staging > 0:
        acc, l_i, m_i = _bf16_segment_full(
            acc, l_i, m_i, q,
            K_staging + bh * stride_kstz, V_staging + bh * stride_vstz,
            stride_kstn, stride_vstn,
            N_staging, sm_log2e, D, BLOCK_KV,
        )
    acc, l_i, m_i = _bf16_segment_full(
        acc, l_i, m_i, q,
        K_window + off_z * stride_kwz + off_h * stride_kwh,
        V_window + off_z * stride_vwz + off_h * stride_vwh,
        stride_kwn, stride_vwn,
        WINDOW, sm_log2e, D, BLOCK_KV,
    )
    out = acc / l_i
    tl.store(Out + off_z * stride_oz + off_h * stride_oh + offs_d * stride_od,
             out.to(Out.type.element_ty))


def d_stage_decode_attention(
    q, k_quant_packed, v_quant_packed, k_scale, v_scale,
    k_topk, v_topk, k_staging, v_staging, k_window, v_window,
    *, v_bits: int = 4, sm_scale: Optional[float] = None,
):
    """[Legacy] D-stage decode with Python-side dequant — kept for fallback only."""
    B, H, _, D = q.shape
    BH = B * H
    if sm_scale is None:
        sm_scale = D ** -0.5

    N_quant = k_quant_packed.shape[1] if k_quant_packed.numel() > 0 else 0
    if N_quant > 0:
        k_rest_deq = dequantize_k_int4_channel(k_quant_packed, k_scale, D).to(q.dtype)
        if v_bits == 4:
            v_rest_deq = dequantize_k_int4_channel(v_quant_packed, v_scale, D).to(q.dtype)
        else:
            v_rest_deq = dequantize_v_int2_channel(v_quant_packed, v_scale, D).to(q.dtype)
    else:
        k_rest_deq = torch.zeros(BH, 0, D, device=q.device, dtype=q.dtype)
        v_rest_deq = torch.zeros(BH, 0, D, device=q.device, dtype=q.dtype)

    N_topk = k_topk.shape[1] if k_topk.numel() > 0 else 0
    N_staging = k_staging.shape[1] if k_staging.numel() > 0 else 0
    WINDOW = k_window.shape[2]

    def ensure(t, shape):
        return t if t.numel() > 0 else torch.zeros(shape, device=q.device, dtype=q.dtype)

    k_rest_deq = ensure(k_rest_deq, (BH, 1, D))
    v_rest_deq = ensure(v_rest_deq, (BH, 1, D))
    k_topk = ensure(k_topk, (BH, 1, D))
    v_topk = ensure(v_topk, (BH, 1, D))
    k_staging = ensure(k_staging, (BH, 1, D))
    v_staging = ensure(v_staging, (BH, 1, D))

    out = torch.empty(B, H, 1, D, device=q.device, dtype=q.dtype)
    grid = (H, B)
    _d_stage_decode_kernel[grid](
        q, k_rest_deq, v_rest_deq, None, None,
        k_topk, v_topk, k_staging, v_staging, k_window, v_window,
        out, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_rest_deq.stride(0), k_rest_deq.stride(1),
        v_rest_deq.stride(0), v_rest_deq.stride(1),
        0, 0,
        k_topk.stride(0), k_topk.stride(1),
        v_topk.stride(0), v_topk.stride(1),
        k_staging.stride(0), k_staging.stride(1),
        v_staging.stride(0), v_staging.stride(1),
        k_window.stride(0), k_window.stride(1), k_window.stride(2),
        v_window.stride(0), v_window.stride(1), v_window.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        N_quant, N_topk, N_staging, WINDOW,
        H=H, D=D, BLOCK_KV=64,
        GROUP_SZ=GROUP_SIZE,
        num_warps=4,
    )
    return out


decode_4seg_attention = d_stage_decode_attention


# ============================================================
# Optimised: fused int4 dequant + 4-segment attention with autotune
# ============================================================

@triton.jit
def _seg_int4_inline(
    acc_even, acc_odd, l_i, m_i,
    q_even, q_odd,                       # [D//2] each, fp32
    K_packed_base, V_packed_base,        # uint8 [N, D//2]
    K_scale_base, V_scale_base,          # fp16  [G, D]
    stride_kn, stride_vn,
    stride_ksg, stride_ksd,              # K_scale layout (G, D)
    stride_vsg, stride_vsd,
    N_seg,
    sm_log2e,
    D_HALF: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GROUP_SZ: tl.constexpr,
):
    """Process the quantized prefix segment with on-chip int4 dequant.

    Even channels (d = 0, 2, 4, ...) live in the HIGH nibble (>> 4);
    odd channels  (d = 1, 3, 5, ...) live in the LOW nibble  (& 0xF).
    We load D//2 unique bytes per row (no duplicate reads) and keep two
    half-D accumulators throughout the kernel.
    """
    offs_d_half = tl.arange(0, D_HALF)            # [0..D/2)
    offs_d_even = 2 * offs_d_half                 # channel ids for the high nibble
    offs_d_odd = 2 * offs_d_half + 1              # channel ids for the low nibble
    offs_kv = tl.arange(0, BLOCK_KV)

    for start in range(0, N_seg, BLOCK_KV):
        ns = start + offs_kv
        mask_n = ns < N_seg
        g = ns // GROUP_SZ                        # [BLOCK_KV] group id per token

        # ---- Load K packed (single contiguous read per row) ----
        packed_K = tl.load(
            K_packed_base + ns[:, None] * stride_kn + offs_d_half[None, :],
            mask=mask_n[:, None], other=0,
        )                                          # [BLOCK_KV, D_HALF] uint8
        # int4 hi/lo unpack; values offset-encoded as q_u = q + 8 ∈ [1..15].
        K_hi = (packed_K >> 4).to(tl.float32) - 8.0   # even channels
        K_lo = (packed_K & 0xF).to(tl.float32) - 8.0  # odd channels

        # Scales for even/odd channels (separate loads — stride-1 along even/odd subsets)
        sk_even = tl.load(
            K_scale_base + g[:, None] * stride_ksg + offs_d_even[None, :] * stride_ksd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        sk_odd = tl.load(
            K_scale_base + g[:, None] * stride_ksg + offs_d_odd[None, :] * stride_ksd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        K_tile_even = K_hi * sk_even              # [BLOCK_KV, D_HALF]
        K_tile_odd = K_lo * sk_odd

        # ---- QK^T over full D, expressed as even-sum + odd-sum ----
        qk = (
            tl.sum(q_even[None, :] * K_tile_even, axis=1)
            + tl.sum(q_odd[None, :] * K_tile_odd, axis=1)
        ) * sm_log2e
        qk = tl.where(mask_n, qk, -1.0e6)

        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        p = tl.math.exp2(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load V packed (same layout as K) ----
        packed_V = tl.load(
            V_packed_base + ns[:, None] * stride_vn + offs_d_half[None, :],
            mask=mask_n[:, None], other=0,
        )
        V_hi = (packed_V >> 4).to(tl.float32) - 8.0
        V_lo = (packed_V & 0xF).to(tl.float32) - 8.0

        sv_even = tl.load(
            V_scale_base + g[:, None] * stride_vsg + offs_d_even[None, :] * stride_vsd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        sv_odd = tl.load(
            V_scale_base + g[:, None] * stride_vsg + offs_d_odd[None, :] * stride_vsd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        V_tile_even = V_hi * sv_even
        V_tile_odd = V_lo * sv_odd

        # ---- PV update ----
        acc_even += tl.sum(p[:, None] * V_tile_even, axis=0)
        acc_odd += tl.sum(p[:, None] * V_tile_odd, axis=0)
        m_i = m_ij

    return acc_even, acc_odd, l_i, m_i


# Auto-tunable configurations. Triton picks the best per (D, ...) at first launch.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_KV": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_KV": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_KV": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_KV": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_KV": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_KV": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_KV": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_KV": 256}, num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["D_HALF", "N_quant", "WINDOW"],
)
@triton.jit
def _fused_decode_kernel(
    Q,
    K_quant, V_quant,
    K_scale, V_scale,
    K_topk, V_topk,
    K_staging, V_staging,
    K_window, V_window,
    Out,
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kqz, stride_kqn,
    stride_vqz, stride_vqn,
    stride_ksz, stride_ksg, stride_ksd,
    stride_vsz, stride_vsg, stride_vsd,
    stride_ktz, stride_ktn, stride_ktd,
    stride_vtz, stride_vtn, stride_vtd,
    stride_kstz, stride_kstn, stride_kstd,
    stride_vstz, stride_vstn, stride_vstd,
    stride_kwz, stride_kwn, stride_kwd,
    stride_vwz, stride_vwn, stride_vwd,
    stride_oz, stride_oh, stride_od,
    N_quant, N_topk, N_staging, WINDOW,
    H_KV: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    D_HALF: tl.constexpr,
    GROUP_SZ: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    off_z = tl.program_id(1).to(tl.int64)
    off_hq = tl.program_id(0).to(tl.int64)
    off_hkv = off_hq // KV_REPEAT
    bh_kv = off_z * H_KV + off_hkv

    offs_d_half = tl.arange(0, D_HALF)
    offs_d_even = 2 * offs_d_half
    offs_d_odd = 2 * offs_d_half + 1

    # Load Q in even/odd halves
    q_base = Q + off_z * stride_qz + off_hq * stride_qh
    q_even = tl.load(q_base + offs_d_even * stride_qd).to(tl.float32)
    q_odd = tl.load(q_base + offs_d_odd * stride_qd).to(tl.float32)

    sm_log2e = sm_scale * 1.44269504

    m_i = tl.full([1], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_even = tl.zeros([D_HALF], dtype=tl.float32)
    acc_odd = tl.zeros([D_HALF], dtype=tl.float32)

    # Seg 1: quantized K4 + V4 (on-chip dequant)
    if N_quant > 0:
        acc_even, acc_odd, l_i, m_i = _seg_int4_inline(
            acc_even, acc_odd, l_i, m_i, q_even, q_odd,
            K_quant + bh_kv * stride_kqz,
            V_quant + bh_kv * stride_vqz,
            K_scale + bh_kv * stride_ksz,
            V_scale + bh_kv * stride_vsz,
            stride_kqn, stride_vqn,
            stride_ksg, stride_ksd,
            stride_vsg, stride_vsd,
            N_quant, sm_log2e,
            D_HALF, BLOCK_KV, GROUP_SZ,
        )

    # Seg 2: topk bf16
    if N_topk > 0:
        acc_even, acc_odd, l_i, m_i = _bf16_segment(
            acc_even, acc_odd, l_i, m_i, q_even, q_odd,
            K_topk + bh_kv * stride_ktz,
            V_topk + bh_kv * stride_vtz,
            stride_ktn, stride_vtn,
            stride_ktd, stride_vtd,
            N_topk, sm_log2e,
            D_HALF, BLOCK_KV,
        )

    # Seg 3: staging bf16
    if N_staging > 0:
        acc_even, acc_odd, l_i, m_i = _bf16_segment(
            acc_even, acc_odd, l_i, m_i, q_even, q_odd,
            K_staging + bh_kv * stride_kstz,
            V_staging + bh_kv * stride_vstz,
            stride_kstn, stride_vstn,
            stride_kstd, stride_vstd,
            N_staging, sm_log2e,
            D_HALF, BLOCK_KV,
        )

    # Seg 4: window bf16 (always present)
    acc_even, acc_odd, l_i, m_i = _bf16_segment(
        acc_even, acc_odd, l_i, m_i, q_even, q_odd,
        K_window + bh_kv * stride_kwz,
        V_window + bh_kv * stride_vwz,
        stride_kwn, stride_vwn,
        stride_kwd, stride_vwd,
        WINDOW, sm_log2e,
        D_HALF, BLOCK_KV,
    )

    # Write output with strided stores at even / odd channel positions
    inv_l = 1.0 / l_i
    out_even = acc_even * inv_l
    out_odd = acc_odd * inv_l
    out_base = Out + off_z * stride_oz + off_hq * stride_oh
    tl.store(out_base + offs_d_even * stride_od, out_even.to(Out.type.element_ty))
    tl.store(out_base + offs_d_odd * stride_od, out_odd.to(Out.type.element_ty))


def fused_decode_attention_k4v4(
    q: torch.Tensor,                # [B, H_Q, 1, D] bf16
    k_quant_packed: torch.Tensor,   # [BH_KV, N_q, D//2] uint8
    v_quant_packed: torch.Tensor,
    k_scale: torch.Tensor,          # [BH_KV, G, D] fp16
    v_scale: torch.Tensor,
    k_topk: torch.Tensor,           # [BH_KV, T, D] bf16 (T may be 0)
    v_topk: torch.Tensor,
    k_staging: torch.Tensor,        # [BH_KV, S, D] bf16
    v_staging: torch.Tensor,
    k_window: torch.Tensor,         # [BH_KV, W, D] bf16
    v_window: torch.Tensor,
    *,
    kv_repeat: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Optimised D-stage decode (K int4 + V int4) — single fused kernel."""
    B, H_Q, _, D = q.shape
    assert D % 2 == 0
    H_KV = H_Q // kv_repeat
    BH_KV = B * H_KV
    assert k_quant_packed.shape[0] == BH_KV, (
        f"K_packed[0]={k_quant_packed.shape[0]} vs BH_KV={BH_KV} mismatch"
    )

    if sm_scale is None:
        sm_scale = D ** -0.5

    N_quant = k_quant_packed.shape[1] if k_quant_packed.numel() > 0 else 0
    N_topk = k_topk.shape[1] if (k_topk is not None and k_topk.numel() > 0) else 0
    N_staging = k_staging.shape[1] if (k_staging is not None and k_staging.numel() > 0) else 0
    WINDOW = k_window.shape[1]

    # Placeholder tensors for empty segments (kernel skips them but still needs valid ptrs)
    def _empty_bf16(shape):
        return torch.zeros(shape, device=q.device, dtype=q.dtype)

    if N_quant == 0:
        k_quant_packed = torch.zeros(BH_KV, 1, D // 2, device=q.device, dtype=torch.uint8)
        v_quant_packed = torch.zeros(BH_KV, 1, D // 2, device=q.device, dtype=torch.uint8)
        k_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)
        v_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)
    if N_topk == 0:
        k_topk = _empty_bf16((BH_KV, 1, D))
        v_topk = _empty_bf16((BH_KV, 1, D))
    if N_staging == 0:
        k_staging = _empty_bf16((BH_KV, 1, D))
        v_staging = _empty_bf16((BH_KV, 1, D))

    out = torch.empty(B, H_Q, 1, D, device=q.device, dtype=q.dtype)

    grid = (H_Q, B)
    _fused_decode_kernel[grid](
        q,
        k_quant_packed, v_quant_packed,
        k_scale, v_scale,
        k_topk, v_topk,
        k_staging, v_staging,
        k_window, v_window,
        out, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_quant_packed.stride(0), k_quant_packed.stride(1),
        v_quant_packed.stride(0), v_quant_packed.stride(1),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        k_topk.stride(0), k_topk.stride(1), k_topk.stride(2),
        v_topk.stride(0), v_topk.stride(1), v_topk.stride(2),
        k_staging.stride(0), k_staging.stride(1), k_staging.stride(2),
        v_staging.stride(0), v_staging.stride(1), v_staging.stride(2),
        k_window.stride(0), k_window.stride(1), k_window.stride(2),
        v_window.stride(0), v_window.stride(1), v_window.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        N_quant, N_topk, N_staging, WINDOW,
        H_KV=H_KV, KV_REPEAT=kv_repeat,
        D_HALF=D // 2, GROUP_SZ=GROUP_SIZE,
    )
    return out


# ============================================================
# Split-K decode kernel (FlashDecoding-style)
#   - Phase 1: scan int4 prefix in NUM_SPLITS parallel chunks per (b, h_q)
#   - Phase 2: reduce splits, then attend over the bf16 "flat" segment
#               (topk + staging + window concatenated by the Python wrapper)
#   - GQA-aware: K/V indexed by h_kv = h_q // KV_REPEAT
#   - uint8 offset-encoded packing (q_unsigned = q_signed + 8):
#       even channels → high nibble (>> 4) - 8
#       odd  channels → low  nibble (& 0xF) - 8
# ============================================================

@triton.jit
def _splitk_scan_kernel(
    Q_ptr,                  # [B, H_Q, D]  bf16
    K_packed_ptr,            # [BH_KV, N_quant, D//2]  uint8 (offset-encoded int4)
    V_packed_ptr,
    K_scale_ptr,             # [BH_KV, G, D]  fp16
    V_scale_ptr,
    Mid_O_ptr,               # [B, H_Q, NUM_SPLITS, D]  fp32
    Mid_M_ptr,               # [B, H_Q, NUM_SPLITS]      fp32
    Mid_L_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kqz, stride_kqn, stride_kqd,
    stride_vqz, stride_vqn, stride_vqd,
    stride_ksz, stride_ksg, stride_ksd,
    stride_vsz, stride_vsg, stride_vsd,
    stride_mo_z, stride_mo_h, stride_mo_s, stride_mo_d,
    stride_mm_z, stride_mm_h, stride_mm_s,
    N_quant,
    H_KV: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    H_Q: tl.constexpr,
    D_HALF: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GROUP_SZ: tl.constexpr,
):
    pid_bh = tl.program_id(0)     # B * H_Q
    pid_split = tl.program_id(1)  # NUM_SPLITS
    b = pid_bh // H_Q
    h_q = pid_bh % H_Q
    h_kv = h_q // KV_REPEAT
    bh_kv = b * H_KV + h_kv

    # ---- Load Q in even/odd halves ----
    offs_d_half = tl.arange(0, D_HALF)
    offs_d_even = 2 * offs_d_half
    offs_d_odd = 2 * offs_d_half + 1
    q_base = Q_ptr + b * stride_qz + h_q * stride_qh
    q_even = tl.load(q_base + offs_d_even * stride_qd).to(tl.float32)
    q_odd = tl.load(q_base + offs_d_odd * stride_qd).to(tl.float32)

    # ---- Online softmax state ----
    # IMPORTANT: m_i / l_i are Python scalars (not [1] blocks) so that we can
    # tl.store them to scalar partials at the end of the kernel.
    sm_log2e = sm_scale * 1.44269504
    m_i = -1.0e6
    l_i = 0.0
    acc_even = tl.zeros([D_HALF], dtype=tl.float32)
    acc_odd = tl.zeros([D_HALF], dtype=tl.float32)

    kq_base = K_packed_ptr + bh_kv * stride_kqz
    vq_base = V_packed_ptr + bh_kv * stride_vqz
    ks_base = K_scale_ptr + bh_kv * stride_ksz
    vs_base = V_scale_ptr + bh_kv * stride_vsz

    start_n_global = pid_split * SPLIT_SIZE

    for kk in range(0, SPLIT_SIZE, BLOCK_KV):
        curr_n = start_n_global + kk
        offs_n = curr_n + tl.arange(0, BLOCK_KV)
        mask_n = offs_n < N_quant
        # Block-level early-exit: if the whole block is past N_quant, skip
        # (Triton can't `break` cleanly, but mask=0 makes load/compute null-ops)
        g = offs_n // GROUP_SZ

        # ---- Load K packed (1 byte per (n, d/2) — no duplicate reads) ----
        k_pack = tl.load(
            kq_base + offs_n[:, None] * stride_kqn + offs_d_half[None, :] * stride_kqd,
            mask=mask_n[:, None], other=0,
        )
        # uint8 → int4 (offset-encoded): even = (hi - 8), odd = (lo - 8)
        K_e = (k_pack >> 4).to(tl.float32) - 8.0
        K_o = (k_pack & 0xF).to(tl.float32) - 8.0

        # ---- Load K scales (even/odd channel subsets via strided load) ----
        sk_e = tl.load(
            ks_base + g[:, None] * stride_ksg + offs_d_even[None, :] * stride_ksd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        sk_o = tl.load(
            ks_base + g[:, None] * stride_ksg + offs_d_odd[None, :] * stride_ksd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        K_even = K_e * sk_e
        K_odd = K_o * sk_o

        # ---- QK^T ----
        qk = (
            tl.sum(q_even[None, :] * K_even, axis=1)
            + tl.sum(q_odd[None, :] * K_odd, axis=1)
        ) * sm_log2e
        qk = tl.where(mask_n, qk, -1.0e6)

        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        # Guard against m_new == -inf (entire partial empty)
        p = tl.math.exp2(qk - m_new)
        alpha = tl.math.exp2(m_i - m_new)
        l_ij = tl.sum(p, axis=0)
        l_i = l_i * alpha + l_ij
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha

        # ---- Load V packed + scales, update PV ----
        v_pack = tl.load(
            vq_base + offs_n[:, None] * stride_vqn + offs_d_half[None, :] * stride_vqd,
            mask=mask_n[:, None], other=0,
        )
        V_e = (v_pack >> 4).to(tl.float32) - 8.0
        V_o = (v_pack & 0xF).to(tl.float32) - 8.0
        sv_e = tl.load(
            vs_base + g[:, None] * stride_vsg + offs_d_even[None, :] * stride_vsd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        sv_o = tl.load(
            vs_base + g[:, None] * stride_vsg + offs_d_odd[None, :] * stride_vsd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        V_even = V_e * sv_e
        V_odd = V_o * sv_o

        acc_even += tl.sum(p[:, None] * V_even, axis=0)
        acc_odd += tl.sum(p[:, None] * V_odd, axis=0)
        m_i = m_new

    # ---- Store partials (Mid_O[d=0,2,...]=acc_even, d=1,3,...=acc_odd via interleave) ----
    acc_inter = tl.interleave(acc_even, acc_odd)  # [D]
    offs_d = tl.arange(0, 2 * D_HALF)
    mo_ptr = (
        Mid_O_ptr + b * stride_mo_z + h_q * stride_mo_h
        + pid_split * stride_mo_s + offs_d * stride_mo_d
    )
    tl.store(mo_ptr, acc_inter)

    mm_ptr = Mid_M_ptr + b * stride_mm_z + h_q * stride_mm_h + pid_split * stride_mm_s
    ml_ptr = Mid_L_ptr + b * stride_mm_z + h_q * stride_mm_h + pid_split * stride_mm_s
    tl.store(mm_ptr, m_i)
    tl.store(ml_ptr, l_i)


@triton.jit
def _reduce_and_flat_kernel(
    Q_ptr,                # [B, H_Q, D]  bf16
    Kf_ptr, Vf_ptr,        # [BH_KV, N_flat, D]  bf16  (topk + staging + window concat)
    Mid_O_ptr, Mid_M_ptr, Mid_L_ptr,
    Out_ptr,               # [B, H_Q, D]  bf16
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kfz, stride_kfn, stride_kfd,
    stride_vfz, stride_vfn, stride_vfd,
    stride_mo_z, stride_mo_h, stride_mo_s, stride_mo_d,
    stride_mm_z, stride_mm_h, stride_mm_s,
    stride_oz, stride_oh, stride_od,
    N_flat,
    H_KV: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_FLAT: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H_Q
    h_q = pid % H_Q
    h_kv = h_q // KV_REPEAT
    bh_kv = b * H_KV + h_kv

    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + b * stride_qz + h_q * stride_qh + offs_d * stride_qd).to(tl.float32)

    sm_log2e = sm_scale * 1.44269504
    m_i = -1.0e6
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # ---- Reduce phase-1 partials ----
    mm_base = Mid_M_ptr + b * stride_mm_z + h_q * stride_mm_h
    ml_base = Mid_L_ptr + b * stride_mm_z + h_q * stride_mm_h
    mo_base = Mid_O_ptr + b * stride_mo_z + h_q * stride_mo_h
    for s in range(0, NUM_SPLITS, BLOCK_SPLITS):
        offs_s = s + tl.arange(0, BLOCK_SPLITS)
        mask_s = offs_s < NUM_SPLITS
        m_parts = tl.load(mm_base + offs_s * stride_mm_s, mask=mask_s, other=-1.0e6)
        l_parts = tl.load(ml_base + offs_s * stride_mm_s, mask=mask_s, other=0.0)
        o_parts = tl.load(
            mo_base + offs_s[:, None] * stride_mo_s + offs_d[None, :] * stride_mo_d,
            mask=mask_s[:, None], other=0.0,
        ).to(tl.float32)

        m_blk = tl.max(m_parts, axis=0)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.math.exp2(m_i - m_new)
        w = tl.math.exp2(m_parts - m_new)
        w = tl.where(mask_s, w, 0.0)
        l_i = l_i * alpha + tl.sum(l_parts * w, axis=0)
        acc = acc * alpha + tl.sum(o_parts * w[:, None], axis=0)
        m_i = m_new

    # ---- Process bf16 flat (topk + staging + window) in one sweep ----
    kf_base = Kf_ptr + bh_kv * stride_kfz
    vf_base = Vf_ptr + bh_kv * stride_vfz
    for start_n in range(0, N_flat, BLOCK_FLAT):
        offs_n = start_n + tl.arange(0, BLOCK_FLAT)
        mask_n = offs_n < N_flat
        k = tl.load(
            kf_base + offs_n[:, None] * stride_kfn + offs_d[None, :] * stride_kfd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        v = tl.load(
            vf_base + offs_n[:, None] * stride_vfn + offs_d[None, :] * stride_vfd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)

        qk = tl.sum(q[None, :] * k, axis=1) * sm_log2e
        qk = tl.where(mask_n, qk, -1.0e6)
        m_curr = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    o_base = Out_ptr + b * stride_oz + h_q * stride_oh
    tl.store(o_base + offs_d * stride_od, out.to(Out_ptr.type.element_ty))


def splitk_decode_attention_k4v4(
    q: torch.Tensor,                # [B, H_Q, 1, D]  bf16
    k_quant_packed: torch.Tensor,   # [BH_KV, N_quant, D//2]  uint8 offset-encoded
    v_quant_packed: torch.Tensor,
    k_scale: torch.Tensor,          # [BH_KV, G, D]  fp16
    v_scale: torch.Tensor,
    k_topk: torch.Tensor,           # [BH_KV, T, D]  bf16  (may be empty)
    v_topk: torch.Tensor,
    k_staging: torch.Tensor,        # [BH_KV, S, D]  bf16
    v_staging: torch.Tensor,
    k_window: torch.Tensor,         # [BH_KV, W, D]  bf16
    v_window: torch.Tensor,
    *,
    kv_repeat: int,
    sm_scale: Optional[float] = None,
    split_size: int = 1024,
    block_kv: int = 128,
) -> torch.Tensor:
    """FlashDecoding-style split-K decode for K int4 + V int4 with bf16 anchors.

    Phase 1 launches  (B*H_Q, NUM_SPLITS)  programs scanning the quantised
    prefix in parallel chunks; phase 2 launches  (B*H_Q,)  programs that
    merge partials and sweep the bf16 segments (topk + staging + window).
    """
    B, H_Q, _, D = q.shape
    assert D % 2 == 0
    H_KV = H_Q // kv_repeat
    BH_KV = B * H_KV
    assert k_quant_packed.shape[0] == BH_KV
    if sm_scale is None:
        sm_scale = D ** -0.5

    N_quant = k_quant_packed.shape[1] if k_quant_packed.numel() > 0 else 0

    # ---- Concatenate bf16 anchors (topk + staging + window) ----
    # Skip empty tensors to avoid an alloc for the trivial case.
    bf16_segs_k = []
    bf16_segs_v = []
    for kk, vv in ((k_topk, v_topk), (k_staging, v_staging), (k_window, v_window)):
        if kk is not None and kk.numel() > 0 and kk.shape[1] > 0:
            bf16_segs_k.append(kk)
            bf16_segs_v.append(vv)
    if len(bf16_segs_k) == 0:
        k_flat = torch.zeros(BH_KV, 1, D, device=q.device, dtype=q.dtype)
        v_flat = torch.zeros(BH_KV, 1, D, device=q.device, dtype=q.dtype)
        N_flat = 0
    elif len(bf16_segs_k) == 1:
        k_flat = bf16_segs_k[0]
        v_flat = bf16_segs_v[0]
        N_flat = int(k_flat.shape[1])
    else:
        k_flat = torch.cat(bf16_segs_k, dim=1)
        v_flat = torch.cat(bf16_segs_v, dim=1)
        N_flat = int(k_flat.shape[1])

    # ---- Phase 1: split-K scan ----
    NUM_SPLITS = max(1, (N_quant + split_size - 1) // split_size)

    if N_quant == 0:
        # No quantised prefix → skip phase 1, do single-pass bf16 attention
        # via a (B*H_Q,) program reducing zero splits + sweeping flat.
        k_quant_packed = torch.zeros(BH_KV, 1, D // 2, device=q.device, dtype=torch.uint8)
        v_quant_packed = torch.zeros(BH_KV, 1, D // 2, device=q.device, dtype=torch.uint8)
        k_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)
        v_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)
        # The phase-1 kernel with N_quant=0 + mask will still write valid
        # zeros into the partials (acc=0, m=-1e6, l=0), which the phase-2
        # reduce treats as "no contribution" correctly via the m=-1e6 guard.

    mid_o = torch.empty(B, H_Q, NUM_SPLITS, D, device=q.device, dtype=torch.float32)
    mid_m = torch.empty(B, H_Q, NUM_SPLITS, device=q.device, dtype=torch.float32)
    mid_l = torch.empty(B, H_Q, NUM_SPLITS, device=q.device, dtype=torch.float32)

    grid_scan = (B * H_Q, NUM_SPLITS)
    _splitk_scan_kernel[grid_scan](
        q,
        k_quant_packed, v_quant_packed,
        k_scale, v_scale,
        mid_o, mid_m, mid_l,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
        v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        N_quant,
        H_KV=H_KV, KV_REPEAT=kv_repeat, H_Q=H_Q,
        D_HALF=D // 2, SPLIT_SIZE=split_size, BLOCK_KV=block_kv,
        GROUP_SZ=GROUP_SIZE,
        num_warps=4, num_stages=3,
    )

    # ---- Phase 2: reduce + bf16 flat ----
    out = torch.empty(B, H_Q, 1, D, device=q.device, dtype=q.dtype)
    BLOCK_SPLITS = min(triton.next_power_of_2(max(NUM_SPLITS, 1)), 128)
    BLOCK_FLAT = 64 if N_flat > 64 else max(N_flat, 16)
    grid_reduce = (B * H_Q,)
    _reduce_and_flat_kernel[grid_reduce](
        q,
        k_flat, v_flat,
        mid_o, mid_m, mid_l,
        out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        N_flat,
        H_KV=H_KV, KV_REPEAT=kv_repeat, H_Q=H_Q,
        D=D, NUM_SPLITS=NUM_SPLITS,
        BLOCK_SPLITS=BLOCK_SPLITS,
        BLOCK_FLAT=BLOCK_FLAT,
        num_warps=4, num_stages=2,
    )
    return out


# ============================================================
# Split-K decode kernel — GQA-grouped Q heads + Tensor Core MMA
#   - One program per (B, H_KV, split) instead of per (B, H_Q, split)
#   - Pack KV_REPEAT Q heads sharing the same KV head into a single tile
#   - Q becomes [KV_REPEAT_PAD, D]; QK and PV use `tl.dot` (Tensor Core)
#   - Pad KV_REPEAT up to KV_REPEAT_TILE (≥16) to satisfy MMA shape requirements
# ============================================================

@triton.jit
def _splitk_scan_kernel_mma(
    Q_ptr,
    K_packed_ptr, V_packed_ptr,
    K_scale_ptr, V_scale_ptr,
    Mid_O_ptr, Mid_M_ptr, Mid_L_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kqz, stride_kqn, stride_kqd,
    stride_vqz, stride_vqn, stride_vqd,
    stride_ksz, stride_ksg, stride_ksd,
    stride_vsz, stride_vsg, stride_vsd,
    stride_mo_z, stride_mo_h, stride_mo_s, stride_mo_d,
    stride_mm_z, stride_mm_h, stride_mm_s,
    N_quant,
    H_KV: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    KV_REPEAT_TILE: tl.constexpr,
    D: tl.constexpr,
    D_HALF: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    GROUP_SZ: tl.constexpr,
    K_BITS: tl.constexpr,   # 4 or 2
    V_BITS: tl.constexpr,   # 4 or 2
):
    pid_bh_kv = tl.program_id(0)        # B * H_KV
    pid_split = tl.program_id(1)
    b = pid_bh_kv // H_KV
    h_kv = pid_bh_kv % H_KV
    bh_kv = b * H_KV + h_kv
    h_q_base = h_kv * KV_REPEAT

    offs_r = tl.arange(0, KV_REPEAT_TILE)
    mask_r = offs_r < KV_REPEAT
    offs_d = tl.arange(0, D)
    offs_d_half = tl.arange(0, D_HALF)
    D_QUARTER: tl.constexpr = D // 4

    # ---- Load Q tile [KV_REPEAT_TILE, D] in bf16 (pad rows masked to 0) ----
    q_ptrs = (
        Q_ptr + b * stride_qz
        + (h_q_base + offs_r)[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_r[:, None], other=0.0)  # bf16 [KV_REPEAT_TILE, D]

    sm_log2e = sm_scale * 1.44269504
    # m_i, l_i are per-Q-head fp32 vectors; padded rows stay at -1e6 / 0
    m_i = tl.full([KV_REPEAT_TILE], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([KV_REPEAT_TILE], dtype=tl.float32)
    acc = tl.zeros([KV_REPEAT_TILE, D], dtype=tl.float32)

    kq_base = K_packed_ptr + bh_kv * stride_kqz
    vq_base = V_packed_ptr + bh_kv * stride_vqz
    ks_base = K_scale_ptr + bh_kv * stride_ksz
    vs_base = V_scale_ptr + bh_kv * stride_vsz

    start_n_global = pid_split * SPLIT_SIZE
    for kk in range(0, SPLIT_SIZE, BLOCK_KV):
        curr_n = start_n_global + kk
        offs_n = curr_n + tl.arange(0, BLOCK_KV)
        mask_n = offs_n < N_quant
        g = offs_n // GROUP_SZ

        # ---- Load + unpack K (gated by K_BITS constexpr) ----
        if K_BITS == 4:
            # 2 channels per byte: high nibble→even, low nibble→odd, offset-encoded
            packed_K = tl.load(
                kq_base + offs_n[:, None] * stride_kqn + offs_d_half[None, :] * stride_kqd,
                mask=mask_n[:, None], other=0,
            )
            K_e = (packed_K >> 4).to(tl.float32) - 8.0   # [BLOCK_KV, D//2]
            K_o = (packed_K & 0xF).to(tl.float32) - 8.0
            K_int = tl.interleave(K_e, K_o)               # [BLOCK_KV, D]
        else:
            # K_BITS == 2: 4 channels per byte (same layout as V int2)
            offs_k_q = tl.arange(0, D_QUARTER)
            packed_K = tl.load(
                kq_base + offs_n[:, None] * stride_kqn + offs_k_q[None, :] * stride_kqd,
                mask=mask_n[:, None], other=0,
            )
            K_b0 = ((packed_K >> 6) & 0x3).to(tl.float32) - 1.5
            K_b1 = ((packed_K >> 4) & 0x3).to(tl.float32) - 1.5
            K_b2 = ((packed_K >> 2) & 0x3).to(tl.float32) - 1.5
            K_b3 = (packed_K & 0x3).to(tl.float32) - 1.5
            K_even = tl.interleave(K_b0, K_b2)            # channels {0,2,4,...}
            K_odd = tl.interleave(K_b1, K_b3)             # channels {1,3,5,...}
            K_int = tl.interleave(K_even, K_odd)          # [BLOCK_KV, D]

        # ---- Load full-D scale once (stride-1 in D) and apply ----
        sk = tl.load(
            ks_base + g[:, None] * stride_ksg + offs_d[None, :] * stride_ksd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)                                  # [BLOCK_KV, D]
        K_deq = (K_int * sk).to(tl.bfloat16)              # bf16 [BLOCK_KV, D]

        # ---- QK^T via Tensor Core ----
        # q [KV_REPEAT_TILE, D] × K_deq.T [D, BLOCK_KV] = [KV_REPEAT_TILE, BLOCK_KV]
        qk = tl.dot(q, K_deq.trans(), allow_tf32=False).to(tl.float32) * sm_log2e
        qk = tl.where(mask_n[None, :], qk, -1.0e6)

        m_ij = tl.max(qk, axis=1)                      # [KV_REPEAT_TILE]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # ---- Load + unpack V. Two paths gated by V_BITS constexpr. ----
        if V_BITS == 4:
            # 2 channels per byte. Same packing as K.
            packed_V = tl.load(
                vq_base + offs_n[:, None] * stride_vqn + offs_d_half[None, :] * stride_vqd,
                mask=mask_n[:, None], other=0,
            )
            V_e = (packed_V >> 4).to(tl.float32) - 8.0   # [BLOCK_KV, D//2]
            V_o = (packed_V & 0xF).to(tl.float32) - 8.0
            V_int = tl.interleave(V_e, V_o)              # [BLOCK_KV, D]
        else:
            # V_BITS == 2: 4 channels per byte. Layout (high→low bits):
            #   d=4k   ← (byte >> 6) & 3
            #   d=4k+1 ← (byte >> 4) & 3
            #   d=4k+2 ← (byte >> 2) & 3
            #   d=4k+3 ← (byte     ) & 3
            # Quantised values are stored as q_u = q + 2 ∈ {0..3}, representing
            # signed values {-2, -1, 0, 1} → after centre-shift  q_u - 1.5
            # gives 4 levels {-1.5, -0.5, 0.5, 1.5}.
            offs_v_q = tl.arange(0, D_QUARTER)
            packed_V = tl.load(
                vq_base + offs_n[:, None] * stride_vqn + offs_v_q[None, :] * stride_vqd,
                mask=mask_n[:, None], other=0,
            )
            V_b0 = ((packed_V >> 6) & 0x3).to(tl.float32) - 1.5   # ch 4k
            V_b1 = ((packed_V >> 4) & 0x3).to(tl.float32) - 1.5   # ch 4k+1
            V_b2 = ((packed_V >> 2) & 0x3).to(tl.float32) - 1.5   # ch 4k+2
            V_b3 = (packed_V & 0x3).to(tl.float32) - 1.5          # ch 4k+3
            # Build [BLOCK_KV, D] via 3-layer interleave: pair (b0,b2) gives the
            # even channels {0,2,4,...}, (b1,b3) gives odd {1,3,5,...}, then
            # interleave those two halves to get [0,1,2,3,4,5,...].
            V_even = tl.interleave(V_b0, V_b2)
            V_odd = tl.interleave(V_b1, V_b3)
            V_int = tl.interleave(V_even, V_odd)         # [BLOCK_KV, D]
        sv = tl.load(
            vs_base + g[:, None] * stride_vsg + offs_d[None, :] * stride_vsd,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.float32)
        V_deq = (V_int * sv).to(tl.bfloat16)             # bf16 [BLOCK_KV, D]

        # ---- PV via Tensor Core ----
        # p [KV_REPEAT_TILE, BLOCK_KV] bf16 × V_deq [BLOCK_KV, D] = [KV_REPEAT_TILE, D]
        p_bf16 = p.to(tl.bfloat16)
        acc += tl.dot(p_bf16, V_deq, allow_tf32=False).to(tl.float32)
        m_i = m_new

    # ---- Store partials for KV_REPEAT valid rows ----
    mo_ptrs = (
        Mid_O_ptr + b * stride_mo_z
        + (h_q_base + offs_r)[:, None] * stride_mo_h
        + pid_split * stride_mo_s
        + offs_d[None, :] * stride_mo_d
    )
    tl.store(mo_ptrs, acc, mask=mask_r[:, None])

    mm_ptrs = (
        Mid_M_ptr + b * stride_mm_z
        + (h_q_base + offs_r) * stride_mm_h
        + pid_split * stride_mm_s
    )
    ml_ptrs = (
        Mid_L_ptr + b * stride_mm_z
        + (h_q_base + offs_r) * stride_mm_h
        + pid_split * stride_mm_s
    )
    tl.store(mm_ptrs, m_i, mask=mask_r)
    tl.store(ml_ptrs, l_i, mask=mask_r)


@triton.jit
def _reduce_and_flat_kernel_mma(
    Q_ptr,
    Kf_ptr, Vf_ptr,
    Mid_O_ptr, Mid_M_ptr, Mid_L_ptr,
    Out_ptr,
    sm_scale,
    stride_qz, stride_qh, stride_qd,
    stride_kfz, stride_kfn, stride_kfd,
    stride_vfz, stride_vfn, stride_vfd,
    stride_mo_z, stride_mo_h, stride_mo_s, stride_mo_d,
    stride_mm_z, stride_mm_h, stride_mm_s,
    stride_oz, stride_oh, stride_od,
    N_flat,
    H_KV: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    KV_REPEAT_TILE: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_FLAT: tl.constexpr,
):
    pid_bh_kv = tl.program_id(0)
    b = pid_bh_kv // H_KV
    h_kv = pid_bh_kv % H_KV
    bh_kv = b * H_KV + h_kv
    h_q_base = h_kv * KV_REPEAT

    offs_r = tl.arange(0, KV_REPEAT_TILE)
    mask_r = offs_r < KV_REPEAT
    offs_d = tl.arange(0, D)

    # ---- Load Q tile bf16 [KV_REPEAT_TILE, D] ----
    q_ptrs = (
        Q_ptr + b * stride_qz
        + (h_q_base + offs_r)[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_r[:, None], other=0.0)

    sm_log2e = sm_scale * 1.44269504
    m_i = tl.full([KV_REPEAT_TILE], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([KV_REPEAT_TILE], dtype=tl.float32)
    acc = tl.zeros([KV_REPEAT_TILE, D], dtype=tl.float32)

    # ---- Reduce phase-1 partials per Q head ----
    # Read NUM_SPLITS partials sequentially (small NUM_SPLITS, fine to iterate).
    for s in range(0, NUM_SPLITS):
        # Per-head m / l at this split
        mm_ptr = (
            Mid_M_ptr + b * stride_mm_z
            + (h_q_base + offs_r) * stride_mm_h
            + s * stride_mm_s
        )
        ml_ptr = (
            Mid_L_ptr + b * stride_mm_z
            + (h_q_base + offs_r) * stride_mm_h
            + s * stride_mm_s
        )
        m_s = tl.load(mm_ptr, mask=mask_r, other=-1.0e6)
        l_s = tl.load(ml_ptr, mask=mask_r, other=0.0)

        # Per-head partial output [KV_REPEAT_TILE, D]
        mo_ptr = (
            Mid_O_ptr + b * stride_mo_z
            + (h_q_base + offs_r)[:, None] * stride_mo_h
            + s * stride_mo_s
            + offs_d[None, :] * stride_mo_d
        )
        acc_s = tl.load(mo_ptr, mask=mask_r[:, None], other=0.0).to(tl.float32)

        m_new = tl.maximum(m_i, m_s)
        alpha = tl.math.exp2(m_i - m_new)
        w = tl.math.exp2(m_s - m_new)
        l_i = l_i * alpha + l_s * w
        acc = acc * alpha[:, None] + acc_s * w[:, None]
        m_i = m_new

    # ---- Process bf16 flat (topk + staging + window) via Tensor Core ----
    kf_base = Kf_ptr + bh_kv * stride_kfz
    vf_base = Vf_ptr + bh_kv * stride_vfz
    for start_n in range(0, N_flat, BLOCK_FLAT):
        offs_n = start_n + tl.arange(0, BLOCK_FLAT)
        mask_n = offs_n < N_flat
        K_flat = tl.load(
            kf_base + offs_n[:, None] * stride_kfn + offs_d[None, :] * stride_kfd,
            mask=mask_n[:, None], other=0.0,
        )                                                  # bf16 [BLOCK_FLAT, D]
        V_flat = tl.load(
            vf_base + offs_n[:, None] * stride_vfn + offs_d[None, :] * stride_vfd,
            mask=mask_n[:, None], other=0.0,
        )

        # q[KV_REPEAT_TILE, D] × K_flat.T[D, BLOCK_FLAT] = [KV_REPEAT_TILE, BLOCK_FLAT]
        qk = tl.dot(q, K_flat.trans(), allow_tf32=False).to(tl.float32) * sm_log2e
        qk = tl.where(mask_n[None, :], qk, -1.0e6)

        m_curr = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), V_flat, allow_tf32=False).to(tl.float32)
        m_i = m_new

    # ---- Normalise + write output ----
    inv_l = 1.0 / l_i
    out = acc * inv_l[:, None]
    o_ptrs = (
        Out_ptr + b * stride_oz
        + (h_q_base + offs_r)[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(Out_ptr.type.element_ty), mask=mask_r[:, None])


def splitk_decode_attention_k4v4_mma(
    q: torch.Tensor,
    k_quant_packed: torch.Tensor,
    v_quant_packed: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_topk: torch.Tensor,
    v_topk: torch.Tensor,
    k_staging: torch.Tensor,
    v_staging: torch.Tensor,
    k_window: torch.Tensor,
    v_window: torch.Tensor,
    *,
    kv_repeat: int,
    k_bits: int = 4,
    v_bits: int = 4,
    sm_scale: Optional[float] = None,
    split_size: int = 1024,
    block_kv: int = 128,
) -> torch.Tensor:
    """Split-K decode with GQA-grouped Q heads + Tensor Core MMA.

    K and V each support {4, 2}-bit packing (K4V4 by default; K4V2, K2V4
    and K2V2 via k_bits / v_bits).
    """
    assert k_bits in (2, 4), f"k_bits must be 2 or 4, got {k_bits}"
    assert v_bits in (2, 4), f"v_bits must be 2 or 4, got {v_bits}"
    B, H_Q, _, D = q.shape
    assert D % 4 == 0     # required for 2-bit packing (4 channels/byte)
    H_KV = H_Q // kv_repeat
    BH_KV = B * H_KV
    assert k_quant_packed.shape[0] == BH_KV
    if sm_scale is None:
        sm_scale = D ** -0.5

    # MMA requires M >= 16 for bf16 (or 8 with reduced throughput). Pad to 16.
    if kv_repeat >= 16:
        KV_REPEAT_TILE = triton.next_power_of_2(kv_repeat)
    else:
        KV_REPEAT_TILE = 16

    N_quant = k_quant_packed.shape[1] if k_quant_packed.numel() > 0 else 0
    K_PACK = D // 2 if k_bits == 4 else D // 4
    V_PACK = D // 2 if v_bits == 4 else D // 4

    # ---- Concat bf16 anchors (topk + staging + window) ----
    bf16_segs_k, bf16_segs_v = [], []
    for kk, vv in ((k_topk, v_topk), (k_staging, v_staging), (k_window, v_window)):
        if kk is not None and kk.numel() > 0 and kk.shape[1] > 0:
            bf16_segs_k.append(kk)
            bf16_segs_v.append(vv)
    if len(bf16_segs_k) == 0:
        k_flat = torch.zeros(BH_KV, 1, D, device=q.device, dtype=q.dtype)
        v_flat = torch.zeros(BH_KV, 1, D, device=q.device, dtype=q.dtype)
        N_flat = 0
    elif len(bf16_segs_k) == 1:
        k_flat = bf16_segs_k[0]; v_flat = bf16_segs_v[0]
        N_flat = int(k_flat.shape[1])
    else:
        k_flat = torch.cat(bf16_segs_k, dim=1)
        v_flat = torch.cat(bf16_segs_v, dim=1)
        N_flat = int(k_flat.shape[1])

    NUM_SPLITS = max(1, (N_quant + split_size - 1) // split_size)
    if N_quant == 0:
        k_quant_packed = torch.zeros(BH_KV, 1, K_PACK, device=q.device, dtype=torch.uint8)
        v_quant_packed = torch.zeros(BH_KV, 1, V_PACK, device=q.device, dtype=torch.uint8)
        k_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)
        v_scale = torch.zeros(BH_KV, 1, D, device=q.device, dtype=torch.float16)

    mid_o = torch.empty(B, H_Q, NUM_SPLITS, D, device=q.device, dtype=torch.float32)
    mid_m = torch.empty(B, H_Q, NUM_SPLITS, device=q.device, dtype=torch.float32)
    mid_l = torch.empty(B, H_Q, NUM_SPLITS, device=q.device, dtype=torch.float32)

    grid_scan = (B * H_KV, NUM_SPLITS)
    _splitk_scan_kernel_mma[grid_scan](
        q, k_quant_packed, v_quant_packed,
        k_scale, v_scale,
        mid_o, mid_m, mid_l,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
        v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        N_quant,
        H_KV=H_KV, KV_REPEAT=kv_repeat, KV_REPEAT_TILE=KV_REPEAT_TILE,
        D=D, D_HALF=D // 2,
        SPLIT_SIZE=split_size, BLOCK_KV=block_kv,
        GROUP_SZ=GROUP_SIZE,
        K_BITS=k_bits, V_BITS=v_bits,
        num_warps=4, num_stages=3,
    )

    out = torch.empty(B, H_Q, 1, D, device=q.device, dtype=q.dtype)
    BLOCK_FLAT = 64 if N_flat > 64 else max(N_flat, 16)
    grid_reduce = (B * H_KV,)
    _reduce_and_flat_kernel_mma[grid_reduce](
        q, k_flat, v_flat,
        mid_o, mid_m, mid_l,
        out, sm_scale,
        q.stride(0), q.stride(1), q.stride(3),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), mid_o.stride(3),
        mid_m.stride(0), mid_m.stride(1), mid_m.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        N_flat,
        H_KV=H_KV, KV_REPEAT=kv_repeat, KV_REPEAT_TILE=KV_REPEAT_TILE,
        D=D, NUM_SPLITS=NUM_SPLITS,
        BLOCK_FLAT=BLOCK_FLAT,
        num_warps=4, num_stages=2,
    )
    return out


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("test_fused_decode_k4v4 (GQA-native, optimised) ...")
    B, H_Q, H_KV, D = 1, 8, 2, 64
    KV_REPEAT = H_Q // H_KV
    BH_KV = B * H_KV
    N_quant = 256
    G = (N_quant + GROUP_SIZE - 1) // GROUP_SIZE
    W = 8

    q = torch.randn(B, H_Q, 1, D, device="cuda", dtype=torch.bfloat16)
    k_quant = torch.randint(0, 256, (BH_KV, N_quant, D // 2), device="cuda", dtype=torch.uint8)
    v_quant = torch.randint(0, 256, (BH_KV, N_quant, D // 2), device="cuda", dtype=torch.uint8)
    k_scale = torch.rand(BH_KV, G, D, device="cuda", dtype=torch.float16) * 0.1
    v_scale = torch.rand(BH_KV, G, D, device="cuda", dtype=torch.float16) * 0.1
    k_topk = torch.randn(BH_KV, 4, D, device="cuda", dtype=torch.bfloat16)
    v_topk = torch.randn(BH_KV, 4, D, device="cuda", dtype=torch.bfloat16)
    k_staging = torch.randn(BH_KV, 2, D, device="cuda", dtype=torch.bfloat16)
    v_staging = torch.randn(BH_KV, 2, D, device="cuda", dtype=torch.bfloat16)
    k_window = torch.randn(BH_KV, W, D, device="cuda", dtype=torch.bfloat16)
    v_window = torch.randn(BH_KV, W, D, device="cuda", dtype=torch.bfloat16)

    out = fused_decode_attention_k4v4(
        q, k_quant, v_quant, k_scale, v_scale,
        k_topk, v_topk, k_staging, v_staging,
        k_window, v_window, kv_repeat=KV_REPEAT,
    )
    assert out.shape == (B, H_Q, 1, D), f"Bad shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in fused output"
    print(f"  PASSED: shape={out.shape}")

    # Quick numerical sanity vs reference (PyTorch dequant + matmul)
    print("test_fused_decode_k4v4 numerical sanity ...")
    # Reference: dequantize then attention
    from src.ops.decode_d_stage import dequantize_k_int4_channel
    k_rest_ref = dequantize_k_int4_channel(k_quant, k_scale, D).to(q.dtype)
    v_rest_ref = dequantize_k_int4_channel(v_quant, v_scale, D).to(q.dtype)
    k_all_ref = torch.cat([k_rest_ref, k_topk, k_staging, k_window], dim=1)
    v_all_ref = torch.cat([v_rest_ref, v_topk, v_staging, v_window], dim=1)
    N_total = k_all_ref.shape[1]
    # GQA expand
    k_all_ref = k_all_ref.reshape(B, H_KV, N_total, D).unsqueeze(2).expand(B, H_KV, KV_REPEAT, N_total, D).reshape(B, H_Q, N_total, D)
    v_all_ref = v_all_ref.reshape(B, H_KV, N_total, D).unsqueeze(2).expand(B, H_KV, KV_REPEAT, N_total, D).reshape(B, H_Q, N_total, D)
    sm = D ** -0.5
    scores = (q @ k_all_ref.transpose(-1, -2)) * sm
    ref_out = torch.softmax(scores, dim=-1) @ v_all_ref

    mse = ((out.float() - ref_out.float()) ** 2).mean().item()
    max_abs = (out.float() - ref_out.float()).abs().max().item()
    print(f"  MSE vs PyTorch reference: {mse:.3e}, max_abs: {max_abs:.3e}")
    assert mse < 1.0, f"MSE too high: {mse}"

    print("test_splitk_decode_k4v4 (FlashDecoding-style) ...")
    out_sk = splitk_decode_attention_k4v4(
        q, k_quant, v_quant, k_scale, v_scale,
        k_topk, v_topk, k_staging, v_staging,
        k_window, v_window, kv_repeat=KV_REPEAT,
        split_size=128, block_kv=128,
    )
    assert out_sk.shape == (B, H_Q, 1, D), f"Bad shape: {out_sk.shape}"
    assert not torch.isnan(out_sk).any(), "NaN detected in split-K output"

    # Numerical match against the same PyTorch reference + the single-kernel fused
    mse_sk_ref = ((out_sk.float() - ref_out.float()) ** 2).mean().item()
    mse_sk_fused = ((out_sk.float() - out.float()) ** 2).mean().item()
    max_abs_sk_ref = (out_sk.float() - ref_out.float()).abs().max().item()
    print(f"  split-K vs PyTorch ref :  MSE={mse_sk_ref:.3e}  max_abs={max_abs_sk_ref:.3e}")
    print(f"  split-K vs fused single:  MSE={mse_sk_fused:.3e}")
    assert mse_sk_ref < 1.0, f"split-K MSE too high vs ref: {mse_sk_ref}"
    assert mse_sk_fused < 1e-2, f"split-K MSE vs fused too high: {mse_sk_fused}"

    print("test_splitk_decode_k4v4_mma (GQA-grouped + Tensor Core) ...")
    out_mma = splitk_decode_attention_k4v4_mma(
        q, k_quant, v_quant, k_scale, v_scale,
        k_topk, v_topk, k_staging, v_staging,
        k_window, v_window, kv_repeat=KV_REPEAT,
        split_size=128, block_kv=128,
    )
    assert out_mma.shape == (B, H_Q, 1, D), f"Bad shape: {out_mma.shape}"
    assert not torch.isnan(out_mma).any(), "NaN detected in MMA output"
    mse_mma_ref = ((out_mma.float() - ref_out.float()) ** 2).mean().item()
    mse_mma_sk = ((out_mma.float() - out_sk.float()) ** 2).mean().item()
    max_abs_mma = (out_mma.float() - ref_out.float()).abs().max().item()
    print(f"  MMA vs PyTorch ref     :  MSE={mse_mma_ref:.3e}  max_abs={max_abs_mma:.3e}")
    print(f"  MMA vs split-K (FFMA)  :  MSE={mse_mma_sk:.3e}")
    assert mse_mma_ref < 1.0, f"MMA MSE too high vs ref: {mse_mma_ref}"
    assert mse_mma_sk < 1e-1, f"MMA MSE vs split-K too high: {mse_mma_sk}"

    # ---------------------------------------------------------------
    # K4V2 path: same K layout, V repacked to int2 (4 channels / byte)
    # ---------------------------------------------------------------
    print("test_splitk_decode_k4v2_mma (V int2 path) ...")
    from src.ops.decode_d_stage import (
        quantize_v_int2_channel,
        dequantize_v_int2_channel,
    )

    # Re-quantise V as int2 from a fresh bf16 reference V tensor so we have a
    # PyTorch reference that exactly matches the kernel's dequant.
    v_ref_bf16 = torch.randn(BH_KV, N_quant, D, device="cuda", dtype=torch.bfloat16)
    v_quant_int2, v_scale_int2 = quantize_v_int2_channel(v_ref_bf16)

    out_mma_v2 = splitk_decode_attention_k4v4_mma(
        q,
        k_quant, v_quant_int2,
        k_scale, v_scale_int2,
        k_topk, v_topk,
        k_staging, v_staging,
        k_window, v_window,
        kv_repeat=KV_REPEAT,
        v_bits=2,
        split_size=128, block_kv=128,
    )
    assert out_mma_v2.shape == (B, H_Q, 1, D), f"Bad shape: {out_mma_v2.shape}"
    assert not torch.isnan(out_mma_v2).any(), "NaN detected in K4V2 MMA output"

    # Build a PyTorch K4V2 reference (dequant K int4 + V int2 → bf16, then attention)
    from src.ops.decode_d_stage import dequantize_k_int4_channel
    k_ref_v2 = dequantize_k_int4_channel(k_quant, k_scale, D).to(q.dtype)
    v_ref_v2 = dequantize_v_int2_channel(v_quant_int2, v_scale_int2, D).to(q.dtype)
    k_all_v2 = torch.cat([k_ref_v2, k_topk, k_staging, k_window], dim=1)
    v_all_v2 = torch.cat([v_ref_v2, v_topk, v_staging, v_window], dim=1)
    N_total_v2 = k_all_v2.shape[1]
    k_all_v2 = (k_all_v2.reshape(B, H_KV, N_total_v2, D).unsqueeze(2)
                .expand(B, H_KV, KV_REPEAT, N_total_v2, D)
                .reshape(B, H_Q, N_total_v2, D))
    v_all_v2 = (v_all_v2.reshape(B, H_KV, N_total_v2, D).unsqueeze(2)
                .expand(B, H_KV, KV_REPEAT, N_total_v2, D)
                .reshape(B, H_Q, N_total_v2, D))
    sm = D ** -0.5
    scores_v2 = (q @ k_all_v2.transpose(-1, -2)) * sm
    ref_out_v2 = torch.softmax(scores_v2, dim=-1) @ v_all_v2

    mse_v2_ref = ((out_mma_v2.float() - ref_out_v2.float()) ** 2).mean().item()
    max_abs_v2 = (out_mma_v2.float() - ref_out_v2.float()).abs().max().item()
    print(f"  K4V2 MMA vs PyTorch ref:  MSE={mse_v2_ref:.3e}  max_abs={max_abs_v2:.3e}")
    assert mse_v2_ref < 1.0, f"K4V2 MMA MSE too high vs ref: {mse_v2_ref}"

    # ---------------------------------------------------------------
    # K2V4 path: K repacked to int2 (4 channels / byte), V stays int4
    # ---------------------------------------------------------------
    print("test_splitk_decode_k2v4_mma (K int2 path) ...")
    # quantize_v_int2_channel works for K too (logic is generic).
    k_ref_bf16 = torch.randn(BH_KV, N_quant, D, device="cuda", dtype=torch.bfloat16)
    k_quant_int2, k_scale_int2 = quantize_v_int2_channel(k_ref_bf16)

    out_mma_k2 = splitk_decode_attention_k4v4_mma(
        q,
        k_quant_int2, v_quant,
        k_scale_int2, v_scale,
        k_topk, v_topk,
        k_staging, v_staging,
        k_window, v_window,
        kv_repeat=KV_REPEAT,
        k_bits=2, v_bits=4,
        split_size=128, block_kv=128,
    )
    assert out_mma_k2.shape == (B, H_Q, 1, D), f"Bad shape: {out_mma_k2.shape}"
    assert not torch.isnan(out_mma_k2).any(), "NaN detected in K2V4 MMA output"

    # PyTorch K2V4 reference
    k_ref_k2 = dequantize_v_int2_channel(k_quant_int2, k_scale_int2, D).to(q.dtype)
    v_ref_k2 = dequantize_k_int4_channel(v_quant, v_scale, D).to(q.dtype)
    k_all_k2 = torch.cat([k_ref_k2, k_topk, k_staging, k_window], dim=1)
    v_all_k2 = torch.cat([v_ref_k2, v_topk, v_staging, v_window], dim=1)
    N_total_k2 = k_all_k2.shape[1]
    k_all_k2 = (k_all_k2.reshape(B, H_KV, N_total_k2, D).unsqueeze(2)
                .expand(B, H_KV, KV_REPEAT, N_total_k2, D)
                .reshape(B, H_Q, N_total_k2, D))
    v_all_k2 = (v_all_k2.reshape(B, H_KV, N_total_k2, D).unsqueeze(2)
                .expand(B, H_KV, KV_REPEAT, N_total_k2, D)
                .reshape(B, H_Q, N_total_k2, D))
    scores_k2 = (q @ k_all_k2.transpose(-1, -2)) * sm
    ref_out_k2 = torch.softmax(scores_k2, dim=-1) @ v_all_k2

    mse_k2_ref = ((out_mma_k2.float() - ref_out_k2.float()) ** 2).mean().item()
    max_abs_k2 = (out_mma_k2.float() - ref_out_k2.float()).abs().max().item()
    print(f"  K2V4 MMA vs PyTorch ref:  MSE={mse_k2_ref:.3e}  max_abs={max_abs_k2:.3e}")
    assert mse_k2_ref < 1.0, f"K2V4 MMA MSE too high vs ref: {mse_k2_ref}"
    print("All tests passed!")
