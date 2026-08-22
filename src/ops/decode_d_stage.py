"""D-stage Decode Attention: per-token symmetric quantization.

q_len=1 decode kernel. Window + topk vertical tokens stay bf16.
Rest KV cache uses per-token quantized K/V (2, 4, or 8 bits).

Quantization: scale[n] = max(|x[n,:]|) / q_max -- one scalar per token.
Packing: INT4 = 2 per byte (even low, odd high), INT2 = 4 per byte.
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_LOG2_E = 1.44269504

GROUP_SIZE = 128  # staging flush threshold (tokens)


# ============================================================
# Triton JIT unpack helpers (split-Q natural-load pattern)
# ============================================================

@triton.jit
def _unpack_int4_even_odd(packed_i8):
    """Unpack two signed int4 from one int8 (even=lo nibble, odd=hi nibble)."""
    x = packed_i8.to(tl.int32)
    lo = (x << 28) >> 28   # sign-extend low 4 bits
    hi = (x >> 4) << 28 >> 28   # sign-extend high 4 bits
    return lo, hi


@triton.jit
def _unpack_int2_quads(packed_i8):
    """Unpack four signed int2 from one int8."""
    x = packed_i8.to(tl.int32)
    v0 = ((x & 0x03) << 30) >> 30
    v1 = (((x >> 2) & 0x03) << 30) >> 30
    v2 = (((x >> 4) & 0x03) << 30) >> 30
    v3 = (((x >> 6) & 0x03) << 30) >> 30
    return v0, v1, v2, v3


# ============================================================
# Per-token symmetric quantization utilities
# ============================================================

def quantize_per_token(x: torch.Tensor, nbits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric quantization.

    Args:
        x: [BH, N, D] bf16/fp16
        nbits: 2, 4, or 8

    Returns:
        packed: [BH, N, D_packed] int8  (D_packed = D for 8-bit, D//2 for 4-bit, D//4 for 2-bit)
        scale: [BH, N] fp16
    """
    BH, N, D = x.shape
    q_max = 2 ** (nbits - 1) - 1
    q_min = -(2 ** (nbits - 1))

    x_f = x.float()
    scale = (x_f.abs().amax(dim=-1).clamp(min=1e-5) / q_max).to(torch.float16)  # [BH, N]

    q = (x_f / scale.float().unsqueeze(-1)).round().clamp(q_min, q_max).to(torch.int8)  # [BH, N, D]

    if nbits == 8:
        return q.contiguous(), scale
    elif nbits == 4:
        return _pack_int4(q), scale
    elif nbits == 2:
        return _pack_int2(q), scale
    else:
        raise ValueError(f"nbits must be 2, 4, or 8, got {nbits}")


def dequantize_per_token(packed: torch.Tensor, scale: torch.Tensor, D: int, nbits: int) -> torch.Tensor:
    """Per-token dequantization.

    Args:
        packed: [BH, N, D_packed] int8
        scale: [BH, N] fp16
        D: original head dimension
        nbits: 2, 4, or 8

    Returns:
        [BH, N, D] fp16
    """
    if nbits == 8:
        q = packed.float()
    elif nbits == 4:
        q = _unpack_int4(packed, D)
    elif nbits == 2:
        q = _unpack_int2(packed, D)
    else:
        raise ValueError(f"nbits must be 2, 4, or 8, got {nbits}")

    return (q * scale.float().unsqueeze(-1)).to(torch.float16)


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack INT4: [BH, N, D] int8 -> [BH, N, D//2] int8.

    Even channels in low nibble, odd channels in high nibble.
    """
    u = q.to(torch.int16) & 0x0F
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).to(torch.int8)
    return packed.contiguous()


def _unpack_int4(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack INT4: [BH, N, D//2] int8 -> [BH, N, D] float32 (sign-extended)."""
    BH, N, _ = packed.shape
    x = packed.to(torch.int32) & 0xFF
    lo = ((x & 0x0F).to(torch.int32) << 28) >> 28
    hi = (((x >> 4) & 0x0F).to(torch.int32) << 28) >> 28
    out = torch.empty(BH, N, D, device=packed.device, dtype=torch.float32)
    out[..., 0::2] = lo.float()
    out[..., 1::2] = hi.float()
    return out


def _pack_int2(q: torch.Tensor) -> torch.Tensor:
    """Pack INT2: [BH, N, D] int8 -> [BH, N, D//4] int8.

    Channels 4i+0..4i+3 packed as bits [1:0], [3:2], [5:4], [7:6].
    """
    u = q.to(torch.int16) & 0x03
    packed = (u[..., 0::4] | (u[..., 1::4] << 2) | (u[..., 2::4] << 4) | (u[..., 3::4] << 6)).to(torch.int8)
    return packed.contiguous()


def _unpack_int2(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Unpack INT2: [BH, N, D//4] int8 -> [BH, N, D] float32 (sign-extended)."""
    BH, N, _ = packed.shape
    x = packed.to(torch.int32) & 0xFF
    v0 = ((x & 0x03).to(torch.int32) << 30) >> 30
    v1 = (((x >> 2) & 0x03).to(torch.int32) << 30) >> 30
    v2 = (((x >> 4) & 0x03).to(torch.int32) << 30) >> 30
    v3 = (((x >> 6) & 0x03).to(torch.int32) << 30) >> 30
    out = torch.empty(BH, N, D, device=packed.device, dtype=torch.float32)
    out[..., 0::4] = v0.float()
    out[..., 1::4] = v1.float()
    out[..., 2::4] = v2.float()
    out[..., 3::4] = v3.float()
    return out


# ============================================================
# Dispatch helpers (backward-compatible interface)
# ============================================================

def _quantize_by_bits(x: torch.Tensor, bits: int):
    """Quantize [BH, N, D] -> (packed, scale [BH, N])."""
    return quantize_per_token(x, bits)


def _dequantize_by_bits(packed, scale, D, bits):
    """Dequantize packed -> [BH, N, D] fp16."""
    return dequantize_per_token(packed, scale, D, bits)


# Backwards-compat aliases — d_stage_cache.py imports these channel-wise names.
def quantize_k_int4_channel(x: torch.Tensor):
    return quantize_per_token(x, 4)


def quantize_v_int2_channel(x: torch.Tensor):
    return quantize_per_token(x, 2)


def dequantize_k_int4_channel(packed: torch.Tensor, scale: torch.Tensor, D: int) -> torch.Tensor:
    return dequantize_per_token(packed, scale, D, 4)


def dequantize_v_int2_channel(packed: torch.Tensor, scale: torch.Tensor, D: int) -> torch.Tensor:
    return dequantize_per_token(packed, scale, D, 2)


def quantize_kv_d_stage(
    k: torch.Tensor,
    v: torch.Tensor,
    k_bits: int = 4,
    v_bits: int = 4,
) -> dict:
    """Quantize K and V per-token."""
    k_packed, k_scale = quantize_per_token(k, k_bits)
    v_packed, v_scale = quantize_per_token(v, v_bits)
    return {
        "k_packed": k_packed, "k_scale": k_scale,
        "v_packed": v_packed, "v_scale": v_scale,
        "k_bits": k_bits, "v_bits": v_bits,
    }


def dequantize_kv_d_stage(quant: dict, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize packed K/V back to fp16."""
    k_deq = dequantize_per_token(quant["k_packed"], quant["k_scale"], D, quant["k_bits"])
    v_deq = dequantize_per_token(quant["v_packed"], quant["v_scale"], D, quant["v_bits"])
    return k_deq, v_deq


# ============================================================
# Tensor Core Multi-Head Decode Kernels (GQA, per-token scales)
# ============================================================
#
# Grid: (BH_KV, ...) — each program handles KV_REPEAT Q heads
# M = max(16, next_power_of_2(kv_repeat)) for tl.dot constraints
# Uses tl.dot for QK and PV: tensor core >> scalar FMA (~3.3x speedup)
#
# Key patterns:
#   QK: split-Q [M, D//k] @ K_unpacked.T [D//k, BLOCK_KV] -> [M, BLOCK_KV]
#   PV: p [M, BLOCK_KV] @ V_unpacked [BLOCK_KV, D//v] -> [M, D//v]
#   bf16: direct tl.dot, no unpack needed
#   Per-token scale: factored OUT of dot product: qk_raw * scale[None, :]


@triton.jit
def _online_softmax_bf16_seg_2d(
    acc, l_i, m_i,       # [M, D] fp32, [M] fp32, [M] fp32
    Q_batch,              # [M, D] bf16
    q_mask,               # [M] bool
    K_ptr, V_ptr,
    stride_k_n, stride_k_d,
    stride_v_n, stride_v_d,
    kv_idx,
    stride_k_bh, stride_v_bh,
    N_seg,                # loop bound (max over batch when batched)
    actual_n_seg,         # actual length for this head (for masking)
    sm,
    M: tl.constexpr, D: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    """Process one bf16 KV segment with 2D online softmax + tensor core."""
    offs_d = tl.arange(0, D)
    offs_kv = tl.arange(0, BLOCK_KV)
    for start_n in range(0, N_seg, BLOCK_KV):
        n_offs = start_n + offs_kv
        n_mask = n_offs < actual_n_seg

        k = tl.load(
            K_ptr + kv_idx * stride_k_bh
            + n_offs[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.bfloat16)  # [BLOCK_KV, D]

        qk = tl.dot(Q_batch, tl.trans(k)) * sm  # [M, BLOCK_KV]
        qk = tl.where(q_mask[:, None] & n_mask[None, :], qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))  # [M]
        p = tl.math.exp2(qk - m_ij[:, None])  # [M, BLOCK_KV]
        l_ij = tl.sum(p, axis=1)  # [M]
        alpha = tl.math.exp2(m_i - m_ij)  # [M]
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]  # [M, D]

        v = tl.load(
            V_ptr + kv_idx * stride_v_bh
            + n_offs[:, None] * stride_v_n + offs_d[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.bfloat16)  # [BLOCK_KV, D]
        acc += tl.dot(p.to(tl.bfloat16), v)  # [M, D]

        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fused_d_stage_decode_kernel(
    Q,                              # [BH_Q, D]
    K_packed, V_packed,             # [BH_KV, N_quant, packed_dim]
    K_scale, V_scale,               # [BH_KV, N_quant] per-token fp16
    K_bf16, V_bf16,                 # [BH_KV, N_bf16, D] merged topk+staging+window
    Out,                            # [BH_Q, D]
    sm_scale,
    stride_q_bh, stride_q_d,
    stride_kp_bh, stride_kp_n, stride_kp_d,
    stride_vp_bh, stride_vp_n, stride_vp_d,
    stride_ks_bh, stride_ks_n,
    stride_vs_bh, stride_vs_n,
    stride_kb_bh, stride_kb_n, stride_kb_d,
    stride_vb_bh, stride_vb_n, stride_vb_d,
    stride_o_bh, stride_o_d,
    N_quant, N_bf16,
    seq_lens_quant_ptr,             # [BH_KV] per-head actual N_quant (or dummy)
    seq_lens_bf16_ptr,              # [BH_KV] per-head actual N_bf16 (or dummy)
    D: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    USE_SEQ_LENS: tl.constexpr = False,
):
    """Fused 4-segment decode with tensor core multi-head batching.

    Grid: (BH_KV,) — each program handles KV_REPEAT Q heads.
    Uses tl.dot for QK and PV. Gather V for quantized segments to keep
    single [M, D] accumulator compatible with bf16 segments.
    """
    pid_kv = tl.program_id(0)

    if USE_SEQ_LENS:
        actual_n_quant = tl.load(seq_lens_quant_ptr + pid_kv)
        actual_n_bf16 = tl.load(seq_lens_bf16_ptr + pid_kv)
    else:
        actual_n_quant = N_quant
        actual_n_bf16 = N_bf16
    offs_m = tl.arange(0, M)
    offs_d = tl.arange(0, D)
    offs_kv_blk = tl.arange(0, BLOCK_KV)
    q_mask = offs_m < KV_REPEAT

    q_base = pid_kv * KV_REPEAT

    # Load Q batch: [M, D] bf16
    Q_batch = tl.load(
        Q + (q_base + offs_m[:, None]) * stride_q_bh + offs_d[None, :] * stride_q_d,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    sm = sm_scale * 1.44269504

    m_i = tl.full([M], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)

    # Pre-split Q for quantized K (compiler DCEs unused branches)
    if K_BITS == 4:
        offs_kd = tl.arange(0, D // 2)
        Q_k_even = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 2) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)  # [M, D//2]
        Q_k_odd = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 2 + 1) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)  # [M, D//2]
    elif K_BITS == 2:
        offs_kd = tl.arange(0, D // 4)
        Q_k0 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k1 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 1) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k2 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 2) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k3 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 3) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)

    # V gather indices (for quantized V — fused kernel uses gather V)
    pack_idx_4 = offs_d // 2
    shift_4 = (offs_d % 2) * 4
    pack_idx_2 = offs_d // 4
    shift_2 = (offs_d % 4) * 2

    # == Segment 1: Quantized prefix (split-Q K + gather V + tl.dot) ==
    for start_n in range(0, N_quant, BLOCK_KV):
        n_offs = start_n + offs_kv_blk
        n_mask = n_offs < actual_n_quant

        k_sc = tl.load(
            K_scale + pid_kv * stride_ks_bh + n_offs * stride_ks_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        # K: split-Q + tl.dot tensor core
        if K_BITS == 4:
            kp = tl.load(
                K_packed + pid_kv * stride_kp_bh + n_offs[:, None] * stride_kp_n
                + offs_kd[None, :] * stride_kp_d,
                mask=n_mask[:, None], other=0,
            ).to(tl.int8)
            k_lo, k_hi = _unpack_int4_even_odd(kp)
            qk = (tl.dot(Q_k_even, tl.trans(k_lo.to(tl.bfloat16)))
                  + tl.dot(Q_k_odd, tl.trans(k_hi.to(tl.bfloat16)))) * k_sc[None, :]
        elif K_BITS == 8:
            k_int8 = tl.load(
                K_packed + pid_kv * stride_kp_bh + n_offs[:, None] * stride_kp_n
                + offs_d[None, :] * stride_kp_d,
                mask=n_mask[:, None], other=0,
            )
            qk = tl.dot(Q_batch, tl.trans(k_int8.to(tl.bfloat16))) * k_sc[None, :]
        elif K_BITS == 2:
            kp = tl.load(
                K_packed + pid_kv * stride_kp_bh + n_offs[:, None] * stride_kp_n
                + offs_kd[None, :] * stride_kp_d,
                mask=n_mask[:, None], other=0,
            ).to(tl.int8)
            k0, k1, k2, k3 = _unpack_int2_quads(kp)
            qk = (tl.dot(Q_k0, tl.trans(k0.to(tl.bfloat16)))
                  + tl.dot(Q_k1, tl.trans(k1.to(tl.bfloat16)))
                  + tl.dot(Q_k2, tl.trans(k2.to(tl.bfloat16)))
                  + tl.dot(Q_k3, tl.trans(k3.to(tl.bfloat16)))) * k_sc[None, :]

        qk = qk * sm
        qk = tl.where(q_mask[:, None] & n_mask[None, :], qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # V: gather mode with tl.dot (keeps [M, D] acc for bf16 segments)
        v_sc = tl.load(
            V_scale + pid_kv * stride_vs_bh + n_offs * stride_vs_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        p_v_sc = (p * v_sc[None, :]).to(tl.bfloat16)  # [M, BLOCK_KV]

        if V_BITS == 4:
            vp = tl.load(
                V_packed + pid_kv * stride_vp_bh + n_offs[:, None] * stride_vp_n
                + pack_idx_4[None, :] * stride_vp_d,
                mask=n_mask[:, None], other=0,
            ).to(tl.int32)
            v_int = (((vp >> shift_4[None, :]) & 0xF) << 28) >> 28
            acc += tl.dot(p_v_sc, v_int.to(tl.bfloat16))
        elif V_BITS == 8:
            v_int8 = tl.load(
                V_packed + pid_kv * stride_vp_bh + n_offs[:, None] * stride_vp_n
                + offs_d[None, :] * stride_vp_d,
                mask=n_mask[:, None], other=0,
            )
            acc += tl.dot(p_v_sc, v_int8.to(tl.bfloat16))
        elif V_BITS == 2:
            vp = tl.load(
                V_packed + pid_kv * stride_vp_bh + n_offs[:, None] * stride_vp_n
                + pack_idx_2[None, :] * stride_vp_d,
                mask=n_mask[:, None], other=0,
            ).to(tl.int32)
            v_int = (((vp >> shift_2[None, :]) & 0x3) << 30) >> 30
            acc += tl.dot(p_v_sc, v_int.to(tl.bfloat16))

        m_i = m_ij

    # == Segment 2: bf16 (merged topk + staging + window) ==
    acc, l_i, m_i = _online_softmax_bf16_seg_2d(
        acc, l_i, m_i, Q_batch, q_mask,
        K_bf16, V_bf16,
        stride_kb_n, stride_kb_d, stride_vb_n, stride_vb_d,
        pid_kv, stride_kb_bh, stride_vb_bh,
        N_bf16, actual_n_bf16, sm, M=M, D=D, BLOCK_KV=BLOCK_KV,
    )

    # Finalize: [M, D] / [M, 1] -> store KV_REPEAT valid rows
    out = acc / l_i[:, None]
    tl.store(
        Out + (q_base + offs_m[:, None]) * stride_o_bh + offs_d[None, :] * stride_o_d,
        out.to(Out.type.element_ty),
        mask=q_mask[:, None],
    )


# ============================================================
# Split-K Flash-Decoding (long-context acceleration)
# ============================================================
#
# Tensor core multi-head version:
# Grid = (BH_KV, S_SPLIT+1) — each program handles M Q heads
# Partial layout: [BH_KV * NUM_SPLITS, M, D] — full M rows per partial
# Reduce: Grid=(BH_Q,), extracts row h from [M, D] partials

@triton.jit
def _splitk_partial_kernel(
    Q,
    K_packed, V_packed,
    K_scale, V_scale,
    K_bf16, V_bf16,                     # [BH_KV, N_bf16, D] merged topk+staging+window
    Partial_out,                        # [BH_KV * NUM_SPLITS * M, D]
    Partial_m,                          # [BH_KV * NUM_SPLITS * M]
    Partial_l,                          # [BH_KV * NUM_SPLITS * M]
    sm_scale,
    stride_q_bh, stride_q_d,
    stride_kp_bh, stride_kp_n, stride_kp_d,
    stride_vp_bh, stride_vp_n, stride_vp_d,
    stride_ks_bh, stride_ks_n,
    stride_vs_bh, stride_vs_n,
    stride_kb_bh, stride_kb_n, stride_kb_d,
    stride_vb_bh, stride_vb_n, stride_vb_d,
    N_quant, N_bf16,
    seq_lens_quant_ptr,
    seq_lens_bf16_ptr,
    D: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_BF16: tl.constexpr,
    S_SPLIT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    USE_SEQ_LENS: tl.constexpr = False,
):
    """Split-K partial with tensor core multi-head batching.

    Grid: (BH_KV, NUM_SPLITS).
    Programs 0..S_SPLIT-1: quantized prefix chunks (split-Q K + natural V + tl.dot).
    Program S_SPLIT: bf16 segment (merged topk + staging + window).
    """
    pid_kv = tl.program_id(0)
    split_id = tl.program_id(1)

    if USE_SEQ_LENS:
        actual_n_quant = tl.load(seq_lens_quant_ptr + pid_kv)
        actual_n_bf16 = tl.load(seq_lens_bf16_ptr + pid_kv)
    else:
        actual_n_quant = N_quant
        actual_n_bf16 = N_bf16

    offs_m = tl.arange(0, M)
    offs_d = tl.arange(0, D)
    offs_kv_blk = tl.arange(0, BLOCK_KV)
    q_mask = offs_m < KV_REPEAT
    q_base = pid_kv * KV_REPEAT

    sm = sm_scale * 1.44269504

    # Load Q batch: [M, D] bf16
    Q_batch = tl.load(
        Q + (q_base + offs_m[:, None]) * stride_q_bh + offs_d[None, :] * stride_q_d,
        mask=q_mask[:, None], other=0.0,
    ).to(tl.bfloat16)

    # Pre-split Q for quantized K (compiler DCEs unused branches)
    if K_BITS == 4:
        offs_kd = tl.arange(0, D // 2)
        Q_k_even = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 2) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k_odd = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 2 + 1) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
    elif K_BITS == 2:
        offs_kd = tl.arange(0, D // 4)
        Q_k0 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k1 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 1) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k2 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 2) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)
        Q_k3 = tl.load(
            Q + (q_base + offs_m[:, None]) * stride_q_bh
            + (offs_kd[None, :] * 4 + 3) * stride_q_d,
            mask=q_mask[:, None], other=0.0,
        ).to(tl.bfloat16)

    m_i = tl.full([M], -1.0e6, dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)

    # All accumulators (compiler DCEs unused based on constexpr V_BITS)
    acc = tl.zeros([M, D], dtype=tl.float32)  # INT8 V + bf16 path
    if V_BITS == 4:
        offs_vd = tl.arange(0, D // 2)
        v_acc0 = tl.zeros([M, D // 2], dtype=tl.float32)
        v_acc1 = tl.zeros([M, D // 2], dtype=tl.float32)
    elif V_BITS == 2:
        offs_vd = tl.arange(0, D // 4)
        v_acc0 = tl.zeros([M, D // 4], dtype=tl.float32)
        v_acc1 = tl.zeros([M, D // 4], dtype=tl.float32)
        v_acc2 = tl.zeros([M, D // 4], dtype=tl.float32)
        v_acc3 = tl.zeros([M, D // 4], dtype=tl.float32)

    if split_id < S_SPLIT:
        # == Quantized prefix chunk (split-Q K + natural V + tl.dot) ==
        start_token = split_id * TOKENS_PER_SPLIT

        for start_n in range(0, TOKENS_PER_SPLIT, BLOCK_KV):
            abs_n = start_token + start_n + offs_kv_blk
            n_mask = abs_n < actual_n_quant

            # Per-token K scale
            k_sc = tl.load(
                K_scale + pid_kv * stride_ks_bh + abs_n * stride_ks_n,
                mask=n_mask, other=0.0,
            ).to(tl.float32)

            # K: split-Q + tl.dot tensor core
            if K_BITS == 4:
                kp = tl.load(
                    K_packed + pid_kv * stride_kp_bh + abs_n[:, None] * stride_kp_n
                    + offs_kd[None, :] * stride_kp_d,
                    mask=n_mask[:, None], other=0,
                ).to(tl.int8)
                k_lo, k_hi = _unpack_int4_even_odd(kp)
                qk = (tl.dot(Q_k_even, tl.trans(k_lo.to(tl.bfloat16)))
                      + tl.dot(Q_k_odd, tl.trans(k_hi.to(tl.bfloat16)))) * k_sc[None, :]
            elif K_BITS == 8:
                k_int8 = tl.load(
                    K_packed + pid_kv * stride_kp_bh + abs_n[:, None] * stride_kp_n
                    + offs_d[None, :] * stride_kp_d,
                    mask=n_mask[:, None], other=0,
                )
                qk = tl.dot(Q_batch, tl.trans(k_int8.to(tl.bfloat16))) * k_sc[None, :]
            elif K_BITS == 2:
                kp = tl.load(
                    K_packed + pid_kv * stride_kp_bh + abs_n[:, None] * stride_kp_n
                    + offs_kd[None, :] * stride_kp_d,
                    mask=n_mask[:, None], other=0,
                ).to(tl.int8)
                k0, k1, k2, k3 = _unpack_int2_quads(kp)
                qk = (tl.dot(Q_k0, tl.trans(k0.to(tl.bfloat16)))
                      + tl.dot(Q_k1, tl.trans(k1.to(tl.bfloat16)))
                      + tl.dot(Q_k2, tl.trans(k2.to(tl.bfloat16)))
                      + tl.dot(Q_k3, tl.trans(k3.to(tl.bfloat16)))) * k_sc[None, :]

            qk = qk * sm
            qk = tl.where(q_mask[:, None] & n_mask[None, :], qk, -1.0e6)

            # 2D online softmax
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            # Per-token V scale + tl.dot
            v_sc = tl.load(
                V_scale + pid_kv * stride_vs_bh + abs_n * stride_vs_n,
                mask=n_mask, other=0.0,
            ).to(tl.float32)
            p_v_sc = (p * v_sc[None, :]).to(tl.bfloat16)  # [M, BLOCK_KV]

            # V: natural load + split accumulators + tl.dot
            if V_BITS == 4:
                v_acc0 = v_acc0 * alpha[:, None]
                v_acc1 = v_acc1 * alpha[:, None]
                vp = tl.load(
                    V_packed + pid_kv * stride_vp_bh + abs_n[:, None] * stride_vp_n
                    + offs_vd[None, :] * stride_vp_d,
                    mask=n_mask[:, None], other=0,
                ).to(tl.int8)
                v_lo, v_hi = _unpack_int4_even_odd(vp)
                v_acc0 += tl.dot(p_v_sc, v_lo.to(tl.bfloat16))  # [M, D//2]
                v_acc1 += tl.dot(p_v_sc, v_hi.to(tl.bfloat16))  # [M, D//2]
            elif V_BITS == 8:
                acc = acc * alpha[:, None]
                v_int8 = tl.load(
                    V_packed + pid_kv * stride_vp_bh + abs_n[:, None] * stride_vp_n
                    + offs_d[None, :] * stride_vp_d,
                    mask=n_mask[:, None], other=0,
                )
                acc += tl.dot(p_v_sc, v_int8.to(tl.bfloat16))  # [M, D]
            elif V_BITS == 2:
                v_acc0 = v_acc0 * alpha[:, None]
                v_acc1 = v_acc1 * alpha[:, None]
                v_acc2 = v_acc2 * alpha[:, None]
                v_acc3 = v_acc3 * alpha[:, None]
                vp = tl.load(
                    V_packed + pid_kv * stride_vp_bh + abs_n[:, None] * stride_vp_n
                    + offs_vd[None, :] * stride_vp_d,
                    mask=n_mask[:, None], other=0,
                ).to(tl.int8)
                vv0, vv1, vv2, vv3 = _unpack_int2_quads(vp)
                v_acc0 += tl.dot(p_v_sc, vv0.to(tl.bfloat16))  # [M, D//4]
                v_acc1 += tl.dot(p_v_sc, vv1.to(tl.bfloat16))
                v_acc2 += tl.dot(p_v_sc, vv2.to(tl.bfloat16))
                v_acc3 += tl.dot(p_v_sc, vv3.to(tl.bfloat16))

            m_i = m_ij
    else:
        # == BF16 segment: merged topk + staging + window (tensor core) ==
        acc, l_i, m_i = _online_softmax_bf16_seg_2d(
            acc, l_i, m_i, Q_batch, q_mask,
            K_bf16, V_bf16,
            stride_kb_n, stride_kb_d, stride_vb_n, stride_vb_d,
            pid_kv, stride_kb_bh, stride_vb_bh,
            N_bf16, actual_n_bf16, sm, M=M, D=D, BLOCK_KV=BLOCK_BF16,
        )

    # Store partial results: [M, D] block per slot
    slot = pid_kv * NUM_SPLITS + split_id
    if split_id < S_SPLIT:
        if V_BITS == 4:
            # Interleave split accumulators back to [M, D]
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 2,
                v_acc0.to(Partial_out.type.element_ty),
            )
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 2 + 1,
                v_acc1.to(Partial_out.type.element_ty),
            )
        elif V_BITS == 2:
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 4,
                v_acc0.to(Partial_out.type.element_ty),
            )
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 4 + 1,
                v_acc1.to(Partial_out.type.element_ty),
            )
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 4 + 2,
                v_acc2.to(Partial_out.type.element_ty),
            )
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_vd[None, :] * 4 + 3,
                v_acc3.to(Partial_out.type.element_ty),
            )
        else:
            tl.store(
                Partial_out + (slot * M + offs_m[:, None]) * D + offs_d[None, :],
                acc.to(Partial_out.type.element_ty),
            )
    else:
        tl.store(
            Partial_out + (slot * M + offs_m[:, None]) * D + offs_d[None, :],
            acc.to(Partial_out.type.element_ty),
        )
    tl.store(Partial_m + slot * M + offs_m, m_i)
    tl.store(Partial_l + slot * M + offs_m, l_i)


@triton.jit
def _splitk_reduce_kernel(
    Partial_out,                        # [BH_KV * NUM_SPLITS * M, D]
    Partial_m,                          # [BH_KV * NUM_SPLITS * M]
    Partial_l,                          # [BH_KV * NUM_SPLITS * M]
    Out,                                # [BH_Q, D]
    D: tl.constexpr,
    M: tl.constexpr,
    KV_REPEAT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Merge NUM_SPLITS partials — extract per-Q-head row from [M, D] blocks."""
    pid_q = tl.program_id(0)  # indexes BH_Q
    pid_kv = pid_q // KV_REPEAT
    h = pid_q % KV_REPEAT  # row within [M, D] block

    offs_d = tl.arange(0, D)

    max_m = tl.full([1], -1.0e6, dtype=tl.float32)
    for s in range(NUM_SPLITS):
        slot = pid_kv * NUM_SPLITS + s
        m_s = tl.load(Partial_m + slot * M + h)
        max_m = tl.maximum(max_m, m_s)

    num = tl.zeros([D], dtype=tl.float32)
    den = tl.zeros([1], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        slot = pid_kv * NUM_SPLITS + s
        m_s = tl.load(Partial_m + slot * M + h)
        l_s = tl.load(Partial_l + slot * M + h)
        alpha = tl.math.exp2(m_s - max_m)
        acc_s = tl.load(Partial_out + (slot * M + h) * D + offs_d).to(tl.float32)
        num += alpha * acc_s
        den += alpha * l_s

    out = num / den
    tl.store(Out + pid_q * D + offs_d, out.to(Out.type.element_ty))


def _choose_s_split(N_quant: int, BH_KV: int) -> int:
    """Choose number of splits for the quantized prefix."""
    if N_quant <= 256:
        return 1
    target = 256
    s = max(1, target // BH_KV)
    max_s = max(1, N_quant // 256)
    s = min(s, max_s, 64)
    s = triton.next_power_of_2(s)
    return min(s, 64)


# Pre-allocated buffer cache to avoid per-call allocation overhead.
_splitk_buf_cache: dict = {}


def _get_splitk_buffers(BH_KV: int, num_splits: int, M: int, D: int,
                        device: torch.device):
    key = (BH_KV, num_splits, M, D, device)
    if key not in _splitk_buf_cache:
        total_slots = BH_KV * num_splits
        _splitk_buf_cache[key] = (
            torch.empty(total_slots * M, D, device=device, dtype=torch.float32),
            torch.empty(total_slots * M, device=device, dtype=torch.float32),
            torch.empty(total_slots * M, device=device, dtype=torch.float32),
        )
    return _splitk_buf_cache[key]


def _packed_dim(D: int, bits: int) -> int:
    if bits == 2:
        return D // 4
    elif bits == 4:
        return D // 2
    else:
        return D


def fused_d_stage_decode_splitk(
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
    k_bits: int,
    v_bits: int,
    kv_repeat: int = 1,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Split-K Flash-Decoding with tensor core multi-head batching."""
    BH_Q, D = q.shape
    BH_KV = BH_Q // kv_repeat
    M = max(16, triton.next_power_of_2(kv_repeat))

    N_quant = k_quant_packed.shape[1]

    # Merge bf16 segments into single contiguous tensor
    k_bf16 = torch.cat([k_topk, k_staging, k_window], dim=1)
    v_bf16 = torch.cat([v_topk, v_staging, v_window], dim=1)
    N_bf16 = k_bf16.shape[1]

    if sm_scale is None:
        sm_scale = D ** -0.5

    BLOCK_QUANT = 128
    BLOCK_BF16 = 64
    s_split = _choose_s_split(N_quant, BH_KV)
    num_splits = s_split + 1

    tps = (N_quant + s_split - 1) // s_split
    tokens_per_split = ((tps + BLOCK_QUANT - 1) // BLOCK_QUANT) * BLOCK_QUANT

    partial_out, partial_m, partial_l = _get_splitk_buffers(
        BH_KV, num_splits, M, D, q.device,
    )
    out = torch.empty(BH_Q, D, device=q.device, dtype=q.dtype)

    # Handle empty quantized segment
    if N_quant == 0:
        pk_d = _packed_dim(D, k_bits)
        pv_d = _packed_dim(D, v_bits)
        k_quant_packed = q.new_zeros(1, 1, pk_d, dtype=torch.int8)
        v_quant_packed = q.new_zeros(1, 1, pv_d, dtype=torch.int8)
        k_scale = q.new_zeros(1, 1, dtype=torch.float16)
        v_scale = q.new_zeros(1, 1, dtype=torch.float16)

    _splitk_partial_kernel[(BH_KV, num_splits)](
        q,
        k_quant_packed, v_quant_packed,
        k_scale, v_scale,
        k_bf16, v_bf16,
        partial_out, partial_m, partial_l,
        sm_scale,
        q.stride(0), q.stride(1),
        k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
        v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        k_bf16.stride(0), k_bf16.stride(1), k_bf16.stride(2),
        v_bf16.stride(0), v_bf16.stride(1), v_bf16.stride(2),
        N_quant, N_bf16,
        q,  # dummy seq_lens_quant_ptr
        q,  # dummy seq_lens_bf16_ptr
        D=D,
        K_BITS=k_bits, V_BITS=v_bits,
        KV_REPEAT=kv_repeat, M=M,
        BLOCK_KV=BLOCK_QUANT,
        BLOCK_BF16=BLOCK_BF16,
        S_SPLIT=s_split,
        NUM_SPLITS=num_splits,
        TOKENS_PER_SPLIT=tokens_per_split,
        USE_SEQ_LENS=False,
        num_warps=4,
        num_stages=2 if (k_bits == 8 and v_bits == 8) else 1,
    )

    _splitk_reduce_kernel[(BH_Q,)](
        partial_out, partial_m, partial_l, out,
        D=D, M=M, KV_REPEAT=kv_repeat, NUM_SPLITS=num_splits,
        num_warps=4,
    )

    return out


def fused_d_stage_decode(
    q: torch.Tensor,                        # [BH_Q, D]
    k_quant_packed: torch.Tensor,           # [BH_KV, N_quant, packed_kd]
    v_quant_packed: torch.Tensor,           # [BH_KV, N_quant, packed_vd]
    k_scale: torch.Tensor,                  # [BH_KV, N_quant] per-token
    v_scale: torch.Tensor,                  # [BH_KV, N_quant] per-token
    k_topk: torch.Tensor,                   # [BH_KV, N_topk, D]
    v_topk: torch.Tensor,                   # [BH_KV, N_topk, D]
    k_staging: torch.Tensor,                # [BH_KV, N_staging, D]
    v_staging: torch.Tensor,                # [BH_KV, N_staging, D]
    k_window: torch.Tensor,                 # [BH_KV, N_window, D]
    v_window: torch.Tensor,                 # [BH_KV, N_window, D]
    k_bits: int,
    v_bits: int,
    kv_repeat: int = 1,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused 4-segment decode: quantized prefix + topk + staging + window.

    Supports K/V bit widths in {2, 4, 8} and GQA via kv_repeat.
    Uses tensor core multi-head batching (tl.dot) for KV sharing.
    Uses split-K Flash-Decoding for long prefixes (>= 512 tokens).
    """
    BH_Q, D = q.shape
    N_quant = k_quant_packed.shape[1]

    # Use split-K for long quantized prefixes
    if N_quant >= 512:
        return fused_d_stage_decode_splitk(
            q, k_quant_packed, v_quant_packed,
            k_scale, v_scale,
            k_topk, v_topk,
            k_staging, v_staging,
            k_window, v_window,
            k_bits, v_bits, kv_repeat, sm_scale,
        )

    BH_KV = BH_Q // kv_repeat
    M = max(16, triton.next_power_of_2(kv_repeat))

    # Merge bf16 segments into single contiguous tensor
    k_bf16 = torch.cat([k_topk, k_staging, k_window], dim=1)
    v_bf16 = torch.cat([v_topk, v_staging, v_window], dim=1)
    N_bf16 = k_bf16.shape[1]

    if sm_scale is None:
        sm_scale = D ** -0.5

    out = torch.empty(BH_Q, D, device=q.device, dtype=q.dtype)
    BLOCK_KV = 64

    # Handle empty quantized segment
    if N_quant == 0:
        pk_d = _packed_dim(D, k_bits)
        pv_d = _packed_dim(D, v_bits)
        k_quant_packed = q.new_zeros(1, 1, pk_d, dtype=torch.int8)
        v_quant_packed = q.new_zeros(1, 1, pv_d, dtype=torch.int8)
        k_scale = q.new_zeros(1, 1, dtype=torch.float16)
        v_scale = q.new_zeros(1, 1, dtype=torch.float16)

    grid = (BH_KV,)
    _fused_d_stage_decode_kernel[grid](
        q,
        k_quant_packed, v_quant_packed,
        k_scale, v_scale,
        k_bf16, v_bf16,
        out,
        sm_scale,
        q.stride(0), q.stride(1),
        k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
        v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        k_bf16.stride(0), k_bf16.stride(1), k_bf16.stride(2),
        v_bf16.stride(0), v_bf16.stride(1), v_bf16.stride(2),
        out.stride(0), out.stride(1),
        N_quant, N_bf16,
        q,  # dummy seq_lens_quant_ptr (not used when USE_SEQ_LENS=False)
        q,  # dummy seq_lens_bf16_ptr
        D=D,
        K_BITS=k_bits,
        V_BITS=v_bits,
        KV_REPEAT=kv_repeat,
        M=M,
        BLOCK_KV=BLOCK_KV,
        USE_SEQ_LENS=False,
        num_warps=4,
        num_stages=2 if (k_bits == 8 and v_bits == 8) else 1,
    )
    return out


def fused_d_stage_decode_batched(
    q: torch.Tensor,                        # [total_BH_Q, D]
    k_quant_packed: torch.Tensor,           # [total_BH_KV, max_N_quant, packed_kd]
    v_quant_packed: torch.Tensor,           # [total_BH_KV, max_N_quant, packed_vd]
    k_scale: torch.Tensor,                  # [total_BH_KV, max_N_quant]
    v_scale: torch.Tensor,                  # [total_BH_KV, max_N_quant]
    k_bf16: torch.Tensor,                   # [total_BH_KV, max_N_bf16, D]
    v_bf16: torch.Tensor,                   # [total_BH_KV, max_N_bf16, D]
    seq_lens_quant: torch.Tensor,           # [total_BH_KV] per-head actual N_quant
    seq_lens_bf16: torch.Tensor,            # [total_BH_KV] per-head actual N_bf16
    k_bits: int,
    v_bits: int,
    kv_repeat: int = 1,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Batched fused 4-segment decode: multiple sequences in one kernel call.

    All sequences' KV tensors are concatenated along the BH dimension and padded
    to max length. Per-head seq_lens tensors tell the kernel each head's actual length.
    """
    BH_Q, D = q.shape
    BH_KV = BH_Q // kv_repeat
    N_quant = k_quant_packed.shape[1]  # max (padded)
    N_bf16 = k_bf16.shape[1]          # max (padded)
    M = max(16, triton.next_power_of_2(kv_repeat))

    if sm_scale is None:
        sm_scale = D ** -0.5

    # Handle empty quantized segment
    if N_quant == 0:
        pk_d = _packed_dim(D, k_bits)
        pv_d = _packed_dim(D, v_bits)
        k_quant_packed = q.new_zeros(1, 1, pk_d, dtype=torch.int8)
        v_quant_packed = q.new_zeros(1, 1, pv_d, dtype=torch.int8)
        k_scale = q.new_zeros(1, 1, dtype=torch.float16)
        v_scale = q.new_zeros(1, 1, dtype=torch.float16)

    out = torch.empty(BH_Q, D, device=q.device, dtype=q.dtype)

    # Split-K only helps when BH_KV is too small to fill SMs.
    # H100 has 132 SMs; when BH_KV >= 66 the base grid already provides
    # sufficient parallelism and the reduce kernel overhead hurts.
    _SM_THRESHOLD = 48
    use_splitk = N_quant >= 512 and BH_KV < _SM_THRESHOLD
    if use_splitk:
        # Split-K path
        BLOCK_QUANT = 128
        BLOCK_BF16 = 64
        s_split = _choose_s_split(N_quant, BH_KV)
        num_splits = s_split + 1
        tps = (N_quant + s_split - 1) // s_split
        tokens_per_split = ((tps + BLOCK_QUANT - 1) // BLOCK_QUANT) * BLOCK_QUANT

        partial_out, partial_m, partial_l = _get_splitk_buffers(
            BH_KV, num_splits, M, D, q.device,
        )
        _splitk_partial_kernel[(BH_KV, num_splits)](
            q,
            k_quant_packed, v_quant_packed,
            k_scale, v_scale,
            k_bf16, v_bf16,
            partial_out, partial_m, partial_l,
            sm_scale,
            q.stride(0), q.stride(1),
            k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
            v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
            k_scale.stride(0), k_scale.stride(1),
            v_scale.stride(0), v_scale.stride(1),
            k_bf16.stride(0), k_bf16.stride(1), k_bf16.stride(2),
            v_bf16.stride(0), v_bf16.stride(1), v_bf16.stride(2),
            N_quant, N_bf16,
            seq_lens_quant, seq_lens_bf16,
            D=D,
            K_BITS=k_bits, V_BITS=v_bits,
            KV_REPEAT=kv_repeat, M=M,
            BLOCK_KV=BLOCK_QUANT,
            BLOCK_BF16=BLOCK_BF16,
            S_SPLIT=s_split,
            NUM_SPLITS=num_splits,
            TOKENS_PER_SPLIT=tokens_per_split,
            USE_SEQ_LENS=True,
            num_warps=4,
            num_stages=2 if (k_bits == 8 and v_bits == 8) else 1,
        )
        _splitk_reduce_kernel[(BH_Q,)](
            partial_out, partial_m, partial_l, out,
            D=D, M=M, KV_REPEAT=kv_repeat, NUM_SPLITS=num_splits,
            num_warps=4,
        )
    else:
        # Non-split-K path — use larger block for long prefixes
        BLOCK_KV = 128 if N_quant >= 512 else 64
        grid = (BH_KV,)
        _fused_d_stage_decode_kernel[grid](
            q,
            k_quant_packed, v_quant_packed,
            k_scale, v_scale,
            k_bf16, v_bf16,
            out,
            sm_scale,
            q.stride(0), q.stride(1),
            k_quant_packed.stride(0), k_quant_packed.stride(1), k_quant_packed.stride(2),
            v_quant_packed.stride(0), v_quant_packed.stride(1), v_quant_packed.stride(2),
            k_scale.stride(0), k_scale.stride(1),
            v_scale.stride(0), v_scale.stride(1),
            k_bf16.stride(0), k_bf16.stride(1), k_bf16.stride(2),
            v_bf16.stride(0), v_bf16.stride(1), v_bf16.stride(2),
            out.stride(0), out.stride(1),
            N_quant, N_bf16,
            seq_lens_quant, seq_lens_bf16,
            D=D,
            K_BITS=k_bits,
            V_BITS=v_bits,
            KV_REPEAT=kv_repeat,
            M=M,
            BLOCK_KV=BLOCK_KV,
            USE_SEQ_LENS=True,
            num_warps=4,
            num_stages=2 if (k_bits == 8 and v_bits == 8) else 1,
        )
    return out
