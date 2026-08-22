"""MInference-style sparse-only baseline.

Same vertical-line identification (compute_vertical_indices) and sliding
window as the hybrid path, but the rest of the prefix is masked to -inf
instead of quantised — isolating what the quantised long tail contributes
beyond a sparse mask.

Deliberately kernel-free (per-head softmax): slow but correct, and prefill
runs once per sample so ablation throughput is acceptable.
"""
from __future__ import annotations

from typing import Optional

import torch


def sparse_only_attention(
    q: torch.Tensor,           # [B, H_q, N, D]
    k: torch.Tensor,           # [B, H_kv, N, D]
    v: torch.Tensor,           # [B, H_kv, N, D]
    vidx: torch.Tensor,        # [B, H_q, top_k]  prefix-relative positions
    *,
    window_size: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Sparse-mask attention: only vert + sliding-window + causal-diag contribute.

    Per-head softmax with explicit mask; per-batch loop avoids materialising
    a [B, H, N, N] tensor.
    """
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    g = H_q // H_kv
    if sm_scale is None:
        sm_scale = D ** -0.5

    if g > 1:
        k_full = k.repeat_interleave(g, dim=1)
        v_full = v.repeat_interleave(g, dim=1)
    else:
        k_full, v_full = k, v

    prefix_len = max(0, N - window_size)
    out = torch.empty_like(q)
    NEG_INF = torch.finfo(q.dtype).min

    # Chunked along the Q dimension to keep peak memory O(BLOCK_M * N) instead
    # of O(N^2). For long context (N > ~10k) the full [N, N] mask + scores
    # tensor easily exceeds available HBM.
    BLOCK_M = 512
    arange_n = torch.arange(N, device=q.device)

    for h in range(H_q):
        for b in range(B):
            q_bh = q[b, h]                              # [N, D]
            k_bh = k_full[b, h]                         # [N, D]
            v_bh = v_full[b, h]                         # [N, D]

            # vidx for this (b, h)
            if vidx.numel() > 0 and vidx.shape[-1] > 0:
                cols = vidx[b, h].to(torch.long)        # [top_k]
            else:
                cols = None

            for m_start in range(0, N, BLOCK_M):
                m_end = min(m_start + BLOCK_M, N)
                q_chunk = q_bh[m_start:m_end]                                 # [Mc, D]
                scores = torch.matmul(q_chunk, k_bh.transpose(-1, -2)) * sm_scale  # [Mc, N]

                q_pos = arange_n[m_start:m_end]                               # [Mc]
                # Build [Mc, N] keep mask (much smaller than [N, N]).
                keep = torch.zeros(m_end - m_start, N, device=q.device, dtype=torch.bool)
                # 1) vertical-line columns (global, span all rows in chunk)
                if cols is not None:
                    keep[:, cols] = True
                # 2) per-query local sliding window [q_pos - W + 1, q_pos]
                if window_size > 0:
                    diff = q_pos[:, None] - arange_n[None, :]                 # [Mc, N]
                    keep |= (diff >= 0) & (diff < window_size)
                # 3) causal mask
                keep &= (arange_n[None, :] <= q_pos[:, None])

                scores = scores.masked_fill(~keep, NEG_INF)
                probs = torch.softmax(scores, dim=-1)
                out[b, h, m_start:m_end] = torch.matmul(probs, v_bh)

                del scores, keep, probs

    return out
