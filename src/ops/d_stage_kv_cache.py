"""DStageKVCache: quantized KV cache for D-stage decode.

Prefill: accumulates bf16 KV (like DynamicCache).
After prefill: quantizes prefix KV at KV-head granularity, preserves window+topk bf16.
Decode: new tokens go to staging buffer, attention uses mixed segments.

Memory layout (all at KV-head granularity, NOT expanded to Q-heads):
  - k/v_quant_packed: [B*H_KV, N_prefix, D_packed] int8
  - k/v_scale: [B*H_KV, N_prefix] fp16 (per-token)
  - k/v_topk: [B*H_KV, topk, D] bf16
  - k/v_staging: [B*H_KV, S, D] bf16
  - k/v_window: [B*H_KV, W, D] bf16
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import torch

from src.ops.decode_d_stage import (
    GROUP_SIZE,
    _quantize_by_bits,
    _dequantize_by_bits,
    fused_d_stage_decode,
)
from src.patch._common import compute_vertical_indices

try:
    from transformers.cache_utils import Cache as _HF_Cache
except Exception:  # transformers not available
    _HF_Cache = object


class DStageLayerCache:
    """Per-layer KV cache with prefill-to-decode phase transition."""

    def __init__(
        self,
        layer_idx: int,
        k_bits: int = 4,
        v_bits: int = 4,
        window_size: int = 256,
        top_ratio: float = 0.05,
        use_vert: bool = True,
    ):
        self.layer_idx = layer_idx
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.window_size = window_size
        self.top_ratio = top_ratio
        self.use_vert = use_vert
        self.phase: str = "prefill"
        self._seq_len: int = 0

        # Prefill state
        self._keys: Optional[torch.Tensor] = None       # [B, H_KV, N, D]
        self._values: Optional[torch.Tensor] = None
        self._query_tail: Optional[torch.Tensor] = None  # [B, H_Q, W, D]

        # Decode segments (KV-head granularity)
        self.k_quant_packed: Optional[torch.Tensor] = None
        self.v_quant_packed: Optional[torch.Tensor] = None
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None
        self.k_topk: Optional[torch.Tensor] = None
        self.v_topk: Optional[torch.Tensor] = None
        self._staging_buf_k: Optional[torch.Tensor] = None
        self._staging_buf_v: Optional[torch.Tensor] = None
        self._staging_ptr: int = 0
        self.k_window: Optional[torch.Tensor] = None
        self.v_window: Optional[torch.Tensor] = None
        self._B: int = 0
        self._H_KV: int = 0
        self._D: int = 0

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accumulate bf16 KV during prefill (identical to DynamicCache)."""
        if self._keys is None:
            self._keys = key_states
            self._values = value_states
        else:
            self._keys = torch.cat([self._keys, key_states], dim=2)
            self._values = torch.cat([self._values, value_states], dim=2)
        self._seq_len = self._keys.shape[2]
        return self._keys, self._values

    def save_query_tail(self, query_states: torch.Tensor):
        """Save last W queries from prefill for topk computation at freeze."""
        W = min(self.window_size, query_states.shape[2])
        self._query_tail = query_states[:, :, -W:, :].detach()

    # ------------------------------------------------------------------
    # Transition: prefill -> decode
    # ------------------------------------------------------------------

    def freeze(self, kv_repeat: int = 1):
        """Quantize prefix, separate window+topk, release bf16 KV."""
        assert self.phase == "prefill" and self._keys is not None

        B, H_KV, N, D = self._keys.shape
        self._B, self._H_KV, self._D = B, H_KV, D
        BH = B * H_KV
        W = min(self.window_size, N)
        prefix_len = N - W
        device = self._keys.device
        dtype = self._keys.dtype

        # Window: ALWAYS preserved in bf16 (all D-stage modes)
        self.k_window = self._keys[:, :, -W:, :].reshape(BH, W, D).contiguous()
        self.v_window = self._values[:, :, -W:, :].reshape(BH, W, D).contiguous()

        if prefix_len > 0:
            k_prefix = self._keys[:, :, :prefix_len, :]    # [B, H_KV, prefix, D]
            v_prefix = self._values[:, :, :prefix_len, :]

            # Top-r% prefix tokens stay in bf16 (k_topk / v_topk) and must be
            # excluded from the quantised prefix, otherwise their attention mass
            # is double-counted across the int4 copy and the bf16 copy.
            if self.use_vert and self._query_tail is not None and kv_repeat >= 1:
                # Average Q across KV group: [B, H_Q, W, D] -> [B, H_KV, W, D]
                q_kv = self._query_tail.reshape(
                    B, H_KV, kv_repeat, -1, D,
                ).mean(dim=2)
                vidx = compute_vertical_indices(
                    q_kv, k_prefix, self.window_size, self.top_ratio,
                )  # [B, H_KV, topk]
                topk_count = int(vidx.shape[-1])

                # Gather bf16 anchors
                gather_d = vidx.unsqueeze(-1).expand(-1, -1, -1, D)
                self.k_topk = k_prefix.gather(2, gather_d).reshape(BH, -1, D).contiguous()
                self.v_topk = v_prefix.gather(2, gather_d).reshape(BH, -1, D).contiguous()

                # Build per-head compact index list of NON-topk positions.
                # Trick: mark topk positions with sentinel -1, sort, drop the
                # first `topk_count` entries (all the -1s, since they sort
                # below any non-negative position id).
                all_pos = torch.arange(
                    prefix_len, device=device, dtype=torch.int64,
                ).view(1, 1, prefix_len).expand(B, H_KV, prefix_len)
                marked = all_pos.scatter(2, vidx.to(torch.int64), -1)
                sorted_marked, _ = marked.sort(dim=2)
                compact_idx = sorted_marked[:, :, topk_count:]   # [B, H_KV, prefix-topk]

                k_compact = k_prefix.gather(
                    2, compact_idx.unsqueeze(-1).expand(-1, -1, -1, D),
                ).reshape(BH, prefix_len - topk_count, D).contiguous()
                v_compact = v_prefix.gather(
                    2, compact_idx.unsqueeze(-1).expand(-1, -1, -1, D),
                ).reshape(BH, prefix_len - topk_count, D).contiguous()
            else:
                self.k_topk = torch.zeros(BH, 0, D, device=device, dtype=dtype)
                self.v_topk = torch.zeros(BH, 0, D, device=device, dtype=dtype)
                k_compact = k_prefix.reshape(BH, prefix_len, D).contiguous()
                v_compact = v_prefix.reshape(BH, prefix_len, D).contiguous()

            # Quantise the compact (topk-excluded) prefix
            self.k_quant_packed, self.k_scale = _quantize_by_bits(k_compact, self.k_bits)
            self.v_quant_packed, self.v_scale = _quantize_by_bits(v_compact, self.v_bits)
        else:
            pack_k = D if self.k_bits == 8 else (D // 2 if self.k_bits == 4 else D // 4)
            pack_v = D if self.v_bits == 8 else (D // 2 if self.v_bits == 4 else D // 4)
            self.k_quant_packed = torch.zeros(BH, 0, pack_k, device=device, dtype=torch.int8)
            self.v_quant_packed = torch.zeros(BH, 0, pack_v, device=device, dtype=torch.int8)
            self.k_scale = torch.zeros(BH, 0, device=device, dtype=torch.float16)
            self.v_scale = torch.zeros(BH, 0, device=device, dtype=torch.float16)
            self.k_topk = torch.zeros(BH, 0, D, device=device, dtype=dtype)
            self.v_topk = torch.zeros(BH, 0, D, device=device, dtype=dtype)

        # Pre-allocated staging ring buffer (avoids torch.cat per token)
        self._staging_buf_k = torch.zeros(BH, GROUP_SIZE, D, device=device, dtype=dtype)
        self._staging_buf_v = torch.zeros(BH, GROUP_SIZE, D, device=device, dtype=dtype)
        self._staging_ptr = 0

        # Release full bf16 KV
        self._keys = None
        self._values = None
        self._query_tail = None
        self.phase = "decode"

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def append_staging(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Append new decode token(s) to pre-allocated staging ring buffer."""
        B, H_KV, S, D = key_states.shape
        BH = B * H_KV
        k = key_states.reshape(BH, S, D)
        v = value_states.reshape(BH, S, D)
        for i in range(S):
            self._staging_buf_k[:, self._staging_ptr, :] = k[:, i, :]
            self._staging_buf_v[:, self._staging_ptr, :] = v[:, i, :]
            self._staging_ptr += 1
            if self._staging_ptr >= GROUP_SIZE:
                self._flush_staging_buffer()
        self._seq_len += S

    def _flush_staging_buffer(self):
        """Quantize full staging buffer and append to quantized prefix."""
        kp, ks = _quantize_by_bits(self._staging_buf_k, self.k_bits)
        vp, vs = _quantize_by_bits(self._staging_buf_v, self.v_bits)
        self.k_quant_packed = torch.cat([self.k_quant_packed, kp], dim=1)
        self.v_quant_packed = torch.cat([self.v_quant_packed, vp], dim=1)
        self.k_scale = torch.cat([self.k_scale, ks], dim=1)
        self.v_scale = torch.cat([self.v_scale, vs], dim=1)
        self._staging_ptr = 0

    def decode_attention(self, query_states: torch.Tensor, kv_repeat: int) -> torch.Tensor:
        """Compute attention for q_len=1 using fused 4-segment Triton kernel.

        Fast path (K4V4): single fused Triton kernel that dequants int4 on-chip
        and walks 4 segments under one online-softmax — no HBM bf16 materialisation,
        no concat, no GQA expansion.

        Fallback path (other bit configs): dequant prefix in PyTorch, concat, expand
        GQA, then matmul (kept for K8V8, K4V2, etc.).

        Args:
            query_states: [B, H_Q, 1, D]
            kv_repeat: H_Q // H_KV
        Returns:
            [B, H_Q, 1, D]
        """
        B, H_KV, D = self._B, self._H_KV, self._D
        H_Q = H_KV * kv_repeat
        BH_KV = B * H_KV

        # Fast path: K/V independently 4- or 2-bit; higher-bit configs (K8V8
        # etc.) fall through to the PyTorch fallback below.
        # HYBRID_ATTN_DECODE_KERNEL picks the kernel:
        #   splitk_mma  — split-K + GQA-grouped Q heads + Tensor Core MMA (default)
        #   splitk      — split-K with per-Q-head program, FFMA path (K4V4 only)
        #   fused       — single-kernel fused dequant+attention (K4V4 only)
        #   pytorch     — PyTorch fallback (no Triton, debug only; all bit widths)
        if self.k_bits in (4, 2) and self.v_bits in (4, 2):
            import os as _os
            _kernel_choice = _os.environ.get("HYBRID_ATTN_DECODE_KERNEL", "splitk_mma").lower()
            if _kernel_choice != "pytorch":
                empty_bf16 = self.k_window.new_zeros(BH_KV, 0, D)
                k_topk = self.k_topk if self.k_topk is not None else empty_bf16
                v_topk = self.v_topk if self.v_topk is not None else empty_bf16
                # Staging buffer contains newly produced KV at decode steps
                # (waiting to be quantized into the prefix). Slice to the active
                # portion via _staging_ptr (0 means empty for the first decode step).
                if self._staging_buf_k is not None and self._staging_ptr > 0:
                    k_staging = self._staging_buf_k[:, :self._staging_ptr, :]
                    v_staging = self._staging_buf_v[:, :self._staging_ptr, :]
                else:
                    k_staging = empty_bf16
                    v_staging = empty_bf16

                # FFMA and fused kernels currently only support K int4 + V int4.
                # If user requested them with K=2 or V=2, fall back to splitk_mma
                # which handles all 4 (K,V) ∈ {4,2}² combinations.
                if _kernel_choice in ("fused", "splitk") and (self.k_bits != 4 or self.v_bits != 4):
                    _kernel_choice = "splitk_mma"

                if _kernel_choice == "fused":
                    from src.ops.d_stage_cache import fused_decode_attention_k4v4
                    return fused_decode_attention_k4v4(
                        query_states,
                        self.k_quant_packed, self.v_quant_packed,
                        self.k_scale, self.v_scale,
                        k_topk, v_topk, k_staging, v_staging,
                        self.k_window, self.v_window,
                        kv_repeat=kv_repeat,
                    )
                if _kernel_choice == "splitk":
                    from src.ops.d_stage_cache import splitk_decode_attention_k4v4
                    return splitk_decode_attention_k4v4(
                        query_states,
                        self.k_quant_packed, self.v_quant_packed,
                        self.k_scale, self.v_scale,
                        k_topk, v_topk, k_staging, v_staging,
                        self.k_window, self.v_window,
                        kv_repeat=kv_repeat,
                    )
                # default: fused per-token d-stage decode. _quantize_by_bits
                # stores k/v_scale as per-token 2D [BH, N_quant], while the
                # d_stage_cache.splitk_*_mma kernels expect a per-channel 3D
                # scale [BH, G, D] — hence the per-token kernel here.
                B_, H_Q_, _, D_ = query_states.shape
                out = fused_d_stage_decode(
                    query_states.reshape(B_ * H_Q_, D_),
                    self.k_quant_packed, self.v_quant_packed,
                    self.k_scale, self.v_scale,
                    k_topk, v_topk, k_staging, v_staging,
                    self.k_window, self.v_window,
                    self.k_bits, self.v_bits,
                    kv_repeat=kv_repeat,
                )
                return out.reshape(B_, H_Q_, 1, D_)
            # else fall through to the PyTorch fallback below

        # ------------------------------------------------------------------
        # Fallback path: dequant in PyTorch + concat + GQA expand + matmul
        # ------------------------------------------------------------------
        dtype = query_states.dtype
        device = query_states.device
        N_prefix = self.k_quant_packed.shape[1] if self.k_quant_packed is not None else 0
        if N_prefix > 0:
            k_prefix = _dequantize_by_bits(
                self.k_quant_packed, self.k_scale, D, self.k_bits,
            ).to(dtype)
            v_prefix = _dequantize_by_bits(
                self.v_quant_packed, self.v_scale, D, self.v_bits,
            ).to(dtype)
        else:
            k_prefix = torch.zeros(BH_KV, 0, D, device=device, dtype=dtype)
            v_prefix = torch.zeros(BH_KV, 0, D, device=device, dtype=dtype)

        segs_k = [k_prefix]
        segs_v = [v_prefix]
        if self.k_topk is not None and self.k_topk.shape[1] > 0:
            segs_k.append(self.k_topk.to(dtype))
            segs_v.append(self.v_topk.to(dtype))
        if (self._staging_buf_k is not None and self._staging_ptr > 0):
            segs_k.append(self._staging_buf_k[:, :self._staging_ptr, :].to(dtype))
            segs_v.append(self._staging_buf_v[:, :self._staging_ptr, :].to(dtype))
        segs_k.append(self.k_window.to(dtype))
        segs_v.append(self.v_window.to(dtype))

        k_all = torch.cat(segs_k, dim=1)
        v_all = torch.cat(segs_v, dim=1)
        N_total = k_all.shape[1]

        k_all = (
            k_all.reshape(B, H_KV, N_total, D)
            .unsqueeze(2)
            .expand(B, H_KV, kv_repeat, N_total, D)
            .reshape(B, H_Q, N_total, D)
        )
        v_all = (
            v_all.reshape(B, H_KV, N_total, D)
            .unsqueeze(2)
            .expand(B, H_KV, kv_repeat, N_total, D)
            .reshape(B, H_Q, N_total, D)
        )

        sm_scale = D ** -0.5
        scores = (query_states @ k_all.transpose(-1, -2)) * sm_scale
        attn_output = torch.softmax(scores, dim=-1) @ v_all
        return attn_output

    def get_seq_length(self) -> int:
        return self._seq_len


class DStageKVCache(_HF_Cache):
    """HF-compatible KV cache with D-stage quantization.

    Usage::

        cache = DStageKVCache(
            num_layers=32, k_bits=4,
            v_bits_fn=lambda i: 4 if i == 0 else 2,
            window_size=256, top_ratio=0.05, use_vert=True,
        )
        outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=256)
    """

    is_compileable = False  # HF generate() compatibility

    def __init__(
        self,
        num_layers: int = 32,
        k_bits: int | Callable[[int], int] = 4,
        v_bits_fn: Callable[[int], int] = lambda i: 4 if i == 0 else 2,
        window_size: int = 256,
        top_ratio: float = 0.05,
        use_vert: bool = True,
    ):
        try:
            super().__init__()
        except TypeError:
            pass
        self.num_layers = num_layers
        self.window_size = window_size
        self.top_ratio = top_ratio
        self.use_vert = use_vert
        k_bits_fn = k_bits if callable(k_bits) else (lambda i, _kb=k_bits: _kb)
        self._layers = [
            DStageLayerCache(
                layer_idx=i,
                k_bits=k_bits_fn(i),
                v_bits=v_bits_fn(i),
                window_size=window_size,
                top_ratio=top_ratio,
                use_vert=use_vert,
            )
            for i in range(num_layers)
        ]
        self._seen_tokens: int = 0

    # ------------------------------------------------------------------
    # HF Cache interface
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard HF cache update (used during prefill only for D-stage)."""
        layer = self._layers[layer_idx]
        if layer.phase == "prefill":
            result = layer.update_prefill(key_states, value_states)
            if layer_idx == 0:
                self._seen_tokens = layer.get_seq_length()
            return result
        else:
            # Decode: append staging (also called from patch forward directly)
            layer.append_staging(key_states, value_states)
            if layer_idx == 0:
                self._seen_tokens = layer.get_seq_length()
            return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._layers):
            return self._layers[layer_idx].get_seq_length()
        return 0

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self):
        for layer in self._layers:
            layer.__init__(
                layer.layer_idx, layer.k_bits, layer.v_bits,
                layer.window_size, layer.top_ratio, layer.use_vert,
            )
        self._seen_tokens = 0

    def get_mask_sizes(
        self,
        query_length: int,
        layer_idx: int = 0,
    ) -> Tuple[int, int]:
        """Return (kv_length, kv_offset) for HF causal mask construction."""
        kv_length = self.get_seq_length(layer_idx)
        if kv_length == 0:
            return query_length, 0
        kv_offset = max(0, kv_length - query_length)
        return kv_length, kv_offset

    def get_layer(self, layer_idx: int) -> DStageLayerCache:
        return self._layers[layer_idx]

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def batch_repeat_interleave(self, repeats: int):
        """For beam search compatibility (not optimized)."""
        raise NotImplementedError("DStageKVCache does not support beam search yet.")

    def batch_select_indices(self, indices: torch.Tensor):
        """For beam search compatibility (not optimized)."""
        raise NotImplementedError("DStageKVCache does not support beam search yet.")

    def crop(self, max_length: int):
        pass

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, idx: int):
        layer = self._layers[idx]
        if layer.phase == "prefill" and layer._keys is not None:
            return (layer._keys, layer._values)
        return (None, None)

    def __iter__(self):
        for i in range(len(self._layers)):
            yield self[i]

    def __bool__(self) -> bool:
        return True


QuantizedKVCache = DStageKVCache
