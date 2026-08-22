"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Vendored from https://github.com/thu-ml/SageAttention (core.py)
Extracted: ``sageattn_qk_int8_pv_fp8_cuda_sm90`` and its dependencies only.
Modified: import paths point to local vendored modules.
"""

import torch
from typing import Any, Optional

from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton
from .quant import per_warp_int8 as per_warp_int8_cuda
from .quant import per_channel_fp8

try:
    from . import sm90_compile
    SM90_ENABLED = True
except Exception:
    SM90_ENABLED = False


def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32
    accumulation, implemented using CUDA SM90 (WGMMA + TMA).

    Parameters
    ----------
    q : torch.Tensor
        Query tensor. HND: ``[B, H_q, N_q, D]``, NHD: ``[B, N_q, H_q, D]``.
    k : torch.Tensor
        Key tensor.   HND: ``[B, H_kv, N_kv, D]``, NHD: ``[B, N_kv, H_kv, D]``.
    v : torch.Tensor
        Value tensor. HND: ``[B, H_kv, N_kv, D]``, NHD: ``[B, N_kv, H_kv, D]``.
    tensor_layout : str
        ``"HND"`` or ``"NHD"``.
    is_causal : bool
        Apply causal mask (only when N_q == N_kv).
    qk_quant_gran : str
        ``"per_warp"`` or ``"per_thread"``.
    sm_scale : Optional[float]
        Softmax scale; defaults to ``1/sqrt(D)``.
    pv_accum_dtype : str
        ``"fp32+fp32"`` (recommended).
    smooth_k : bool
        Subtract key mean before quantisation.
    return_lse : bool
        Return log-sum-exp alongside output.

    Returns
    -------
    torch.Tensor  (or tuple if return_lse)
    """
    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure your GPUs have compute capability 9.0."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in (torch.float16, torch.bfloat16), "Input tensors must be float16 or bfloat16."
    assert qk_quant_gran in ("per_warp", "per_thread"), "qk_quant_gran must be 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_causal = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif 64 < head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, \
        "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og ** -0.5

    seq_dim = 1 if _tensor_layout == 0 else 2
    nh_dim = 2 if _tensor_layout == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(
                    q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)
                ).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(
                    q, km_broadcast.transpose(2, 3)
                ).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128,
        )
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
            q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128,
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    # pad v to multiple of 128
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
        else:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
    elif pv_accum_dtype == "fp32+fp32":
        lse = sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
            _tensor_layout, _is_causal, _qk_quant_gran, sm_scale, _return_lse,
        )

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
