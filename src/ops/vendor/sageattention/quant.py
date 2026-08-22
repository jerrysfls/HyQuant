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

Vendored from https://github.com/thu-ml/SageAttention
Modified: replaced ``from . import _fused`` with JIT compilation via
``torch.utils.cpp_extension.load()``.
"""

import os
import torch
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# JIT-compile the fused CUDA extension on first import
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_csrc = os.path.join(_dir, "csrc")

_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

from torch.utils.cpp_extension import load as _load_ext

_fused = _load_ext(
    name="sageattention_fused",
    sources=[
        os.path.join(_csrc, "fused", "pybind.cpp"),
        os.path.join(_csrc, "fused", "fused.cu"),
    ],
    extra_include_paths=[_csrc],
    extra_cflags=["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={_ABI}"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        f"-D_GLIBCXX_USE_CXX11_ABI={_ABI}",
    ],
    verbose=False,
)


# ---------------------------------------------------------------------------
# Quantization helpers (unchanged from upstream)
# ---------------------------------------------------------------------------

def per_block_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: Optional[float] = None,
    tensor_layout: str = "HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    sm_scale *= 1.44269504

    _fused.quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale


def per_warp_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    km: Optional[torch.Tensor] = None,
    BLKQ: int = 128,
    WARPQ: int = 32,
    BLKK: int = 64,
    tensor_layout: str = "HND",
):
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, ((qo_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, BLKQ, WARPQ, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale


def per_channel_fp8(
    v: torch.Tensor,
    tensor_layout: str = "HND",
    scale_max: float = 448.0,
    smooth_v: bool = True,
):
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype, device=v.device)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    _fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)

    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
    vm = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)

    if smooth_v:
        _fused.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        _fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None
