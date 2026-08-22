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
Modified: replaced ``from . import _qattn_sm90`` with JIT compilation via
``torch.utils.cpp_extension.load()``.
"""

import os
import torch
from torch.utils.cpp_extension import load as _load_ext

# ---------------------------------------------------------------------------
# JIT-compile the SM90 attention CUDA extension on first import.
# WGMMA instructions require sm_90a (architecture-specific).  PyTorch's JIT
# auto-detects the GPU and adds -gencode for sm_90 (without 'a'), which fails.
# We override TORCH_CUDA_ARCH_LIST to force sm_90a only.
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_csrc = os.path.join(_dir, "csrc")

_ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

_prev_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

_qattn_sm90 = _load_ext(
    name="sageattention_qattn_sm90",
    sources=[
        os.path.join(_csrc, "qattn", "pybind_sm90.cpp"),
        os.path.join(_csrc, "qattn", "qk_int_sv_f8_cuda_sm90.cu"),
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
    extra_ldflags=["-lcuda"],
    verbose=False,
)

# Restore original TORCH_CUDA_ARCH_LIST
if _prev_arch_list is None:
    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = _prev_arch_list


# ---------------------------------------------------------------------------
# torch.library.custom_op wrappers (unchanged from upstream)
# ---------------------------------------------------------------------------

@torch.library.custom_op("sageattention_sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf", mutates_args=("output",), device_types="cuda")
def qk_int8_sv_f8_accum_f32_attn_inst_buf(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _qattn_sm90.qk_int8_sv_f8_accum_f32_attn_inst_buf(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )


@torch.library.register_fake("sageattention_sm90::qk_int8_sv_f8_accum_f32_attn_inst_buf")
def qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    batch_size = query.size(0)
    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)
    if return_lse:
        lse = torch.empty((batch_size, num_qo_heads, qo_len), dtype=torch.float32, device=query.device)
    else:
        lse = torch.empty((0))
    return lse


@torch.library.custom_op("sageattention_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", mutates_args=("output",), device_types="cuda")
def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        query, key, value, output, query_scale, key_scale, value_scale,
        tensor_layout, is_causal, qk_quant_gran, sm_scale, return_lse
    )


@torch.library.register_fake("sageattention_sm90::qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf")
def qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_lse: int,
) -> torch.Tensor:
    return qk_int8_sv_f8_accum_f32_attn_inst_buf_fake_impl(
        query, key, value, output, query_scale, key_scale, tensor_layout,
        is_causal, qk_quant_gran, sm_scale, return_lse
    )
