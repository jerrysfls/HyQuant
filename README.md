# HyQuant: Hybrid-Precision Quantization for LLM Attention

Official implementation of **"HyQuant: Hybrid-Precision Quantization for LLM Attention"** (EMNLP 2026).

## Overview

Uniformly pushing LLM attention to very low bit-widths often breaks long-context
chain-of-thought reasoning: a single bit-width for all tokens amplifies errors at a
small number of accuracy-critical positions while wasting precision budget elsewhere.

HyQuant is built on a simple, recurring observation: attention maps are dominated by
**persistent vertical lines** — a tiny fraction of key positions (typically < 5%) that
are repeatedly attended to by most queries. On Llama-3.1-8B / Qwen3-8B, the global
top-5% key positions plus a 128-token local window already cover **over 80%** of total
attention mass, and quantization errors are most destructive exactly at these
high-score positions.

HyQuant therefore keeps this tiny critical set in full precision and quantizes the
remaining majority to low bits, with three components:

1. **Vertical-line-aware retention** — a lightweight running column-mass score
   (updated every 64 tokens) identifies the top-ρ vertical-line positions online,
   with only 3–5% runtime overhead.
2. **Prefill: fused hybrid-precision attention** — the bulk of the attention GEMMs run
   in low precision (INT8/FP8, INT4/FP4 depending on backend) while vertical-line and
   local-window tiles stay in FP16/BF16, fused into a single FlashAttention-style
   Triton kernel.
3. **Decode: hybrid low-bit KV cache + fused attention** — the KV cache is stored in
   K4V4 for the majority and FP16 for vertical-line / window positions;
   dequantization is fused into the decode attention kernel (online softmax over
   quantized blocks), cutting memory traffic and bandwidth pressure.

Across Qwen3-8B, Qwen3-32B, Llama-3.1-8B-Instruct, and GLM-4-9B-0414 on LongBench,
GSM8K, and MATH500, HyQuant achieves **1.32×–3.58× decode-kernel speedup** and
**1.04×–1.17× end-to-end decode speedup** while maintaining near-full-precision
accuracy — and clearly improving over strict low-bit baselines.

## Partial Results

### Accuracy: LongBench v1 (average score)

HyQuant uses the K4V4 + top-5% vertical-line + local-window configuration. Baselines
are full-precision FlashAttention-2 (FA2) and representative low-bit methods
(KIVI, KVTuner, SageAttention); "–" means the baseline was not applicable to that model.

| Method | Qwen3-8B (thinking) | Llama-3.1-8B-Instruct | GLM-4-9B-0414 | Qwen3-32B |
|---|---:|---:|---:|---:|
| FA2 (full precision) | 44.59 | 46.63 | 44.83 | 48.61 |
| KIVI | 37.68 | – | – | 41.98 |
| SageAttention | 38.13 | 45.68 | 45.75 | – |
| KVTuner (4-bit) | 40.45 | 45.66 | 45.63 | 48.22 |
| **HyQuant (K4V4, top-5%)** | **45.04** | **46.73** | **45.78** | 48.46 |

HyQuant stays within noise of full precision (occasionally slightly above it, which we
treat as evaluation variance) while strict low-bit baselines degrade — most visibly on
Qwen3-8B long-CoT thinking mode (44.59 → 37.68 for KIVI vs. 45.04 for HyQuant).

### Efficiency: decode attention kernel latency (ms/token, H100)

| Prefix length | FA2 | HyQuant | Speedup |
|---:|---:|---:|---:|
| 1,024 | 0.224 | 0.170 | 1.32× |
| 2,048 | 0.416 | 0.173 | 2.40× |
| 4,096 | 0.805 | 0.263 | 3.06× |
| 8,192 | 1.604 | 0.478 | 3.36× |
| 16,384 | 3.181 | 0.903 | 3.52× |
| 32,768 | 6.354 | 1.775 | **3.58×** |

### Efficiency: end-to-end decode speed relative to FA2

| Prefix length | KIVI | KVTuner | HyQuant |
|---:|---:|---:|---:|
| 1,024 | 0.72× | 0.78× | **1.04×** |
| 8,192 | 0.73× | 0.80× | **1.12×** |
| 32,768 | 0.69× | 0.72× | **1.17×** |

Unlike KV-quantization baselines whose dequantization overhead makes decoding *slower*
than FA2, HyQuant's fused dequant-attention kernel turns the bandwidth savings into
real end-to-end gains.

### Prefill numerical error

With FA2 as reference, retaining the top-5% vertical-line tokens and the local window
in full precision reduces the layer-wise MSE of the attention output by a large factor
compared with SageAttention across layers, and brings uniform 4-bit error down to near
the 8-bit level from 1K to 32K context.

### Overhead

- Vertical-line identification: 3–5% of total runtime (amortized, every 64 tokens).
- Query buffer: at most 64 FP16 query vectors (64 × H_Q × d × 2 bytes).
- Keeping 5% of tokens in FP16 grows the KV cache by ~15% vs. strict 4-bit — offset in
  practice by the speed and accuracy gains.

## Installation

### Requirements

| Component | Minimum | Tested |
|---|---|---|
| Python | 3.10 | 3.10.19 |
| CUDA Toolkit | 12.0 | 12.8 |
| PyTorch | 2.3 | 2.10.0+cu128 |
| Triton | 2.2 | 3.6.0 |
| Transformers | 4.45 | 4.52.0 |
| flash-attn (optional, FA2 baseline) | 2.5 | 2.8.3 |

GPU: SM80+ (A100/H100/RTX 40x0). The FP8-PV prefill path requires SM90 (H100-class).
All paper experiments were run on a single NVIDIA H100 80GB.

### Setup

```bash
conda create -n hyquant python=3.10 -y
conda activate hyquant

# PyTorch (CUDA 12.8 wheels; pulls the matching Triton automatically)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install transformers==4.52.0 accelerate==1.12.0 datasets==4.5.0 \
    tokenizers==0.21.4 safetensors einops numpy pandas matplotlib rouge tqdm

# Optional: FA2 baseline (builds against the installed torch)
pip install ninja packaging
pip install flash-attn==2.8.3 --no-build-isolation
```

For a different CUDA toolkit, swap the index URL (e.g. `.../whl/cu126`).

### Optional: baseline methods

Only needed to reproduce the baseline rows in the tables:

```bash
# KIVI / quantized-cache baselines via HF QuantizedCache
pip install hqq==0.2.8.post1 bitsandbytes

# MInference comparison (eval/eval_longbench_minference.py)
git clone https://github.com/microsoft/MInference && pip install -e ./MInference

# KVTuner baseline (provides the flexible_quant package)
git clone https://github.com/cmd2001/KVTuner && pip install -e ./KVTuner/flexible_quant
```

### Datasets

```bash
python eval/download_longbench_v2.py    # LongBench; GSM8K/MATH500 load via `datasets`
```

## Quick Start

### Patch a HuggingFace model

HyQuant is applied as an in-place attention patch; Qwen3, Llama-3, and GLM-4 are
auto-detected:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.patch import patch_attention_forward_model
from src.ops.d_stage_kv_cache import DStageKVCache

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Qwen3-8B", torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
).cuda()

# Full HyQuant configuration: K4V4 cache + top-5% vertical lines + 256-token window
model = patch_attention_forward_model(
    model, mode="hybrid_full", window_size=256, top_ratio=0.05,
)

# Decode-stage modes need the hybrid quantized KV cache
cache = DStageKVCache(
    num_layers=model.config.num_hidden_layers,
    k_bits=4, v_bits_fn=lambda layer_idx: 4,
    window_size=256, top_ratio=0.05, use_vert=True,
)
outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=1024)
```

### Mode reference

| Mode | Meaning |
|---|---|
| `dense` / `full_attention` | Full-precision baselines (HF attention / Triton FP16) |
| `sage_w0` / `sage_w` | INT8 SageAttention-style prefill, without / with local window |
| `vquant_vert` | Prefill hybrid precision with vertical-line retention |
| `d_k4v4` / `d_k8v8` / `d_k4v2` / `d_k2v4` | Strict low-bit decode KV cache (K/V bit-widths) |
| `d_k4v4_vert` (etc.) | Low-bit decode cache + full-precision vertical lines |
| `hybrid_full` | **Full HyQuant** (K4V4 + vertical lines + window) |
| `kivi` / `kvtuner` | Baselines via HF quantized cache (eval scripts only) |

Key hyperparameters: `window_size` (local full-precision window, default 256) and
`top_ratio` (vertical-line retention ratio ρ, default 0.05 = top-5%).

## Reproducing the Paper

```bash
cd script

# LongBench v1 (Tables 2–5)
./run.sh longbench --model-path /path/to/model --data-dir /path/to/longbench

# Math reasoning (GSM8K / MATH500)
./run.sh math --model-path /path/to/model --data-dir /path/to/data \
  --dataset math500 --mode hybrid_full
```

Or call an evaluator directly:

```bash
python eval/eval_longbench.py \
  --model_path /path/to/Qwen3-8B --data_dir /path/to/longbench \
  --task hotpotqa --mode hybrid_full --window_size 256 --top_ratio 0.05
```

## Repository Structure

```
HyQuant/
├── src/
│   ├── patch/                       # In-place attention patches
│   │   ├── _common.py               #   mode dispatch, vertical-line scoring, shared utils
│   │   ├── patch_qwen.py            #   Qwen3
│   │   ├── patch_llama.py           #   Llama-3
│   │   └── patch_glm.py             #   GLM-4
│   └── ops/                         # Triton kernels & KV caches
│       ├── sage_unified.py          #   INT8-QK + FP16-PV prefill attention (+ window/vertical variants)
│       ├── sage_unified_fp8.py      #   FP8 prefill variants (SM90)
│       ├── sage_fp8_attention.py    #   INT8-QK + FP8-PV prefill attention (SM90)
│       ├── decode_d_stage.py        #   fused dequant + decode attention kernels
│       ├── d_stage_kv_cache.py      #   hybrid low-bit KV cache (DStageKVCache)
│       └── vendor/                  #   vendored SageAttention baseline
├── eval/                            # Evaluators & run scripts
│   ├── eval_longbench.py            #   LongBench v1
│   ├── eval_longbench_minference.py #   LongBench vs. MInference baseline
│   ├── eval_math_unified.py         #   GSM8K / MATH500 (unified entry)
│   ├── eval_mcq.py                  #   multiple-choice QA
│   ├── grader.py, parser.py         #   math answer grading utilities
│   ├── aggregate_*.py               #   result aggregation
│   └── download_longbench_v2.py     #   dataset download
├── script/                          # Env setup, dataset download, run wrappers
│   ├── install_env.sh               #   one-command venv setup
│   ├── download_datasets.sh         #   LongBench v1 etc.
│   ├── run.sh                       #   unified eval entry (longbench / math)
│   └── smoke_test_model.py          #   quick model sanity check
├── requirements.txt                 # Python dependencies (used by install_env.sh)
├── LICENSE
└── README.md
```

## Citation

```bibtex
@inproceedings{hyquant2026,
  title     = {HyQuant: Hybrid-Precision Quantization for LLM Attention},
  author    = {Ding, Jiatong and Xing, Bingxin and Zhang, Yu and Ding, Dian and
               Yi, Xiaodong and Ouyang, Xianbin and Zhou, Feihu and Zhang, Kun and
               Guo, Zhenyu and Pan, Hao and Xue, Guangtao and Zhang, Yiming},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
