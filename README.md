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

1. **Vertical-line-aware retention** — at the end of prefill, the last `W` query
   vectors are mean-pooled and dotted against the non-window prefix keys; the top-ρ
   positions become the vertical-line set. This is a single matmul per layer, and the
   set is then frozen and reused for the whole decode phase.
2. **Prefill: fused hybrid-precision attention** — the bulk of QKᵀ runs on INT8 tensor
   cores (per-block scales, K-smoothing), PV runs in FP16 (or FP8 on SM90), while the
   vertical-line and local-window tiles stay in FP16/BF16. The three segments share
   one FlashAttention-style online softmax inside a single Triton kernel.
3. **Decode: hybrid low-bit KV cache + fused attention** — at the prefill→decode
   boundary the cache is reorganized into four segments: the quantized prefix (K4V4,
   per-token scales), the frozen vertical-line tokens (BF16), the last `W` prefill
   tokens (BF16), and a staging buffer that holds the most recent generated tokens in
   BF16 and flushes them to 4-bit in groups of 128. Dequantization is fused into the
   split-K decode kernel (one online softmax across all four segments), so no
   full-precision KV is ever materialized.

Across Qwen3-8B, Qwen3-32B, Llama-3.1-8B-Instruct, and GLM-4-9B-0414 on LongBench,
GSM8K, and MATH500, HyQuant achieves **1.32×–3.58× decode-kernel speedup** and
**1.04×–1.17× end-to-end decode speedup** while maintaining near-full-precision
accuracy — and clearly improving over strict low-bit baselines.

## Results

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

With FA2 as reference, the hybrid INT8 prefill kernel that retains the top-5%
vertical-line tokens and the local window in FP16 reduces the layer-wise MSE of the
attention output by a large factor compared with SageAttention across layers.
Prefill latency is on par with SageAttention.

### Overhead

- Vertical-line identification: one mean-pooled query × prefix-key matmul per layer
  at the prefill→decode boundary; 3–5% of total runtime in our measurements.
- Query buffer: the last `W` (default 256) query vectors per layer are kept in BF16
  during prefill and freed once the cache is frozen.
- Keeping 5% of tokens in BF16 grows the KV cache by ~15% vs. strict 4-bit — offset in
  practice by the speed and accuracy gains.

## Installation

### Requirements

| Component | Minimum | Tested |
|---|---|---|
| Python | 3.10 | 3.10 |
| CUDA Toolkit | 12.0 | 12.6 |
| PyTorch | 2.3 | 2.6.0+cu126 |
| Triton | 2.2 | 3.2.0 |
| Transformers | 4.45 | — |
| flash-attn (optional, FA2 baseline) | 2.5 | 2.7.2 |

GPU: SM80+ (A100/H100/RTX 40x0). The FP8-PV prefill path requires SM90 (H100-class).
All paper experiments were run on a single NVIDIA H100 80GB.

### Option A: one-command setup

```bash
cd script
./install_env.sh                    # creates ./.venv, installs torch cu126 + requirements
# ./install_env.sh --skip-flash-attn   # if you don't need the FA2 baseline
source activate_env.sh
```

Non-default CUDA versions: `./install_env.sh --torch-index-url https://download.pytorch.org/whl/cu124`.

### Option B: manual

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt     # add --no-build-isolation if flash-attn builds from source
```

### Datasets

```bash
cd script
./download_datasets.sh              # LongBench v1 etc.; see also eval/download_longbench_v2.py
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

# Full HyQuant configuration: hybrid INT8 prefill + K4V4 cache
# + top-5% vertical lines + 256-token window
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

| Mode | Prefill | Decode KV cache |
|---|---|---|
| `dense` / `full_attention` / `flash_attention_2` | full precision | full precision |
| `sage_w0` / `sage_w` | INT8 QK + FP16 PV, without / with local window | full precision |
| `vquant_vert` | INT8 QK + FP16 PV + FP16 vertical lines & window | full precision |
| `d_k4v4` / `d_k8v8` / `d_k4v2` / `d_k2v4` | full precision | strict low-bit (K/V bit-widths as named) |
| `d_k4v4_vert` (etc.) | full precision | low-bit + BF16 vertical lines & window |
| **`hybrid_full`** | same as `vquant_vert` | K4V4 + BF16 vertical lines & window (**full HyQuant**) |
| `hybrid_k4v2` / `hybrid_k2v4` | same as `vquant_vert` | K4V2 / K2V4 + BF16 vertical lines & window |
| `kivi` / `kvtuner` | full precision | HF `HQQQuantizedCache` baselines (eval scripts only) |

Key hyperparameters: `window_size` (local full-precision window, default 256) and
`top_ratio` (vertical-line retention ratio ρ, default 0.05 = top-5%).

## Reproducing the Paper

LongBench v1 (Tables 2–5). Pass `--modes` explicitly; the wrapper's default mode
list predates the current mode names.

```bash
cd script
./run.sh longbench --model-path /path/to/Qwen3-8B --data-dir /path/to/longbench \
  --modes flash_attention_2,kivi,kvtuner,sage_w,hybrid_full \
  --window-size 256 --top-ratio 0.05
```

Or call the evaluator directly for a single task:

```bash
python eval/eval_longbench.py \
  --model_path /path/to/Qwen3-8B --data_dir /path/to/longbench \
  --task hotpotqa --mode hybrid_full --window_size 256 --top_ratio 0.05
```

Note: the math evaluators under `eval/` (`eval_math_unified.py`,
`eval_math500_llama3.py`) currently accept only the legacy prefill-only mode names and
have not been migrated to the `hybrid_*` modes; use `eval_longbench.py` for the
HyQuant configuration.

## Repository Structure

```
HyQuant/
├── src/
│   ├── patch/                  # In-place attention patches
│   │   ├── _common.py          #   mode dispatch, vertical-line scoring, shared utils
│   │   ├── patch_qwen.py       #   Qwen3
│   │   ├── patch_llama.py      #   Llama-3
│   │   └── patch_glm.py        #   GLM-4
│   └── ops/                    # Triton kernels & KV caches
│       ├── sage_unified.py     #   INT8-QK + FP16-PV prefill attention (+ window/vertical variants)
│       ├── sage_fp8_attention.py #  INT8-QK + FP8-PV prefill attention (SM90)
│       ├── decode_d_stage.py   #   fused dequant + split-K decode attention kernels
│       └── d_stage_kv_cache.py #   hybrid low-bit KV cache (DStageKVCache, freeze/staging logic)
├── eval/                       # LongBench, GSM8K/MATH500, MCQ evaluators
├── script/                     # Env setup, dataset download, run wrappers
├── requirements.txt
└── LICENSE                     # MIT
```

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@inproceedings{hyquant2026,
  title     = {HyQuant: Hybrid-Precision Quantization for LLM Attention},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```
