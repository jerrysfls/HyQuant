#!/usr/bin/env python3
"""LongBench v1 evaluator with MInference attention patching.

Mirrors eval_longbench.py output format exactly (per-sample JSONL + summary JSON).
Replaces the in-repo `patch_attention_forward_model(...)` step with MInference's
`MInference(attn_type=..., kv_type=..., model_name=<HF id>)`.

Notes on model support (current MInference release):
  - meta-llama/Llama-3.1-8B-Instruct   : supported natively (attn_type=minference)
  - THUDM/glm-4-9b-chat-1m             : supported natively
  - Qwen3-8B                           : NOT in MODEL2PATH. Falls back to
    attn_type=hf or dense (no sparse attention applied). Set --attn_type
    accordingly. Passing --minference_model_name lets you override the
    HF id MInference uses to look up its pattern config.

Usage:
  python eval/eval_longbench_minference.py \
    --model_path /path/to/Llama-3.1-8B-Instruct \
    --minference_model_name meta-llama/Llama-3.1-8B-Instruct \
    --task qasper --attn_type minference --kv_type dense \
    --data_dir /path/to/datasets \
    --output_file results/minference_qasper_llama31.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse prompts / metrics / postprocess / data loader from the in-repo evaluator
# so output is bit-for-bit comparable across attention backends.
from eval.eval_longbench import (  # noqa: E402
    TASK_CONFIG,
    build_prompt,
    get_ground_truth,
    load_longbench_v1,
    postprocess_prediction,
)

# Optional: point MINFERENCE_ROOT at a source checkout of MInference;
# otherwise the pip-installed package is used.
_MINFERENCE_ROOT = os.environ.get("MINFERENCE_ROOT", "")
if _MINFERENCE_ROOT and os.path.isdir(_MINFERENCE_ROOT) and _MINFERENCE_ROOT not in sys.path:
    sys.path.insert(0, _MINFERENCE_ROOT)
from minference import MInference  # noqa: E402


# Default mapping from local checkpoint dirname → MInference HF id used to look
# up the pattern config. Users override via --minference_model_name.
_DEFAULT_MN_NAME = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "GLM-4-9B-0414":         "THUDM/glm-4-9b-chat-1m",   # closest match in MODEL2PATH
    "qwen3-8B":              "Qwen/Qwen2.5-7B-Instruct", # NOT a real match — see notes
}


def parse_args():
    parser = argparse.ArgumentParser("LongBench evaluator (MInference)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=[t for t, c in TASK_CONFIG.items() if c["version"] == "v1"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--attn_type", type=str, default="minference",
                        choices=["minference", "hf", "dense", "a_shape", "tri_shape",
                                 "flexprefill", "xattention", "inf_llm",
                                 "tri_mix", "tri_mix_minference"])
    parser.add_argument("--kv_type", type=str, default="dense",
                        choices=["dense", "streamingllm", "snapkv",
                                 "pyramidkv", "quest", "retr_attn", "kivi"])
    parser.add_argument("--minference_model_name", type=str, default=None,
                        help="HF id used by MInference to look up the pattern "
                             "config. If omitted, inferred from the model_path "
                             "basename via a small built-in table.")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_input_len", type=int, default=24000)
    parser.add_argument("--skip_too_long", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.output_file is None:
        args.output_file = (
            f"results/longbench_{args.task}_minference_{args.attn_type}.jsonl"
        )
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    if args.minference_model_name is None:
        base = os.path.basename(os.path.normpath(args.model_path))
        args.minference_model_name = _DEFAULT_MN_NAME.get(base, base)

    cfg = TASK_CONFIG[args.task]
    if cfg["version"] != "v1":
        raise ValueError("This script only handles LongBench v1 tasks.")

    print(f"=== Config ===")
    print(f"  model_path:             {args.model_path}")
    print(f"  minference_model_name:  {args.minference_model_name}")
    print(f"  attn_type:              {args.attn_type}")
    print(f"  kv_type:                {args.kv_type}")
    print(f"  task:                   {args.task}")
    print(f"  output_file:            {args.output_file}")

    # Load data
    print(f"Loading {args.task} from {args.data_dir} ...")
    samples = load_longbench_v1(args.data_dir, args.task, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # MInference requires flash_attention_2 for its kernels (or sdpa for some
    # attn_types). flash_attention_2 is the documented default.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Apply MInference patch. For attn_types in OTHER_ATTENTION_TYPES (e.g. "hf",
    # "dense"), MODEL2PATH lookup is skipped — useful for unsupported models.
    print(f"Patching with MInference: attn_type={args.attn_type}, kv_type={args.kv_type}")
    mn = MInference(
        attn_type=args.attn_type,
        kv_type=args.kv_type,
        model_name=args.minference_model_name,
    )
    model = mn(model)

    # Evaluate
    metric_fn = cfg["metric_fn"]
    max_gen = cfg["max_gen"]
    scores = []

    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    sum_total_throughput = 0.0
    sum_decode_throughput = 0.0
    n_throughput = 0
    n_truncated = 0
    n_skipped = 0

    f_out = open(args.output_file, "w", encoding="utf-8")
    pbar = tqdm(enumerate(samples), total=len(samples),
                desc=f"{args.task}/{args.attn_type}")

    try:
        for idx, item in pbar:
            prompt = build_prompt(args.task, item, tokenizer)
            gold = get_ground_truth(args.task, item)

            tok_full = tokenizer(prompt, truncation=False,
                                 return_tensors="pt").input_ids[0]
            orig_len = int(tok_full.shape[0])
            if orig_len > args.max_input_len:
                if args.skip_too_long:
                    n_skipped += 1
                    continue
                half = args.max_input_len // 2
                prompt = (
                    tokenizer.decode(tok_full[:half], skip_special_tokens=True)
                    + tokenizer.decode(tok_full[-half:], skip_special_tokens=True)
                )
                n_truncated += 1

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=args.max_input_len).to(model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                from transformers import StoppingCriteria, StoppingCriteriaList

                class _TTFTHook(StoppingCriteria):
                    def __init__(self):
                        self.first_token_time = None
                    def __call__(self, input_ids, scores, **kw):
                        if self.first_token_time is None:
                            torch.cuda.synchronize()
                            self.first_token_time = time.perf_counter()
                        return False

                hook = _TTFTHook()
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    stopping_criteria=StoppingCriteriaList([hook]),
                )
                torch.cuda.synchronize()
                t_total = time.perf_counter() - t_start
                t_prefill = (hook.first_token_time - t_start) \
                    if hook.first_token_time is not None else t_total

            gen_ids = output[0, input_len:]
            gen_len = gen_ids.shape[0]
            raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            prediction = postprocess_prediction(args.task, raw_pred)

            score = metric_fn(prediction, gold)
            scores.append(score)

            t_decode = max(t_total - t_prefill, 1e-6)
            decode_tps = max(gen_len - 1, 1) / t_decode if gen_len > 1 else 0.0
            total_tps = (input_len + gen_len) / max(t_total, 1e-6)

            total_prefill_time += t_prefill
            total_decode_time += t_decode
            total_input_tokens += input_len
            total_output_tokens += gen_len
            sum_total_throughput += total_tps
            if gen_len > 1:
                sum_decode_throughput += decode_tps
                n_throughput += 1

            result = {
                "idx": idx,
                "score": score,
                "prediction": prediction,
                "raw_prediction": raw_pred,
                "gold": gold,
                "input_len": input_len,
                "gen_len": int(gen_len),
                "t_prefill_s": t_prefill,
                "t_total_s": t_total,
                "decode_tps": decode_tps,
                "total_tps": total_tps,
                "task": args.task,
                "mode": f"minference:{args.attn_type}+{args.kv_type}",
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            del inputs, output, gen_ids
            torch.cuda.empty_cache()
    finally:
        f_out.close()

    avg_score = sum(scores) / max(len(scores), 1) * 100
    overall_prefill_tps = total_input_tokens / max(total_prefill_time, 1e-6)
    overall_decode_tps = (
        max(total_output_tokens - len(scores), 0) / max(total_decode_time, 1e-6)
    )
    avg_decode_tps = sum_decode_throughput / max(n_throughput, 1)
    avg_total_tps = sum_total_throughput / max(len(scores), 1)

    summary = {
        "task": args.task,
        "mode": f"minference:{args.attn_type}+{args.kv_type}",
        "model_path": args.model_path,
        "minference_model_name": args.minference_model_name,
        "samples": len(scores),
        "n_truncated": n_truncated,
        "n_skipped": n_skipped,
        "max_input_len": args.max_input_len,
        "metric_name": cfg["metric_name"],
        "metric_value": avg_score,
        "avg_input_len": total_input_tokens / max(len(scores), 1),
        "avg_output_len": total_output_tokens / max(len(scores), 1),
        "overall_prefill_tps": overall_prefill_tps,
        "overall_decode_tps": overall_decode_tps,
        "avg_decode_tps": avg_decode_tps,
        "avg_total_tps": avg_total_tps,
    }
    summary_path = args.output_file.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task: {args.task} | Mode: minference:{args.attn_type}+{args.kv_type}")
    print(f"Samples: {len(scores)} | {cfg['metric_name']}: {avg_score:.2f}%")
    print(f"Truncated: {n_truncated} | Skipped: {n_skipped} | max_input_len: {args.max_input_len}")
    print(f"Avg input len: {summary['avg_input_len']:.0f} | Avg output len: {summary['avg_output_len']:.0f}")
    print(f"Prefill: {overall_prefill_tps:.1f} tok/s | Decode: {overall_decode_tps:.1f} tok/s")
    print(f"Avg decode tps (per-sample): {avg_decode_tps:.1f} | Avg total tps: {avg_total_tps:.1f}")
    print(f"Output: {args.output_file}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
