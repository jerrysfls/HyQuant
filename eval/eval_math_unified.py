#!/usr/bin/env python3
"""Unified math evaluator for Math500, GSM8K, AIME24, AIME25.

Usage:
  python eval/eval_math_unified.py --model_path /path/to/model \
    --dataset math500 --mode sage --output_file results/math500_sage.jsonl
"""
import argparse, json, os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser import extract_answer, parse_ground_truth
from grader import math_equal
from src.patch import patch_attention_forward_model

DATASET_CONFIG = {
    "math500": {"repo": "math-ai/math500", "split": "test", "parser": "math", "q_field": "problem", "a_field": "answer"},
    "gsm8k": {"repo": "openai/gsm8k", "split": "test", "subset": "main", "parser": "gsm8k", "q_field": "question", "a_field": "answer"},
    "aime24": {"repo": "math-ai/aime24", "split": "train", "parser": "aime24", "q_field": "problem", "a_field": "answer"},
    "aime25": {"repo": "math-ai/aime25", "split": "train", "parser": "aime24", "q_field": "problem", "a_field": "answer"},
}


def load_dataset_samples(name, data_dir, max_samples=-1):
    cfg = DATASET_CONFIG[name]
    # Try local files first
    local_paths = [
        os.path.join(data_dir, cfg.get("repo", "").split("/")[-1]),
        os.path.join(data_dir, name),
    ]
    for lp in local_paths:
        if os.path.isdir(lp):
            # Look for parquet or jsonl
            for f in os.listdir(lp):
                if f.endswith(".parquet"):
                    import pandas as pd
                    df = pd.read_parquet(os.path.join(lp, f))
                    samples = df.to_dict("records")
                    if 0 < max_samples < len(samples):
                        samples = samples[:max_samples]
                    return samples

    # Fallback to HF
    from datasets import load_dataset
    subset = cfg.get("subset", None)
    if subset:
        ds = load_dataset(cfg["repo"], subset, split=cfg["split"])
    else:
        ds = load_dataset(cfg["repo"], split=cfg["split"])
    samples = [dict(r) for r in ds]
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]
    return samples


def build_prompt(tokenizer, problem):
    messages = [{"role": "user", "content": f"Problem:\n{problem}\n\nPlease solve step by step. Put your final answer in \\boxed{{}}."}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"Problem:\n{problem}\n\nAnswer:"


def main():
    parser = argparse.ArgumentParser("Unified math eval")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing local dataset caches/downloads",
    )
    parser.add_argument("--mode", required=True, choices=["dense", "full_attention", "int8_pv_fp16", "int8_pv_fp16_window", "int8_pv_fp8", "int8_pv_fp16_vertical"])
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--top_ratio", type=float, default=0.05)
    parser.add_argument("--patch_target", type=str, default="auto", choices=["auto", "qwen", "llama", "glm4"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = DATASET_CONFIG[args.dataset]

    patch_mode = args.mode

    print(f"Loading model and patching with mode={patch_mode}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="cuda", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = patch_attention_forward_model(
        model,
        mode=patch_mode,
        window_size=args.window_size,
        top_ratio=args.top_ratio,
        patch_target=args.patch_target,
    )
    model.to("cuda")
    model.eval()

    print(f"Loading dataset: {args.dataset}...")
    samples = load_dataset_samples(args.dataset, args.data_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    correct = 0
    total = 0

    with open(args.output_file, "w") as f_out:
        for item in tqdm(samples, desc=f"{args.dataset}/{args.mode}"):
            problem = item.get(cfg["q_field"], "")
            prompt = build_prompt(tokenizer, problem)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=131072 - args.max_new_tokens).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            pred_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            pred_ans = extract_answer(pred_text, data_name=cfg["parser"])
            gt_raw = parse_ground_truth(item, data_name=cfg["parser"])
            gt_ans = gt_raw[1] if isinstance(gt_raw, tuple) else str(gt_raw)

            try:
                is_correct = math_equal(str(pred_ans), str(gt_ans))
            except:
                is_correct = False

            total += 1
            if is_correct:
                correct += 1

            f_out.write(json.dumps({"correct": is_correct, "pred": str(pred_ans), "gold": str(gt_ans), "dataset": args.dataset, "mode": args.mode}, ensure_ascii=False) + "\n")
            f_out.flush()

    acc = correct / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"{args.dataset} | {args.mode} | {correct}/{total} = {acc:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
