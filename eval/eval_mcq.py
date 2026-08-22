#!/usr/bin/env python3
"""MCQ evaluator for MMLU-Pro.

Usage:
  python eval/eval_mcq.py --model_path /path/to/model \
    --dataset mmlu_pro --mode sage --output_file results/mmlu_pro_sage.jsonl
"""
import argparse, json, os, re, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.patch import patch_attention_forward_model

DATASET_CONFIG = {
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "split": "test"},
}


def load_mmlu_pro(data_dir, max_samples=-1):
    local = os.path.join(data_dir, "mmlu_pro")
    if os.path.isdir(local):
        for f in os.listdir(local):
            if f.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(os.path.join(local, f))
                samples = df.to_dict("records")
                if 0 < max_samples < len(samples):
                    samples = samples[:max_samples]
                return samples
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    samples = [dict(r) for r in ds]
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]
    return samples


def build_mcq_prompt(tokenizer, item):
    question = item.get("question", "")
    options = item.get("options", [])
    letters = "ABCDEFGHIJ"
    choices = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
    content = f"Question: {question}\n\n{choices}\n\nAnswer with the letter only:"
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return content


def extract_choice(text):
    text = text.strip()
    m = re.match(r'^([A-J])\b', text.upper())
    if m:
        return m.group(1)
    m = re.search(r'\b([A-J])\b', text.upper()[:20])
    if m:
        return m.group(1)
    return text.strip()[:1].upper()


def main():
    parser = argparse.ArgumentParser("MCQ eval")
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
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    patch_mode = args.mode

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
    model.eval()

    samples = load_mmlu_pro(args.data_dir, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    correct = 0
    total = 0

    with open(args.output_file, "w") as f_out:
        for item in tqdm(samples, desc=f"{args.dataset}/{args.mode}"):
            prompt = build_mcq_prompt(tokenizer, item)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            pred_text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            pred = extract_choice(pred_text)
            gold = str(item.get("answer", "")).strip().upper()
            is_correct = (pred == gold)

            total += 1
            if is_correct:
                correct += 1

            f_out.write(json.dumps({"correct": is_correct, "pred": pred, "gold": gold, "dataset": args.dataset, "mode": args.mode}, ensure_ascii=False) + "\n")
            f_out.flush()

    acc = correct / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"{args.dataset} | {args.mode} | {correct}/{total} = {acc:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
