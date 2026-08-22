import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parser import extract_answer, parse_ground_truth
from grader import math_equal
from src.patch.patch_qwen import patch_qwen3_attention_forward_model

SYSTEM_PROMPT = "You are a helpful assistant."


def build_prompt(tokenizer, problem_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem_text}\n\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def coerce_official_answer(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (tuple, list)):
        for t in reversed(x):
            s = coerce_official_answer(t)
            if s:
                return s
        return ""
    if isinstance(x, dict):
        for k in ["answer", "final", "gt", "prediction", "parsed", "value", "text"]:
            if k in x:
                s = coerce_official_answer(x[k])
                if s:
                    return s
        return str(x).strip()
    return str(x).strip()


def load_done_ids_and_stats(output_file: str):
    done_ids = set()
    all_total = 0
    all_correct = 0

    if not os.path.exists(output_file):
        return done_ids, all_total, all_correct

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            uid = obj.get("unique_id", None)
            if uid is None:
                continue

            done_ids.add(uid)
            all_total += 1
            if bool(obj.get("is_correct", False)):
                all_correct += 1

    return done_ids, all_total, all_correct


def load_model_and_tokenizer(
    model_path: str,
    mode: str,
    window_size: int,
    top_ratio: float,
    report_stats: bool,
):
    print(f"Loading Tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading Model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = patch_qwen3_attention_forward_model(
        model,
        mode=mode,
        window_size=window_size,
        top_ratio=top_ratio,
        report_stats=report_stats,
    )
    model.eval()
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Qwen3 with HyQuant on Math500")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Math500 jsonl dataset",
    )
    parser.add_argument("--output_file", type=str, default="qwen3_8b_math500_vquant_results.jsonl")

    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive sample index in dataset")
    parser.add_argument("--end_idx", type=int, default=-1, help="Exclusive sample index in dataset, -1 means to end")

    parser.add_argument("--max_new_tokens", type=int, default=1024 * 20)
    parser.add_argument("--do_sample", action="store_true", default=False)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "int8_pv_fp16", "int8_pv_fp16_window", "int8_pv_fp8", "int8_pv_fp16_vertical", "full_attention"],
        default="int8_pv_fp16_window",
    )
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--top_ratio", type=float, default=0.05)
    parser.add_argument("--report_stats", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    done_ids, all_total_prev, all_correct_prev = load_done_ids_and_stats(args.output_file)
    if done_ids:
        print(f"[Resume] Found existing results: {all_total_prev} lines, done unique_id: {len(done_ids)}")
        print(
            f"[Resume] Previous accuracy: {all_correct_prev}/{all_total_prev} = "
            f"{100.0 * all_correct_prev / max(1, all_total_prev):.2f}%"
        )
    else:
        print("[Resume] No existing result file, start fresh.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        mode=args.mode,
        window_size=args.window_size,
        top_ratio=args.top_ratio,
        report_stats=args.report_stats,
    )

    print(f"Loading Data from: {args.data_path}")
    data = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    total_samples = len(data)
    start_idx = max(0, args.start_idx)
    end_idx = total_samples if args.end_idx < 0 else min(args.end_idx, total_samples)

    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, total={total_samples}")

    sliced = data[start_idx:end_idx]
    print(f"Total samples in dataset: {total_samples}")
    print(f"Evaluating range: [{start_idx}, {end_idx}) -> {len(sliced)} samples")
    print(f"mode={args.mode}, window_size={args.window_size}, top_ratio={args.top_ratio}")

    new_total = 0
    new_correct = 0

    open_mode = "a" if os.path.exists(args.output_file) else "w"
    with open(args.output_file, open_mode, encoding="utf-8") as f_out:
        for local_idx, item in tqdm(enumerate(sliced, start=start_idx), total=len(sliced), desc="Evaluating(HyQuant)"):
            problem = item.get("problem", "")
            unique_id = item.get("unique_id", local_idx)

            if unique_id in done_ids:
                continue

            gt_original = item.get("answer", "")
            gt_official_raw = parse_ground_truth(item, data_name="math")
            gt_official_answer = coerce_official_answer(gt_official_raw)
            if not gt_official_answer and gt_original:
                gt_official_answer = coerce_official_answer(gt_original)

            input_text = build_prompt(tokenizer, problem)
            inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )

            gen_only = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
            response_text = tokenizer.decode(gen_only[0], skip_special_tokens=True)

            pred_official_raw = extract_answer(response_text, data_name="math")
            pred_extracted_answer = coerce_official_answer(pred_official_raw)

            if pred_extracted_answer and gt_official_answer:
                try:
                    is_correct = math_equal(pred_extracted_answer, gt_official_answer)
                    err_msg = ""
                except Exception as e:
                    is_correct = False
                    err_msg = f"{type(e).__name__}: {str(e)}"
            else:
                is_correct = False
                err_msg = ""

            new_total += 1
            if is_correct:
                new_correct += 1

            result_item = {
                "gt_official_answer": gt_official_answer,
                "pred_extracted_answer": pred_extracted_answer,
                "is_correct": is_correct,
                "unique_id": unique_id,
                "sample_idx": local_idx,
                "problem": problem,
                "gt_original": gt_original,
                "error": err_msg,
                "full_response": response_text,
                "gt_official_raw": gt_official_raw,
                "pred_official_raw": pred_official_raw,
                "meta": {
                    "model_path": args.model_path,
                    "data_path": args.data_path,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "mode": args.mode,
                    "window_size": args.window_size,
                    "top_ratio": args.top_ratio,
                    "report_stats": args.report_stats,
                },
            }

            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()
            done_ids.add(unique_id)

    new_acc = 100.0 * new_correct / max(1, new_total)

    _, all_total, all_correct = load_done_ids_and_stats(args.output_file)
    all_acc = 100.0 * all_correct / max(1, all_total)

    print("\n" + "=" * 60)
    print("HyQuant evaluation finished")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Output: {args.output_file}")
    print("-" * 60)
    print(f"[This run] evaluated: {new_total}, correct: {new_correct}, acc: {new_acc:.2f}%")
    print(f"[All runs] total: {all_total}, correct: {all_correct}, acc: {all_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
