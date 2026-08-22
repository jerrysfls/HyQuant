#!/usr/bin/env python3
"""LongBench v1 & v2 evaluator for attention method comparison.

Covers the LongBench v1 tasks (QA / summarization / classification / code,
see TASK_CONFIG) plus LongBench v2 multiple choice, across the attention
modes listed in --mode (patched modes, FA2 baseline, KIVI/KVTuner caches).

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/eval_longbench.py \
    --model_path /path/to/model \
    --task qasper --mode hybrid_full --data_dir /path/to/data \
    --output_file results/qasper_hybrid_full.jsonl
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.patch import patch_attention_forward_model

# ============================================================
# Metrics
# ============================================================

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation/articles/extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s.strip()


def _f1_pair(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, ground_truths) -> float:
    """Max F1 over a list of reference answers (LongBench convention)."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not ground_truths:
        return 0.0
    return max(_f1_pair(prediction, g) for g in ground_truths)


def _rouge_l_pair(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(normalize_answer(prediction) == normalize_answer(ground_truth))
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction, ground_truths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    if not ground_truths:
        return 0.0
    return max(_rouge_l_pair(prediction, g) for g in ground_truths)


def accuracy_score(prediction, ground_truths) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    pred = prediction.strip().upper()
    for gold in ground_truths:
        g = gold.strip().upper()
        for p in [pred, pred[:1]]:
            if p in {"A", "B", "C", "D"} and p == g:
                return 1.0
        if pred == g:
            return 1.0
    return 0.0


def paragraph_retrieval_score(prediction, ground_truths) -> float:
    """Match 'Paragraph N' from prediction against gold (e.g. 'Paragraph 5')."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    m = re.search(r"[Pp]aragraph\s*(\d+)", prediction)
    if m is None:
        nums = re.findall(r"\d+", prediction)
        pred_num = nums[0] if nums else None
    else:
        pred_num = m.group(1)
    if pred_num is None:
        return 0.0
    for gold in ground_truths:
        gm = re.search(r"\d+", gold)
        if gm and gm.group() == pred_num:
            return 1.0
    return 0.0


# ============================================================
# Official LongBench v1 prompt templates
# (mirrors THUDM/LongBench config/dataset2prompt.json)
# ============================================================

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. "
        "Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there "
        "in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph "
        "the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be "
        "like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}


TASK_CONFIG = {
    # LongBench v1 — QA tasks (F1) — max_gen from official LongBench DATASET2MAXLEN
    "qasper": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 128, "version": "v1"},
    "hotpotqa": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 32, "version": "v1"},
    "2wikimqa": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 32, "version": "v1"},
    "musique": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 32, "version": "v1"},
    "multifieldqa_en": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 64, "version": "v1"},
    "narrativeqa": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 128, "version": "v1"},
    "triviaqa": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 32, "version": "v1"},
    # LongBench v1 — Summarization tasks (ROUGE-L)
    "gov_report": {"metric_fn": rouge_l_score, "metric_name": "ROUGE-L", "max_gen": 512, "version": "v1"},
    "multi_news": {"metric_fn": rouge_l_score, "metric_name": "ROUGE-L", "max_gen": 512, "version": "v1"},
    "qmsum": {"metric_fn": rouge_l_score, "metric_name": "ROUGE-L", "max_gen": 512, "version": "v1"},
    "samsum": {"metric_fn": rouge_l_score, "metric_name": "ROUGE-L", "max_gen": 128, "version": "v1"},
    # LongBench v1 — Classification/Retrieval (Accuracy)
    "trec": {"metric_fn": accuracy_score, "metric_name": "Accuracy", "max_gen": 64, "version": "v1"},
    "passage_count": {"metric_fn": accuracy_score, "metric_name": "Accuracy", "max_gen": 32, "version": "v1"},
    "passage_retrieval_en": {"metric_fn": paragraph_retrieval_score, "metric_name": "Accuracy", "max_gen": 32, "version": "v1"},
    # LongBench v1 — Code tasks (EM-ish via F1)
    "lcc": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 64, "version": "v1"},
    "repobench-p": {"metric_fn": f1_score, "metric_name": "F1", "max_gen": 64, "version": "v1"},
    # LongBench v2 (multiple choice) — short answer suffices
    "longbench_v2": {"metric_fn": accuracy_score, "metric_name": "Accuracy", "max_gen": 32, "version": "v2"},
}

# ============================================================
# Data Loading
# ============================================================

def load_longbench_v1(data_dir: str, task: str, max_samples: int = -1):
    """Load LongBench v1 JSONL from downloaded repo."""
    # Try multiple possible locations
    candidates = [
        os.path.join(data_dir, "longbench", "data", f"{task}.jsonl"),
        os.path.join(data_dir, "longbench", f"{task}.jsonl"),
        os.path.join(data_dir, f"{task}.jsonl"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        # Try loading via datasets library
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task, split="test")
            samples = [dict(row) for row in ds]
            if 0 < max_samples < len(samples):
                samples = samples[:max_samples]
            return samples
        except Exception:
            pass
        raise FileNotFoundError(
            f"Cannot find {task}.jsonl in {data_dir}. "
            f"Tried: {candidates}. "
            "Download with: huggingface_hub.snapshot_download('THUDM/LongBench', ...)"
        )
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]
    return samples


def load_longbench_v2(data_dir: str, max_samples: int = -1):
    """Load LongBench v2 from downloaded repo."""
    candidates = [
        os.path.join(data_dir, "longbench_v2", "data.jsonl"),
        os.path.join(data_dir, "longbench_v2", "data", "data.jsonl"),
    ]
    # Also search for parquet
    parquet_dir = os.path.join(data_dir, "longbench_v2", "data")
    if os.path.isdir(parquet_dir):
        for f in os.listdir(parquet_dir):
            if f.endswith(".parquet"):
                candidates.append(os.path.join(parquet_dir, f))

    # Also try .json (single JSON array)
    json_candidates = [
        os.path.join(data_dir, "longbench_v2", "data.json"),
    ]
    candidates.extend(json_candidates)

    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break

    if path is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench-v2", split="train")
            samples = [dict(row) for row in ds]
            if 0 < max_samples < len(samples):
                samples = samples[:max_samples]
            return samples
        except Exception:
            pass
        raise FileNotFoundError(
            f"Cannot find LongBench-v2 data in {data_dir}. Tried: {candidates}"
        )

    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        samples = df.to_dict("records")
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if not isinstance(samples, list):
            raise ValueError(f"Expected JSON array in {path}")
    else:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]
    return samples


# ============================================================
# Prompt Construction
# ============================================================

def build_prompt_v2_mcq(item: dict) -> str:
    ctx = item.get("context", "")
    q = item.get("question", "")
    choices = []
    for letter in ["A", "B", "C", "D"]:
        c = item.get(f"choice_{letter}", "")
        if c:
            choices.append(f"{letter}. {c}")
    choices_text = "\n".join(choices)
    return (
        f"{ctx}\n\n"
        f"Question: {q}\n"
        f"{choices_text}\n"
        f"Answer with a single letter (A, B, C, or D):"
    )


def build_prompt(task: str, item: dict, tokenizer) -> str:
    if task == "longbench_v2":
        user_content = build_prompt_v2_mcq(item)
    elif task in DATASET2PROMPT:
        template = DATASET2PROMPT[task]
        user_content = template.format(
            context=item.get("context", ""),
            input=item.get("input", ""),
        )
    else:
        # Fallback: generic QA
        user_content = (
            f"{item.get('context', '')}\n\nQuestion: {item.get('input', '')}\nAnswer:"
        )

    messages = [{"role": "user", "content": user_content}]
    # Qwen3-style: disable thinking via apply_chat_template kwarg if supported,
    # else append /no_think directive.
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizers don't accept enable_thinking
        messages[0]["content"] = messages[0]["content"] + " /no_think"
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return user_content
    except Exception:
        return user_content


def get_ground_truth(task: str, item: dict):
    """Return a list of acceptable answers (LongBench supports multi-reference)."""
    if task == "longbench_v2":
        return [item.get("answer", "")]
    answers = item.get("answers", [])
    if isinstance(answers, list) and answers:
        return [a for a in answers if isinstance(a, str)]
    a = item.get("answer", "")
    return [a] if a else []


# ============================================================
# Prediction post-processing (robust answer extraction)
# ============================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*?(?=$)", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 thinking traces)."""
    text = _THINK_RE.sub("", text)
    # If generation truncated before </think>, drop the unclosed block too.
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def postprocess_prediction(task: str, raw: str) -> str:
    """Strip thinking and truncate to the relevant answer span."""
    text = strip_thinking(raw)
    # Trim a leading 'Answer:' / 'The answer is' the model sometimes echoes
    text = re.sub(r"^\s*(answer|the answer is|final answer)\s*[:\-]?\s*",
                  "", text, flags=re.IGNORECASE).strip()

    # For short-answer QA / classification tasks, take just the first non-empty line
    short_tasks = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
                   "2wikimqa", "musique", "triviaqa", "trec",
                   "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
                   "longbench_v2"}
    if task in short_tasks:
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return text.strip()
    # Summarization tasks keep full text
    return text.strip()


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("LongBench evaluator")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASK_CONFIG.keys()))
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing LongBench or LongBench-v2 data",
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["full_attention", "sage", "vquant_window", "vquant_vertical",
                                 "sage_w", "sage_w0", "sage_vert", "vquant_vert", "sparse_only",
                                 "flash_attention_2", "dense",
                                 "d_k4v4", "d_k4v4_vert", "d_k8v8",
                                 "d_k4v2", "d_k4v2_vert",
                                 "d_k2v4", "d_k2v4_vert",
                                 "hybrid_full", "hybrid_k4v2", "hybrid_k2v4",
                                 "kivi", "kvtuner"])
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--top_ratio", type=float, default=0.05)
    parser.add_argument("--patch_target", type=str, default="auto", choices=["auto", "qwen", "llama", "glm4"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument(
        "--max_input_len", type=int, default=24000,
        help="Max input tokens; longer inputs are middle-truncated (LongBench convention). "
             "Set to a high value (e.g. 120000) to effectively disable.",
    )
    parser.add_argument(
        "--skip_too_long", action="store_true",
        help="Skip samples whose tokenized prompt exceeds --max_input_len "
             "instead of middle-truncating them.",
    )
    parser.add_argument(
        "--force_output_len", type=int, default=0,
        help="Throughput benchmark mode: force EXACTLY N generated tokens per sample "
             "(uses min_new_tokens=max_new_tokens=N, ignores EOS). Bypasses task max_gen. "
             "Metric values become meaningless (model often loops/repeats); use only "
             "to measure decode tok/s in a regime where freeze() is amortised. "
             "0 = disabled (default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # -------------------------------------------------------------
    # Distributed init (tensor parallel via torchrun)
    # -------------------------------------------------------------
    import torch.distributed as dist
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", "0"))
    USE_TP = WORLD_SIZE > 1

    # Do NOT init dist here — transformers 4.52's `initialize_tensor_parallelism`
    # has a bug (UnboundLocalError on `current_device`) that triggers when dist is
    # already initialized before from_pretrained. Let HF call init_process_group
    # itself inside from_pretrained(tp_plan="auto").
    if USE_TP:
        torch.cuda.set_device(LOCAL_RANK)
    IS_MAIN = (RANK == 0)

    def log(msg):
        if IS_MAIN:
            print(msg, flush=True)

    import transformers
    log(f"=== Versions ===")
    log(f"  python: {sys.version.split()[0]}")
    log(f"  torch: {torch.__version__}")
    log(f"  transformers: {transformers.__version__}")
    log(f"=== Distributed env ===")
    log(f"  WORLD_SIZE={WORLD_SIZE}, RANK={RANK}, LOCAL_RANK={LOCAL_RANK}, USE_TP={USE_TP}")
    log(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    log(f"  torch.cuda.device_count()={torch.cuda.device_count()}")
    if not USE_TP:
        log("  >>> WARNING: WORLD_SIZE=1 — running in SINGLE-GPU mode (no TP).")
        log("      To enable TP, launch with: torchrun --nproc_per_node=N ...")

    if args.output_file is None:
        args.output_file = f"results/longbench_{args.task}_{args.mode}.jsonl"
    if IS_MAIN:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    cfg = TASK_CONFIG[args.task]

    # Load data (all ranks read the same JSONL — deterministic, no broadcast needed)
    log(f"Loading {args.task} data from {args.data_dir}...")
    if cfg["version"] == "v1":
        samples = load_longbench_v1(args.data_dir, args.task, args.max_samples)
    else:
        samples = load_longbench_v2(args.data_dir, args.max_samples)
    log(f"Loaded {len(samples)} samples")

    # Load model
    log(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Three loading paths:
    #   1) USE_TP=1 (with torchrun)        → tensor parallel via tp_plan
    #   2) USE_DEVICE_MAP=auto/balanced/manual → multi-GPU layer sharding (no offload)
    #   3) default                         → single-device load on cuda:0
    _device_map_env = os.environ.get("USE_DEVICE_MAP", "").strip().lower()
    if USE_TP:
        load_kwargs["tp_plan"] = "auto"
    elif _device_map_env in ("auto", "balanced", "balanced_low_0", "sequential", "manual"):
        ngpu = torch.cuda.device_count()
        if _device_map_env == "manual":
            # Manual placement via accelerate's dispatch_model (low-level API).
            # We do NOT pass device_map to from_pretrained because in transformers
            # 4.52, dict-form device_map is silently ignored (hf_device_map=None,
            # all weights stay on CPU). Instead we load on CPU then dispatch.
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            n_layers = getattr(_cfg, "num_hidden_layers", None)
            if n_layers is None:
                raise RuntimeError("manual device_map requires num_hidden_layers in config")
            per_gpu = (n_layers + ngpu - 1) // ngpu
            _MANUAL_DMAP = {
                "model.embed_tokens": 0,
                "model.rotary_emb": 0,
                "model.norm": ngpu - 1,
                "lm_head": ngpu - 1,
            }
            for i in range(n_layers):
                _MANUAL_DMAP[f"model.layers.{i}"] = min(i // per_gpu, ngpu - 1)
            log(f"Will MANUALLY dispatch {n_layers} layers across {ngpu} GPUs ({per_gpu}/GPU) after CPU load")
            # Important: don't set device_map on load_kwargs — load on CPU first.
        else:
            load_kwargs["device_map"] = _device_map_env
            _max_per_gpu = os.environ.get("MAX_GPU_MEM", "").strip()
            if _max_per_gpu:
                # Generous cap to PREVENT offload (large enough that all weights
                # fit on GPUs comfortably; no cpu key → accelerate refuses to offload).
                load_kwargs["max_memory"] = {i: _max_per_gpu for i in range(ngpu)}
                log(f"device_map='{_device_map_env}' max_memory={_max_per_gpu}/GPU x{ngpu}")
            else:
                log(f"device_map='{_device_map_env}' (no max_memory)")
    else:
        load_kwargs["device_map"] = "cuda"

    # Auto-fallback: device_map + flash_attention_2 has a known bug in transformers 4.52
    # (FA2 op dispatched to CPU when accelerate uses init_empty_weights). PyTorch SDPA on
    # H100 transparently uses the FA2 kernel via the SDP dispatcher, so quality/perf are
    # equivalent and no CPU dispatch issues. Force sdpa whenever sharding is enabled.
    _force_sdpa = bool(_device_map_env)  # only when device_map mode is on
    _attn_impl_baseline = "sdpa" if _force_sdpa else "flash_attention_2"

    if args.mode == "flash_attention_2":
        log(f"Loading FA2 baseline (attn_implementation={_attn_impl_baseline}, no patching)")
        load_kwargs["attn_implementation"] = _attn_impl_baseline
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    elif args.mode in ("kivi", "kvtuner"):
        # KIVI/KVTuner baselines: standard attention with HuggingFace's
        # HQQ-backed quantised KV cache (no attention patch). Bit width is set
        # below when constructing the per-sample cache.
        log(f"Loading {args.mode} baseline ({_attn_impl_baseline} + HQQQuantizedCache, no patching)")
        load_kwargs["attn_implementation"] = _attn_impl_baseline
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    else:
        load_kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        log(f"Patching with mode={args.mode}, window_size={args.window_size}, top_ratio={args.top_ratio}")
        model = patch_attention_forward_model(
            model,
            mode=args.mode,
            window_size=args.window_size,
            top_ratio=args.top_ratio,
            patch_target=args.patch_target,
        )
    # NOTE: do NOT call model.to("cuda") here.
    # With device_map="cuda" (or "auto" / tp_plan), accelerate hooks have already
    # placed every parameter on the right device. A redundant .to("cuda") collapses
    # a multi-GPU sharded model back to cuda:0 (bug we hit on Qwen3-32B + 4 GPUs).
    #
    # MANUAL device_map dispatch (transformers 4.52 ignores dict device_map in
    # from_pretrained, so we do it explicitly here via accelerate.dispatch_model).
    if _device_map_env == "manual" and "_MANUAL_DMAP" in dir():
        from accelerate import dispatch_model
        log(f"Manually dispatching model to {len(set(_MANUAL_DMAP.values()))} GPUs via accelerate.dispatch_model ...")
        model = dispatch_model(model, device_map=_MANUAL_DMAP)
        log("dispatch_model done")
    # Only force-move to cuda:0 if accelerate hasn't already placed the model
    # (hf_device_map is set by device_map="auto"/"balanced"/manual).
    # Calling .to("cuda") on a sharded model collapses everything to GPU 0.
    model.to("cuda")
    model.eval()

    # Diagnostic: where did weights actually land?
    try:
        hfdm = getattr(model, "hf_device_map", None)
        log(f"model.hf_device_map = {hfdm}")
    except Exception as _e:
        log(f"hf_device_map probe failed: {_e}")
    try:
        p = next(model.parameters())
        log(f"sample param device = {p.device}, dtype = {p.dtype}")
        # Sample several params from different points
        for name in ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                     "model.layers.30.self_attn.q_proj.weight", "lm_head.weight"]:
            try:
                t = model.get_parameter(name)
                log(f"  {name} -> {t.device}")
            except AttributeError:
                pass
    except Exception as _e:
        log(f"param device probe failed: {_e}")

    # Verify each GPU's actual memory footprint after load
    log("=== Per-GPU memory after load ===")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used_gb = (total - free) / 1e9
        log(f"  GPU {i}: {used_gb:.1f} GB used / {total/1e9:.1f} GB total")
    if USE_TP and torch.cuda.device_count() > 1:
        free_0, total_0 = torch.cuda.mem_get_info(0)
        used_0 = (total_0 - free_0) / 1e9
        if used_0 > 50:
            log(f"  >>> WARNING: GPU 0 has {used_0:.1f} GB used — TP may NOT have sharded! "
                "Check transformers version supports tp_plan for Qwen3.")

    # Evaluate
    metric_fn = cfg["metric_fn"]
    max_gen = cfg["max_gen"]
    # Throughput-benchmark override: force exactly N generated tokens per sample
    if args.force_output_len and args.force_output_len > 0:
        max_gen = args.force_output_len
        log(f"=== FORCED OUTPUT LENGTH = {args.force_output_len} (throughput-only mode) ===")
        log("    F1/ROUGE/Acc numbers will be meaningless (model may loop / repeat).")
    scores = []

    # Throughput tracking
    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    sum_total_throughput = 0.0
    sum_decode_throughput = 0.0
    n_throughput = 0

    # D-stage cache helpers
    is_d_stage = args.mode in {
        "d_k4v4", "d_k4v4_vert", "d_k8v8",
        "d_k4v2", "d_k4v2_vert",
        "d_k2v4", "d_k2v4_vert",
        "hybrid_full", "hybrid_k4v2", "hybrid_k2v4",
    }
    if is_d_stage:
        from src.ops.d_stage_kv_cache import DStageKVCache
        num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 32))
        _D_BITS = {
            "d_k4v4": (4, 4),
            "d_k4v4_vert": (4, 4),
            "d_k8v8": (8, 8),
            "d_k4v2": (4, 2),
            "d_k4v2_vert": (4, 2),
            "d_k2v4": (2, 4),
            "d_k2v4_vert": (2, 4),
            "hybrid_full": (4, 4),
            "hybrid_k4v2": (4, 2),
            "hybrid_k2v4": (2, 4),
        }
        _k_bits, _v_bits = _D_BITS[args.mode]
        _use_vert = args.mode in {
            "d_k4v4_vert", "d_k4v2_vert", "d_k2v4_vert",
            "hybrid_full", "hybrid_k4v2", "hybrid_k2v4",
        }

        def _new_dstage_cache():
            return DStageKVCache(
                num_layers=num_hidden_layers,
                k_bits=_k_bits,
                v_bits_fn=lambda i, _vb=_v_bits: _vb,
                window_size=args.window_size,
                top_ratio=args.top_ratio,
                use_vert=_use_vert,
            )

    # KIVI / KVTuner cache helpers (HF HQQQuantizedCache proxies)
    is_qcache = args.mode in {"kivi", "kvtuner"}
    if is_qcache:
        _QCACHE_BITS = {"kivi": 2, "kvtuner": 4}
        _qcache_nbits = _QCACHE_BITS[args.mode]
        log(f"=== {args.mode}: HQQQuantizedCache nbits={_qcache_nbits}, "
            f"residual_length={args.window_size} ===")

        def _new_qcache():
            from transformers.cache_utils import (
                HQQQuantizedCache,
                QuantizedCacheConfig,
            )
            cfg = QuantizedCacheConfig(
                backend="HQQ",
                nbits=_qcache_nbits,
                axis_key=0,
                axis_value=0,
                q_group_size=64,
                residual_length=args.window_size,
                compute_dtype=torch.bfloat16,
                device="cuda",
            )
            return HQQQuantizedCache(cfg)

    # Rank 0 opens output file; other ranks discard writes via /dev/null
    f_out = open(args.output_file, "w", encoding="utf-8") if IS_MAIN else open(os.devnull, "w")

    sample_iter = tqdm(enumerate(samples), total=len(samples),
                       desc=f"{args.task}/{args.mode}") if IS_MAIN else enumerate(samples)

    n_truncated = 0
    n_skipped = 0

    try:
        for idx, item in sample_iter:
            prompt = build_prompt(args.task, item, tokenizer)
            gold = get_ground_truth(args.task, item)

            # Length policy: check tokenized length; either skip or middle-truncate.
            tok_full = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            orig_len = int(tok_full.shape[0])
            if orig_len > args.max_input_len:
                if args.skip_too_long:
                    n_skipped += 1
                    if IS_MAIN:
                        sample_iter.write(
                            f"[SKIP] idx={idx} input_len={orig_len} > {args.max_input_len}"
                        ) if hasattr(sample_iter, "write") else None
                    continue
                # Middle-truncate: keep first half + last half (LongBench convention).
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
                # Single generate(); TTFT captured via StoppingCriteria hook
                # fired right after the first decode-token is produced.
                from transformers import StoppingCriteria, StoppingCriteriaList

                class _TTFTHook(StoppingCriteria):
                    def __init__(self):
                        self.first_token_time = None
                    def __call__(self, input_ids, scores, **kw):
                        if self.first_token_time is None:
                            torch.cuda.synchronize()
                            self.first_token_time = time.perf_counter()
                        return False  # never stop

                gen_kwargs_full = dict(max_new_tokens=max_gen, do_sample=False)
                if args.force_output_len and args.force_output_len > 0:
                    # Force the model to emit exactly N tokens, ignoring EOS.
                    gen_kwargs_full["min_new_tokens"] = args.force_output_len
                if is_d_stage:
                    gen_kwargs_full["past_key_values"] = _new_dstage_cache()
                elif is_qcache:
                    gen_kwargs_full["past_key_values"] = _new_qcache()
                hook = _TTFTHook()
                gen_kwargs_full["stopping_criteria"] = StoppingCriteriaList([hook])
                torch.cuda.synchronize()
                if USE_TP:
                    dist.barrier()
                t_start = time.perf_counter()
                output = model.generate(**inputs, **gen_kwargs_full)
                torch.cuda.synchronize()
                if USE_TP:
                    dist.barrier()
                t_total = time.perf_counter() - t_start
                t_prefill = (
                    (hook.first_token_time - t_start)
                    if hook.first_token_time is not None
                    else t_total
                )

            gen_ids = output[0, input_len:]
            gen_len = gen_ids.shape[0]
            raw_pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            prediction = postprocess_prediction(args.task, raw_pred)

            score = metric_fn(prediction, gold)
            scores.append(score)

            # Compute throughput
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

            if IS_MAIN:
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
                    "mode": args.mode,
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

            # Free per-sample tensors / cache to reduce fragmentation
            del inputs, output, gen_ids
            if "past_key_values" in gen_kwargs_full:
                del gen_kwargs_full["past_key_values"]
            torch.cuda.empty_cache()
    finally:
        f_out.close()

    avg_score = sum(scores) / max(len(scores), 1) * 100
    # Aggregate throughput (two views)
    overall_prefill_tps = total_input_tokens / max(total_prefill_time, 1e-6)
    overall_decode_tps = max(total_output_tokens - len(scores), 0) / max(total_decode_time, 1e-6)
    avg_decode_tps = sum_decode_throughput / max(n_throughput, 1)
    avg_total_tps = sum_total_throughput / max(len(scores), 1)

    summary = {
        "task": args.task,
        "mode": args.mode,
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
    if IS_MAIN:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Task: {args.task} | Mode: {args.mode}")
        print(f"Samples: {len(scores)} | {cfg['metric_name']}: {avg_score:.2f}%")
        print(f"Truncated: {n_truncated} | Skipped: {n_skipped} | max_input_len: {args.max_input_len}")
        print(f"Avg input len: {summary['avg_input_len']:.0f} | Avg output len: {summary['avg_output_len']:.0f}")
        print(f"Prefill: {overall_prefill_tps:.1f} tok/s | Decode: {overall_decode_tps:.1f} tok/s")
        print(f"Avg decode tps (per-sample): {avg_decode_tps:.1f} | Avg total tps: {avg_total_tps:.1f}")
        print(f"Output: {args.output_file}")
        print(f"Summary: {summary_path}")
        print(f"{'='*60}")

    # Clean shutdown
    if USE_TP and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
