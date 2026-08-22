#!/usr/bin/env python3
"""Unified eval over the P-stage / D-stage configurations. Single GPU, all datasets sequentially.

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval/eval_9configs.py --config pd_full --output_dir results/9configs/
"""
import argparse, json, os, sys, re, string
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.patch.patch_llama import patch_llama3_attention_forward_model
from src.ops.d_stage_kv_cache import DStageKVCache

# ============================================================
# Configurations
# ============================================================
_D_STAGE_CONFIGS = {"d_k4v4", "d_k4v4_vert", "d_k8v8"}

CONFIGS = {
    # P-stage baselines
    "pd_full":           {"patch_mode": "full_attention"},
    "p_int8_pv_fp16_win":{"patch_mode": "int8_pv_fp16_window"},
    "p_int8_pv_fp16_vert":{"patch_mode": "int8_pv_fp16_vertical"},
    "p_int8_pv_fp16":   {"patch_mode": "int8_pv_fp16"},
    # D-stage: 3 configs
    "d_k4v4":            {"patch_mode": "d_k4v4"},
    "d_k4v4_vert":       {"patch_mode": "d_k4v4_vert"},
    "d_k8v8":            {"patch_mode": "d_k8v8"},
}

# D-stage KV cache config per config name
_D_STAGE_CACHE_CFG = {
    "d_k4v4":        {"k_bits": 4, "v_bits_fn": lambda i: 4, "use_vert": False},
    "d_k4v4_vert":   {"k_bits": 4, "v_bits_fn": lambda i: 4, "use_vert": True},
    "d_k8v8":        {"k_bits": 8, "v_bits_fn": lambda i: 8, "use_vert": False},
}


def _make_d_stage_cache(config_name, model):
    """Create a fresh D-stage KV cache for the given config."""
    cfg = _D_STAGE_CACHE_CFG[config_name]

    return DStageKVCache(
        num_layers=model.config.num_hidden_layers,
        k_bits=cfg["k_bits"],
        v_bits_fn=cfg["v_bits_fn"],
        window_size=256,
        top_ratio=0.05,
        use_vert=cfg["use_vert"],
    )

# ============================================================
# Datasets
# ============================================================
LB_V1_TASKS = [
    "qasper", "hotpotqa", "2wikimqa", "musique", "multifieldqa_en",
    "narrativeqa", "triviaqa",
    "gov_report", "multi_news", "qmsum", "samsum",
    "trec", "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]
MATH_TASKS = ["math500", "gsm8k", "aime24", "aime25"]
MCQ_TASKS = ["mmlu_pro"]

ALL_TASKS = LB_V1_TASKS + ["longbench_v2"] + MATH_TASKS + MCQ_TASKS

# ============================================================
# Metrics (copied from eval_longbench.py)
# ============================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split()).strip()

def f1_score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(normalize_answer(pred) == normalize_answer(gold))
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0: return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)

def rouge_l(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(normalize_answer(pred) == normalize_answer(gold))
    m, n = len(pt), len(gt)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1]+1 if pt[i-1]==gt[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    p, r = lcs/m, lcs/n
    return 2*p*r/(p+r)

TASK_METRICS = {}
for t in ["qasper","hotpotqa","2wikimqa","musique","multifieldqa_en","narrativeqa","triviaqa","lcc","repobench-p"]:
    TASK_METRICS[t] = ("F1", f1_score, 128)
for t in ["gov_report","multi_news"]:
    TASK_METRICS[t] = ("ROUGE-L", rouge_l, 512)
for t in ["qmsum","samsum"]:
    TASK_METRICS[t] = ("ROUGE-L", rouge_l, 256)
for t in ["trec","passage_count","passage_retrieval_en"]:
    TASK_METRICS[t] = ("Acc", lambda p,g: float(p.strip().upper()[:len(g)]==g.strip().upper()), 32)
TASK_METRICS["longbench_v2"] = ("Acc", lambda p,g: float(p.strip().upper()[:1]==g.strip().upper()[:1]), 16)
for t in MATH_TASKS:
    TASK_METRICS[t] = ("EM", None, 2048)  # uses math_equal
TASK_METRICS["mmlu_pro"] = ("Acc", lambda p,g: float(re.match(r'^([A-J])', p.strip().upper()).group(1)==g.strip().upper() if re.match(r'^([A-J])', p.strip().upper()) else False), 16)

# ============================================================
# Data loading
# ============================================================
def load_task_data(task, data_dir):
    if task in LB_V1_TASKS:
        path = os.path.join(data_dir, "longbench", "data", f"{task}.jsonl")
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    elif task == "longbench_v2":
        path = os.path.join(data_dir, "longbench_v2", "data.json")
        with open(path) as f:
            return json.load(f)
    elif task in MATH_TASKS:
        local = os.path.join(data_dir, task)
        for fn in os.listdir(local):
            if fn.endswith(".parquet"):
                import pandas as pd
                return pd.read_parquet(os.path.join(local, fn)).to_dict("records")
        from datasets import load_dataset
        cfg = {"math500":("math-ai/math500","test",None), "gsm8k":("openai/gsm8k","test","main"),
               "aime24":("math-ai/aime24","test",None), "aime25":("math-ai/aime25","test",None)}
        repo, split, subset = cfg[task]
        ds = load_dataset(repo, subset, split=split) if subset else load_dataset(repo, split=split)
        return [dict(r) for r in ds]
    elif task == "mmlu_pro":
        local = os.path.join(data_dir, "mmlu_pro")
        for fn in os.listdir(local):
            if fn.endswith(".parquet"):
                import pandas as pd
                return pd.read_parquet(os.path.join(local, fn)).to_dict("records")
        from datasets import load_dataset
        return [dict(r) for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")]
    raise FileNotFoundError(f"Cannot load {task}")

def build_prompt(task, item, tokenizer):
    if task in LB_V1_TASKS:
        ctx = item.get("context","")
        inp = item.get("input","")
        if task in ["gov_report","multi_news","qmsum","samsum"]:
            content = f"Read and summarize:\n\n{ctx}\n\nSummary:"
        else:
            content = f"Read and answer:\n\n{ctx}\n\nQuestion: {inp}\nAnswer:"
    elif task == "longbench_v2":
        ctx = item.get("context","")
        q = item.get("question","")
        choices = "\n".join(f"{c}. {item.get(f'choice_{c}','')}" for c in "ABCD")
        content = f"{ctx}\n\nQuestion: {q}\n{choices}\nAnswer:"
    elif task in MATH_TASKS:
        p = item.get("problem", item.get("question",""))
        content = f"Problem:\n{p}\n\nSolve step by step. Put final answer in \\boxed{{}}."
    elif task == "mmlu_pro":
        q = item.get("question","")
        opts = item.get("options",[])
        choices = "\n".join(f"{'ABCDEFGHIJ'[i]}. {o}" for i,o in enumerate(opts))
        content = f"Question: {q}\n\n{choices}\n\nAnswer with letter only:"
    else:
        content = str(item)

    msgs = [{"role":"user","content":content}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except:
        return content

def get_gold(task, item):
    if task in LB_V1_TASKS:
        ans = item.get("answers",[])
        return ans[0] if isinstance(ans,list) and ans else item.get("answer","")
    elif task == "longbench_v2":
        return item.get("answer","")
    elif task in MATH_TASKS:
        return str(item.get("answer",""))
    elif task == "mmlu_pro":
        return str(item.get("answer",""))
    return ""

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing LongBench/LongBench-v2/math/MMLU-Pro data",
    )
    parser.add_argument("--output_dir", default="results/9configs")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

    cfg = CONFIGS[args.config]
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS

    # Load model + patch P-stage
    pm = cfg["patch_mode"]
    print(f"Config: {args.config} | patch_mode={pm}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map="cuda", torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    )
    model = patch_llama3_attention_forward_model(model, mode=pm, window_size=256, top_ratio=0.05)
    model.eval()

    for task in tasks:
        outfile = os.path.join(args.output_dir, f"{args.config}_{task}.jsonl")
        if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            print(f"[SKIP] {outfile} exists")
            continue

        try:
            data = load_task_data(task, args.data_dir)
        except Exception as e:
            print(f"[FAIL] {task}: {e}")
            continue

        if 0 < args.max_samples < len(data):
            data = data[:args.max_samples]

        metric_name, metric_fn, max_gen = TASK_METRICS.get(task, ("F1", f1_score, 128))
        scores = []

        with open(outfile, "w") as fout:
            for item in tqdm(data, desc=f"{args.config}/{task}"):
                prompt = build_prompt(task, item, tokenizer)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=131072-max_gen).to(model.device)
                with torch.no_grad():
                    gen_kwargs = dict(max_new_tokens=max_gen, do_sample=False)
                    if args.config in _D_STAGE_CONFIGS:
                        gen_kwargs["past_key_values"] = _make_d_stage_cache(args.config, model)
                    out = model.generate(**inputs, **gen_kwargs)
                pred = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                gold = get_gold(task, item)

                if metric_fn:
                    sc = metric_fn(pred, gold)
                else:
                    # math_equal
                    from grader import math_equal
                    from parser import extract_answer
                    pred_ans = extract_answer(pred, data_name="math")
                    try: sc = float(math_equal(str(pred_ans), str(gold)))
                    except: sc = 0.0

                scores.append(sc)
                fout.write(json.dumps({"score":sc,"pred":pred[:200],"gold":gold,"task":task,"config":args.config}, ensure_ascii=False)+"\n")
                fout.flush()

        avg = sum(scores)/max(len(scores),1)*100
        print(f"  {task}: {metric_name}={avg:.1f}% (n={len(scores)})")

    print(f"\nDone. Results in {args.output_dir}/")

if __name__ == "__main__":
    main()
