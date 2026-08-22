#!/usr/bin/env python3
"""Aggregate per-(task, mode) _summary.json and print a Δscore table.

Reads <results_dir>/<task>_<mode>_summary.json files, extracts the headline
metric (F1 or ROUGE-L or Accuracy), and prints a wide table with one row per
task and one column per mode. Also reports Δ vs a chosen baseline mode.
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional


def load_summary(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_score(summary: Dict) -> Optional[float]:
    """Extract the headline percentage score from a summary dict."""
    # eval_longbench.py writes "metric_value" (already in percent, e.g. 90.73)
    for key in ("metric_value", "avg_score", "score", "f1", "rouge_l", "accuracy"):
        if key in summary:
            return float(summary[key])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", required=True)
    p.add_argument("--baseline", default="flash_attention_2",
                   help="Mode to use as the reference for Δscore.")
    args = p.parse_args()

    # Load scores: scores[task][mode] = float | None
    scores: Dict[str, Dict[str, Optional[float]]] = {}
    metrics: Dict[str, str] = {}  # task -> metric name (F1 / ROUGE-L / Accuracy)
    for task in args.tasks:
        scores[task] = {}
        for mode in args.modes:
            summ = load_summary(os.path.join(args.results_dir, f"{task}_{mode}_summary.json"))
            scores[task][mode] = get_score(summ) if summ else None
            if summ and "metric_name" in summ and task not in metrics:
                metrics[task] = summ["metric_name"]

    # ---- Absolute score table ----
    col_w = max(12, max(len(m) for m in args.modes) + 1)
    print()
    print(f"{'Task':<18}{'Metric':<10}" + "".join(f"{m:>{col_w}}" for m in args.modes))
    print("-" * (18 + 10 + col_w * len(args.modes)))
    for task in args.tasks:
        row = f"{task:<18}{metrics.get(task, '?'):<10}"
        for mode in args.modes:
            v = scores[task][mode]
            row += f"{('n/a' if v is None else f'{v:6.2f}'):>{col_w}}"
        print(row)

    # ---- Δ vs baseline ----
    if args.baseline not in args.modes:
        print(f"\n[warn] baseline mode '{args.baseline}' not in --modes; skipping Δ table")
        return
    print()
    print(f"Δ vs {args.baseline} (positive = quantized method better)")
    print(f"{'Task':<18}{'Metric':<10}" + "".join(
        f"{m:>{col_w}}" for m in args.modes if m != args.baseline))
    print("-" * (18 + 10 + col_w * (len(args.modes) - 1)))
    for task in args.tasks:
        base = scores[task].get(args.baseline)
        row = f"{task:<18}{metrics.get(task, '?'):<10}"
        for mode in args.modes:
            if mode == args.baseline:
                continue
            v = scores[task][mode]
            if v is None or base is None:
                row += f"{'n/a':>{col_w}}"
            else:
                d = v - base
                sign = "+" if d >= 0 else ""
                row += f"{sign}{d:5.2f}".rjust(col_w)
        print(row)

    # ---- Per-mode mean across tasks ----
    print()
    print(f"{'Mean':<18}{'-':<10}" + "".join(
        f"{m:>{col_w}}" for m in args.modes))
    row = f"{'Mean (abs)':<18}{'':<10}"
    for mode in args.modes:
        vals = [scores[t][mode] for t in args.tasks if scores[t][mode] is not None]
        row += f"{(f'{sum(vals)/len(vals):6.2f}' if vals else 'n/a'):>{col_w}}"
    print(row)

    # Sanity warning about missing runs
    missing = [
        (t, m) for t in args.tasks for m in args.modes
        if scores[t][m] is None
    ]
    if missing:
        print()
        print(f"[warn] missing {len(missing)} (task, mode) summaries:")
        for t, m in missing[:10]:
            print(f"        {t} / {m}")
        if len(missing) > 10:
            print(f"        ... and {len(missing) - 10} more")


if __name__ == "__main__":
    main()
