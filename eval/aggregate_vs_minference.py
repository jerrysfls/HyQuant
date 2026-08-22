#!/usr/bin/env python3
"""Unified comparison: hybrid_attention configs vs MInference attn variants.

Reads two parallel results directories (one per evaluator), merges by (task, label),
and prints a side-by-side table with both score and prefill speed.
"""
import argparse
import json
import os
from typing import Dict, Optional


def load_summary(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fmt(v, w=7):
    if v is None:
        return f"{'n/a':>{w}}"
    return f"{v:>{w}.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid_dir", required=True)
    p.add_argument("--minference_dir", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--hybrid_labels", nargs="+", required=True)
    p.add_argument("--minference_labels", nargs="+", required=True)
    args = p.parse_args()

    # ---- Collect summaries ----
    rows = {}   # task -> {label_full: {"score":..., "ttp":..., "kind":...}}
    metric_by_task = {}
    for task in args.tasks:
        rows[task] = {}
        for lbl in args.hybrid_labels:
            s = load_summary(os.path.join(args.hybrid_dir, f"{task}_{lbl}_summary.json"))
            rows[task][f"H:{lbl}"] = {
                "score": s.get("metric_value") if s else None,
                "tps":   s.get("overall_prefill_tps") if s else None,
                "kind":  "hybrid",
            }
            if s and task not in metric_by_task:
                metric_by_task[task] = s.get("metric_name", "?")
        for lbl in args.minference_labels:
            s = load_summary(os.path.join(args.minference_dir, f"{task}_{lbl}_summary.json"))
            rows[task][f"M:{lbl}"] = {
                "score": s.get("metric_value") if s else None,
                "tps":   s.get("overall_prefill_tps") if s else None,
                "kind":  "minference",
            }
            if s and task not in metric_by_task:
                metric_by_task[task] = s.get("metric_name", "?")

    all_labels = [f"H:{l}" for l in args.hybrid_labels] + \
                 [f"M:{l}" for l in args.minference_labels]
    col_w = max(12, max(len(l) for l in all_labels) + 1)

    # ---- Score table ----
    print("\n" + "=" * 60)
    print("  Accuracy (% score)")
    print("=" * 60)
    print(f"{'Task':<18}{'Metric':<10}" + "".join(f"{l:>{col_w}}" for l in all_labels))
    print("-" * (28 + col_w * len(all_labels)))
    for task in args.tasks:
        row = f"{task:<18}{metric_by_task.get(task, '?'):<10}"
        for lbl in all_labels:
            row += fmt(rows[task][lbl]["score"], col_w)
        print(row)

    # Mean (abs)
    row = f"{'Mean':<18}{'':<10}"
    for lbl in all_labels:
        vals = [rows[t][lbl]["score"] for t in args.tasks if rows[t][lbl]["score"] is not None]
        row += fmt(sum(vals)/len(vals) if vals else None, col_w)
    print(row)

    # ---- Speed table ----
    print("\n" + "=" * 60)
    print("  Prefill speed (tok/s)")
    print("=" * 60)
    print(f"{'Task':<18}{'':<10}" + "".join(f"{l:>{col_w}}" for l in all_labels))
    print("-" * (28 + col_w * len(all_labels)))
    for task in args.tasks:
        row = f"{task:<18}{'':<10}"
        for lbl in all_labels:
            v = rows[task][lbl]["tps"]
            row += f"{('n/a' if v is None else f'{v:.0f}'):>{col_w}}"
        print(row)

    # Mean
    row = f"{'Mean':<18}{'':<10}"
    for lbl in all_labels:
        vals = [rows[t][lbl]["tps"] for t in args.tasks if rows[t][lbl]["tps"] is not None]
        row += f"{('n/a' if not vals else f'{sum(vals)/len(vals):.0f}'):>{col_w}}"
    print(row)

    # ---- Δ vs fa2 (accuracy) and ratio vs fa2 (speed) ----
    if "H:fa2" in all_labels:
        print("\n" + "=" * 60)
        print("  Δaccuracy vs fa2  /  speedup × vs fa2")
        print("=" * 60)
        print(f"{'Task':<18}{'Metric':<10}" + "".join(
            f"{l:>{col_w}}" for l in all_labels if l != "H:fa2"))
        for task in args.tasks:
            base_s = rows[task]["H:fa2"]["score"]
            base_t = rows[task]["H:fa2"]["tps"]
            srow = f"{task:<18}{metric_by_task.get(task, '?'):<10}"
            spdrow = f"{'  speed':<18}{'':<10}"
            for lbl in all_labels:
                if lbl == "H:fa2":
                    continue
                s = rows[task][lbl]["score"]
                t = rows[task][lbl]["tps"]
                ds = (s - base_s) if (s is not None and base_s is not None) else None
                rt = (t / base_t) if (t and base_t) else None
                srow += f"{('n/a' if ds is None else f'{ds:+.2f}'):>{col_w}}"
                spdrow += f"{('n/a' if rt is None else f'{rt:.2f}x'):>{col_w}}"
            print(srow)
            print(spdrow)

    # ---- Missing runs ----
    missing = [(t, l) for t in args.tasks for l in all_labels
               if rows[t][l]["score"] is None]
    if missing:
        print()
        print(f"[warn] missing {len(missing)} (task, label) summaries:")
        for t, l in missing[:15]:
            print(f"        {t} / {l}")
        if len(missing) > 15:
            print(f"        ... +{len(missing)-15} more")


if __name__ == "__main__":
    main()
