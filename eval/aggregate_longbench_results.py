#!/usr/bin/env python3
"""Aggregate per-(task, mode) summary JSONs into a single comparison table."""
import argparse
import glob
import json
import os
from collections import defaultdict


TASK_LABEL = {
    "narrativeqa": "NarrQA",
    "qasper": "Qasper",
    "multifieldqa_en": "MF-en",
    "hotpotqa": "HotpotQA",
    "2wikimqa": "2Wiki",
    "musique": "Musique",
    "qmsum": "QMSum",
    "multi_news": "MNews",
    "triviaqa": "Trivia",
    "samsum": "SAMSum",
    "passage_retrieval_en": "Pretrieve-en",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    summaries = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*_summary.json"))):
        with open(path) as f:
            summaries.append(json.load(f))

    if not summaries:
        print(f"No summary files found in {args.results_dir}")
        return

    by_mode = defaultdict(dict)
    for s in summaries:
        by_mode[s["mode"]][s["task"]] = s

    modes = sorted(by_mode.keys())
    tasks_present = sorted({s["task"] for s in summaries},
                           key=lambda t: list(TASK_LABEL.keys()).index(t)
                                          if t in TASK_LABEL else 999)

    # ----- Quality table -----
    print("\n" + "=" * 80)
    print("  QUALITY (F1 / ROUGE-L / Accuracy, %)")
    print("=" * 80)
    header = f"{'Task':<14}" + "".join(f"{m:>20}" for m in modes)
    print(header)
    print("-" * len(header))
    for task in tasks_present:
        label = TASK_LABEL.get(task, task)
        row = f"{label:<14}"
        for mode in modes:
            s = by_mode[mode].get(task)
            if s is None:
                row += f"{'--':>20}"
            else:
                row += f"{s['metric_value']:>18.2f} {s['metric_name'][:1]}"
        print(row)

    # Mean over tasks
    print("-" * len(header))
    row = f"{'Mean':<14}"
    for mode in modes:
        vals = [by_mode[mode][t]["metric_value"] for t in tasks_present
                if t in by_mode[mode]]
        row += f"{(sum(vals)/max(len(vals),1)):>18.2f}  "
    print(row)

    # ----- Throughput table (decode tok/s, overall) -----
    print("\n" + "=" * 80)
    print("  THROUGHPUT — decode (tok/s, higher is better)")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for task in tasks_present:
        label = TASK_LABEL.get(task, task)
        row = f"{label:<14}"
        for mode in modes:
            s = by_mode[mode].get(task)
            if s is None:
                row += f"{'--':>20}"
            else:
                row += f"{s['overall_decode_tps']:>20.2f}"
        print(row)
    print("-" * len(header))
    row = f"{'Mean':<14}"
    for mode in modes:
        vals = [by_mode[mode][t]["overall_decode_tps"] for t in tasks_present
                if t in by_mode[mode]]
        row += f"{(sum(vals)/max(len(vals),1)):>20.2f}"
    print(row)

    # ----- Throughput table (prefill tok/s) -----
    print("\n" + "=" * 80)
    print("  THROUGHPUT — prefill (tok/s, higher is better)")
    print("=" * 80)
    print(header)
    print("-" * len(header))
    for task in tasks_present:
        label = TASK_LABEL.get(task, task)
        row = f"{label:<14}"
        for mode in modes:
            s = by_mode[mode].get(task)
            if s is None:
                row += f"{'--':>20}"
            else:
                row += f"{s['overall_prefill_tps']:>20.2f}"
        print(row)
    print("-" * len(header))
    row = f"{'Mean':<14}"
    for mode in modes:
        vals = [by_mode[mode][t]["overall_prefill_tps"] for t in tasks_present
                if t in by_mode[mode]]
        row += f"{(sum(vals)/max(len(vals),1)):>20.2f}"
    print(row)

    # Save aggregate CSV
    csv_path = os.path.join(args.results_dir, "aggregate.csv")
    with open(csv_path, "w") as f:
        f.write("task,mode,metric_name,metric_value,avg_input_len,avg_output_len,"
                "prefill_tps,decode_tps,total_tps\n")
        for task in tasks_present:
            for mode in modes:
                s = by_mode[mode].get(task)
                if s is None:
                    continue
                f.write(f"{task},{mode},{s['metric_name']},{s['metric_value']:.4f},"
                        f"{s['avg_input_len']:.1f},{s['avg_output_len']:.1f},"
                        f"{s['overall_prefill_tps']:.2f},"
                        f"{s['overall_decode_tps']:.2f},"
                        f"{s['avg_total_tps']:.2f}\n")
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
