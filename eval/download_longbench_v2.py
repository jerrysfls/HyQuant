#!/usr/bin/env python3
"""Download zai-org/LongBench-v2 into the local datasets dir.

Usage:
  python eval/download_longbench_v2.py
  # or
  python eval/download_longbench_v2.py --dest /custom/path
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dest",
        default="datasets/longbench_v2",
        help="Local directory to populate.",
    )
    p.add_argument(
        "--repo_id",
        default="zai-org/LongBench-v2",
        help="HF repo id (defaults to zai-org/LongBench-v2).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (or set $HF_TOKEN).",
    )
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install -U huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.dest, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {args.dest} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.dest,
        local_dir_use_symlinks=False,
        token=args.token,
        max_workers=8,
        # Filter to actual data files to save bandwidth; LongBench-v2 ships parquet/jsonl
        allow_patterns=["*.json", "*.jsonl", "*.parquet", "*.md", "data/**"],
    )
    print(f"Done. Snapshot at: {path}")

    # Show a quick listing
    for root, _, files in os.walk(args.dest):
        rel = os.path.relpath(root, args.dest)
        for f in files:
            full = os.path.join(rel, f) if rel != "." else f
            size_mb = os.path.getsize(os.path.join(root, f)) / 1e6
            print(f"  {full}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
