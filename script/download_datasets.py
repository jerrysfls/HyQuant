#!/usr/bin/env python3

"""Download and validate evaluation datasets used by this repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.third_party_datasets import dataset_local_dir, run_download

TARGET_DATASETS = ("longbench", "longbench_v2", "math500", "gsm8k")
LONG_BENCH_V1_TASKS = ("qasper", "hotpotqa", "gov_report")


@dataclass
class ValidationResult:
    ok: bool
    summary: str
    details: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download and validate core evaluation datasets")
    parser.add_argument(
        "--data-root",
        default=os.path.join(ROOT, "data"),
        help="Target directory for downloaded datasets. Defaults to <repo>/data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download datasets even if the target directory already exists.",
    )
    return parser.parse_args()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_first_jsonl_record(path: Path) -> tuple[dict, int]:
    first = None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            count += 1
            if first is None:
                first = json.loads(line)
    if first is None:
        raise ValueError(f"No JSONL records found in {path}")
    return first, count


def _read_first_json_array_record(path: Path) -> tuple[dict, int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    if not payload:
        raise ValueError(f"No JSON records found in {path}")
    return dict(payload[0]), len(payload)


def _read_first_parquet_record(path: Path) -> tuple[dict, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    if pq is not None:
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        if row_count < 1:
            raise ValueError(f"No parquet rows found in {path}")
        batch = next(parquet.iter_batches(batch_size=1))
        first = batch.to_pylist()[0]
        return dict(first), row_count

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Parquet validation requires pyarrow or pandas. Install dependencies with script/install_env.sh first."
        ) from exc

    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"No parquet rows found in {path}")
    return dict(frame.iloc[0].to_dict()), int(len(frame))


def _preview_keys(sample: dict) -> str:
    keys = list(sample.keys())[:6]
    return ", ".join(keys) if keys else "<no keys>"


def validate_longbench_v1(data_root: Path) -> ValidationResult:
    target_dir = Path(dataset_local_dir("longbench", str(data_root)))
    details = [f"local_dir={target_dir}"]
    missing = []

    for task in LONG_BENCH_V1_TASKS:
        path = _first_existing(
            (
                target_dir / "data" / f"{task}.jsonl",
                target_dir / f"{task}.jsonl",
            )
        )
        if path is None:
            missing.append(task)
            continue

        sample, row_count = _read_first_jsonl_record(path)
        details.append(
            f"task={task} path={path.relative_to(data_root)} rows={row_count} sample_keys={_preview_keys(sample)}"
        )

    if missing:
        return ValidationResult(
            ok=False,
            summary="missing required LongBench v1 task files",
            details=details + [f"missing_tasks={', '.join(missing)}"],
        )

    return ValidationResult(
        ok=True,
        summary=f"validated {len(LONG_BENCH_V1_TASKS)} LongBench v1 tasks",
        details=details,
    )


def validate_longbench_v2(data_root: Path) -> ValidationResult:
    target_dir = Path(dataset_local_dir("longbench_v2", str(data_root)))
    details = [f"local_dir={target_dir}"]
    candidates = [
        target_dir / "data.jsonl",
        target_dir / "data" / "data.jsonl",
        target_dir / "data.json",
    ]

    parquet_dir = target_dir / "data"
    if parquet_dir.is_dir():
        candidates.extend(sorted(parquet_dir.glob("*.parquet")))

    path = _first_existing(candidates)
    if path is None:
        return ValidationResult(
            ok=False,
            summary="could not find LongBench-v2 data.json, data.jsonl, or parquet files",
            details=details,
        )

    if path.suffix == ".json":
        sample, row_count = _read_first_json_array_record(path)
    elif path.suffix == ".jsonl":
        sample, row_count = _read_first_jsonl_record(path)
    elif path.suffix == ".parquet":
        sample, row_count = _read_first_parquet_record(path)
    else:
        return ValidationResult(
            ok=False,
            summary=f"unsupported LongBench-v2 file type: {path.name}",
            details=details,
        )

    details.append(f"path={path.relative_to(data_root)} rows={row_count} sample_keys={_preview_keys(sample)}")
    return ValidationResult(ok=True, summary="validated LongBench-v2 dataset", details=details)


def validate_parquet_dataset(name: str, data_root: Path) -> ValidationResult:
    target_dir = Path(dataset_local_dir(name, str(data_root)))
    details = [f"local_dir={target_dir}"]
    parquet_files = sorted(target_dir.glob("*.parquet"))
    if not parquet_files:
        return ValidationResult(
            ok=False,
            summary=f"no parquet files found for {name}",
            details=details,
        )

    sample, row_count = _read_first_parquet_record(parquet_files[0])
    details.append(
        f"path={(parquet_files[0]).relative_to(data_root)} rows={row_count} sample_keys={_preview_keys(sample)}"
    )
    return ValidationResult(ok=True, summary=f"validated {name} parquet dataset", details=details)


def validate_dataset(name: str, data_root: Path) -> ValidationResult:
    if name == "longbench":
        return validate_longbench_v1(data_root)
    if name == "longbench_v2":
        return validate_longbench_v2(data_root)
    if name in {"math500", "gsm8k"}:
        return validate_parquet_dataset(name, data_root)
    return ValidationResult(ok=False, summary=f"no validator registered for {name}", details=[])


def attempt_download(name: str, data_root: Path, force: bool) -> tuple[str, int, str]:
    try:
        return run_download(name, data_root=str(data_root), force=force)
    except FileNotFoundError as exc:
        return (
            "failed",
            127,
            f"Required download command not found while downloading {name}: {exc}. "
            "Install the Hugging Face CLI and ensure the `hf` command is available.",
        )
    except Exception as exc:
        return (
            "failed",
            1,
            f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print("== Dataset Download and Validation ==")
    print(f"repo_root:  {ROOT}")
    print(f"data_root:  {data_root}")
    print(f"datasets:   {' '.join(TARGET_DATASETS)}")
    print(f"force:      {args.force}")

    downloaded = 0
    skipped = 0
    failed = 0
    validated = 0

    for name in TARGET_DATASETS:
        print("")
        print(f"[dataset] {name}")

        status, code, message = attempt_download(name, data_root, args.force)
        if status == "downloaded":
            downloaded += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

        print(f"download_status={status}")
        print(f"local_dir={dataset_local_dir(name, str(data_root))}")
        if message:
            trimmed = message if len(message) <= 1500 else message[:1500] + "...<truncated>"
            print(trimmed)

        if status == "failed":
            print(f"download_exit_code={code}")
            continue

        try:
            validation = validate_dataset(name, data_root)
        except Exception as exc:
            failed += 1
            print(f"validation_status=failed")
            print(f"validation_error={type(exc).__name__}: {exc}")
            continue

        print(f"validation_status={'ok' if validation.ok else 'failed'}")
        print(f"validation_summary={validation.summary}")
        for detail in validation.details:
            print(detail)

        if validation.ok:
            validated += 1
        else:
            failed += 1

    print("")
    print(
        f"[summary] downloaded={downloaded} skipped={skipped} validated={validated} failed={failed} "
        f"data_root={data_root}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
