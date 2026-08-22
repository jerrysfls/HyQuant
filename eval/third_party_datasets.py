"""Unified third-party dataset registry and loading interface.

This module is intentionally evaluation-framework agnostic. It provides:
- a central dataset registry for the third-party datasets mentioned in this repo,
- helper APIs to build download commands,
- a lightweight normalization layer that converts heterogeneous samples into a
  common dictionary shape for downstream evaluators.

It does not claim that every dataset here already has a complete evaluator in
this repository. The goal is to make dataset discovery and loading consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo_id: str
    source_type: str = "hf"
    task_type: str = "qa"
    split: Optional[str] = None
    subset: Optional[str] = None
    local_dir_name: Optional[str] = None
    parser_name: Optional[str] = None
    gated: bool = False
    notes: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "math500": DatasetSpec(
        name="math500",
        repo_id="math-ai/math500",
        split="test",
        local_dir_name="math500",
        parser_name="math",
        notes="Current repo mainline dataset.",
    ),
    "gsm8k": DatasetSpec(
        name="gsm8k",
        repo_id="openai/gsm8k",
        subset="main",
        split="test",
        local_dir_name="gsm8k",
        parser_name="gsm8k",
    ),
    "aime24": DatasetSpec(
        name="aime24",
        repo_id="math-ai/aime24",
        split="train",
        local_dir_name="aime24",
        parser_name="aime24",
    ),
    "aime25": DatasetSpec(
        name="aime25",
        repo_id="math-ai/aime25",
        split="train",
        local_dir_name="aime25",
        parser_name="aime25",
    ),
    "amc23": DatasetSpec(
        name="amc23",
        repo_id="math-ai/amc23",
        split="train",
        local_dir_name="amc23",
        parser_name="amc23",
    ),
    "minerva_math": DatasetSpec(
        name="minerva_math",
        repo_id="math-ai/minervamath",
        split="train",
        local_dir_name="minervamath",
        parser_name="minerva_math",
    ),
    "mmlu": DatasetSpec(
        name="mmlu",
        repo_id="cais/mmlu",
        task_type="mcq",
        split="test",
        local_dir_name="mmlu",
        parser_name="mmlu",
    ),
    "mmlu_stem": DatasetSpec(
        name="mmlu_stem",
        repo_id="TIGER-Lab/MMLU-STEM",
        task_type="mcq",
        split="test",
        local_dir_name="mmlu_stem",
        parser_name="mmlu_stem",
    ),
    "mmlu_pro": DatasetSpec(
        name="mmlu_pro",
        repo_id="TIGER-Lab/MMLU-Pro",
        task_type="mcq",
        split="test",
        local_dir_name="mmlu_pro",
        parser_name="mmlu",
    ),
    "gpqa": DatasetSpec(
        name="gpqa",
        repo_id="Idavidrein/gpqa",
        task_type="mcq",
        split="train",
        local_dir_name="gpqa",
        gated=True,
        notes="Access is gated on Hugging Face.",
    ),
    "humaneval": DatasetSpec(
        name="humaneval",
        repo_id="openai/openai_humaneval",
        task_type="code",
        split="test",
        local_dir_name="humaneval",
    ),
    "livecodebench": DatasetSpec(
        name="livecodebench",
        repo_id="livecodebench/code_generation_lite",
        task_type="code",
        split="test",
        local_dir_name="livecodebench_codegen_lite",
        notes="Use a pinned version_tag when loading via datasets.",
    ),
    "longbench": DatasetSpec(
        name="longbench",
        repo_id="THUDM/LongBench",
        task_type="long_context",
        subset="qasper",
        split="test",
        local_dir_name="longbench",
    ),
    "longbench_v2": DatasetSpec(
        name="longbench_v2",
        repo_id="THUDM/LongBench-v2",
        task_type="long_context",
        split="train",
        local_dir_name="longbench_v2",
    ),
    "ruler": DatasetSpec(
        name="ruler",
        repo_id="https://github.com/NVIDIA/RULER.git",
        source_type="git",
        task_type="long_context",
        local_dir_name="RULER",
    ),
    "infibench": DatasetSpec(
        name="infibench",
        repo_id="https://github.com/infi-coder/infibench-evaluation-harness.git",
        source_type="git",
        task_type="long_context",
        local_dir_name="InfiBench",
    ),
    "math": DatasetSpec(
        name="math",
        repo_id="EleutherAI/hendrycks_math",
        split="test",
        local_dir_name="hendrycks_math",
        parser_name="math",
    ),
    "humaneval_plus": DatasetSpec(
        name="humaneval_plus",
        repo_id="evalplus/humanevalplus",
        task_type="code",
        split="test",
        local_dir_name="humanevalplus",
    ),
    "mbpp": DatasetSpec(
        name="mbpp",
        repo_id="google-research-datasets/mbpp",
        task_type="code",
        split="test",
        local_dir_name="mbpp",
    ),
    "multipl_e": DatasetSpec(
        name="multipl_e",
        repo_id="nuprl/MultiPL-E",
        task_type="code",
        split="test",
        local_dir_name="multipl_e",
    ),
    "olympiadbench": DatasetSpec(
        name="olympiadbench",
        repo_id="Hothan/OlympiadBench",
        split="train",
        local_dir_name="olympiadbench",
        parser_name="olympiadbench",
    ),
    "collegemath": DatasetSpec(
        name="collegemath",
        repo_id="cheongmyeong17/MathScale-CollegeMath-cleaned",
        split="train",
        local_dir_name="collegemath",
        parser_name="college_math",
        notes="Public cleaned mirror, not treated as canonical by this repo.",
    ),
    "cmath": DatasetSpec(
        name="cmath",
        repo_id="weitianwen/cmath",
        split="train",
        local_dir_name="cmath",
        parser_name="cmath",
    ),
    "aqua": DatasetSpec(
        name="aqua",
        repo_id="https://github.com/google-deepmind/AQuA.git",
        source_type="git",
        task_type="mcq",
        local_dir_name="aqua_rat",
        parser_name="aqua",
    ),
    "gaokao2023_math_en": DatasetSpec(
        name="gaokao2023_math_en",
        repo_id="MARIO-Math-Reasoning/Gaokao2023-Math-En",
        split="train",
        local_dir_name="gaokao2023_math_en",
        parser_name="gaokao2023en",
    ),
}


def list_supported_datasets() -> list[str]:
    return sorted(DATASET_SPECS)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.strip().lower().replace("-", "_")
    if key not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Supported datasets: {', '.join(list_supported_datasets())}")
    return DATASET_SPECS[key]


def get_parser_name(name: str) -> str:
    spec = get_dataset_spec(name)
    return spec.parser_name or spec.name


def build_download_command(name: str, data_root: str = "$DATA_ROOT") -> str:
    spec = get_dataset_spec(name)
    local_dir_name = spec.local_dir_name or spec.name
    local_dir = f"{data_root.rstrip('/')}/{local_dir_name}"
    if spec.source_type == "git":
        return f'git clone {spec.repo_id} "{local_dir}"'
    return f'hf download --repo-type dataset {spec.repo_id} --local-dir "{local_dir}"'


def dataset_local_dir(name: str, data_root: str) -> str:
    spec = get_dataset_spec(name)
    local_dir_name = spec.local_dir_name or spec.name
    return os.path.join(data_root, local_dir_name)


def run_download(
    name: str,
    *,
    data_root: str,
    force: bool = False,
) -> tuple[str, int, str]:
    spec = get_dataset_spec(name)
    target_dir = dataset_local_dir(name, data_root)
    os.makedirs(data_root, exist_ok=True)

    if os.path.exists(target_dir) and os.listdir(target_dir) and not force:
        return "skipped", 0, f"already exists: {target_dir}"

    if spec.source_type == "git":
        cmd = ["git", "clone", spec.repo_id, target_dir]
    else:
        hf_bin = os.path.expanduser("~/.local/bin/hf")
        if os.path.exists(hf_bin):
            cmd = [hf_bin, "download", "--repo-type", "dataset", spec.repo_id, "--local-dir", target_dir]
        else:
            cmd = ["hf", "download", "--repo-type", "dataset", spec.repo_id, "--local-dir", target_dir]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return "downloaded", proc.returncode, output.strip()
    return "failed", proc.returncode, output.strip()


def _maybe_import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading third-party datasets requires the `datasets` package. "
            "Install it with `python -m pip install -U datasets`."
        ) from exc
    return load_dataset


def _choose_default_split(dataset_obj: Any, spec: DatasetSpec, split: Optional[str]) -> Any:
    if split is not None:
        return dataset_obj
    if not isinstance(dataset_obj, dict):
        return dataset_obj
    preferred = spec.split or "test"
    if preferred in dataset_obj:
        return dataset_obj[preferred]
    first_key = next(iter(dataset_obj))
    return dataset_obj[first_key]


def _first_present(example: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return default


def _normalize_choices(example: dict[str, Any]) -> Any:
    choices = _first_present(example, ("choices", "options"))
    if isinstance(choices, dict):
        return choices
    if isinstance(choices, list):
        return {chr(ord("A") + idx): value for idx, value in enumerate(choices)}
    option_keys = [key for key in ("A", "B", "C", "D", "E") if key in example]
    if option_keys:
        return {key: example[key] for key in option_keys}
    return None


def _normalize_answer(example: dict[str, Any]) -> Any:
    answer = _first_present(
        example,
        (
            "answer",
            "correct",
            "target",
            "label",
            "expected_output",
            "canonical_solution",
            "solution",
            "reference_solution",
            "final_answer",
            "output",
        ),
    )
    if isinstance(answer, list) and answer:
        return answer[0]
    if isinstance(answer, int) and _normalize_choices(example):
        return chr(ord("A") + answer)
    return answer


def _normalize_problem(example: dict[str, Any]) -> str:
    problem = _first_present(
        example,
        (
            "problem",
            "question",
            "prompt",
            "instruction",
            "input",
            "query",
            "description",
        ),
        default="",
    )
    if isinstance(problem, list):
        problem = "\n".join(str(item) for item in problem)
    return str(problem)


def normalize_example(dataset_name: str, example: dict[str, Any], index: int) -> dict[str, Any]:
    spec = get_dataset_spec(dataset_name)
    unique_id = _first_present(
        example,
        (
            "unique_id",
            "id",
            "task_id",
            "question_id",
            "problem_id",
            "idx",
        ),
        default=index,
    )
    normalized = {
        "dataset": spec.name,
        "parser_name": get_parser_name(spec.name),
        "unique_id": unique_id,
        "problem": _normalize_problem(example),
        "question": _normalize_problem(example),
        "answer": _normalize_answer(example),
        "choices": _normalize_choices(example),
        "solution": _first_present(example, ("solution", "rationale", "explanation")),
        "test": _first_present(example, ("test", "test_list", "entry_point")),
        "raw_example": example,
    }
    return normalized


def iter_dataset_records(
    name: str,
    *,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    local_path: Optional[str] = None,
    limit: Optional[int] = None,
    **load_kwargs: Any,
) -> Iterable[dict[str, Any]]:
    spec = get_dataset_spec(name)
    if spec.source_type != "hf":
        raise ValueError(
            f"Dataset '{name}' is registered as a git-based benchmark source. "
            "Use `build_download_command()` and the benchmark's own loader/evaluator."
        )

    load_dataset = _maybe_import_datasets()
    repo_or_path = local_path or spec.repo_id
    subset_name = subset if subset is not None else spec.subset
    split_name = split if split is not None else spec.split

    dataset_obj = load_dataset(repo_or_path, subset_name, split=split_name, **load_kwargs)
    if split_name is None:
        dataset_obj = _choose_default_split(dataset_obj, spec, split)

    for index, example in enumerate(dataset_obj):
        if limit is not None and index >= limit:
            break
        yield normalize_example(spec.name, dict(example), index)


def load_dataset_records(
    name: str,
    *,
    split: Optional[str] = None,
    subset: Optional[str] = None,
    local_path: Optional[str] = None,
    limit: Optional[int] = None,
    **load_kwargs: Any,
) -> list[dict[str, Any]]:
    return list(
        iter_dataset_records(
            name,
            split=split,
            subset=subset,
            local_path=local_path,
            limit=limit,
            **load_kwargs,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Third-party dataset loading interface")
    parser.add_argument("--list", action="store_true", help="List registered datasets.")
    parser.add_argument("--dataset", type=str, help="Dataset name to inspect or load.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--local_path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--data_root", type=str, default="data/third_party")
    parser.add_argument("--show_download", action="store_true", help="Print the recommended download command.")
    parser.add_argument("--load_samples", action="store_true", help="Load and normalize a few samples.")
    parser.add_argument("--dump_json", action="store_true", help="Dump normalized samples as JSON.")
    parser.add_argument("--download", action="store_true", help="Download the selected dataset into --data_root.")
    parser.add_argument(
        "--download_all_public",
        action="store_true",
        help="Download all registry datasets except gated entries. Git-based benchmark repos are included.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the target directory already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name in list_supported_datasets():
            spec = get_dataset_spec(name)
            print(f"{name:18s} source={spec.source_type:4s} repo={spec.repo_id}")
        return

    if args.download_all_public:
        ok = 0
        skipped = 0
        failed = 0
        for name in list_supported_datasets():
            spec = get_dataset_spec(name)
            if spec.gated:
                print(f"[skip] {name}: gated dataset, download manually after access is granted.")
                skipped += 1
                continue
            status, code, message = run_download(name, data_root=args.data_root, force=args.force)
            if status == "downloaded":
                ok += 1
                print(f"[ok]   {name}: {dataset_local_dir(name, args.data_root)}")
            elif status == "skipped":
                skipped += 1
                print(f"[skip] {name}: {message}")
            else:
                failed += 1
                print(f"[fail] {name}: exit_code={code}")
                if message:
                    print(message[:1000])
        print(f"[summary] downloaded={ok} skipped={skipped} failed={failed} data_root={args.data_root}")
        return

    if not args.dataset:
        raise SystemExit("Provide --dataset <name> or use --list.")

    spec = get_dataset_spec(args.dataset)
    print(f"name={spec.name}")
    print(f"source_type={spec.source_type}")
    print(f"repo_id={spec.repo_id}")
    print(f"task_type={spec.task_type}")
    print(f"default_subset={spec.subset}")
    print(f"default_split={spec.split}")
    print(f"parser_name={get_parser_name(spec.name)}")
    if spec.gated:
        print("gated=true")
    if spec.notes:
        print(f"notes={spec.notes}")
    if args.show_download:
        print(build_download_command(spec.name, data_root=args.data_root))

    if args.download:
        if spec.gated:
            raise SystemExit(
                f"Dataset '{spec.name}' is gated. Request access on the dataset page first, then rerun the download."
            )
        status, code, message = run_download(spec.name, data_root=args.data_root, force=args.force)
        print(f"download_status={status}")
        if status == "failed":
            print(f"exit_code={code}")
            if message:
                print(message)
        else:
            print(f"local_dir={dataset_local_dir(spec.name, args.data_root)}")
        if not (args.load_samples or args.dump_json):
            return

    if spec.source_type != "hf":
        return

    if not (args.load_samples or args.dump_json):
        return

    rows = load_dataset_records(
        spec.name,
        split=args.split,
        subset=args.subset,
        local_path=args.local_path,
        limit=args.limit,
    )
    print(f"loaded_rows={len(rows)}")
    if args.dump_json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, default=str))
        return

    for row in rows:
        preview = {
            "unique_id": row["unique_id"],
            "problem": row["problem"][:160],
            "answer": row["answer"],
        }
        print(json.dumps(preview, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
