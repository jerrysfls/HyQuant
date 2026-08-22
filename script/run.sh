#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

usage() {
  cat <<EOF
Usage:
  cd script
  ./run.sh <command> [options]

Commands:
  longbench   Run LongBench tasks across one or more GPUs
  9configs    Run eval/eval_9configs.py with multi-GPU batching
  math        Run eval/eval_math_unified.py
  mcq         Run eval/eval_mcq.py
  python      Run any repo-relative Python script under the project root
  help        Show this message

Examples:
  ./run.sh longbench --model-path /path/to/model --data-dir /path/to/data
  ./run.sh 9configs --model-path /path/to/model --data-dir /path/to/data --num-gpus 8
  ./run.sh math --model-path /path/to/model --data-dir /path/to/data --dataset math500 --mode full_attention
  ./run.sh mcq --model-path /path/to/model --data-dir /path/to/data --mode int8_pv_fp16_window
  ./run.sh python benchmarks/bench_decode_10k.py
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2-}"
  [[ -n "$value" ]] || fail "Missing value for ${flag}"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer, got '${value}'"
}

run_longbench() {
  local model_path=""
  local data_dir=""
  local results_dir="${PROJECT_DIR}/results/longbench"
  local tasks_csv="qasper,hotpotqa,gov_report,longbench_v2"
  local modes_csv="full_attention,int8_pv_fp16,int8_pv_fp16_window,int8_pv_fp16_vertical"
  local window_size="256"
  local top_ratio="0.05"
  local max_samples="-1"
  local num_gpus="8"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)
        require_value "$1" "${2-}"
        model_path="$2"
        shift 2
        ;;
      --data-dir)
        require_value "$1" "${2-}"
        data_dir="$2"
        shift 2
        ;;
      --results-dir)
        require_value "$1" "${2-}"
        results_dir="$2"
        shift 2
        ;;
      --tasks)
        require_value "$1" "${2-}"
        tasks_csv="$2"
        shift 2
        ;;
      --modes)
        require_value "$1" "${2-}"
        modes_csv="$2"
        shift 2
        ;;
      --window-size)
        require_value "$1" "${2-}"
        window_size="$2"
        shift 2
        ;;
      --top-ratio)
        require_value "$1" "${2-}"
        top_ratio="$2"
        shift 2
        ;;
      --max-samples)
        require_value "$1" "${2-}"
        max_samples="$2"
        shift 2
        ;;
      --num-gpus)
        require_value "$1" "${2-}"
        num_gpus="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        fail "Unknown option for longbench: $1"
        ;;
    esac
  done

  [[ -n "$model_path" ]] || fail "--model-path is required for longbench"
  [[ -n "$data_dir" ]] || fail "--data-dir is required for longbench"
  require_positive_int "--num-gpus" "$num_gpus"

  mkdir -p "$results_dir"

  local -a tasks=()
  local -a modes=()
  IFS=',' read -r -a tasks <<< "$tasks_csv"
  IFS=',' read -r -a modes <<< "$modes_csv"

  local gpu=0
  local round=1
  local total_jobs=$(( ${#tasks[@]} * ${#modes[@]} ))
  local job=0

  echo "== LongBench =="
  echo "script dir:  $SCRIPT_DIR"
  echo "project dir: $PROJECT_DIR"
  echo "model:       $model_path"
  echo "data:        $data_dir"
  echo "results:     $results_dir"
  echo "tasks:       ${tasks[*]}"
  echo "modes:       ${modes[*]}"
  echo "num gpus:    $num_gpus"
  echo "jobs:        $total_jobs"

  for task in "${tasks[@]}"; do
    for mode in "${modes[@]}"; do
      job=$((job + 1))

      local output_file="${results_dir}/longbench_${task}_${mode}.jsonl"
      local log_file="${results_dir}/longbench_${task}_${mode}.log"

      echo "[Job ${job}/${total_jobs}] GPU=${gpu} task=${task} mode=${mode}"

      CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_DIR/eval/eval_longbench.py" \
        --model_path "$model_path" \
        --task "$task" \
        --mode "$mode" \
        --data_dir "$data_dir" \
        --window_size "$window_size" \
        --top_ratio "$top_ratio" \
        --max_samples "$max_samples" \
        --output_file "$output_file" \
        > "$log_file" 2>&1 &

      gpu=$(( (gpu + 1) % num_gpus ))
      if [[ "$gpu" -eq 0 ]]; then
        echo "--- Waiting for round ${round} ---"
        wait
        round=$((round + 1))
      fi
    done
  done

  if [[ "$gpu" -ne 0 ]]; then
    echo "--- Waiting for final round ---"
    wait
  fi

  echo "LongBench finished. Logs and JSONL files are in ${results_dir}"
}

run_9configs() {
  local model_path=""
  local data_dir=""
  local output_dir="${PROJECT_DIR}/results/9configs"
  local configs_csv="pd_full,p_int8_pv_fp16_win,p_int8_pv_fp16_vert,p_int8_pv_fp16,d_k4v4,d_k4v4_vert,d_k8v8"
  local tasks=""
  local max_samples="-1"
  local num_gpus="8"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)
        require_value "$1" "${2-}"
        model_path="$2"
        shift 2
        ;;
      --data-dir)
        require_value "$1" "${2-}"
        data_dir="$2"
        shift 2
        ;;
      --output-dir)
        require_value "$1" "${2-}"
        output_dir="$2"
        shift 2
        ;;
      --configs)
        require_value "$1" "${2-}"
        configs_csv="$2"
        shift 2
        ;;
      --tasks)
        require_value "$1" "${2-}"
        tasks="$2"
        shift 2
        ;;
      --max-samples)
        require_value "$1" "${2-}"
        max_samples="$2"
        shift 2
        ;;
      --num-gpus)
        require_value "$1" "${2-}"
        num_gpus="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        fail "Unknown option for 9configs: $1"
        ;;
    esac
  done

  [[ -n "$model_path" ]] || fail "--model-path is required for 9configs"
  [[ -n "$data_dir" ]] || fail "--data-dir is required for 9configs"
  require_positive_int "--num-gpus" "$num_gpus"

  mkdir -p "$output_dir"

  local -a configs=()
  IFS=',' read -r -a configs <<< "$configs_csv"

  local gpu=0
  local batch=1
  local total_jobs="${#configs[@]}"
  local job=0

  echo "== 9configs =="
  echo "script dir:  $SCRIPT_DIR"
  echo "project dir: $PROJECT_DIR"
  echo "model:       $model_path"
  echo "data:        $data_dir"
  echo "output:      $output_dir"
  echo "configs:     ${configs[*]}"
  echo "num gpus:    $num_gpus"

  for config in "${configs[@]}"; do
    job=$((job + 1))
    local log_file="${output_dir}/${config}_all.log"
    local -a extra_args=()
    if [[ -n "$tasks" ]]; then
      extra_args+=(--tasks "$tasks")
    fi

    echo "[Job ${job}/${total_jobs}] GPU=${gpu} config=${config}"

    CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_DIR/eval/eval_9configs.py" \
      --config "$config" \
      --model_path "$model_path" \
      --data_dir "$data_dir" \
      --output_dir "$output_dir" \
      --max_samples "$max_samples" \
      "${extra_args[@]}" \
      > "$log_file" 2>&1 &

    gpu=$(( (gpu + 1) % num_gpus ))
    if [[ "$gpu" -eq 0 ]]; then
      echo "--- Waiting for batch ${batch} ---"
      wait
      batch=$((batch + 1))
    fi
  done

  if [[ "$gpu" -ne 0 ]]; then
    echo "--- Waiting for final batch ---"
    wait
  fi

  echo "9configs finished. Logs and JSONL files are in ${output_dir}"
}

run_math() {
  local model_path=""
  local data_dir=""
  local dataset=""
  local mode=""
  local results_dir="${PROJECT_DIR}/results/math"
  local output_file=""
  local window_size="256"
  local top_ratio="0.05"
  local max_samples="-1"
  local max_new_tokens="2048"
  local gpu="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)
        require_value "$1" "${2-}"
        model_path="$2"
        shift 2
        ;;
      --data-dir)
        require_value "$1" "${2-}"
        data_dir="$2"
        shift 2
        ;;
      --dataset)
        require_value "$1" "${2-}"
        dataset="$2"
        shift 2
        ;;
      --mode)
        require_value "$1" "${2-}"
        mode="$2"
        shift 2
        ;;
      --results-dir)
        require_value "$1" "${2-}"
        results_dir="$2"
        shift 2
        ;;
      --output-file)
        require_value "$1" "${2-}"
        output_file="$2"
        shift 2
        ;;
      --window-size)
        require_value "$1" "${2-}"
        window_size="$2"
        shift 2
        ;;
      --top-ratio)
        require_value "$1" "${2-}"
        top_ratio="$2"
        shift 2
        ;;
      --max-samples)
        require_value "$1" "${2-}"
        max_samples="$2"
        shift 2
        ;;
      --max-new-tokens)
        require_value "$1" "${2-}"
        max_new_tokens="$2"
        shift 2
        ;;
      --gpu)
        require_value "$1" "${2-}"
        gpu="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        fail "Unknown option for math: $1"
        ;;
    esac
  done

  [[ -n "$model_path" ]] || fail "--model-path is required for math"
  [[ -n "$data_dir" ]] || fail "--data-dir is required for math"
  [[ -n "$dataset" ]] || fail "--dataset is required for math"
  [[ -n "$mode" ]] || fail "--mode is required for math"

  mkdir -p "$results_dir"
  if [[ -z "$output_file" ]]; then
    output_file="${results_dir}/${dataset}_${mode}.jsonl"
  fi

  echo "== Math =="
  echo "dataset:     $dataset"
  echo "mode:        $mode"
  echo "gpu:         $gpu"
  echo "output file: $output_file"

  CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_DIR/eval/eval_math_unified.py" \
    --model_path "$model_path" \
    --dataset "$dataset" \
    --data_dir "$data_dir" \
    --mode "$mode" \
    --window_size "$window_size" \
    --top_ratio "$top_ratio" \
    --max_samples "$max_samples" \
    --max_new_tokens "$max_new_tokens" \
    --output_file "$output_file"
}

run_mcq() {
  local model_path=""
  local data_dir=""
  local dataset="mmlu_pro"
  local mode=""
  local results_dir="${PROJECT_DIR}/results/mcq"
  local output_file=""
  local window_size="256"
  local top_ratio="0.05"
  local max_samples="-1"
  local gpu="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model-path)
        require_value "$1" "${2-}"
        model_path="$2"
        shift 2
        ;;
      --data-dir)
        require_value "$1" "${2-}"
        data_dir="$2"
        shift 2
        ;;
      --dataset)
        require_value "$1" "${2-}"
        dataset="$2"
        shift 2
        ;;
      --mode)
        require_value "$1" "${2-}"
        mode="$2"
        shift 2
        ;;
      --results-dir)
        require_value "$1" "${2-}"
        results_dir="$2"
        shift 2
        ;;
      --output-file)
        require_value "$1" "${2-}"
        output_file="$2"
        shift 2
        ;;
      --window-size)
        require_value "$1" "${2-}"
        window_size="$2"
        shift 2
        ;;
      --top-ratio)
        require_value "$1" "${2-}"
        top_ratio="$2"
        shift 2
        ;;
      --max-samples)
        require_value "$1" "${2-}"
        max_samples="$2"
        shift 2
        ;;
      --gpu)
        require_value "$1" "${2-}"
        gpu="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        fail "Unknown option for mcq: $1"
        ;;
    esac
  done

  [[ -n "$model_path" ]] || fail "--model-path is required for mcq"
  [[ -n "$data_dir" ]] || fail "--data-dir is required for mcq"
  [[ -n "$mode" ]] || fail "--mode is required for mcq"

  mkdir -p "$results_dir"
  if [[ -z "$output_file" ]]; then
    output_file="${results_dir}/${dataset}_${mode}.jsonl"
  fi

  echo "== MCQ =="
  echo "dataset:     $dataset"
  echo "mode:        $mode"
  echo "gpu:         $gpu"
  echo "output file: $output_file"

  CUDA_VISIBLE_DEVICES="$gpu" python "$PROJECT_DIR/eval/eval_mcq.py" \
    --model_path "$model_path" \
    --dataset "$dataset" \
    --data_dir "$data_dir" \
    --mode "$mode" \
    --window_size "$window_size" \
    --top_ratio "$top_ratio" \
    --max_samples "$max_samples" \
    --output_file "$output_file"
}

run_repo_python() {
  local target_script="${1-}"
  [[ -n "$target_script" ]] || fail "python command requires a repo-relative script path"
  shift

  echo "== Python passthrough =="
  echo "script: ${target_script}"

  python "$PROJECT_DIR/$target_script" "$@"
}

main() {
  local command="${1-help}"
  if [[ $# -gt 0 ]]; then
    shift
  fi

  case "$command" in
    longbench)
      run_longbench "$@"
      ;;
    9configs|nineconfigs)
      run_9configs "$@"
      ;;
    math)
      run_math "$@"
      ;;
    mcq)
      run_mcq "$@"
      ;;
    python)
      run_repo_python "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      fail "Unknown command '${command}'. Run './run.sh help' for usage."
      ;;
  esac
}

main "$@"
