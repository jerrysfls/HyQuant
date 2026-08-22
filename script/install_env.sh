#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

fail() {
  echo "Error: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  cd script
  ./install_env.sh [options]

Options:
  --python <bin>             Python executable to use for venv creation
  --venv-dir <path>          Virtualenv directory
  --torch-version <ver>      Torch version to install
  --torch-index-url <url>    PyTorch wheel index URL
  --skip-flash-attn          Skip flash-attn installation
  -h, --help                 Show this help

Environment variable equivalents:
  PYTHON_BIN
  VENV_DIR
  TORCH_VERSION
  TORCH_INDEX_URL
  INSTALL_FLASH_ATTN=0

Examples:
  ./install_env.sh
  ./install_env.sh --python python3.10
  ./install_env.sh --skip-flash-attn
EOF
}

require_value() {
  local flag="$1"
  local value="${2-}"
  [[ -n "$value" ]] || fail "Missing value for ${flag}"
}

default_python="python3.10"
if ! command -v "$default_python" >/dev/null 2>&1; then
  default_python="python3"
fi

python_bin="${PYTHON_BIN:-$default_python}"
venv_dir="${VENV_DIR:-$PROJECT_DIR/.venv}"
torch_version="${TORCH_VERSION:-2.6.0}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
install_flash_attn="${INSTALL_FLASH_ATTN:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      require_value "$1" "${2-}"
      python_bin="$2"
      shift 2
      ;;
    --venv-dir)
      require_value "$1" "${2-}"
      venv_dir="$2"
      shift 2
      ;;
    --torch-version)
      require_value "$1" "${2-}"
      torch_version="$2"
      shift 2
      ;;
    --torch-index-url)
      require_value "$1" "${2-}"
      torch_index_url="$2"
      shift 2
      ;;
    --skip-flash-attn)
      install_flash_attn="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

command -v "$python_bin" >/dev/null 2>&1 || fail "Python executable not found: $python_bin"

echo "== hybrid_attention environment setup =="
echo "script dir:        $SCRIPT_DIR"
echo "project dir:       $PROJECT_DIR"
echo "python:            $python_bin"
echo "venv dir:          $venv_dir"
echo "torch version:     $torch_version"
echo "torch index url:   $torch_index_url"
echo "install flash-attn ${install_flash_attn}"

"$python_bin" -m venv "$venv_dir"
source "$venv_dir/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==${torch_version}" --index-url "$torch_index_url"

tmp_requirements=""
cleanup() {
  if [[ -n "$tmp_requirements" && -f "$tmp_requirements" ]]; then
    rm -f "$tmp_requirements"
  fi
}
trap cleanup EXIT

if [[ "$install_flash_attn" == "1" ]]; then
  python -m pip install --no-build-isolation -r "$PROJECT_DIR/requirements.txt"
else
  tmp_requirements="$(mktemp)"
  grep -v '^flash-attn' "$PROJECT_DIR/requirements.txt" > "$tmp_requirements"
  python -m pip install -r "$tmp_requirements"
fi

cat <<EOF

Environment is ready.

Activate:
  source "$SCRIPT_DIR/activate_env.sh"

Run tests:
  cd "$SCRIPT_DIR"
  ./test_llama.sh --model-path /path/to/llama
  ./test_qwen.sh --model-path /path/to/qwen
EOF
