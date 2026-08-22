#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  # Prefer the repo-local environment created by install_env.sh.
  source "$PROJECT_DIR/.venv/bin/activate"
fi

python "$SCRIPT_DIR/download_datasets.py" "$@"
