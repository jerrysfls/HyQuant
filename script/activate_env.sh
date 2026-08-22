#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script instead of executing it:"
  echo "  source script/activate_env.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Virtualenv not found: $VENV_DIR"
  echo "Run ./script/install_env.sh first."
  return 1
fi

source "$VENV_DIR/bin/activate"

case ":${PYTHONPATH:-}:" in
  *":$PROJECT_DIR:"*) ;;
  *)
    if [[ -n "${PYTHONPATH:-}" ]]; then
      export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    else
      export PYTHONPATH="$PROJECT_DIR"
    fi
    ;;
esac

cd "$PROJECT_DIR" || return 1

echo "Activated hybrid_attention environment."
echo "VENV: $VENV_DIR"
echo "PYTHONPATH includes: $PROJECT_DIR"
