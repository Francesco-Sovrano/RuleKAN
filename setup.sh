#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a local Python environment and install this project's dependencies.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' was not found. Install Python 3 or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "Error: requirements.txt not found in $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo
echo "Setup complete."
echo "Activate the environment with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "Run the Sum-Product KAN example with:"
echo "  cd \"$ROOT_DIR\" && python -m examples.example_sum_product_kan"
echo "Run the older RuleMask example with:"
echo "  cd \"$ROOT_DIR\" && python -m examples.example_rulemask_product"
