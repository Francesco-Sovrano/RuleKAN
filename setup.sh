#!/usr/bin/env bash
set -euo pipefail

# Create a local development environment for the RuleKAN Python package.
# Benchmark baselines are installed separately with ./benchmarks/setup.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEV="${INSTALL_DEV:-1}"
INSTALL_EXAMPLES="${INSTALL_EXAMPLES:-0}"

log() { printf '\n==> %s\n' "$*"; }
fail() { echo "Error: $*" >&2; exit 1; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "'$PYTHON_BIN' was not found. Install Python 3.10 or later, or set PYTHON_BIN."
fi

"$PYTHON_BIN" - <<'PY' || exit 1
import sys
if sys.version_info < (3, 10):
    raise SystemExit("RuleKAN requires Python 3.10 or later")
PY

[[ -f "$ROOT_DIR/pyproject.toml" ]] || fail "pyproject.toml not found in $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading Python build tooling"
python -m pip install --upgrade pip setuptools wheel

EXTRAS=()
if [[ "$INSTALL_DEV" == "1" ]]; then
  EXTRAS+=(dev)
fi
if [[ "$INSTALL_EXAMPLES" == "1" ]]; then
  EXTRAS+=(examples)
fi

if [[ ${#EXTRAS[@]} -eq 0 ]]; then
  SPEC="$ROOT_DIR"
else
  IFS=,
  SPEC="$ROOT_DIR[${EXTRAS[*]}]"
  unset IFS
fi

log "Installing RuleKAN in editable mode"
python -m pip install -e "$SPEC"

log "Checking RuleKAN import"
python - <<'PY'
import rulekan
from rulekan import SumProductKAN, PowerRuleKAN
print("rulekan:", rulekan.__version__)
print("SumProductKAN:", SumProductKAN.__name__)
print("PowerRuleKAN:", PowerRuleKAN.__name__)
PY

echo
echo "RuleKAN setup complete."
echo "Activate with: source \"$VENV_DIR/bin/activate\""
echo "Test the package only with: python -m examples.quickstart"
echo "Install benchmark baselines separately with: ./benchmarks/setup.sh"
