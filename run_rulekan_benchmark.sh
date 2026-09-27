#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.env-baselines}"
PROFILE="${1:-standard}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/current}"
DEVICE="${RULEKAN_DEVICE:-cpu}"

if [[ ! -d "$ENV_DIR" ]]; then
  echo "Error: benchmark environment not found: $ENV_DIR" >&2
  echo "Create it with: ./benchmarks/setup.sh" >&2
  echo "Or set ENV_DIR to an existing benchmark environment." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if ! python - <<'PYDEPS' >/dev/null 2>&1
import importlib
for name in ("rulekan", "pytest", "yaml", "pandas", "matplotlib", "scipy", "sklearn", "torch", "ucimlrepo", "pmlb"):
    importlib.import_module(name)
PYDEPS
then
  echo "Error: $ENV_DIR is not a complete RuleKAN benchmark environment." >&2
  echo "Rebuild or repair it with: VENV_DIR=\"$ENV_DIR\" ./benchmarks/setup.sh" >&2
  exit 1
fi

check_optional_baselines() {
  python - "$ROOT_DIR/benchmarks/configs/default.yaml" "$PROFILE" <<'PYDEPS'
import sys, yaml
from benchmarks.config_utils import resolve_profile
from benchmarks.run_benchmark import _validate_optional_model_dependencies
cfg = yaml.safe_load(open(sys.argv[1]))
profile = resolve_profile(cfg, sys.argv[2])
_validate_optional_model_dependencies(list(profile.get("models", [])))
PYDEPS
}

if ! check_optional_baselines; then
  echo >&2
  echo "Repair the selected baseline stack with:" >&2
  echo "  VENV_DIR=\"$ENV_DIR\" ./benchmarks/setup.sh" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

cleanup() {
  python -m benchmarks.aggregate --run-dir "$RUN_DIR" --quiet >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "[test] running unit tests"
  (cd "$ROOT_DIR" && python -m pytest -q tests)
fi

echo "[benchmark] profile=$PROFILE"
echo "[benchmark] environment=$ENV_DIR"
echo "[benchmark] results=$RUN_DIR"
echo "[benchmark] device=$DEVICE"
echo "[benchmark] live table=$RUN_DIR/summary.md"
echo "[benchmark] live figures=$RUN_DIR/figures/*.pdf"

echo
(cd "$ROOT_DIR" && python -m benchmarks.run_benchmark \
  --config "$ROOT_DIR/benchmarks/configs/default.yaml" \
  --profile "$PROFILE" \
  --run-dir "$RUN_DIR" \
  --device "$DEVICE" \
  --aggregate-every 999999 \
  --resume \
  --reuse-completed \
  "$@")

echo
echo "Benchmark complete."
echo "  runs:      $RUN_DIR/runs.csv"
echo "  summary:   $RUN_DIR/summary.md"
echo "  latest:    $RUN_DIR/latest_results.md"
echo "  figures:   $RUN_DIR/figures/ (PDF only)"
