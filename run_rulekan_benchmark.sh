#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${1:-standard}"
if [[ $# -gt 0 ]]; then shift; fi
RUN_DIR="${RESULT_DIR:-$ROOT_DIR/benchmark_results/current}"
DEVICE="${RULEKAN_DEVICE:-cpu}"

# Profile names are defined by benchmarks/configs/default.yaml. The Python
# runner validates the requested profile directly against that configuration.


if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found" >&2
  exit 1
fi

if [[ ! -d "$ENV_DIR" ]]; then
  echo "[setup] creating virtual environment: $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

STAMP="$ENV_DIR/.rulekan_benchmark_requirements.stamp"
REQ_HASH="$(python - "$ROOT_DIR/requirements.txt" "$ROOT_DIR/benchmarks/requirements-benchmark.txt" <<'PYHASH'
import hashlib, pathlib, sys
h = hashlib.sha256()
for name in sys.argv[1:]:
    h.update(pathlib.Path(name).read_bytes())
print(h.hexdigest())
PYHASH
)"
if [[ ! -f "$STAMP" ]] || [[ "$(cat "$STAMP")" != "$REQ_HASH" ]]; then
  echo "[setup] installing/updating dependencies"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r "$ROOT_DIR/requirements.txt" -r "$ROOT_DIR/benchmarks/requirements-benchmark.txt"
  printf '%s\n' "$REQ_HASH" > "$STAMP"
else
  echo "[setup] dependencies unchanged"
fi

# SO=$(find .env/lib/python3.12/site-packages/pyoperon \
#   -name 'pyoperon*.so' -print -quit)

# brew install zlib zstd

# if otool -L "$SO" | grep -q '@rpath/libz.1.dylib'; then
#   install_name_tool \
#     -change '@rpath/libz.1.dylib' '/usr/lib/libz.1.dylib' \
#     "$SO"
# fi

# if otool -L "$SO" | grep -q '@rpath/libc++.1.dylib'; then
#   install_name_tool \
#     -change '@rpath/libc++.1.dylib' '/usr/lib/libc++.1.dylib' \
#     "$SO"
# fi

# codesign --force --sign - "$SO"

# python -c "from pyoperon.sklearn import SymbolicRegressor; print('Operon OK')"

# A cached environment can have a matching requirements stamp while still
# missing runtime/test tools. Verify core imports explicitly.
if ! python - <<'PYDEPS' >/dev/null 2>&1
import importlib
for name in ("pytest", "yaml", "pandas", "matplotlib", "scipy", "sklearn", "torch", "ucimlrepo", "pmlb"):
    importlib.import_module(name)
PYDEPS
then
  echo "[setup] environment is missing required runtime/test packages; repairing"
  python -m pip install -r "$ROOT_DIR/requirements.txt" -r "$ROOT_DIR/benchmarks/requirements-benchmark.txt"
fi

# Validate only optional external baselines selected by this profile. In
# particular, importing pyoperon.sklearn catches broken native-library wheels
# before hundreds of jobs are launched.
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
if ! check_optional_baselines >/dev/null 2>&1; then
  echo "[setup] selected external baseline dependency is missing/broken; repairing"
  python -m pip install -r "$ROOT_DIR/benchmarks/requirements-benchmark.txt"
  check_optional_baselines
fi

mkdir -p "$RUN_DIR"

# Always refresh aggregate outputs on exit, including interrupted runs.
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
