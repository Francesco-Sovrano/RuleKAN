#!/usr/bin/env bash
set -euo pipefail

# Create the benchmark environment and install external baseline dependencies.
#
# This environment is separate from the package-only environment created by
# ../setup.sh. The default benchmark interpreter is Python 3.12.  A few symbolic-regression packages
# need special handling on macOS:
#   * RILS-ROLS must be built with pybind11 visible (no PEP-517 isolation).
#   * PyOperon 0.6.1 is attempted from its binary wheel. A failed PyOperon
#     install/import is reported but does not abort setup. On macOS, the
#     published wheel can optionally be repaired with OPERON_MACOS_FIX=1.
#   * uDSR/DSO is intentionally NOT installed into this Python-3.12 venv; its
#     upstream package pins legacy NumPy/Numba versions.  Keep it in a separate
#     Python-3.9 venv if needed.
#
# Useful overrides:
#   PYTHON_BIN=python3.12 ./benchmarks/setup.sh
#   VENV_DIR=/path/to/venv ./benchmarks/setup.sh
#   INSTALL_OPERON=0 ./benchmarks/setup.sh
#   OPERON_MACOS_FIX=1 ./benchmarks/setup.sh
#   INSTALL_RILS_ROLS=0 ./benchmarks/setup.sh
#   INSTALL_SYMBOLIC_KAN=0 ./benchmarks/setup.sh
#   INSTALL_PYSR=0 ./benchmarks/setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.env-baselines}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
INSTALL_OPERON="${INSTALL_OPERON:-1}"
INSTALL_RILS_ROLS="${INSTALL_RILS_ROLS:-1}"
INSTALL_SYMBOLIC_KAN="${INSTALL_SYMBOLIC_KAN:-1}"
INSTALL_PYSR="${INSTALL_PYSR:-1}"
OPERON_MACOS_FIX="${OPERON_MACOS_FIX:-0}"
PYOPERON_VERSION="${PYOPERON_VERSION:-0.6.1}"
OPERON_AVAILABLE=0
SYMBOLIC_KAN_REPO="${SYMBOLIC_KAN_REPO:-https://github.com/sfaroughi3/Pub_Symbolic_KANs.git}"
SYMBOLIC_KAN_COMMIT="${SYMBOLIC_KAN_COMMIT:-9481a82}"
SYMBOLIC_KAN_DIR="${SYMBOLIC_KAN_DIR:-$ROOT_DIR/external/Pub_Symbolic_KANs}"

log() { printf '\n==> %s\n' "$*"; }
warn() { echo "Warning: $*" >&2; }
fail() { echo "Error: $*" >&2; exit 1; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "'$PYTHON_BIN' was not found. Install Python 3.12 or set PYTHON_BIN."
fi

[[ -f "$ROOT_DIR/pyproject.toml" ]] || fail "pyproject.toml not found in $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading Python build tooling"
python -m pip install --upgrade pip setuptools wheel

log "Installing RuleKAN and benchmark test support"
python -m pip install -e "$ROOT_DIR[dev]"

# Compiler/toolchain sanity check.  The source-built baselines require a C/C++
# compiler.  On macOS this is provided by the Xcode Command Line Tools.
if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! xcrun --find clang >/dev/null 2>&1; then
    fail "Apple Command Line Tools are missing. Run 'xcode-select --install', then rerun ./benchmarks/setup.sh."
  fi
fi

BENCH_REQ="$ROOT_DIR/benchmarks/requirements-benchmark.txt"
if [[ -f "$BENCH_REQ" ]]; then
  log "Installing benchmark dependencies that are safe in the main Python 3.12 venv"
  python -m pip install -r "$BENCH_REQ"
else
  log "Installing standard benchmark dependencies"
  python -m pip install \
    ucimlrepo pytest pmlb \
    psrn pybind11 \
    'scikit-build-core>=0.11.6' 'nanobind>=2.11.0' cmake ninja \
    'git+https://github.com/marcobuhler/SR-KAN.git'
fi

if [[ "$INSTALL_PYSR" == "1" ]]; then
  log "Installing PySR 2.2.1"
  # Avoid importing PySR during setup: JuliaCall performs Julia discovery when
  # PySR is initialized and may provision Julia if no suitable runtime is found.
  python -m pip install "pysr==2.2.1"
else
  log "Skipping PySR (INSTALL_PYSR=0)"
fi

install_symbolic_kan() {
  log "Installing/checking official Symbolic-KAN source"
  command -v git >/dev/null 2>&1 || fail "git is required to install Symbolic-KAN"

  mkdir -p "$ROOT_DIR/external"

  if [[ -d "$SYMBOLIC_KAN_DIR" && ! -d "$SYMBOLIC_KAN_DIR/.git" ]]; then
    # Older local bundles sometimes contained an unversioned source snapshot.
    # Do not silently trust it for a reproducible benchmark: replace it with
    # the pinned upstream checkout used by the adapter.
    log "Replacing unversioned Symbolic-KAN snapshot with pinned upstream checkout"
    rm -rf "$SYMBOLIC_KAN_DIR"
  fi

  if [[ ! -d "$SYMBOLIC_KAN_DIR/.git" ]]; then
    git clone "$SYMBOLIC_KAN_REPO" "$SYMBOLIC_KAN_DIR"
  else
    # Keep an existing clone reproducible even if the user previously checked
    # out another branch/commit.
    if ! git -C "$SYMBOLIC_KAN_DIR" remote get-url origin >/dev/null 2>&1; then
      git -C "$SYMBOLIC_KAN_DIR" remote add origin "$SYMBOLIC_KAN_REPO"
    fi
    git -C "$SYMBOLIC_KAN_DIR" fetch --quiet origin --tags
  fi

  # Resolve the exact commit used by the benchmark integration. If the local
  # clone does not yet contain it, fetch that object explicitly.
  if ! git -C "$SYMBOLIC_KAN_DIR" cat-file -e "${SYMBOLIC_KAN_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$SYMBOLIC_KAN_DIR" fetch --quiet origin "$SYMBOLIC_KAN_COMMIT"
  fi
  git -C "$SYMBOLIC_KAN_DIR" checkout --quiet --detach "$SYMBOLIC_KAN_COMMIT"

  local required="$SYMBOLIC_KAN_DIR/Exp_reaction_diffusion/symKanTraining.py"
  [[ -f "$required" ]] || fail "Official Symbolic-KAN checkout is incomplete: missing $required"

  local actual expected
  actual="$(git -C "$SYMBOLIC_KAN_DIR" rev-parse HEAD)"
  expected="$(git -C "$SYMBOLIC_KAN_DIR" rev-parse "${SYMBOLIC_KAN_COMMIT}^{commit}")"
  [[ "$actual" == "$expected" ]] || \
    fail "Symbolic-KAN checkout mismatch: expected $expected, got $actual"

  printf '%s\n' "$actual" > "$SYMBOLIC_KAN_DIR/UPSTREAM_COMMIT.txt"
  echo "Symbolic-KAN: OK (${actual:0:12})"
}

install_rils_rols() {
  if (cd "$ROOT_DIR" && python -c 'import rils_rols' >/dev/null 2>&1); then
    log "RILS-ROLS is already importable; skipping rebuild"
    return 0
  fi

  log "Installing RILS-ROLS without build isolation"
  # Its build imports pybind11 while preparing the extension, so pybind11 must
  # already be visible in this environment.
  python -m pip install --upgrade pybind11
  python -m pip install --no-build-isolation rils-rols

  python - <<'PY'
import rils_rols
print("RILS-ROLS OK:", rils_rols.__file__)
PY
}

repair_pyoperon_macos() {
  [[ "$(uname -s)" == "Darwin" ]] || {
    warn "OPERON_MACOS_FIX=1 has no effect outside macOS."
    return 1
  }

  local tool
  for tool in otool install_name_tool codesign; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      warn "$tool is missing; install the Apple Command Line Tools to repair PyOperon."
      return 1
    fi
  done

  if ! command -v brew >/dev/null 2>&1; then
    warn "Homebrew is required for the optional PyOperon macOS repair (zlib/zstd)."
    return 1
  fi

  local so
  if ! so="$(python - <<'PY'
import sysconfig
from pathlib import Path
root = Path(sysconfig.get_paths()["purelib"]) / "pyoperon"
hits = list(root.glob("pyoperon*.so"))
if len(hits) != 1:
    raise SystemExit(f"Expected exactly one PyOperon extension, found: {hits}")
print(hits[0])
PY
)"; then
    warn "Could not locate the installed PyOperon extension for repair."
    return 1
  fi

  local changed=0
  local zlib_rpath zstd_rpath

  if otool -L "$so" | grep -Fq '@rpath/libz.1.dylib'; then
    if ! brew list zlib >/dev/null 2>&1 && ! brew install zlib; then
      warn "Could not install Homebrew zlib required by the PyOperon wheel."
      return 1
    fi
    zlib_rpath="$(brew --prefix zlib)/lib"
    if ! otool -l "$so" | grep -Fq "$zlib_rpath"; then
      log "Adding PyOperon zlib rpath: $zlib_rpath"
      if ! install_name_tool -add_rpath "$zlib_rpath" "$so"; then
        warn "Could not add the zlib rpath to PyOperon."
        return 1
      fi
      changed=1
    fi
  fi

  if otool -L "$so" | grep -Fq '@rpath/libzstd.1.dylib'; then
    if ! brew list zstd >/dev/null 2>&1 && ! brew install zstd; then
      warn "Could not install Homebrew zstd required by the PyOperon wheel."
      return 1
    fi
    zstd_rpath="$(brew --prefix zstd)/lib"
    if ! otool -l "$so" | grep -Fq "$zstd_rpath"; then
      log "Adding PyOperon zstd rpath: $zstd_rpath"
      if ! install_name_tool -add_rpath "$zstd_rpath" "$so"; then
        warn "Could not add the zstd rpath to PyOperon."
        return 1
      fi
      changed=1
    fi
  fi

  if otool -L "$so" | grep -Fq '@rpath/libc++.1.dylib'; then
    log "Pointing PyOperon at the macOS system libc++"
    if ! install_name_tool \
      -change '@rpath/libc++.1.dylib' \
      '/usr/lib/libc++.1.dylib' \
      "$so"; then
      warn "Could not rewrite the PyOperon libc++ dependency."
      return 1
    fi
    changed=1
  fi

  if [[ "$changed" == "1" ]]; then
    log "Re-signing modified PyOperon extension"
    if ! codesign --force --sign - "$so"; then
      warn "Could not ad-hoc sign the modified PyOperon extension."
      return 1
    fi
  fi

  if ! codesign --verify --strict "$so"; then
    warn "The repaired PyOperon extension failed code-signature verification."
    return 1
  fi
}

pyoperon_importable() {
  (
    cd "$ROOT_DIR"
    python - <<'PY' >/dev/null 2>&1
import pyoperon
from pyoperon.sklearn import SymbolicRegressor
PY
  )
}

report_pyoperon_failure() {
  warn "PyOperon ${PYOPERON_VERSION} is unavailable; continuing without the Operon baseline."
  if [[ "$(uname -s)" == "Darwin" && "$OPERON_MACOS_FIX" != "1" ]]; then
    warn "The macOS wheel may have unresolved native-library rpaths. Rerun with OPERON_MACOS_FIX=1 to apply the optional Homebrew/rpath/code-signing repair."
  fi
  warn "You can also disable the attempt explicitly with INSTALL_OPERON=0."

  if [[ "$(uname -s)" == "Darwin" ]]; then
    local so
    so="$(find "$VENV_DIR" -path '*/site-packages/pyoperon/pyoperon*.so' -print -quit 2>/dev/null || true)"
    if [[ -n "$so" ]]; then
      echo "PyOperon linked libraries:" >&2
      otool -L "$so" >&2 || true
    fi
  fi
}

install_pyoperon() {
  if pyoperon_importable; then
    log "PyOperon is already importable; skipping installation"
    OPERON_AVAILABLE=1
    return 0
  fi

  log "Attempting PyOperon ${PYOPERON_VERSION} binary-wheel installation"
  if ! python -m pip install --only-binary=:all: "pyoperon==${PYOPERON_VERSION}"; then
    report_pyoperon_failure
    return 0
  fi

  if pyoperon_importable; then
    OPERON_AVAILABLE=1
    echo "PyOperon: OK (${PYOPERON_VERSION})"
    return 0
  fi

  if [[ "$(uname -s)" == "Darwin" && "$OPERON_MACOS_FIX" == "1" ]]; then
    log "Applying optional macOS PyOperon runtime-linkage repair"
    if repair_pyoperon_macos && pyoperon_importable; then
      OPERON_AVAILABLE=1
      echo "PyOperon: OK after macOS repair (${PYOPERON_VERSION})"
      return 0
    fi
  fi

  report_pyoperon_failure
  return 0
}

if [[ "$INSTALL_SYMBOLIC_KAN" == "1" ]]; then
  install_symbolic_kan
fi

if [[ "$INSTALL_RILS_ROLS" == "1" ]]; then
  install_rils_rols
fi

if [[ "$INSTALL_OPERON" == "1" ]]; then
  install_pyoperon
fi

log "Checking modern symbolic-regression imports"
python - <<'PY'
for name in ["torch", "psrn"]:
    mod = __import__(name)
    print(f"{name}: OK ({getattr(mod, '__file__', 'built-in')})")
PY

if [[ "$INSTALL_RILS_ROLS" == "1" ]]; then
  python - <<'PY'
import rils_rols
print("rils_rols: OK", rils_rols.__file__)
PY
fi

if [[ "$INSTALL_PYSR" == "1" ]]; then
  python - <<'PY'
from importlib.metadata import version
print("pysr: installed", version("pysr"), "(not imported during setup)")
PY
else
  echo "PySR installation skipped (INSTALL_PYSR=0)."
fi

if [[ "$INSTALL_OPERON" == "1" ]]; then
  if [[ "$OPERON_AVAILABLE" == "1" ]]; then
    echo "pyoperon: OK"
  else
    echo "pyoperon: unavailable; benchmark setup continued without Operon"
  fi
else
  echo "PyOperon installation skipped (INSTALL_OPERON=0)."
fi

# Symbolic-KAN is an external pinned source checkout, not a pip package.
if [[ "$INSTALL_SYMBOLIC_KAN" == "1" ]]; then
  [[ -f "$SYMBOLIC_KAN_DIR/Exp_reaction_diffusion/symKanTraining.py" ]] || \
    fail "Symbolic-KAN source disappeared after installation: $SYMBOLIC_KAN_DIR"
  echo "symbolic_kan source: OK ($SYMBOLIC_KAN_DIR)"
else
  echo "Symbolic-KAN installation skipped (INSTALL_SYMBOLIC_KAN=0)."
fi

echo
echo "Benchmark setup complete."
echo "Activate the environment with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "Run a benchmark with:"
echo "  cd \"$ROOT_DIR\" && ./run_rulekan_benchmark.sh quick"
