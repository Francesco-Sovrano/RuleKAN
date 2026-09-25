#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the project virtual environment and install the benchmark stack.
#
# The default environment is Python 3.12.  A few symbolic-regression packages
# need special handling on macOS:
#   * RILS-ROLS must be built with pybind11 visible (no PEP-517 isolation).
#   * PyOperon wheels can contain unusable macOS rpaths, so we build v0.6.1
#     from source and add the venv's lib directory to the extension rpath.
#   * uDSR/DSO is intentionally NOT installed into this Python-3.12 venv; its
#     upstream package pins legacy NumPy/Numba versions.  Keep it in a separate
#     Python-3.9 venv if needed.
#
# Useful overrides:
#   PYTHON_BIN=python3.12 ./setup.sh
#   VENV_DIR=/path/to/venv ./setup.sh
#   INSTALL_BENCHMARK_DEPS=0 ./setup.sh       # core project only
#   INSTALL_OPERON=0 ./setup.sh               # skip source-built PyOperon
#   INSTALL_RILS_ROLS=0 ./setup.sh            # skip RILS-ROLS
#   INSTALL_SYMBOLIC_KAN=0 ./setup.sh          # skip official Symbolic-KAN checkout

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
INSTALL_BENCHMARK_DEPS="${INSTALL_BENCHMARK_DEPS:-1}"
INSTALL_OPERON="${INSTALL_OPERON:-1}"
INSTALL_RILS_ROLS="${INSTALL_RILS_ROLS:-1}"
INSTALL_SYMBOLIC_KAN="${INSTALL_SYMBOLIC_KAN:-1}"
PYOPERON_TAG="${PYOPERON_TAG:-v0.6.1}"
PYOPERON_DIR="${PYOPERON_DIR:-$ROOT_DIR/external/pyoperon}"
SYMBOLIC_KAN_REPO="${SYMBOLIC_KAN_REPO:-https://github.com/sfaroughi3/Pub_Symbolic_KANs.git}"
SYMBOLIC_KAN_COMMIT="${SYMBOLIC_KAN_COMMIT:-9481a82}"
SYMBOLIC_KAN_DIR="${SYMBOLIC_KAN_DIR:-$ROOT_DIR/external/Pub_Symbolic_KANs}"

log() { printf '\n==> %s\n' "$*"; }
fail() { echo "Error: $*" >&2; exit 1; }

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "'$PYTHON_BIN' was not found. Install Python 3.12 or set PYTHON_BIN."
fi

[[ -f "$ROOT_DIR/requirements.txt" ]] || fail "requirements.txt not found in $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading Python build tooling"
python -m pip install --upgrade pip setuptools wheel

log "Installing RuleKAN core dependencies"
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ "$INSTALL_BENCHMARK_DEPS" != "1" ]]; then
  echo
  echo "Core setup complete (benchmark dependencies skipped)."
  echo "Activate with: source \"$VENV_DIR/bin/activate\""
  exit 0
fi

# Compiler/toolchain sanity check.  The source-built baselines require a C/C++
# compiler.  On macOS this is provided by the Xcode Command Line Tools.
if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! xcrun --find clang >/dev/null 2>&1; then
    fail "Apple Command Line Tools are missing. Run 'xcode-select --install', then rerun setup.sh."
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
    'pysr==2.2.1' psrn pybind11 \
    'scikit-build-core>=0.11.6' 'nanobind>=2.11.0' cmake ninja \
    'git+https://github.com/marcobuhler/SR-KAN.git'
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

patch_pyoperon_dependency_probe() {
  local script="$PYOPERON_DIR/script/dependencies.py"
  [[ -f "$script" ]] || fail "PyOperon dependency helper not found: $script"

  # Upstream's Conda-oriented helper invokes `cmake-package-check`.  A plain
  # Python venv does not provide that executable.  Falling back to False is
  # correct here: it simply tells the helper to build/install the dependency.
  python - "$script" <<'PY'
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
s = p.read_text()

if "import shutil" not in s:
    if "import subprocess" in s:
        s = s.replace("import subprocess", "import subprocess\nimport shutil", 1)
    else:
        s = "import shutil\n" + s

if 'shutil.which("cmake-package-check")' not in s:
    pattern = r"(def check_installed\(name\):\n)"
    replacement = (
        r"\1"
        '    if shutil.which("cmake-package-check") is None:\n'
        '        return False\n'
    )
    s2, n = re.subn(pattern, replacement, s, count=1)
    if n != 1:
        raise RuntimeError("Could not locate check_installed(name) in dependencies.py")
    s = s2

p.write_text(s)
print("patched", p)
PY
}

patch_pyoperon_macos_rpath() {
  [[ "$(uname -s)" == "Darwin" ]] || return 0

  command -v install_name_tool >/dev/null 2>&1 || \
    fail "install_name_tool is missing; install the Apple Command Line Tools."

  local so
  so="$(python - <<'PY'
from pathlib import Path
import site

hits = []
for base in site.getsitepackages():
    hits.extend(Path(base).glob("pyoperon/pyoperon*.so"))
print(hits[0] if hits else "")
PY
)"
  [[ -n "$so" && -f "$so" ]] || fail "Could not locate the installed pyoperon extension"

  if ! otool -l "$so" | grep -Fq "$VENV_DIR/lib"; then
    log "Adding $VENV_DIR/lib to PyOperon's macOS runtime search path"
    install_name_tool -add_rpath "$VENV_DIR/lib" "$so"
  fi

  # If Homebrew is available, add its general lib directory as a secondary
  # search path.  The source-built dependencies should normally resolve from
  # $VENV_DIR/lib, so this is only a fallback.
  if command -v brew >/dev/null 2>&1; then
    local brew_lib
    brew_lib="$(brew --prefix)/lib"
    if [[ -d "$brew_lib" ]] && ! otool -l "$so" | grep -Fq "$brew_lib"; then
      install_name_tool -add_rpath "$brew_lib" "$so" || true
    fi
  fi
}

install_pyoperon() {
  if (
    cd "$ROOT_DIR"
    python - <<'PY' >/dev/null 2>&1
import pyoperon
from pyoperon.sklearn import SymbolicRegressor
PY
  ); then
    log "PyOperon is already importable; skipping rebuild"
    return 0
  fi

  log "Installing PyOperon $PYOPERON_TAG from source"
  command -v git >/dev/null 2>&1 || fail "git is required to install PyOperon"

  # scikit-build-core/nanobind are the upstream build backend requirements;
  # cmake+ninja make the build independent of system CMake packaging.
  python -m pip install --upgrade \
    'scikit-build-core>=0.11.6' 'nanobind>=2.11.0' cmake ninja

  mkdir -p "$(dirname "$PYOPERON_DIR")"
  if [[ -d "$PYOPERON_DIR/.git" ]]; then
    git -C "$PYOPERON_DIR" fetch --tags --quiet
    git -C "$PYOPERON_DIR" checkout --quiet "$PYOPERON_TAG"
  else
    rm -rf "$PYOPERON_DIR"
    git clone --depth 1 --branch "$PYOPERON_TAG" \
      https://github.com/heal-research/pyoperon.git "$PYOPERON_DIR"
  fi

  patch_pyoperon_dependency_probe

  (
    cd "$PYOPERON_DIR"
    export CC="${CC:-clang}"
    export CXX="${CXX:-clang++}"
    # Upstream builds its C++ dependencies into the active environment.
    python script/dependencies.py
    python -m pip install . --no-build-isolation
  )

  patch_pyoperon_macos_rpath

  # Run the import test from the RuleKAN root so the source checkout does not
  # shadow the installed extension package.
  if ! (
    cd "$ROOT_DIR"
    python - <<'PY'
import pyoperon
from pyoperon.sklearn import SymbolicRegressor
print("PyOperon OK:", pyoperon.__file__)
PY
  ); then
    echo "PyOperon import failed after installation." >&2
    if [[ "$(uname -s)" == "Darwin" ]]; then
      local so
      so="$(find "$VENV_DIR" -path '*/site-packages/pyoperon/pyoperon*.so' -print -quit)"
      if [[ -n "$so" ]]; then
        echo "Linked libraries for $so:" >&2
        otool -L "$so" >&2 || true
      fi
    fi
    return 1
  fi
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

if [[ "$INSTALL_OPERON" == "1" ]]; then
  (
    cd "$ROOT_DIR"
    python - <<'PY'
import pyoperon
from pyoperon.sklearn import SymbolicRegressor
print("pyoperon: OK", pyoperon.__file__)
PY
  )
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
echo "Setup complete."
echo "Activate the environment with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "Run the Sum-Product KAN example with:"
echo "  cd \"$ROOT_DIR\" && python -m examples.example_sum_product_kan"
echo "Run the older RuleMask example with:"
echo "  cd \"$ROOT_DIR\" && python -m examples.example_rulemask_product"
