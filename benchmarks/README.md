# Benchmark environment

The benchmark suite uses a separate Python environment from the installable RuleKAN package.

Create the default Python 3.12 benchmark environment from the repository root:

```bash
./benchmarks/setup.sh
source .env-baselines/bin/activate
```

The setup installs RuleKAN in editable mode, the directly installable dependencies in `benchmarks/requirements-benchmark.txt`, and the external baselines that require special installation steps.

The following environment variables control the source-based baseline installations:

```bash
INSTALL_OPERON=0 ./benchmarks/setup.sh
OPERON_MACOS_FIX=1 ./benchmarks/setup.sh
INSTALL_RILS_ROLS=0 ./benchmarks/setup.sh
INSTALL_SYMBOLIC_KAN=0 ./benchmarks/setup.sh
INSTALL_PYSR=0 ./benchmarks/setup.sh
```

PySR `2.2.1` is installed by default, but setup verifies only the installed Python distribution and does not import PySR. Importing or running PySR later invokes JuliaCall's Julia discovery and may provision Julia if no suitable runtime is available. Set `INSTALL_PYSR=0` to omit PySR.

`VENV_DIR` changes the benchmark environment directory and `PYTHON_BIN` changes the Python executable. The default environment is `.env-baselines` and the default interpreter is `python3.12`.

The setup uses these fixed integration points:

- Symbolic-KAN: `sfaroughi3/Pub_Symbolic_KANs`, commit `9481a82`.
- PyOperon: setup attempts the binary wheel `pyoperon==0.6.1`. If the wheel cannot be installed or imported, setup prints a warning and continues without Operon. On macOS, `OPERON_MACOS_FIX=1` enables an optional repair that adds Homebrew zlib/zstd runtime paths, points libc++ at the macOS system library, and ad-hoc re-signs the modified extension.
- RILS-ROLS: installed with `--no-build-isolation` after `pybind11` is available.
- SR-KAN: installed from the authors' repository through `requirements-benchmark.txt`.

uDSR/DSO is not installed in this environment because its upstream dependency constraints require an older Python/NumPy/Numba stack.

Run a configured benchmark after activating the environment:

```bash
./run_rulekan_benchmark.sh quick
./run_rulekan_benchmark.sh research_modern
```

`run_rulekan_benchmark.sh` uses `.env-baselines` by default. Set `ENV_DIR` when using another benchmark environment.
