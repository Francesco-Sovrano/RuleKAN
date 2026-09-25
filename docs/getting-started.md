# Getting started

## Requirements

The repository uses Python and PyTorch. `setup.sh` defaults to Python 3.12 and
creates a virtual environment at `.env`.

For the complete supported benchmark setup:

```bash
./setup.sh
source .env/bin/activate
```

Besides installing Python dependencies, the script reconstructs the gitignored
third-party source checkouts used by the benchmark. It checks out official
Symbolic-KAN at commit `9481a82` under `external/Pub_Symbolic_KANs`, clones and
builds PyOperon under `external/pyoperon`, repairs its macOS runtime search path,
and installs RILS-ROLS with build isolation disabled.

Equivalent core-only installation:

```bash
python3.12 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`benchmarks/requirements-benchmark.txt` and
`benchmarks/requirements-modern-sr.txt` contain the directly pip-installable
subset only. They intentionally omit RILS-ROLS and PyOperon, which need the
special handling in `setup.sh`, and Symbolic-KAN, which is a pinned source
checkout rather than a Python package.

The package named `srkan` on PyPI is not the dependency used by the benchmark.
To install only the official SR-KAN implementation expected by the adapter:

```bash
python -m pip install -r benchmarks/requirements-srkan.txt
```

uDSR/DSO is intentionally not installed into the main Python-3.12 environment:
its upstream package pins legacy NumPy/Numba versions. Use a separate compatible
environment if the uDSR baseline is required.

## Verify the checkout

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The `run_rulekan_benchmark.sh` wrapper executes the tests automatically unless `SKIP_TESTS=1` is set. Direct `python -m benchmarks.run_benchmark` commands do not run the test suite.

## Run an example

Examples are Python modules under `examples/`:

```bash
python -m examples.example_sum_product_kan
python -m examples.example_rulemask_product
```

Command-line examples expose their options with `--help`:

```bash
python -m examples.example_simple --help
python -m examples.example_feynman --help
```

Run module commands from the repository root so local imports and relative data paths resolve consistently.

## Run a benchmark

The shell entry point has the form:

```bash
./run_rulekan_benchmark.sh PROFILE [runner options]
```

Examples:

```bash
./run_rulekan_benchmark.sh quick
./run_rulekan_benchmark.sh standard
./run_rulekan_benchmark.sh research
./run_rulekan_benchmark.sh fuzzy
./run_rulekan_benchmark.sh long_expression
./run_rulekan_benchmark.sh width_sensitivity
```

`benchmarks/configs/default.yaml` defines the profiles. `research` uses three seeds, 32 tasks, 19 methods, synthetic train/validation/test sizes of `1600/400/500`, per-job timeout `2400` seconds, shared width `12`, grid `12`, and the `target_core` symbolic library.

The runner can also be called directly:

```bash
python -m benchmarks.run_benchmark \
  --config benchmarks/configs/default.yaml \
  --profile research \
  --models rulekan,rulekan_adaptive,power_rulekan,sisp \
  --tasks mixed_rank4,power_ratio_product \
  --seeds 0,1,2 \
  --run-dir benchmark_results/example
```

Display the registered profile, suite, and method names with:

```bash
python -m benchmarks.run_benchmark --profile research --list
```

## Run the component ablation

The benchmark component ablation is defined by the `ablation` profile:

```bash
./run_rulekan_benchmark.sh ablation
```

It uses seeds `0,1,2`, 14 tasks, and 14 RuleKAN variants. The profile extends `research`, so the RuleKAN control inherits the research data sizes, timeout, shared capacity, grid, symbolic library, and model settings.

To write the run to a different directory:

```bash
RESULT_DIR="$PWD/benchmark_results/ablation_run" \
  ./run_rulekan_benchmark.sh ablation
```

To select only part of the matrix:

```bash
python -m benchmarks.run_benchmark \
  --profile ablation \
  --models rulekan,rulekan_no_product,rulekan_no_pruning,rulekan_no_gmp \
  --tasks same_var_exp_sin,mixed_rank4 \
  --seeds 0 \
  --run-dir benchmark_results/ablation_subset
```

The complete ablation matrix and the standalone Feynman OFAT script are documented in [Ablations](benchmarks/ablations.md).

## Run the standalone Feynman OFAT ablation

The separate OFAT command varies hidden width, regularization, pruning iterations, and seed around a fixed baseline:

```bash
python -m tools.ablation_ofat --help
```

A typical invocation is:

```bash
python -m tools.ablation_ofat \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --max_datasets 10 \
  --dataset_select_seed 123 \
  --device cpu \
  --output_csv benchmark_results/ofat_ablation/results.csv
```

The default OFAT grid contains 15 configurations and five symbolic-regression methods, producing 75 method-runs per selected dataset.

## Devices and CPU parallelism

CPU is the default. When `--workers` is omitted, CPU execution uses approximately half of the available logical CPUs as independent experiment workers. Each CPU experiment uses one PyTorch/BLAS thread by default.

```bash
./run_rulekan_benchmark.sh research --device cpu --workers 8 --cpu-threads-per-job 1
./run_rulekan_benchmark.sh research --device cuda
./run_rulekan_benchmark.sh research --device mps
```

The standalone OFAT tool accepts the same device names and its own `--workers`, `--cpu_fraction`, and `--cpu_threads_per_job` controls.

## Result directories

The benchmark runner defaults to:

```text
benchmark_results/current/
```

A run directory contains per-condition records and generated aggregates:

```text
runs/                         JSON record for each scheduled task/model/seed condition
logs/                         stdout/stderr for each condition
figures/                      aggregate PDF figures
benchmark_config_snapshot.yaml
runs.csv
summary.csv
summary.md
symbolic_runs.csv
symbolic_summary.csv
latest_results.md
STATUS.txt
incomplete_runs.csv
failure_summary.csv
skipped_incompatible_jobs.csv
skipped_incompatible_jobs.json
```

Additional files are generated when the selected profile contains fuzzy, ablation, width-sensitivity, library-sensitivity, redundancy, or statistical-comparison data.

## Inspect existing results without running experiments

Read the status file:

```bash
cat benchmark_results/current/STATUS.txt
```

Print incomplete and failed conditions:

```bash
python tools/explain_benchmark_status.py benchmark_results/current
```

Regenerate aggregate CSV, Markdown, and PDF outputs from the existing JSON records:

```bash
python -m benchmarks.aggregate --run-dir benchmark_results/current
```

`benchmarks.aggregate` reads existing result records. It does not train models or schedule benchmark jobs.

## Resume and reuse

The benchmark runner writes a source/configuration build fingerprint into each condition record. With the default `--resume`, a completed record is reused only when its stored fingerprint matches the active build.

Relevant options are:

```text
--resume / --no-resume
--reuse-completed / --no-reuse-completed
--verbose-resume / --no-verbose-resume
--show-build-fingerprint / --no-show-build-fingerprint
```

Use `--reuse-completed` when completed records should remain reusable even if the active source/configuration fingerprint differs:

```bash
./run_rulekan_benchmark.sh research --reuse-completed
```

Missing, failed, or otherwise non-reusable conditions can still be scheduled. `--no-resume` forces the selected matrix to execute again.

## Configuration

The benchmark configuration is `benchmarks/configs/default.yaml`. Profile inheritance is resolved by `benchmarks/config_utils.py`.

The main configuration groups are:

```text
profiles.<name>.seeds
profiles.<name>.models
profiles.<name>.tasks or profiles.<name>.suites
profiles.<name>.data
profiles.<name>.timeout_seconds
profiles.<name>.model_config
profiles.<name>.shared_settings
```

Configuration field references are under [`docs/reference/`](reference/configuration.md).
