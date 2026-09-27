# Getting started

## Requirements

RuleKAN requires Python 3.10 or later. The package bootstrap uses `python3` by default; the benchmark bootstrap uses Python 3.12 by default.

Install the library from a source checkout with:

```bash
python -m pip install .
```

For development, use an editable install with the test dependency:

```bash
python -m pip install -e ".[dev]"
```

The installed distribution is named `rulekan`, and the preferred public import namespace is also `rulekan`:

```python
from rulekan import SumProductKAN, PowerRuleKAN
```


For a local package/development environment:

```bash
./setup.sh
source .env/bin/activate
```

`setup.sh` installs RuleKAN in editable mode with the development extra. Set `INSTALL_DEV=0` for runtime dependencies only, or `INSTALL_EXAMPLES=1` to install the example extra.

For the benchmark baselines, use the separate Python 3.12 environment:

```bash
./benchmarks/setup.sh
source .env-baselines/bin/activate
```

`benchmarks/setup.sh` installs RuleKAN in editable mode and the benchmark dependency stack. Official Symbolic-KAN is pinned to commit `9481a82`; RILS-ROLS is installed without build isolation; and PySR `2.2.1` is installed without importing it during setup. PyOperon `0.6.1` is attempted from its binary wheel, but a failed install or import is reported without aborting the rest of setup. On macOS, rerun with `OPERON_MACOS_FIX=1` to apply the optional Homebrew/rpath/code-signing repair. `INSTALL_OPERON=0`, `INSTALL_RILS_ROLS=0`, `INSTALL_SYMBOLIC_KAN=0`, and `INSTALL_PYSR=0` skip the corresponding components.

The package named `srkan` on PyPI is not the dependency used by the benchmark. To install only the official SR-KAN implementation expected by the adapter:

```bash
python -m pip install -r benchmarks/requirements-srkan.txt
```

uDSR/DSO is not installed into the main Python-3.12 benchmark environment because its upstream dependency constraints require older NumPy/Numba versions. Use a separate compatible environment for model identifier `udsr`.

## Verify the checkout

Run the test suite from the repository root:

```bash
python -m pytest -q
```

The `run_rulekan_benchmark.sh` wrapper uses `.env-baselines` by default and executes the tests automatically unless `SKIP_TESTS=1` is set. Direct `python -m benchmarks.run_benchmark` commands do not run the test suite.

## Run an example

Examples are Python modules under `examples/`:

```bash
python -m examples.example_sum_product_kan
python -m examples.example_rulemask_product
python -m examples.example_feynman --help
```

`examples.example_simple` depends on the external `pykan` package. Install the optional example dependencies before running it:

```bash
python -m pip install -e ".[examples]"
python -m examples.example_simple --help
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

`benchmarks/configs/default.yaml` defines the profiles. `research` uses three seeds, 32 tasks, 19 methods, synthetic train/validation/test sizes of `1600/400/500`, per-job timeout `2400` seconds, shared width `12`, grid `12`, maximum factor order `q=3`, and the ten-operator `target_core` / `core10` symbolic library. The 30-task analytic comparison uses the six analytic suites and the 23 `research_modern.main_comparison_models` listed in [Benchmark protocol](benchmarks/protocol.md).

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
  --feynman_root rulekan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv rulekan/datasets/FeynmanEquations.csv \
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

## Inspect a completed run

For a run directory containing completed or partial records, inspect its status file:

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

The benchmark runner writes a source/configuration build fingerprint into each condition record. With the default `--resume`, completed records are reused even when that fingerprint differs from the active build; the mismatch is reported for provenance.

Relevant options are:

```text
--resume / --no-resume
--reuse-completed / --no-reuse-completed
--verbose-resume / --no-verbose-resume
--show-build-fingerprint / --no-show-build-fingerprint
```

Use `--no-reuse-completed` when a deliberate method/configuration change should force strict same-build reruns:

```bash
./run_rulekan_benchmark.sh research --no-reuse-completed
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
