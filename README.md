# RuleKAN

RuleKAN is a symbolic-regression system built around sparse sums of multiplicative rules. A numerical `SumProductKAN` learns which variables participate together, and symbolic search replaces the numerical factors with analytic univariate functions while preserving the learned support structure.

A RuleKAN predictor has the form

\[
\hat y(x)=b+\sum_{r=1}^{R} a_r\prod_{j=1}^{m_r} g_{rj}(x_{v_{rj}}),
\]

where each rule is a product of univariate factors. The implementation also includes adaptive support recovery, depth-2 composition, structure-independent symbolic pursuit, and `PowerRuleKAN` models that apply integer powers and reciprocals to complete symbolic RuleKAN bases.

## Repository layout

```text
symbolic_kan/                 model implementations and symbolic utilities
benchmarks/                   benchmark tasks, model adapters, configuration, runner, aggregation
benchmarks/configs/           benchmark profiles and shared settings
tools/                        standalone analysis and diagnostic commands
examples/                     executable examples and experimental scripts
tests/                        unit and regression tests
docs/                         model, algorithm, benchmark, and API documentation
benchmark_results/current/    main preserved benchmark run
benchmark_results/archive/    additional preserved run directories
run_rulekan_benchmark.sh      benchmark entry point
setup.sh                      core environment bootstrap
requirements.txt              core Python dependencies
```

Generated benchmark files are kept under `benchmark_results/`; they are not expected in the repository root.

## Installation

Python 3.12 is the default used by `setup.sh`.

```bash
./setup.sh
source .env/bin/activate
python -m pytest -q
```

For the full benchmark environment, `setup.sh` installs the normal Python
packages and also reconstructs the gitignored third-party source checkouts under
`external/`: official Symbolic-KAN is pinned to commit `9481a82`, and PyOperon
is cloned/built from source with the macOS runtime-path repair used by this
project. RILS-ROLS is installed separately with build isolation disabled.
These third-party directories should not be committed to RuleKAN.

`benchmarks/requirements-benchmark.txt` contains only dependencies that are safe
to install directly into the main Python-3.12 virtual environment. It does
**not** by itself install RILS-ROLS, PyOperon, or Symbolic-KAN; use `./setup.sh`
for the complete benchmark setup.

The official SR-KAN implementation can also be installed on its own with:

```bash
python -m pip install -r benchmarks/requirements-srkan.txt
```

`benchmarks/requirements-modern-sr.txt` is likewise only the directly
pip-installable subset for the additional modern baselines. uDSR/DSO is
intentionally not installed into the main Python-3.12 environment because its
upstream package pins legacy NumPy/Numba versions; run it from a separate
compatible environment if that baseline is required.

## Examples

Run examples from the repository root as modules:

```bash
python -m examples.example_sum_product_kan
python -m examples.example_rulemask_product
python -m examples.example_simple --help
python -m examples.example_feynman --help
```

## Benchmark profiles

The executable benchmark definition is `benchmarks/configs/default.yaml`. The shell entry point accepts a profile name followed by benchmark-runner options.

```bash
./run_rulekan_benchmark.sh quick
./run_rulekan_benchmark.sh research
./run_rulekan_benchmark.sh research_modern
./run_rulekan_benchmark.sh ablation
./run_rulekan_benchmark.sh fuzzy
./run_rulekan_benchmark.sh long_expression
./run_rulekan_benchmark.sh width_sensitivity
```

The main profiles include:

| Profile | Seeds | Tasks | Methods | Purpose |
|---|---:|---:|---:|---|
| `quick` | 1 | 3 | 10 | smoke run |
| `standard` | 3 | 28 | 12 | broad capability benchmark |
| `research` | 3 | 32 | 19 | primary research matrix |
| `research_modern` | 3 | 32 | 23 | research matrix + Symbolic-KAN, PSE, RILS-ROLS, uDSR |
| `ablation` | 3 | 14 | 14 | RuleKAN component ablation |
| `fuzzy` | 3 | 7 | 13 | fuzzy-rule benchmark |
| `long_expression` | 3 | 2 | 19 | expression-length stress test |
| `width_sensitivity` | 3 | 8 | 12 | shared-capacity width sweep |

CPU is the default device. Examples:

```bash
./run_rulekan_benchmark.sh research --device cpu --workers 8
./run_rulekan_benchmark.sh research --device cuda
./run_rulekan_benchmark.sh research --device mps
```

Direct Python execution supports task, method, and seed overrides:

```bash
python -m benchmarks.run_benchmark \
  --config benchmarks/configs/default.yaml \
  --profile research \
  --models rulekan,rulekan_adaptive,power_rulekan,sisp \
  --tasks mixed_rank4,power_ratio_product \
  --seeds 0,1,2 \
  --run-dir benchmark_results/example
```

List the registered profiles, suites, and methods with:

```bash
python -m benchmarks.run_benchmark --profile research --list
```

## Ablations

There are two separate ablation entry points.

The configured component ablation uses the benchmark harness and the `ablation` profile:

```bash
./run_rulekan_benchmark.sh ablation
```

It schedules 14 RuleKAN variants on 14 tasks for seeds `0,1,2`. It inherits the `research` data sizes, timeout, shared width, grid, symbolic library, and RuleKAN baseline configuration.

The standalone one-factor-at-a-time Feynman ablation is:

```bash
python -m tools.ablation_ofat --help
```

It requires local Feynman text datasets under `symbolic_kan/datasets/` by default and writes CSV output under `benchmark_results/ofat_ablation/`. Full commands, dataset layout, parameter grids, and output columns are documented in [`docs/benchmarks/ablations.md`](docs/benchmarks/ablations.md).

## Existing benchmark results

The default run directory is:

```text
benchmark_results/current/
```

The preserved run can be inspected without executing experiments:

```bash
cat benchmark_results/current/STATUS.txt
python tools/explain_benchmark_status.py benchmark_results/current
```

Aggregate tables and figures can be regenerated from the existing per-condition JSON records without retraining models:

```bash
python -m benchmarks.aggregate --run-dir benchmark_results/current
```

The benchmark runner normally reuses a completed condition only when its stored build fingerprint matches the active source and configuration. To allow completed records from a different fingerprint to be reused, pass `--reuse-completed`:

```bash
./run_rulekan_benchmark.sh ablation --reuse-completed
```

This option affects completed records only; conditions that are absent or incomplete can still be scheduled.

A run directory can contain:

```text
runs/                         one JSON record per scheduled condition
logs/                         stdout/stderr per condition
figures/                      generated PDF figures
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

## Documentation

Start with [`docs/getting-started.md`](docs/getting-started.md). The complete documentation index is in [`docs/README.md`](docs/README.md).
