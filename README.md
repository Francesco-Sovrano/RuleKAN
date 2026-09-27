# RuleKAN: Addressing the Symbolic Expressivity Gap in KANs for Symbolic Regression

RuleKAN is a symbolic-regression method for recovering analytic structure that can be hidden by edge-wise symbolic extraction in Kolmogorov-Arnold Networks (KANs). A flexible KAN edge can approximate several analytic factors of the same variable as one univariate function, while multivariate interactions, fuzzy gate-and-branch structure, whole-expression powers or reciprocals, and nested composition require symbolic structure beyond an independent edge replacement.

The implementation separates numerical interaction discovery from symbolic factorization. **RuleKAN** trains a KAN-derived sum-product numerical model to identify variable supports, then searches explicit sums of products of analytic factors inside those supports. **RuleSISP** (Structure-Independent Symbolic Pursuit) searches the same symbolic language without restricting candidate variable multisets to learned supports. Optional bounded extensions add integer powers, reciprocals and ratios of complete symbolic bases, and one additional analytic composition level.

A flat symbolic model has the form

\[
\hat y(x)=b+\sum_{r=1}^{R}a_r\prod_{s=1}^{m_r}
\psi_{k_{rs}}(\beta_{rs}x_{j_{rs}}+\gamma_{rs}),
\qquad m_r\le q,
\]

where each factor uses one analytic primitive \(\psi_k\), an affine input chart, and one input variable. Repeated occurrences of the same variable are allowed during symbolic search. This permits structures such as \(e^x\sin x\) even when the numerical stage represented the full product with one learned univariate function. A two-branch fuzzy rule, \((1-g)u+gv\), is also a sum of products and can be represented with explicit gate and branch roles.

## Method

RuleKAN uses three bounded stages.

**Stage 1 - numerical interaction discovery.** `SumProductKAN` fits an overcomplete sum of numerical product terms. Sparse rule and factor gates are hardened and pruned. The symbolic stage receives variable supports, not the shapes of the learned spline or RBF edge functions.

**Stage 2 - symbolic sum-product search.** Each retained support is expanded into admissible variable multisets through maximum factor order `q`, including repeated variables and multiple additive terms with the same support. Candidate factors are selected from an analytic vocabulary, fitted as `g(beta*x + gamma)`, discretized, refitted, and selected in the context of the complete additive model. RuleSISP replaces learned-support conditioning with enumeration of all variable multisets through order `q`.

**Stage 3 - bounded scope and depth extensions.** `PowerRuleKAN` applies signed integer powers to complete Stage-2 bases, which represents reciprocals and ratios. `RuleKAN-Comp`, `RuleSISP-Comp`, and `PowerRuleKAN-Comp` permit one additional analytic composition around selected symbolic bases.

The principal implemented variants are:

| Model identifier | Numerical support source | Symbolic family |
|---|---|---|
| `rulekan` | spline `SumProductKAN` | support-conditioned sums of products |
| `rulekan_fast` | Gaussian-RBF `SumProductKAN` | same Stage-2 language as `rulekan` |
| `rulekan_adaptive` | retained and validation-selected Stage-1 support evidence | support-conditioned sums of products |
| `rulekan_omp_full` | learned supports | RuleKAN with OMP-style Stage-2 term selection |
| `sisp` | no learned-support restriction | all variable multisets through order `q` |
| `rulekan_comp` | learned supports | RuleKAN plus one composition level |
| `sisp_comp` | no learned-support restriction | SISP plus one composition level |
| `power_rulekan` | effective RuleKAN support bank | signed integer powers, reciprocals, and ratios of symbolic bases |
| `power_rulekan_comp` | effective RuleKAN support bank | powered/ratio grammar with composed bases |

## Installation

Python 3.12 is the default environment used by `setup.sh`.

Core environment:

```bash
INSTALL_BENCHMARK_DEPS=0 ./setup.sh
source .env/bin/activate
python -m pytest -q
```

Full benchmark environment:

```bash
./setup.sh
source .env/bin/activate
python -m pytest -q
```

The full setup installs directly compatible benchmark packages and handles source-based dependencies used by selected baselines. The official Symbolic-KAN source is pinned to commit `9481a82`. PyOperon is built from source when required, and RILS-ROLS is installed without build isolation. uDSR/DSO is not installed into the Python-3.12 environment because its upstream dependency constraints require a separate compatible environment.

## Examples

Run examples from the repository root:

```bash
python -m examples.example_sum_product_kan
python -m examples.example_rulemask_product
python -m examples.example_simple --help
python -m examples.example_feynman --help
```

The lower-level Python API is described in [`docs/reference/python-api.md`](docs/reference/python-api.md).

## Analytic evaluation configuration

The 30-task analytic evaluation uses six suites defined in `benchmarks/specs.py`:

| Suite | Tasks | Capacity being tested |
|---|---:|---|
| `synthetic_core` | 6 | same-variable and cross-variable products, repeated factors, multi-term mixtures |
| `fuzzy_rules` | 7 | gate-and-branch factorization, shared variables, nested and product-valued branches |
| `power_expression_stress` | 4 | whole-expression squares, reciprocals, and ratios |
| `nested_stress` | 5 | one additional analytic composition level |
| `kan_canonical` | 3 | canonical KAN function-fitting targets |
| `feynman` | 5 | compact physics-style equations |

Each analytic task uses seeds `0,1,2` and `1600/400/500` train/validation/test observations. The controlled RuleKAN/KAN vocabulary is `core10` / `target_core`:

```text
x, x^2, 1/x, 1/x^2, sqrt, log, exp, sin, cos, tanh
```

The maximum factor order is `q=3`, the shared numerical width is `W=12`, and the research timeout is 2400 seconds per task/model/seed job. Compound shortcuts are excluded from `core10`; scope and composition must therefore be represented by the explicit Stage-3 mechanisms when needed.

The 23-method main comparison is the `research_modern` profile's `main_comparison_models` evaluated on the six analytic suites above. An exact command is:

```bash
python -m benchmarks.run_benchmark \
  --config benchmarks/configs/default.yaml \
  --profile research_modern \
  --suites fuzzy_rules,synthetic_core,power_expression_stress,nested_stress,kan_canonical,feynman \
  --models rulekan,rulekan_comp,rulekan_adaptive,rulekan_fast,rulekan_omp_full,sisp,sisp_comp,power_rulekan,power_rulekan_comp,autosym,fastkan_autosym,gsr,fastkan_gsr,gmp,srkan,symbolic_kan,rils_rols,sindy,parfam,eql,pysr,operon,multkan_deep_gsr \
  --seeds 0,1,2 \
  --run-dir benchmark_results/analytic_30
```

`research_modern` without overrides schedules 32 tasks because it also includes two small real-data tasks, and it schedules 25 models because `pse` and `anfis` are available outside the 23-method main comparison. The profile named `paper` is a separate 19-task, 7-method KAN-extraction configuration using the `paper25` vocabulary.

Common benchmark profiles resolve to:

| Profile | Seeds | Tasks | Scheduled methods | Timeout/job |
|---|---:|---:|---:|---:|
| `quick` | 1 | 3 | 10 | 900 s |
| `standard` | 3 | 28 | 12 | 2400 s |
| `research` / `main` | 3 | 32 | 19 | 2400 s |
| `research_modern` | 3 | 32 | 25 | 2400 s |
| `fuzzy` | 3 | 7 | 13 | 2400 s |
| `ablation` | 3 | 14 | 14 | 2400 s |
| `width_sensitivity` | 3 | 8 | 12 | 5400 s |
| `paper` | 3 | 19 | 7 | 5400 s |
| `full` | 5 | 32 | 17 | 5400 s |

Inspect the registered profiles, suites, and model identifiers with:

```bash
python -m benchmarks.run_benchmark --profile research --list
```

Run a profile through the shell entry point:

```bash
./run_rulekan_benchmark.sh research
./run_rulekan_benchmark.sh fuzzy
./run_rulekan_benchmark.sh ablation
```

CPU is the default device. Device and worker overrides are accepted by the runner:

```bash
./run_rulekan_benchmark.sh research --device cpu --workers 8
./run_rulekan_benchmark.sh research --device cuda
./run_rulekan_benchmark.sh research --device mps
```

Benchmark output is created under `benchmark_results/` by default. A run directory contains per-condition JSON records, logs, aggregate CSV files, `summary.md`, `latest_results.md`, status diagnostics, and generated PDF figures. Aggregate outputs can be regenerated without retraining:

```bash
python -m benchmarks.aggregate --run-dir benchmark_results/current
```

Completed records are reusable during resume. Use `--no-reuse-completed` when completed jobs must be recomputed under the active source/configuration fingerprint.

## Repository layout

```text
symbolic_kan/                 RuleKAN, RuleSISP, power/composition models, symbolic utilities
benchmarks/                   task definitions, model adapters, configuration, runner, aggregation
benchmarks/configs/           benchmark profiles and shared controls
tools/                        analysis, diagnostics, and sensitivity commands
examples/                     executable examples
tests/                        unit and regression tests
docs/                         method, benchmark, configuration, and API reference
run_rulekan_benchmark.sh      benchmark shell entry point
setup.sh                      environment bootstrap
requirements.txt              core Python dependencies
```

Runtime outputs such as `.env/`, `external/`, and `benchmark_results/` are created locally and are not required to be present in a source archive.

## Documentation

- [`docs/README.md`](docs/README.md): method overview, terminology, and reference index
- [`docs/getting-started.md`](docs/getting-started.md): installation and execution
- [`docs/theory/model-class.md`](docs/theory/model-class.md): Stage-1, Stage-2, and Stage-3 model classes
- [`docs/algorithms/numerical-training.md`](docs/algorithms/numerical-training.md): numerical interaction discovery and pruning
- [`docs/algorithms/symbolic-search.md`](docs/algorithms/symbolic-search.md): support expansion, GMP, hard proposals, GSR/OMP, and SISP
- [`docs/algorithms/power-rulekan.md`](docs/algorithms/power-rulekan.md): powers, reciprocals, and ratios
- [`docs/algorithms/composition-rescue.md`](docs/algorithms/composition-rescue.md): bounded composition
- [`docs/benchmarks/protocol.md`](docs/benchmarks/protocol.md): splits, controlled settings, profiles, execution, and reproducibility
- [`docs/reference/configuration.md`](docs/reference/configuration.md): configuration schema
- [`docs/reference/python-api.md`](docs/reference/python-api.md): Python API
