# Source layout

## Top-level paths

| Path | Responsibility |
|---|---|
| `rulekan/` | RuleKAN, RuleSISP, PowerRuleKAN, composition models, KAN/MultKAN layers, and symbolic utilities |
| `benchmarks/` | task definitions, model adapters, configuration profiles, execution, and aggregation |
| `tools/` | diagnostics, sensitivity utilities, smoke tests, and OFAT ablation |
| `examples/` | executable examples |
| `tests/` | unit and regression tests |
| `docs/` | method, benchmark, configuration, and API reference |
| `run_rulekan_benchmark.sh` | benchmark shell entry point |
| `setup.sh` | RuleKAN package/development environment |
| `benchmarks/setup.sh` | Python-3.12 benchmark environment and external baseline setup |
| `pyproject.toml` | installable package metadata and core runtime dependencies |
| `requirements.txt` | convenience editable development install; runtime dependencies are declared in `pyproject.toml` |

`.env/`, `.env-baselines/`, `external/`, and `benchmark_results/` are runtime directories created locally when needed; they are not required to exist in a source checkout.

## Model code

| Path | Responsibility |
|---|---|
| `rulekan/sum_product_kan.py` | `SumProductKAN`, training stages, pruning, support evidence, Stage-2 GMP/GSR, SISP structure search, affine partitions, and formula export |
| `rulekan/power_rulekan.py` | PowerRuleKAN structures, signed integer powers, reciprocals, ratios, and polishing |
| `rulekan/composition_rulekan.py` | bounded depth-2 symbolic composition structures and fitting |
| `rulekan/MultKAN.py` | MultKAN implementation and KAN-based symbolic-extraction controls |
| `rulekan/KANLayer.py` | spline KAN edge layer |
| `rulekan/gated_kan.py` | gated KAN mechanisms used by baseline paths |
| `rulekan/rule_mask.py` | rule-mask product mechanisms used by examples and RuleKAN utilities |
| `rulekan/utils.py` | symbolic libraries and shared utilities |

`rulekan/` is the implementation and public Python package namespace.

## Benchmark code

| Path | Responsibility |
|---|---|
| `benchmarks/configs/default.yaml` | profile definitions, per-model settings, shared capacity, and symbolic-library selection |
| `benchmarks/specs.py` | task formulas, domains, suite assignments, annotations, data loading, and splitting |
| `benchmarks/models.py` | model adapters, RuleKAN/RuleSISP/PowerRuleKAN pipelines, external baselines, and metrics |
| `benchmarks/run_one.py` | one task/model/seed execution and result serialization |
| `benchmarks/run_benchmark.py` | matrix construction, dependency checks, resume policy, subprocess execution, and timeout handling |
| `benchmarks/aggregate.py` | aggregate tables, statistical summaries, Markdown summaries, and PDF figures |
| `benchmarks/build_info.py` | source version and benchmark build fingerprint |
| `benchmarks/config_utils.py` | profile inheritance and configuration resolution |
| `benchmarks/device_utils.py` | device selection and CPU worker/thread policy |
| `benchmarks/anfis.py` | ANFIS baseline |
| `benchmarks/eql_baseline.py` | EQL-Div baseline |
| `benchmarks/sindy_baseline.py` | SINDy-12 and uncapped SINDy conditions |
| `benchmarks/symbolic_kan_baseline.py` | official Symbolic-KAN adapter |
| `benchmarks/vocabulary_audit.py` | cross-method symbolic-vocabulary audit |

## Entry points and tools

| Path | Responsibility |
|---|---|
| `run_rulekan_benchmark.sh` | profile execution, environment checks, tests, and aggregation hook |
| `setup.sh` | package/development virtual environment |
| `benchmarks/setup.sh` | Python-3.12 benchmark virtual environment and external baseline dependency setup |
| `tools/ablation_ofat.py` | one-factor-at-a-time Feynman ablation |
| `tools/explain_benchmark_status.py` | incomplete and failed benchmark-record report |
| `tools/redundancy_report.py` | redundancy analysis for a run directory |
| `tools/grid_sensitivity_rulekan.py` | RuleKAN grid sensitivity |
| `tools/benchmark_native_rational_smoke.py` | native rational benchmark smoke test |
| `tools/compare_rational_denominator_target.py` | rational-denominator comparison utility |
| `tools/smoke_native_rational.py` | native rational smoke test |

Package entry points are run from the repository root:

```bash
python -m benchmarks.run_benchmark --profile quick
python -m tools.ablation_ofat --help
python -m examples.example_sum_product_kan
```

## Result layout

The default run directory is `benchmark_results/current/`. It is created by `run_rulekan_benchmark.sh` or `benchmarks.run_benchmark`. Per-condition JSON files are written under `runs/`, process logs under `logs/`, and aggregate figures under `figures/`. Aggregate CSV and Markdown files are written directly in the run directory.
