# Source layout

## Top-level directories

| Path | Responsibility |
|---|---|
| `symbolic_kan/` | RuleKAN, PowerRuleKAN, MultKAN, numerical layers, symbolic utilities |
| `benchmarks/` | benchmark tasks, model adapters, profiles, orchestration, aggregation |
| `tools/` | standalone diagnostics, smoke tests, reports, and OFAT ablation |
| `examples/` | executable examples and experimental scripts |
| `tests/` | unit and regression tests |
| `docs/` | model, algorithm, benchmark, and API documentation |
| `benchmark_results/current/` | main preserved benchmark run |
| `benchmark_results/archive/` | additional preserved run directories |

## Model code

| Path | Responsibility |
|---|---|
| `symbolic_kan/sum_product_kan.py` | SumProductKAN numerical model, training stages, pruning, support evidence, GMP/GSR, SISP grammar, affine-partition recovery, symbolic formula export |
| `symbolic_kan/power_rulekan.py` | PowerRuleKAN data structures, integer powers, reciprocals, ratios, polishing |
| `symbolic_kan/composition_rulekan.py` | depth-2 symbolic composition structures |
| `symbolic_kan/MultKAN.py` | MultKAN implementation and symbolic-extraction baselines |
| `symbolic_kan/KANLayer.py` | spline KAN edge layer |
| `symbolic_kan/gated_kan.py` | gated symbolic/numerical mechanisms used by KAN baselines |
| `symbolic_kan/rule_mask.py` | rule-mask product mechanisms used by example and legacy paths |
| `symbolic_kan/utils.py` | symbolic library and shared utilities |

The public package imports are re-exported from `symbolic_kan/__init__.py`.

## Benchmark code

| Path | Responsibility |
|---|---|
| `benchmarks/configs/default.yaml` | profile definitions, method settings, shared capacity and symbolic libraries |
| `benchmarks/specs.py` | task definitions, formulas, annotations, data loading and splitting |
| `benchmarks/models.py` | method adapters, RuleKAN training pipeline, PowerRuleKAN search, metrics |
| `benchmarks/run_one.py` | one task/model/seed execution and JSON provenance |
| `benchmarks/run_benchmark.py` | matrix construction, dependency checks, resume, subprocess execution, timeout handling |
| `benchmarks/aggregate.py` | aggregate tables, statistical summaries, Markdown summaries, PDF figures |
| `benchmarks/build_info.py` | code version and benchmark build fingerprint |
| `benchmarks/config_utils.py` | profile inheritance and configuration resolution |
| `benchmarks/device_utils.py` | device selection and CPU worker/thread policy |
| `benchmarks/task_catalog.csv` | machine-readable task catalog |

## Entry points and tools

| Path | Responsibility |
|---|---|
| `run_rulekan_benchmark.sh` | benchmark environment setup and profile execution |
| `setup.sh` | core virtual-environment setup |
| `tools/ablation_ofat.py` | standalone one-factor-at-a-time Feynman ablation |
| `tools/explain_benchmark_status.py` | report incomplete and failed benchmark records |
| `tools/redundancy_report.py` | method redundancy analysis from an existing run directory |
| `tools/grid_sensitivity_rulekan.py` | grid-sensitivity utility |
| `tools/benchmark_native_rational_smoke.py` | native rational benchmark smoke test |
| `tools/compare_rational_denominator_target.py` | rational-denominator comparison utility |
| `tools/smoke_native_rational.py` | native rational smoke utility |

Run package-based entry points from the repository root, for example:

```bash
python -m benchmarks.run_benchmark --profile quick
python -m tools.ablation_ofat --help
python -m examples.example_sum_product_kan
```

## Result layout

`benchmark_results/current/` follows the run-directory contract used by `benchmarks.run_benchmark` and `benchmarks.aggregate`. Per-condition JSON files live in `runs/`, process logs in `logs/`, and aggregate figures in `figures/`. Aggregate CSV and Markdown files are stored directly in the run directory.
