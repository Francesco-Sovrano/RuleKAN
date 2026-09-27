# Benchmark protocol

## Configuration and profile resolution

`benchmarks/configs/default.yaml` is the executable benchmark configuration. Profiles may inherit from another profile with `extends`. Nested mappings are merged recursively; scalars and lists in the child profile replace the parent value.

`benchmarks.config_utils.resolve_profile()` resolves inheritance before execution. `benchmarks.run_benchmark` expands a resolved profile into task/model/seed jobs and launches each job in a separate subprocess.

Registered suites, models, and profiles are available with:

```bash
python -m benchmarks.run_benchmark --profile research --list
```

## Data splits

Synthetic regression and fuzzy tasks are generated independently for each seed from the domains in `benchmarks/specs.py`. Inputs are standardized from training-set statistics. Synthetic regression targets retain their native scale.

Real and tabular tasks use seeded train/temporary and validation/test splits. Input standardization is fitted only on the training split. Regression targets are standardized from training-target statistics; binary targets are not standardized.

Split roles are fixed:

- **training:** numerical fitting and continuous symbolic parameter fitting;
- **validation:** checkpoint selection, pruning, symbolic candidate selection, support augmentation, composition selection, power/ratio selection, reciprocal-domain admissibility, and cleanup;
- **test:** final predictive and structural evaluation after model selection.

Some internal KAN APIs use `test_input` and `test_label` names for held-out tensors. `BenchmarkData.dataset_dict()` maps the benchmark validation split to those internal fields. The benchmark test split is not passed to those training APIs.

## Shared capacity and symbolic controls

`W` denotes the shared numerical width, while `q` denotes maximum product order. Under shared capacity:

- RuleKAN-family models receive `n_rules = W`;
- shallow MultKAN divides `W` between additive and multiplication units;
- deep MultKAN preserves total hidden width `W` in each configured hidden layer;
- vanilla KAN receives hidden width `W`;
- ANFIS uses `W` fuzzy rules when shared capacity is enabled.

The `research` family of profiles uses:

```yaml
capacity:
  policy: fixed
  width: 12
  mult_units: 4
max_product_order: 3
grid: 12
symbolic_library: target_core
symbolic_hybrid_hard_screening: true
symbolic_residual_structure_topk: 1
symbolic_joint_scale_refit: true
symbolic_rule_budget_at_least_width: true
```

`target_core` is an alias of the ten-primitive `core10` vocabulary:

```text
x, x^2, 1/x, 1/x^2, sqrt, log, exp, sin, cos, tanh
```

Each RuleKAN factor applies one primitive to an affine argument. Compound shortcuts are excluded from this controlled vocabulary.

## Thirty-task analytic comparison

The controlled analytic comparison uses six suites and 30 noise-free tasks:

| Suite | Count |
|---|---:|
| `synthetic_core` | 6 |
| `fuzzy_rules` | 7 |
| `power_expression_stress` | 4 |
| `nested_stress` | 5 |
| `kan_canonical` | 3 |
| `feynman` | 5 |
| **Total** | **30** |

The analytic protocol uses seeds `0,1,2`, `1600/400/500` train/validation/test observations, `W=12`, grid `12`, `q=3`, `core10`, and a 2400-second per-job limit.

The 23 main comparison model identifiers are:

```text
rulekan
rulekan_comp
rulekan_adaptive
rulekan_fast
rulekan_omp_full
sisp
sisp_comp
power_rulekan
power_rulekan_comp
autosym
fastkan_autosym
gsr
fastkan_gsr
gmp
srkan
symbolic_kan
rils_rols
sindy
parfam
eql
pysr
operon
multkan_deep_gsr
```

These are the `research_modern.main_comparison_models`. The exact 30-task matrix is run by overriding the two extra small real-data tasks and the two additional scheduled models in `research_modern`:

```bash
python -m benchmarks.run_benchmark \
  --config benchmarks/configs/default.yaml \
  --profile research_modern \
  --suites fuzzy_rules,synthetic_core,power_expression_stress,nested_stress,kan_canonical,feynman \
  --models rulekan,rulekan_comp,rulekan_adaptive,rulekan_fast,rulekan_omp_full,sisp,sisp_comp,power_rulekan,power_rulekan_comp,autosym,fastkan_autosym,gsr,fastkan_gsr,gmp,srkan,symbolic_kan,rils_rols,sindy,parfam,eql,pysr,operon,multkan_deep_gsr \
  --seeds 0,1,2 \
  --run-dir benchmark_results/analytic_30
```

`research_modern` without overrides selects 32 tasks and 25 scheduled models. Its `main_comparison_models` field contains the 23 identifiers above. The additional scheduled models are `pse` and `anfis`; the additional tasks are the two `real_small` tasks.

SINDy in the main comparison is `sindy`, which is limited to 12 active non-bias library terms. `sindy_unconstrained` uses the same feature dictionary and optimizer without the final support cap and is a separate capacity-sensitivity condition.

### Metrics and inference

Predictive accuracy is test NRMSE, defined as test RMSE divided by the standard deviation of the test target. Each analytic task is summarized by the median over its three seeds. Failed runs remain in the 30-task predictive protocol through a per-task worst-error penalty before the seed median is computed. Pairwise predictive tests use two-sided Wilcoxon signed-rank tests on paired log-NRMSE ratios across the 30 task medians, with Benjamini-Hochberg false-discovery-rate correction across the 14 external-baseline comparisons.

The seven fuzzy tasks additionally use expanded-rule F1 and exact gate-and-branch recovery. Failed structural runs contribute zero. Exact recovery is evaluated over the 21 task-seed runs with paired exact McNemar tests and the same false-discovery-rate correction. [Metrics and figures](metrics-and-figures.md) specifies the scoring and distinguishes this 30-task protocol from the generic all-pairs Holm diagnostics produced by `benchmarks.aggregate`.

## Profiles

Resolved counts for the principal profiles are:

| Profile | Seeds | Tasks | Scheduled methods | Timeout/job | Scope |
|---|---:|---:|---:|---:|---|
| `quick` | 1 | 3 | 10 | 900 s | smoke test |
| `standard` | 3 | 28 | 12 | 2400 s | fuzzy, product, nested, canonical, physics, and small real-data tasks |
| `research` | 3 | 32 | 19 | 2400 s | RuleKAN/SISP/PowerRuleKAN research matrix plus two small real-data tasks |
| `main` | 3 | 32 | 19 | 2400 s | inherited alias of `research` |
| `research_modern` | 3 | 32 | 25 | 2400 s | research matrix with contemporary baselines; 23 models are marked for main comparison |
| `fuzzy` | 3 | 7 | 13 | 2400 s | fuzzy-rule suite |
| `paper` | 3 | 19 | 7 | 5400 s | KAN extraction subset using `paper25` |
| `ablation` | 3 | 14 | 14 | 2400 s | component ablation |
| `width_sensitivity` | 3 | 8 | 12 | 5400 s | shared-width sweep over `W={6,8,10,12,16,24}` |
| `power_expression_quick` | 3 | 4 | 3 | 900 s | power/ratio subset |
| `full` | 5 | 32 | 17 | 5400 s | broad matrix including UCI binary tasks |

The resolved `research` model list is:

```text
rulekan
rulekan_comp
rulekan_adaptive
rulekan_fast
rulekan_omp_full
sisp
sisp_comp
power_rulekan
power_rulekan_comp
autosym
fastkan_autosym
gsr
fastkan_gsr
gmp
srkan
pysr
operon
anfis
multkan_deep_gsr
```

Additional configured profiles cover deep MultKAN comparisons, reduced-budget smoke runs, width and library sensitivity, pursuit ablations, numerical logic compression, composition ablation, and baseline repair.

## Ablation configuration

`ablation` extends `research` and replaces the selected models and tasks. Seeds, synthetic split sizes, timeout, RuleKAN configurations, shared width, grid, factor order, symbolic library, hard-screening controls, and symbolic rule-budget policy are inherited from `research`.

The `2^3` factorial uses:

```text
rulekan
rulekan_no_product
rulekan_no_pruning
rulekan_no_gmp
rulekan_no_product_no_pruning
rulekan_no_product_no_gmp
rulekan_no_pruning_no_gmp
rulekan_no_product_no_pruning_no_gmp
```

Additional ablation identifiers isolate one-shot pruning, backfitting, repeated-variable symbolic products, Gaussian-RBF numerical factors, OMP-linear pursuit, and graph-redundancy controls.

## Width sensitivity

`width_sensitivity` shares the `research` seeds, synthetic split sizes, grid, `q=3`, and `target_core` vocabulary, but uses a separate optimization schedule and a 5400-second job limit. The `W=12` row in this profile is therefore part of the width-sensitivity configuration rather than an exact duplicate of the `research` row.

Key differences are:

| Setting | `research` | `width_sensitivity` |
|---|---:|---:|
| timeout/job | 2400 s | 5400 s |
| RuleKAN `stage_scale` | 0.24 | 0.12 |
| RuleKAN `plateau_rounds` | 4 | 2 |
| RuleKAN `numeric_lbfgs_steps` | 40 | default 0 |
| rank continuation | enabled | not enabled by the profile |
| hybrid hard screening | enabled | not enabled by the profile |
| residual structure top-k | 1 | not enabled by the profile |
| joint scale refit | enabled | not enabled by the profile |
| width | fixed `W=12` | `W={6,8,10,12,16,24}` |

## Execution and process isolation

Each task/model/seed condition runs in a separate subprocess. CPU is the default device. CPU workers use approximately half of available logical CPUs unless `--workers` or `--cpu-fraction` changes the worker policy. Each CPU subprocess uses one PyTorch/BLAS thread by default. CUDA and MPS use one benchmark subprocess at a time unless explicitly changed.

Timeout handling terminates the experiment process group. Runtime exceptions are written to result JSON with traceback information. `--keep-going` continues with remaining jobs after a failure.

## Resume and provenance

Each result record includes source/configuration provenance fields such as:

```text
code_version
benchmark_build_fingerprint
profile
task
model
seed
host
platform
torch_version
```

Resume is enabled by default. Completed JSON records are reused even when the active build fingerprint differs; the stored fingerprint remains available in the result metadata. Use `--no-reuse-completed` to recompute completed conditions under the active source/configuration state. `--verbose-resume` prints per-job reuse decisions, and `--show-build-fingerprint` prints the active fingerprint.

A scheduled job writes a `running` record before completion. An interrupted job therefore remains visible as incomplete rather than disappearing from the run directory.

## Aggregation and status files

`benchmarks/aggregate.py` generates aggregate CSV files, Markdown summaries, status diagnostics, and PDF figures from per-condition JSON records.

Typical run-directory outputs include:

```text
runs/
logs/
figures/
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

A status summary can be printed with:

```bash
python tools/explain_benchmark_status.py benchmark_results/current
```

## External symbolic-regression dependencies

`benchmarks/requirements-benchmark.txt` contains dependencies that can be installed directly into the Python-3.12 benchmark environment. `benchmarks/setup.sh` creates `.env-baselines` and handles source-based or special-install dependencies used by selected baselines:

- Symbolic-KAN: pinned upstream checkout under `external/Pub_Symbolic_KANs`, commit `9481a82`;
- PyOperon: the setup script attempts the binary wheel `pyoperon==0.6.1`; failure is reported without aborting the remaining benchmark setup. On macOS, `OPERON_MACOS_FIX=1` enables an optional repair of zlib/zstd runtime paths and the system libc++ dependency followed by ad-hoc re-signing;
- RILS-ROLS: installation without build isolation.

SR-KAN is installed from the authors' repository and is checked for the expected `regressor` and `SympyEvaluator` API. The package published on PyPI under the name `srkan` is not the implementation used by the benchmark adapter.

Optional dependency checks are performed only for selected models. Operon is import-tested through `pyoperon.sklearn`; PSE through `psrn`; RILS-ROLS through `rils_rols.rils_rols`; uDSR through `dso`; SINDy through `pysindy`; and ParFam through `parfam`. EQL is implemented in-tree with PyTorch and SymPy.

uDSR is registered as model identifier `udsr` but is not selected by the default `research_modern` profile. Its upstream dependency constraints require a separate compatible environment.

## Commands

Run a configured profile:

```bash
./run_rulekan_benchmark.sh research
./run_rulekan_benchmark.sh ablation
./run_rulekan_benchmark.sh fuzzy
```

Run selected methods and tasks:

```bash
python -m benchmarks.run_benchmark \
  --profile research \
  --models rulekan,rulekan_adaptive,power_rulekan,sisp \
  --tasks mixed_rank4,power_ratio_product \
  --seeds 0,1,2 \
  --run-dir benchmark_results/subset
```

Run one suite:

```bash
python -m benchmarks.run_benchmark \
  --profile fuzzy \
  --suites fuzzy_rules \
  --run-dir benchmark_results/fuzzy
```

## SINDy term budget

`sindy` uses the same `core10` elementary atoms and task-specific interaction order as the controlled comparison, with at most 12 active non-bias terms. For each STLSQ threshold candidate, an over-budget support is ranked by empirical RMS contribution, reduced to the 12 strongest non-bias terms, jointly OLS-refit, and scored on validation data. The bias term does not consume the 12-term budget.

`sindy_unconstrained` uses the same dictionary, threshold grid, data split, and STLSQ procedure without the final support cap. Formula term count and serialized length are recorded for the capacity comparison.
