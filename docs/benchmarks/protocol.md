# Benchmark protocol

## Configuration and profile resolution

`benchmarks/configs/default.yaml` is the executable benchmark configuration. A profile may inherit from another profile with

```yaml
extends: parent_profile
```

Nested mappings are merged recursively. Scalar values replace parent values. Lists such as `models`, `tasks`, `suites`, `width_values`, and `library_values` replace the corresponding parent list.

`benchmarks.config_utils.resolve_profile()` performs this merge before execution. `benchmarks.run_benchmark` expands the resolved profile into task/model/seed jobs and launches each job in a separate subprocess.

## Data splits

Synthetic regression and fuzzy tasks are generated independently for each seed from the ranges declared in `benchmarks/specs.py`. Inputs are standardized using the training-set mean and standard deviation. Synthetic regression targets remain in their native scale.

Real and tabular tasks use a seeded train/temporary split followed by a seeded validation/test split. Input standardization is fitted on the training split. Regression targets are standardized with training-target statistics; binary targets are not standardized.

The split roles are:

- **training:** numerical fitting and continuous symbolic parameter fitting;
- **validation:** checkpoint selection, pruning decisions, symbolic candidate selection, adaptive support rescue, composition rescue, powered-expression selection, reciprocal-domain admissibility, and validation-gated cleanup;
- **test:** final predictive and structural evaluation after model selection is complete.

Some internal KAN APIs call their held-out tensors `test_input` and `test_label`. `BenchmarkData.dataset_dict()` maps the benchmark validation split to those internal names. The benchmark test split is not passed to those training APIs.

## Shared capacity and symbolic controls

Profiles may enable `shared_settings` to align comparable controls across method families.

`W` denotes total hidden/rule width, not product arity. Under the shared-capacity resolver:

- RuleKAN-family models receive `n_rules = W`;
- shallow MultKAN divides `W` between additive and multiplication units;
- deep MultKAN preserves total width `W` in each configured hidden layer;
- vanilla KAN receives hidden width `W`;
- ANFIS uses `W` fuzzy rules when the shared capacity applies;
- product arity remains task-specific through `TaskSpec.max_factors` when `max_product_order: task` is active.

Shared settings can also align grid resolution, symbolic vocabulary, hard-screening controls, and minimum symbolic rule budget.

The `research` profile uses:

```yaml
capacity:
  policy: fixed
  width: 12
  mult_units: 4
max_product_order: task
grid: 12
symbolic_library: target_core
symbolic_hybrid_hard_screening: true
symbolic_residual_structure_topk: 1
symbolic_joint_scale_refit: true
symbolic_rule_budget_at_least_width: true
```

`target_core` is the 14-operator benchmark library documented in [Symbolic library](../reference/symbolic-library.md).

## Principal profiles

The task counts below are the resolved number of tasks selected by each profile.

| Profile | Seeds | Tasks | Methods | Timeout/job | Purpose |
|---|---:|---:|---:|---:|---|
| `quick` | 1 | 3 | 10 | 900 s | small execution smoke test |
| `standard` | 3 | 28 | 12 | 2400 s | fuzzy, synthetic, nested, canonical KAN, Feynman-style, and small real-data benchmark |
| `research` | 3 | 32 | 19 | 2400 s | principal benchmark; adds powered-expression tasks and RuleKAN/SISP/PowerRuleKAN variants |
| `main` | 3 | 32 | 19 | 2400 s | alias of `research` through profile inheritance |
| `fuzzy` | 3 | 7 | 13 | 2400 s | complete fuzzy-rule suite |
| `paper` | 3 | 19 | 7 | 5400 s | reference KAN symbolic-extraction comparison using the 25-operator library and its own capacity settings |
| `ablation` | 3 | 14 | 14 | 2400 s | RuleKAN component ablation on a representative research-task subset |
| `width_sensitivity` | 3 | 8 | 12 | 5400 s | shared-width sweep over `W={6,8,10,12,16,24}` |
| `power_expression_quick` | 3 | 4 | 3 | 900 s | focused powered-expression comparison |
| `full` | 5 | 32 | 17 | 5400 s | broad benchmark including UCI binary tasks |

The resolved `research` method list is:

```text
rulekan
rulekan_comp
rulekan_adaptive
rulekan_fast
rulekan_omp_full
sisp
sisp_comp
sisp_fast
power_rulekan
power_rulekan_comp
autosym
fastkan_autosym
gsr
fastkan_gsr
gmp
pysr
operon
anfis
multkan_deep_gsr
```

Additional profiles are defined for exhaustive method matrices and controlled analyses:

| Profile | Resolved scope |
|---|---|
| `research_exhaustive` | 32 research tasks, 24 methods |
| `fuzzy_exhaustive` | 7 fuzzy tasks, 19 methods |
| `deep_multkan_quick`, `deep_multkan` | shallow/deep multiplication comparisons |
| `ablation_quick` | one-seed reduced-budget ablation smoke test |
| `width_sensitivity_quick` | one-seed reduced-budget width sweep over `W={6,8,10,12}` |
| `library_sensitivity_quick` | symbolic vocabulary sweep over `core14`, `medium20`, and `research26` |
| `controlled_compare`, `controlled_compare_quick` | matched-capacity comparisons |
| `omp_ablation`, `fuzzy_omp_ablation` | RuleKAN pursuit/extractor variants |
| `deep_ablation` | shallow/deep MultKAN symbolic pipelines |
| `logic_compressed`, `fuzzy_logic_compressed` | numerical logic-compression variants |
| `composition_ablation_quick`, `composition_ablation` | flat versus depth-2 symbolic composition variants |
| `baseline_repair` | external `operon` and `anfis` rows on the research task/data configuration |

Use

```bash
python -m benchmarks.run_benchmark --list --profile research
```

to print registered suites, models, and profiles.

## Research and ablation comparability

`ablation` extends `research` and overrides only `description`, `models`, `suites`, and `tasks`. After profile resolution, the following are exactly equal between the two profiles:

- seeds;
- data split sizes and real-data sample cap;
- per-job timeout;
- `model_config.rulekan`;
- `model_config.rulekan_fast`;
- all inherited RuleKAN pursuit-variant configurations;
- shared capacity, grid, symbolic library, hard-screening controls, residual-structure control, joint scale refit, and symbolic rule-budget policy.

The `rulekan` row in `ablation` is therefore the same configured RuleKAN treatment as `rulekan` in `research`; the ablation profile changes the task subset and includes knockout model identifiers. The knockout adapters receive the inherited RuleKAN configuration and override only the mechanism named by the ablation identifier.

The core factorial is:

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

Additional ablations isolate one-shot pruning, symbolic backfitting, repeated-variable symbolic products, Gaussian-RBF numerical factors, OMP-linear pursuit, and graph redundancy controls.

## Research and width-sensitivity comparability

`width_sensitivity` is not an exact `research` reproduction with only `W` changed. It shares the research seeds, synthetic data sizes, grid, task-dependent product order, and `target_core` symbolic vocabulary, but it uses a separate sensitivity schedule.

The principal differences from `research` are:

| Setting | `research` | `width_sensitivity` |
|---|---:|---:|
| timeout/job | 2400 s | 5400 s |
| RuleKAN `stage_scale` | 0.24 | 0.12 |
| RuleKAN `plateau_rounds` | 4 | 2 |
| RuleKAN `numeric_lbfgs_steps` | 40 | default 0 |
| RuleKAN rank continuation | enabled, caps 3 and 6 | not enabled by the profile |
| shared hybrid hard screening | enabled | not enabled by the profile |
| shared residual structure top-k | 1 | not enabled by the profile |
| shared joint scale refit | enabled | not enabled by the profile |
| width control | fixed `W=12`, `mult_units=4` | sweep `W={6,8,10,12,16,24}`, multiplication units derived from `mult_fraction=0.375`, minimum 2 |

The `W=12` sensitivity condition should therefore be interpreted inside the width-sensitivity experiment, not as the research baseline row.

## Execution and process isolation

Each scheduled task/model/seed condition runs in its own subprocess. CPU is the default device. CPU workers use approximately 50% of available logical CPUs unless `--workers` or `--cpu-fraction` overrides the policy. Each CPU subprocess uses one PyTorch/BLAS thread by default. CUDA and MPS default to one benchmark subprocess at a time.

Timeouts terminate the complete experiment process group. Runtime exceptions are written to result JSON with traceback. With `--keep-going`, remaining jobs continue after failures.

## Resume and provenance

Each result record contains:

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

Resume reuses a completed JSON only when its stored build fingerprint matches the active source/configuration fingerprint. `--reuse-completed` explicitly permits reuse after a source or configuration change. `--verbose-resume` prints per-job resume decisions. `--show-build-fingerprint` prints the build fingerprint that is always retained in the result JSON.

A run begins with a `running` JSON. If the parent process or machine stops before terminal status is written, that record remains `running` and is reported as incomplete.

## Aggregation and status files

`benchmarks/aggregate.py` regenerates aggregate CSV files, `summary.md`, and PDF figures. RuleKAN-family numerical and symbolic metrics are normalized into stage-consistent aggregate columns.

A run directory contains status diagnostics including:

```text
STATUS.txt
incomplete_runs.csv
failure_summary.csv
skipped_incompatible_jobs.csv
skipped_incompatible_jobs.json
```

For an existing run directory:

```bash
python tools/explain_benchmark_status.py benchmark_results/current
```

prints non-completed task/model/seed records with their status, error, elapsed time, result path, and log path.

## Optional external symbolic-regression dependencies

`benchmarks/requirements-benchmark.txt` installs the external benchmark dependencies, including PySR, Operon, and the official SR-KAN implementation. The pinned evolutionary baselines are:

```text
pysr==2.2.1
pyoperon==0.6.1
```

SR-KAN is installed from the authors' GitHub repository; `benchmarks/requirements-srkan.txt` provides the narrow SR-KAN-only install. The unrelated PyPI package named `srkan` does not provide the API expected by the benchmark adapter.

Selected external baselines are checked before job construction. Operon is import-tested through `pyoperon.sklearn`; SR-KAN is checked for the expected `regressor` and `SympyEvaluator` API; PySR installation is checked without importing the Julia bridge during preflight. Operator grammars and compute budgets are profile-controlled.

## Commands

Run the complete research profile:

```bash
./run_rulekan_benchmark.sh research --no-resume
```

Run the full ablation profile:

```bash
./run_rulekan_benchmark.sh ablation --no-resume
```

The ablation matrix, output files, resume behavior, and standalone OFAT command are specified in [Ablations](ablations.md).

Run selected research methods:

```bash
python -m benchmarks.run_benchmark \
  --profile research \
  --models rulekan,rulekan_adaptive,power_rulekan,sisp \
  --seeds 0,1,2 \
  --run-dir benchmark_results/research_rulekan_family
```

Run one suite:

```bash
python -m benchmarks.run_benchmark \
  --profile fuzzy \
  --suites fuzzy_rules \
  --run-dir benchmark_results/fuzzy
```
