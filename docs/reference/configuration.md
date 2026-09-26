# Configuration reference

`benchmarks/configs/default.yaml` defines benchmark profiles, model-specific parameters, shared controls, and sensitivity sweeps. The benchmark runner resolves a profile before constructing jobs.

## Profile inheritance

A profile may inherit another profile:

```yaml
child_profile:
  extends: parent_profile
```

Nested mappings are merged recursively. Scalars replace parent scalars. Lists replace parent lists rather than being appended. This applies to `models`, `tasks`, `suites`, `width_values`, and `library_values`.

For example, `ablation` extends `research` and replaces its model/task lists. Because it does not redefine the RuleKAN model configuration or shared settings, the resolved RuleKAN baseline is exactly the research baseline.

## Profile fields

| Field | Meaning |
|---|---|
| `description` | Human-readable profile description. |
| `extends` | Optional parent profile name. |
| `seeds` | Integer random seeds. |
| `models` | Model identifiers scheduled by default. |
| `suites` | Task-suite names used when an explicit `tasks` list is absent. |
| `tasks` | Explicit task identifiers; when present, these determine the task matrix. |
| `data.train_n` | Synthetic training-set size. |
| `data.val_n` | Synthetic validation-set size. |
| `data.test_n` | Synthetic test-set size. |
| `data.max_real_samples` | Maximum retained samples for real-data tasks. |
| `timeout_seconds` | Per-job subprocess timeout. |
| `model_config` | Model-specific configuration mapping. |
| `shared_settings` | Cross-family capacity, grid, product-order, symbolic-library, and symbolic-budget controls. |
| `width_values` | Width values expanded by width-sensitivity profiles. |
| `library_values` | Symbolic-library values expanded by library-sensitivity profiles. |

## Model configuration selection

For a scheduled model, `benchmarks/run_one.py` selects its configuration in this order:

1. exact `model_config.<model_id>` entry, if present;
2. family fallback for deep MultKAN, PowerRuleKAN, SISP, RuleKAN-RBF, or RuleKAN identifiers;
3. an empty mapping if no family configuration applies.

RuleKAN ablation identifiers begin with `rulekan`, so they inherit `model_config.rulekan` unless an exact model-specific entry is provided. Their trainer then applies the ablation override.

PowerRuleKAN inherits the RuleKAN numerical/support-discovery configuration because the powered grammar is applied after the RuleKAN base stage. SISP similarly inherits the RuleKAN or RuleKAN-RBF precursor settings while replacing the symbolic support grammar.

## Shared settings

A typical controlled configuration is:

```yaml
shared_settings:
  enabled: true
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

`capacity.width` is total rule/hidden width, not product order. The shared resolver maps it to each compatible method family. `max_product_order: 3` fixes the maximum product order across tasks rather than deriving it from target-specific `TaskSpec.max_factors` metadata.

For width sweeps, `capacity.mult_fraction` and `capacity.min_mult_units` may replace a fixed `mult_units` count so MultKAN multiplication capacity scales with `W`.

## Symbolic-library aliases

Accepted aliases include:

```text
compact
core / core10 / target_core / research
medium / medium16
paper25
research26 / full / full26
```

`research` is an alias for the controlled 10-operator elementary `core10` / `target_core` library. `research26` selects the larger 26-operator vocabulary.

## Command-line overrides

`python -m benchmarks.run_benchmark` accepts:

```text
--config
--profile
--run-dir
--models
--suites
--tasks
--seeds
--device
--workers
--cpu-fraction
--cpu-threads-per-job
--aggregate-every
--edge-policy
--gmp-refinement-policy
--resume / --no-resume
--reuse-completed / --no-reuse-completed
--verbose-resume / --no-verbose-resume
--show-build-fingerprint / --no-show-build-fingerprint
--keep-going / --no-keep-going
--list
```

`--models`, `--tasks`, `--suites`, and `--seeds` replace the corresponding selection for that invocation.

The per-job runner additionally accepts `--shared-width`, `--shared-library`, and `--cpu-threads`. The orchestrator uses these arguments when expanding sensitivity profiles and controlling per-process CPU threads.

## Provenance

Each run directory contains a benchmark configuration snapshot. Every result JSON stores a build fingerprint derived from source and configuration inputs. Resume reuses completed records by default even across fingerprint changes; pass `--no-reuse-completed` to require an exact active-build match.

The complete resolved baseline values for RuleKAN are documented in [RuleKAN-family configuration](configuration-rulekan.md). Numerical, symbolic, power, and baseline-specific fields are documented in the adjacent reference pages.
