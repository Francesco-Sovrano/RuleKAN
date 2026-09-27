# RuleKAN-family configuration

RuleKAN configuration is resolved from three sources:

1. model-specific values under `profile.model_config`;
2. profile `shared_settings`, which can override shared capacity, grid, symbolic library, product order, and selected symbolic-search controls;
3. adapter defaults in `benchmarks/models.py` for values omitted by the resolved profile.

For ablation identifiers beginning with `rulekan`, `benchmarks/run_one.py` supplies the resolved `model_config.rulekan` configuration unless the profile defines an explicit configuration for that exact identifier. The ablation adapter then applies only its mechanism-specific override.

## Research RuleKAN baseline

The resolved `research` profile defines the following RuleKAN values before shared-setting resolution:

| Field | Value |
|---|---:|
| `stage_scale` | `0.24` |
| `iterative_pruning` | `true` |
| `prune_refit_steps` | `60` |
| `prune_stabilize_rounds` | `2` |
| `prune_stabilize_steps` | `30` |
| `prune_candidates` | `4` |
| `plateau_rounds` | `4` |
| `plateau_steps` | `100` |
| `plateau_patience` | `2` |
| `numeric_lbfgs_steps` | `40` |
| `symbolic` | `true` |
| `symbolic_gmp_steps` | `70` |
| `symbolic_tuple_steps` | `25` |
| `symbolic_commit_steps` | `90` |
| `symbolic_commit_lbfgs` | `10` |
| `symbolic_backfit_steps` | `45` |
| `symbolic_backfit_lbfgs` | `5` |
| `symbolic_final_steps` | `160` |
| `symbolic_final_lbfgs` | `25` |
| `symbolic_cleanup_seconds` | `30` |
| `symbolic_cleanup_trials` | `12` |
| `symbolic_trial_steps` | `55` |
| `symbolic_max_rules` | `12` |
| `symbolic_rank_continuation` | `true` |
| `symbolic_rank_continuation_caps` | `[3, 6]` |

Its shared settings resolve the common benchmark controls to:

| Field | Value |
|---|---|
| `capacity.policy` | `fixed` |
| `capacity.width` | `12` |
| `capacity.mult_units` | `4` |
| `max_product_order` | `3` |
| `grid` | `12` |
| `symbolic_library` | `target_core` |
| `symbolic_hybrid_hard_screening` | `true` |
| `symbolic_residual_structure_topk` | `1` |
| `symbolic_joint_scale_refit` | `true` |
| `symbolic_rule_budget_at_least_width` | `true` |

For RuleKAN, shared width sets `n_rules=12`; the fixed maximum product order sets `max_factors_override=3` for every task; `target_core` resolves to the ten-operator `core10` library.

## Ablation inheritance

The full `ablation` profile contains `extends: research`. It replaces only its model list and task list. The resolved `rulekan` and `rulekan_fast` configurations, data settings, seeds, timeout, and shared settings are therefore identical to `research`.

Ablation adapters apply these overrides:

| Identifier | Override |
|---|---|
| `rulekan_no_product` | `max_factors_override=1` |
| `rulekan_no_pruning` | `pruning_mode="none"` |
| `rulekan_one_shot_prune` | `pruning_mode="one_shot"` |
| `rulekan_no_gmp` | `use_gmp_preselection=false` |
| `rulekan_no_backfit` | `symbolic_backfit=false` |
| `rulekan_no_self_product` | `allow_symbolic_self_products=false` |
| `rulekan_no_product_no_pruning` | product order one and pruning disabled |
| `rulekan_no_product_no_gmp` | product order one and GMP disabled |
| `rulekan_no_pruning_no_gmp` | pruning and GMP disabled |
| `rulekan_no_product_no_pruning_no_gmp` | product order one, pruning disabled, GMP disabled |

`rulekan_fast`, `rulekan_omp_linear`, and `rulekan_graph` are comparative variants rather than knockout identifiers. They use the inherited research budgets with their basis, pursuit, or graph-specific behavior.

## Detailed controls

- [Numerical configuration](configuration-numerical.md): rule bank, product order, spline/RBF settings, training schedule, pruning, support evidence, numerical manifold constraints, and redundancy controls.
- [Symbolic configuration](configuration-symbolic.md): symbolic library, GMP, hard proposals, pursuit, backfitting, rank continuation, affine partitions, adaptive rescue, and composition rescue.
- [PowerRuleKAN configuration](configuration-power.md): integer powers, reciprocals, ratios, transformed-target bases, and powered composition rescue.
- [Configuration reference](configuration.md): profile inheritance, shared settings, sensitivity sweeps, and command-line overrides.
