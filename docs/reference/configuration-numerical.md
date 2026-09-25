# RuleKAN numerical configuration

These fields control the numerical `SumProductKAN` precursor used by RuleKAN-family methods. Values come from the resolved benchmark profile and shared settings; omitted values use the adapter defaults in `benchmarks/models.py`.

## Model size and numerical basis

| Field | Meaning | Adapter default |
|---|---|---|
| `n_rules` | Number of numerical rule slots. Shared benchmark width overrides this for RuleKAN. | `max(10, 2*input_dim+4)` |
| `max_factors_override` | Maximum factor slots per rule. Shared `max_product_order: 3` fixes this at three in the controlled benchmark profiles. | task `max_factors` |
| `grid` | Spline/RBF grid resolution. | `12` |
| `k` | Spline order used by spline numerical factors. | `3` |
| `grid_range` | Standardized input range used to initialize the numerical basis. | `[-2.5, 2.5]` |
| `lr` | Base learning rate used to construct the staged numerical schedule. | `0.002` |
| `stage_scale` | Multiplicative scale applied to the default numerical training-stage lengths. | `0.15` |
| `rbf_train_grid` | Train RBF grid locations for the Fast/RBF basis. | `true` |
| `rbf_train_width` | Train RBF widths for the Fast/RBF basis. | `true` |
| `rbf_width_scale` | Initial RBF width scale. | `1.0` |
| `init_factor_open_prob` | Initial probability that an optional factor gate is open. | `0.95` |

The `research` profile overrides `stage_scale` to `0.24`; shared settings set `n_rules=12`, `grid=12`, and fixed product order `q=3`.

## Gradient stabilization and numerical sparsity

| Field | Meaning | Adapter default |
|---|---|---:|
| `numeric_factor_gradient_scale` | Straight-through gradient scale for factor gates. | `4.0` |
| `numeric_product_gradient_scale` | Gradient scale for product-rule paths. | `2.0` |
| `numeric_product_gradient_power` | Exponent used by product-gradient gain normalization. | `1.0` |
| `numeric_product_gradient_max_gain` | Maximum product-gradient amplification. | `8.0` |
| `numeric_rule_l0_multiplier` | Multiplier on rule L0 pressure during compression/consolidation. | `1.0` |
| `numeric_factor_l0_multiplier` | Multiplier on factor L0 pressure. | `1.0` |
| `numeric_group_lasso_multiplier` | Multiplier on contribution group-lasso pressure. | `1.0` |
| `contribution_correlation` | Correlation redundancy penalty applied to later numerical stages. | `0.0` |
| `graph_redundancy` | Graph/span redundancy penalty applied to later numerical stages. | `0.0` |
| `contribution_correlation_threshold` | Correlation threshold used by redundancy controls. | `0.95` |

These penalties alter numerical compression; they do not change the symbolic support-admissibility rule.

## Pruning and numerical polishing

`pruning_mode` accepts `iterative`, `one_shot`, or `none`. If it is omitted, `iterative_pruning=true` selects `iterative`; `iterative_pruning=false` selects `none`.

| Field | Meaning | Adapter default |
|---|---|---:|
| `pruning_mode` | Explicit pruning policy. | derived from `iterative_pruning` |
| `iterative_pruning` | Backward-compatible switch used when `pruning_mode` is absent. | `true` |
| `prune_refit_steps` | Gradient refit steps after candidate structural deletions. | `60` |
| `prune_lr` | Learning rate for prune refits. | `2e-4` |
| `prune_stabilize_rounds` | Stabilization passes after accepted deletions. | `2` |
| `prune_stabilize_steps` | Refit steps per stabilization pass. | `30` |
| `prune_recovery_probes` | Recovery probes used during pruning. | `1` |
| `prune_candidates` | Maximum candidate deletions considered per pass. | `4` |
| `prune_local_budget` | Relative local MSE budget for a deletion. | `0.01` |
| `prune_global_budget` | Relative global MSE budget for a deletion. | `0.03` |
| `plateau_rounds` | Maximum hard numerical plateau-polish rounds. | `1` |
| `plateau_steps` | Gradient steps per plateau round. | `80` |
| `plateau_lr` | Learning rate for plateau polishing. | `3e-4` |
| `plateau_min_improvement` | Minimum relative improvement used by plateau stopping. | `0.001` |
| `plateau_patience` | Number of insufficient-improvement rounds allowed. | `2` |
| `numeric_lbfgs_steps` | Final hard numerical LBFGS steps. | `0` |
| `numeric_target_rmse` | Optional numerical target RMSE for early termination logic. | `0.0` |

The `research` profile uses four plateau rounds and 40 final LBFGS steps. The full `ablation` profile inherits those values exactly.

## Pre-pruning support evidence and support extraction

Before destructive pruning, the benchmark records high-recall support evidence from factor gates. Final active supports and pre-pruning evidence are combined by the learned-support route.

| Field | Meaning | Adapter default |
|---|---|---:|
| `symbolic_support_factor_open_threshold` | Factor-open threshold used when capturing support evidence. | `0.20` |
| `symbolic_learned_support_max` | Maximum retained support classes for learned-support symbolic search. | dimension/rule dependent |
| `symbolic_learned_support_importance_mix` | Mixture weight used when ranking support evidence sources. | `0.5` |
| `symbolic_numeric_structure_max` | Optional cap on retained numerical structures; `0` means no extra cap. | `0` |

The support stage records distinct-variable supports. Repeated-variable multiplicity is introduced later by symbolic grammar construction rather than inferred from numerical same-variable factorization.

## Optional numerical symbolic-manifold constraint

`numeric_symbolic_manifold=true` adds an auxiliary penalty that attracts active numerical edge shapes toward the configured symbolic families while the forward model remains numerical. The option is disabled by default.

| Field | Meaning | Adapter default |
|---|---|---:|
| `numeric_symbolic_manifold` | Enable the numerical-to-symbolic manifold constraint. | `false` |
| `manifold_warmup_scale` | Scale for the unconstrained warmup duration. | `0.5` |
| `manifold_hardening_start` | Initial symbolic hardening level in compression. | `0.0` |
| `manifold_hardening_mid` | Intermediate hardening level. | `0.65` |
| `manifold_hardening_end` | Final hardening level. | `1.0` |
| `manifold_temperature_start` | Initial operator temperature. | `1.2` |
| `manifold_temperature_mid` | Intermediate operator temperature. | `0.55` |
| `manifold_temperature_end` | Final operator temperature. | `0.25` |
| `manifold_symbolic_lr_scale` | Learning-rate multiplier for auxiliary symbolic parameters. | `3.0` |
| `manifold_dual_init` | Initial dual coefficient in the compression stage. | `3e-4` |
| `manifold_dual_hard_init` | Initial dual coefficient in the harder consolidation stage. | `8e-4` |
| `manifold_rho` | Dual update scale during compression. | `3e-4` |
| `manifold_hard_rho` | Dual update scale during consolidation. | `8e-4` |
| `manifold_tolerance` | Constraint tolerance. | `0.02` |
| `manifold_dual_every` | Steps between dual updates. | `50` |
| `manifold_dual_max` | Maximum dual coefficient. | `0.03` |
| `manifold_max_samples` | Maximum samples used for manifold matching. | `256` |
| `manifold_projection_lr` | Learning rate used by projection/polish code paths. | `2e-4` |

## Optional numerical logic compression

`numeric_logic_compression=true` attempts to remove complete hard numerical rules after the standard numerical fit. A deletion is retained only if the configured validation error bounds are satisfied.

| Field | Meaning | Adapter default |
|---|---|---:|
| `numeric_logic_compression` | Enable rollback-safe numerical rule deletion. | `false` |
| `numeric_logic_min_rules` | Minimum number of numerical rules retained. | `1` |
| `numeric_logic_max_deletions` | Maximum accepted deletions; `0` permits the implementation-defined full pass. | `0` |
| `numeric_logic_candidate_trials` | Number of deletion candidates tested per pass. | `4` |
| `numeric_logic_lbfgs_steps` | LBFGS refit steps after a candidate deletion. | `24` |
| `numeric_logic_relative_rmse_tolerance` | Relative validation-RMSE tolerance for deletion acceptance. | `0.25` |
| `numeric_logic_nrmse_tolerance` | Absolute validation-NRMSE tolerance. | `0.003` |

## Redundancy cleanup

| Field | Meaning | Adapter default |
|---|---|---:|
| `redundancy_cleanup` | Enable symbolic contribution redundancy cleanup. | `true` |
| `redundancy_corr` | Pairwise contribution-correlation threshold. | `0.995` |
| `redundancy_span_r2` | Span-redundancy R² threshold. | `0.995` |
| `symbolic_elimination_rel_mse_budget` | Relative MSE budget for removing a symbolic rule. | `0.02` |
| `symbolic_min_rule_improvement_rel` | Minimum relative improvement required for a retained symbolic rule. | `0.002` |

`rulekan_graph` enables a small nonzero `graph_redundancy` penalty and uses a contribution-correlation threshold of `0.97` unless the profile overrides them.
