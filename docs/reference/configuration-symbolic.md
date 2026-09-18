# RuleKAN symbolic-search configuration

RuleKAN symbolic search converts numerical support evidence into a sparse symbolic sum of products. The configuration controls the symbolic operator library, multiplicity/rank expansion, proposal generation, whole-rule pursuit, continuous refitting, validation-gated rescue, and cleanup.

## Core symbolic controls

| Field | Meaning | Adapter default |
|---|---|---|
| `symbolic` | Run symbolic conversion after the numerical stage. | `false` |
| `symbolic_library` | Operator families available to univariate symbolic factors. | compact 11-operator tuple |
| `symbolic_min_rules` | Minimum number of committed symbolic rules. | `1` |
| `symbolic_max_rules` | Maximum symbolic rules before shared-budget adjustment. | numerical `n_rules` |
| `allow_symbolic_self_products` | Permit repeated occurrences of the same variable in one symbolic product. | `true` |
| `symbolic_max_rules_per_structure` | Maximum independently parameterized rules assigned to one support/multiplicity structure. | `2` |
| `symbolic_rank_continuation` | Permit staged expansion of symbolic rank. | `true` |
| `symbolic_rank_continuation_caps` | Intermediate rank caps used by continuation. | `[3, 6]` |
| `symbolic_target_rmse` | Target symbolic RMSE used by pursuit stopping logic. | `1e-4` |
| `symbolic_seed` | Deterministic symbolic-search seed. | numerical seed + `100003` |

The `research` profile sets `symbolic=true`, `symbolic_max_rules=12`, `symbolic_rank_continuation=true`, and caps `[3,6]`. Shared settings enforce a symbolic rule budget at least equal to width `W=12`.

## GMP operator proposal

GMP proposes operator tuples for complete product rules before exact hard fitting.

| Field | Meaning | Adapter default |
|---|---|---:|
| `use_gmp_preselection` | Enable GMP proposal screening. | `true` |
| `symbolic_gmp_steps` | GMP optimization steps. | `60` |
| `symbolic_gmp_lr` | Optional GMP learning rate. | low-level default unless set |
| `symbolic_gmp_topk` | Overall GMP tuple/proposal cap. | structure-dependent |
| `symbolic_gmp_unary_topk` | Unary proposal cap. | structure-dependent |
| `symbolic_gmp_self_topk` | Repeated-variable proposal cap. | structure-dependent |
| `symbolic_gmp_relaxation_mode` | Relaxation used for discrete operator choice. | `soft` |
| `symbolic_gmp_atom_backward_normalization` | Backward-gradient normalization policy for symbolic atoms. | `none` |
| `symbolic_gmp_gumbel_noise_scale` | Gumbel perturbation scale when used by the relaxation. | `1.0` |
| `symbolic_gmp_complexity_weight` | Direct complexity penalty in GMP scoring. | `0.0` |
| `symbolic_gmp_complexity_logit_prior` | Complexity prior applied to operator logits. | `0.0` |
| `symbolic_gmp_nonlinearity_logit_prior` | Nonlinearity prior applied to operator logits. | `0.0` |
| `symbolic_gmp_curvature_weight` | Curvature-based proposal preference. | `0.0` |
| `symbolic_gmp_nonlinearity_weight` | Nonlinearity-based proposal preference. | `0.0` |
| `symbolic_gmp_identity_chart` | Affine chart used to initialize identity-family proposals. | `raw` |
| `symbolic_tuple_steps` | Continuous refinement steps for operator tuples. | `20` |

The `research` RuleKAN budget uses `symbolic_gmp_steps=70` and `symbolic_tuple_steps=25`.

## Hard operator proposals and hybrid screening

Hard proposal paths evaluate exact operator families rather than only relaxed GMP logits.

```text
symbolic_hard_proposal_union
symbolic_hard_proposal_top_tuples
symbolic_hard_proposal_beam
symbolic_hard_proposal_samples
symbolic_hard_proposal_max_per_factor
symbolic_hybrid_hard_screening
symbolic_residual_hard_proposal_refresh
symbolic_hard_residual_operator_rescue
symbolic_hard_screen_start_step
symbolic_hard_screen_matching_steps
symbolic_hard_screen_global_candidates
symbolic_hard_screen_structure_topk
symbolic_hard_screen_beam
symbolic_hard_screen_top_tuples
symbolic_hard_screen_samples
symbolic_hard_screen_residual_rescue
```

`symbolic_hard_proposal_*` controls the initial hard tuple union. `symbolic_hard_screen_*` controls later residual-stage hard screening. The `research` shared configuration enables `symbolic_hybrid_hard_screening`; the adapter's ordinary RuleKAN default is disabled unless the profile enables it.

## Whole-rule pursuit and continuous refitting

| Field | Meaning | Adapter default |
|---|---|---:|
| `symbolic_trial_steps` | Gradient steps for evaluating a candidate rule. | `50` |
| `symbolic_max_rule_candidates` | Candidate-rule cap evaluated by the pursuit. | `16` |
| `symbolic_commit_steps` | Gradient steps after committing a rule. | `80` |
| `symbolic_commit_lbfgs` | LBFGS steps after rule commitment. | `10` |
| `symbolic_backfit` | Refit previously committed rules after additions. | `true` |
| `symbolic_backfit_beam` | Number of backfit alternatives retained. | `3` |
| `symbolic_backfit_steps` | Gradient steps in backfitting. | `40` |
| `symbolic_backfit_lbfgs` | LBFGS steps in backfitting. | `5` |
| `symbolic_final_steps` | Final joint gradient-polish steps. | `120` |
| `symbolic_final_lbfgs` | Final joint LBFGS steps. | `20` |
| `symbolic_cleanup_seconds` | Wall-clock limit for cleanup search. | `20` |
| `symbolic_cleanup_trials` | Maximum cleanup trials. | `10` |
| `symbolic_beam` | General symbolic beam width. | `4` |
| `symbolic_beam_max_per_structure` | Beam cap per support/multiplicity structure. | `1` |
| `symbolic_joint_scale_refit` | Jointly refit additive rule scales during symbolic pursuit. | disabled unless enabled by caller/profile |
| `pursuit_mode` | Symbolic pursuit policy: GSR or an OMP variant. | model-specific |
| `omp_extra_steps` | Additional optimization steps used by OMP paths. | `60` |
| `symbolic_takeover` | Symbolic conversion route, such as learned-support GSR or matching pursuit. | model-family dependent |
| `symbolic_use_compressed_numeric_structures` | Use post-compression numerical structures as the symbolic source. | `false` |

The `research` RuleKAN profile uses `55` trial steps, `90/10` commit gradient/LBFGS steps, `45/5` backfit steps, `160/25` final steps, and `30` seconds with `12` cleanup trials. Shared settings enable joint scale refitting.

## Residual structure and rank controls

```text
symbolic_structure_diverse_beam
symbolic_residual_structure_topk
symbolic_residual_structure_mass
symbolic_residual_structure_min
symbolic_residual_structure_max
symbolic_residual_structure_gap_rel
symbolic_residual_operator_rescue
symbolic_residual_rescue_gmp_topk
symbolic_residual_rescue_gmp_steps
symbolic_residual_rescue_beam
symbolic_residual_rescue_steps
symbolic_residual_rescue_lbfgs
symbolic_max_rules_per_structure
symbolic_rank_continuation
symbolic_rank_continuation_caps
```

These controls allow pursuit to revisit additional structures or allocate more additive rank when the current residual remains structured. They do not permit a distinct-variable support outside the effective RuleKAN support bank. `research` sets `symbolic_residual_structure_topk=1` through shared settings.

## Interaction-shape screening and initial block pursuit

```text
symbolic_interaction_shape_screening
symbolic_interaction_shape_topk
symbolic_interaction_shape_bins
symbolic_interaction_shape_min_rank1
symbolic_interaction_shape_require_multiple
symbolic_initial_block_pursuit
symbolic_initial_block_pool
symbolic_initial_block_pair_beam
symbolic_initial_block_refit_steps
symbolic_initial_block_lbfgs_steps
symbolic_initial_block_consolidate_steps
symbolic_initial_block_consolidate_lbfgs_steps
```

Interaction-shape screening summarizes residual behavior over low-dimensional variable grids and can prioritize supports whose residual surface is compatible with one or several product mechanisms. Initial block pursuit can jointly initialize two rules when a single greedy rule is insufficient. Both mechanisms operate only on admissible supports.

The ordinary adapter defaults enable interaction-shape screening and initial block pursuit; a profile can set `symbolic_interaction_shape_topk=0` to suppress interaction proposals.

## Affine-partition recovery

Affine-partition recovery searches for complementary identity gates such as `u` and `1-u` inside already admissible supports.

```text
symbolic_affine_partition_rescue
symbolic_affine_partition_family_beam
symbolic_affine_partition_seed_topk
symbolic_affine_partition_max_samples
symbolic_affine_partition_max_support_pairs
symbolic_affine_partition_refine_steps
symbolic_affine_partition_refine_lr
symbolic_affine_partition_lbfgs_steps
symbolic_affine_partition_final_polish_topk
symbolic_affine_partition_final_polish_steps
symbolic_affine_partition_final_polish_lbfgs_steps
symbolic_affine_partition_equivalence_rel_mse
symbolic_affine_partition_equivalence_nrmse
symbolic_affine_partition_cancellation_weight
symbolic_affine_partition_complexity_weight
symbolic_affine_partition_min_improvement_rel
```

The learned-support GSR path enables affine-partition rescue by default. The search receives the same `allowed_supports` contract as the rest of RuleKAN symbolic search and cannot introduce an unseen variable combination.

## Adaptive high-recall support rescue

`RuleKAN Adaptive` enables `symbolic_validation_rescue`. The rescue rebuilds a symbolic bank from pre-pruning numerical support evidence when the primary symbolic model is insufficient on validation data.

| Field | Meaning | Ordinary adapter default |
|---|---|---:|
| `symbolic_validation_rescue` | Enable validation-gated high-recall support rescue. | `false` |
| `symbolic_validation_rescue_nrmse` | Validation-NRMSE trigger. | `0.03` |
| `symbolic_validation_rescue_force_nrmse` | Higher validation-NRMSE threshold for forced rescue consideration. | `0.10` |
| `symbolic_validation_rescue_numeric_ratio` | Symbolic-to-numerical validation ratio trigger. | `1.35` |
| `symbolic_validation_rescue_support_max` | Optional cap on rescued support classes; `0` means derived. | `0` |
| `symbolic_validation_rescue_max_rules` | Maximum rescued symbolic rules. | derived from ordinary budget |
| `symbolic_validation_rescue_min_improvement_rel` | Minimum relative validation improvement required to replace the incumbent. | `0.005` |

Rescue-specific search budgets can be set with:

```text
symbolic_validation_rescue_gmp_steps
symbolic_validation_rescue_tuple_steps
symbolic_validation_rescue_trial_steps
symbolic_validation_rescue_commit_steps
symbolic_validation_rescue_commit_lbfgs
symbolic_validation_rescue_backfit_steps
symbolic_validation_rescue_backfit_lbfgs
symbolic_validation_rescue_final_steps
symbolic_validation_rescue_final_lbfgs
symbolic_validation_rescue_initial_block
symbolic_validation_rescue_initial_pool
symbolic_validation_rescue_pair_beam
symbolic_validation_rescue_hard_beam
symbolic_validation_rescue_hard_tuples
```

When omitted, the adapter derives rescue budgets from the ordinary symbolic-search settings with minimum rescue budgets.

## Depth-2 composition rescue

`RuleKAN-Comp` and `SISP-Comp` enable `symbolic_composition_rescue`. The ordinary RuleKAN and SISP variants keep it disabled.

| Field | Meaning | Adapter default |
|---|---|---:|
| `symbolic_composition_rescue` | Enable one validation-gated depth-2 symbolic composition. | `false` |
| `symbolic_composition_trigger_nrmse` | Validation-NRMSE trigger for trying composition. | `0.003` |
| `symbolic_composition_min_improvement_rel` | Relative validation improvement required for replacement. | `0.002` |
| `symbolic_composition_coarse_topk` | Coarse candidate count. | `18` |
| `symbolic_composition_family_topk` | Operator families retained per coarse candidate. | `4` |
| `symbolic_composition_max_families` | Maximum family combinations considered. | `220` |
| `symbolic_composition_shallow_steps` | Shallow composition refinement steps. | `80` |
| `symbolic_composition_shallow_lbfgs` | Shallow LBFGS steps. | `20` |
| `symbolic_composition_deep_topk` | Candidates retained for deep refinement. | `6` |
| `symbolic_composition_deep_steps` | Deep refinement gradient steps. | `250` |
| `symbolic_composition_deep_lbfgs` | Deep refinement LBFGS steps. | `80` |
| `symbolic_composition_correction_topk` | Flat correction rules considered alongside the composition. | `4` |

Composition changes symbolic depth but preserves the support contract used to construct the inner expression.
