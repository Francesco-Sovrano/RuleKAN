# PowerRuleKAN configuration

PowerRuleKAN starts from the canonical effective RuleKAN support bank and searches an outer grammar over complete symbolic bases. A powered model can contain terms of the form

\[
a\prod_t B_t(x)^{p_t},\qquad p_t\in\mathbb Z\setminus\{0\},
\]

where every base \(B_t\) is itself a support-conditioned RuleKAN symbolic expression. Negative powers are reciprocals and are subject to domain-margin checks.

## Outer power search

| Field | Meaning | Adapter default |
|---|---|---:|
| `power_values` | Integer exponents considered by the outer grammar. | `(-3,-2,-1,1,2,3)` |
| `power_outer_rules` | Maximum number of powered outer terms. | `2` |
| `power_target_epsilon` | Numerical floor used by transformed-target construction. | `1e-7` |
| `power_reciprocal_epsilon` | Numerical epsilon used in safe reciprocal evaluation. | `1e-8` |
| `power_reciprocal_margin` | Minimum absolute base value required on train+validation coordinates for negative powers. | `0.02` |
| `power_reciprocal_barrier` | Barrier weight used while polishing reciprocal bases. | `0.02` |
| `power_hard_polish_steps` | Hard powered-expression polishing steps. | `32` |
| `power_hard_polish_lr` | Learning rate for hard powered-expression polishing. | `0.002` |
| `power_ratio_enabled` | Permit ratio candidates. | `true` |
| `power_baseline_validation_rescue` | Allow the ordinary RuleKAN base to use validation-gated support rescue before outer search. | `true` |

The ordinary RuleKAN solution is retained as an explicit `p=1` candidate. PowerRuleKAN therefore compares powered and ratio candidates against the flat support-conditioned model rather than requiring the outer search to rediscover it.

## Ratio search

```text
power_ratio_pilot_ridge
power_ratio_pilot_supports
power_ratio_base_max_rules
power_ratio_cross_steps
power_ratio_quotient_steps
power_ratio_polish_lr
power_ratio_barrier
```

The ratio pilot identifies support subsets for numerator and denominator candidates. Subsequent numerator and denominator symbolic fits remain restricted to support classes licensed by the effective RuleKAN support contract. Ratio polishing jointly refines the selected numerator/denominator expression while enforcing reciprocal safety.

## Transformed-target base search

Transformed-target and ratio-base fits use the `power_base_*` controls when present:

```text
power_base_max_rules
power_base_max_rules_per_structure
power_base_gmp_steps
power_base_gmp_topk
power_base_gmp_unary_topk
power_base_gmp_self_topk
power_base_tuple_steps
power_base_interaction_topk
power_base_interaction_bins
power_base_interaction_min_rank1
power_base_hard_tuples
power_base_hard_beam
power_base_hard_samples
power_base_hard_per_factor
power_base_initial_pool
power_base_pair_beam
power_base_pair_steps
power_base_consolidate_steps
power_base_beam
power_base_trial_steps
power_base_commit_steps
power_base_backfit_steps
power_base_final_steps
power_base_rescue_topk
power_base_rescue_gmp_steps
power_base_rescue_steps
power_base_cleanup_seconds
power_base_cleanup_trials
```

These fields specialize the ordinary RuleKAN symbolic-search budgets for transformed targets such as square roots or reciprocals of the target. When a `power_base_*` value is absent, the adapter derives the corresponding base-search setting from the ordinary RuleKAN symbolic configuration.

## Reciprocal-domain selection

`power_reciprocal_margin` is evaluated on training plus validation inputs during candidate selection. A negative-power candidate whose base approaches zero inside this selection domain is inadmissible.

After the expression is fixed, the benchmark computes the same minimum absolute base value on test inputs and records:

```text
reciprocal_domain_margin
reciprocal_domain_margin_train_val
reciprocal_domain_margin_test
reciprocal_domain_test_valid
```

The test-domain diagnostic does not participate in model selection.

## PowerRuleKAN-Comp

`power_rulekan_comp` enables `power_composition_rescue`. Composition may be used for the ordinary `p=1` base, transformed-power bases, and ratio numerator/denominator bases, while all inner expressions remain support-conditioned.

The powered composition controls are:

```text
power_composition_rescue
power_composition_trigger_nrmse
power_composition_min_improvement_rel
power_composition_coarse_topk
power_composition_family_topk
power_composition_max_families
power_composition_shallow_steps
power_composition_shallow_lbfgs
power_composition_deep_topk
power_composition_deep_steps
power_composition_deep_lbfgs
power_composition_correction_topk
```

If a `power_composition_*` value is omitted, the corresponding `symbolic_composition_*` value is used. The flat base remains the incumbent unless the composed candidate improves validation error by the configured criterion.
