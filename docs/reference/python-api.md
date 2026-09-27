# Python API

## `SumProductKAN`

```python
from symbolic_kan import SumProductKAN
```

Constructor:

```python
SumProductKAN(
    in_dim,
    n_rules=10,
    max_factors=2,
    grid=16,
    k=3,
    grid_range=(-2.1, 2.1),
    numeric_basis="spline",       # "spline" or "rbf"
    rbf_train_grid=True,
    rbf_train_width=True,
    rbf_width_scale=1.0,
    symbolic_library=...,
    min_order=1,
    init_rule_open_prob=0.95,
    init_factor_open_prob=0.95,
    init_spline_prob=0.995,
    variable_temperature=1.2,
    operator_temperature=1.2,
    rule_temperature=2/3,
    factor_temperature=2/3,
    symbolic_gradient_scale=4.0,
    symbolic_product_gradient_scale=0.0,
    numeric_factor_gradient_scale=0.0,
    numeric_product_gradient_scale=0.0,
    numeric_product_gradient_power=1.0,
    numeric_product_gradient_max_gain=8.0,
    stochastic_rule_gates=False,
    diverse_structure_init=True,
    allow_self_products=True,
    structure_init_margin=0.75,
    seed=0,
    device="cpu",
)
```

Methods:

```text
forward(x, return_details=False)
discretize(rule_threshold=0.5, force_symbolic=False, freeze_gates=True)
undiscretize()
hard_structure(variable_names=None)
symbolic_formula(variable_names=None, input_mean=None, input_std=None, digits=5, simplify=False)
diagnostics()
expected_rule_count()
expected_factor_count()
initialize_symbolic_manifold_seeds_()
symbolic_manifold_distance(...)
```

## Numerical training

```python
from symbolic_kan import default_sum_product_schedule, fit_sum_product_kan
```

`default_sum_product_schedule(base_lr=2e-3, symbolic=True)` returns `SumProductTrainingStage` objects. The benchmark calls it with `symbolic=False` and performs the later symbolic search separately.

```python
history = fit_sum_product_kan(
    model,
    train_x,
    train_y,
    val_x,
    val_y,
    stages=stages,
    restore_best_each_stage=True,
)
```

## Support evidence and symbolic grammar

```python
from symbolic_kan import (
    capture_numeric_support_evidence,
    learned_numeric_support_classes,
    learned_structure_symbolic_bank,
)
```

Typical sequence:

```python
evidence = capture_numeric_support_evidence(model, val_x)
classes = learned_numeric_support_classes(model, evidence=evidence)
bank = learned_structure_symbolic_bank(classes, max_factors=model.max_factors)
```

`classes` groups by distinct-variable support. `bank` expands each support into repeated-variable multiplicity patterns.

## RuleKAN learned-support symbolic search

```python
from symbolic_kan import learned_support_symbolic_gsr

symbolic_model, history = learned_support_symbolic_gsr(
    model,
    train_x, train_y,
    val_x, val_y,
    structure_candidates=bank,
    allowed_supports=[c["support"] for c in classes],
)
```

This entry point validates every candidate support and automatically forwards the same support contract to the affine-partition rescue.

## Generic matching pursuit / SISP engine

```python
from symbolic_kan import mandatory_symbolic_matching_pursuit
```

`mandatory_symbolic_matching_pursuit` is the generic whole-rule search. Passing an independent complete structure bank yields SISP behavior. Passing learned RuleKAN structures directly is possible, but RuleKAN callers should use `learned_support_symbolic_gsr` so support validation remains centralized.

## GMP

```python
from symbolic_kan import gmp_symbolic_operator_preselection
```

GMP receives data, a structure list and a symbolic operator library. It returns per-structure operator proposals used by the hard symbolic search.

## `PowerRuleKAN`

```python
from symbolic_kan import PowerRuleKAN
```

Constructor:

```python
PowerRuleKAN(
    bases,
    terms,                      # each term is [(base_index, integer_power), ...]
    scales=None,
    bias=0.0,
    reciprocal_epsilon=1e-8,
)
```

Methods:

```text
forward(x)
chosen_powers()
reciprocal_domain_margin(x)
symbolic_formula(variable_names=None, input_mean=None, input_std=None, digits=8, simplify=False)
```

Helpers:

```text
inverse_power_target(y, p, eps=1e-8)
safe_integer_power(z, p, eps=1e-8)
harden_symbolic_base(base)
hard_power_polish(...)
hard_ratio_polish(...)
```

The complete PowerRuleKAN benchmark optimizer, including transformed-target base fitting, ratio pilots and outer atom pursuit, is implemented by `benchmarks.models.train_power_rulekan`.

## Benchmark adapters

`benchmarks.models` exposes the principal adapters:

```text
train_rulekan
train_rulekan_fast
train_sisp
train_sisp_fast
train_power_rulekan
```

`run_model` dispatches benchmark model identifiers to their adapters. `fuzzy_rule_recovery_scores` computes structural fuzzy metrics for final SumProductKAN and PowerRuleKAN expressions.
