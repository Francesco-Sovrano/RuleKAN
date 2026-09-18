# Baseline configuration

These controls configure the MultKAN symbolic pipelines, external symbolic-regression systems and numerical baselines. Profile inheritance and CLI overrides are described in [Configuration reference](configuration.md).

## MultKAN pipeline configuration

Common keys used by `AutoSym`, `GSR`, `GMP` and their FastKAN/deep variants include:

```text
steps
lr
lamb
prune_iters
node_th
edge_th
width_additive
mult_units
mult_arity
gate_top_k_start
top_k_gates
gating_entropy
gating_l1
edge_policy
gmp_refinement_policy
symbolic_trial_steps
deep_width_1
deep_width_2
deep_mult_units
deep_mult_arity
```

The deep keys apply only to the late-multiplication MultKAN variants.

## Common execution and optimization controls

Several adapters accept generic controls in addition to their model-specific settings:

```text
progress
log_every
verbosity
validation_every
min_lr
numeric_target_rmse
reg_metric
```

`progress`, `log_every` and `verbosity` affect reporting only. `validation_every`, `min_lr`, `numeric_target_rmse` and `reg_metric` affect training/early-stopping behavior in adapters that read them.

## PySR configuration

```text
niterations
populations
population_size
maxsize
maxdepth
timeout_seconds
deterministic
parallelism
```

The benchmark adapter also accepts explicit `unary_operators` and `binary_operators` lists when supplied by a profile.

## Operon configuration

```text
allowed_symbols
population_size
generations
tournament_size
objectives
max_evaluations
optimizer_iterations
max_length
max_depth
max_time
n_threads
model_selection
model_selection_criterion
```

`allowed_symbols` is passed directly to PyOperon as its comma-separated primitive set.

## ANFIS configuration

```text
anfis_rules
min_sigma
max_sigma
kmeans_n_init
epochs
lr
ridge
patience
min_delta_rel
premise_l2
```

`anfis_rules` is set to the shared benchmark width when shared capacity is enabled. Gaussian antecedent centers are initialized with K-means. Premise parameters are updated by Adam; first-order Sugeno consequents are refit by ridge least squares. Validation MSE selects the checkpoint.

## Numerical baseline configuration

`vanilla_kan` uses `steps`, `lr`, `grid`, `k`, and `hidden`. The MLP adapter uses `width`, `steps`, and `lr`.
