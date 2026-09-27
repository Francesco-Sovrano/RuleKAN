# Benchmark methods

The benchmark compares RuleKAN variants with structure-independent symbolic pursuit, MultKAN symbolic-extraction pipelines, external symbolic-regression systems, and numerical predictors. Model identifiers are passed through `--models` and are defined in `benchmarks/models.py`.

## RuleKAN family

| Identifier | Method |
|---|---|
| `rulekan` | B-spline numerical RuleKAN followed by support-conditioned symbolic GMP/GSR. |
| `rulekan_comp` | RuleKAN followed by a validation-gated depth-2 symbolic composition rescue. |
| `rulekan_adaptive` | RuleKAN with validation-gated higher-recall support rescue from pre-pruning numerical evidence. |
| `rulekan_fast` | **RuleKAN-RBF**: RuleKAN with a FastKAN-style Gaussian-RBF numerical edge bank; the symbolic language and support conditioning are unchanged. |
| `rulekan_hybrid` | RuleKAN with GMP proposals unioned with deterministic hard multistart proposals. |
| `rulekan_fast_hybrid` | RuleKAN-RBF counterpart of `rulekan_hybrid`. |
| `rulekan_fast_adaptive` | RuleKAN-RBF counterpart of `rulekan_adaptive`. |
| `rulekan_distill` | B-spline numerical RuleKAN with numerical-to-symbolic distillation takeover. |
| `rulekan_fast_distill` | RuleKAN-RBF counterpart of `rulekan_distill`. |
| `rulekan_graph` | RuleKAN with contribution-correlation/span redundancy controls in the GSR path. |
| `rulekan_fast_graph` | RuleKAN-RBF counterpart of `rulekan_graph`. |

RuleKAN primary search is support-conditioned. Repeated-variable multiplicity and symbolic rank may expand within a retained support. Configured validation-gated augmentation may additionally use nonempty support subsets, pre-pruning supports, or bounded gate-aware support unions derived from Stage-1 evidence. RuleSISP alone enumerates arbitrary variable multisets through order `q`.

The `research` profile includes `rulekan_comp`, `sisp_comp`, and `power_rulekan_comp` as separate methods alongside their flat counterparts.

## OMP variants

The OMP variants use the same numerical support source and symbolic grammar as the corresponding RuleKAN model. They change the pursuit/refinement policy.

| Identifier | Pursuit mode |
|---|---|
| `rulekan_omp_linear` | `omp_linear` |
| `rulekan_omp_nonlinear` | `omp_nonlinear` |
| `rulekan_omp_full` | `omp_full` |
| `rulekan_fast_omp_linear` | RBF numerical precursor with `omp_linear` |
| `rulekan_fast_omp_nonlinear` | RBF numerical precursor with `omp_nonlinear` |
| `rulekan_fast_omp_full` | RBF numerical precursor with `omp_full` |

## SISP

`SISP` means Structure-Independent Symbolic Pursuit. It uses the same whole-rule symbolic search machinery but supplies the complete variable-multiset grammar through the configured maximum product order instead of a RuleKAN support bank.

| Identifier | Numerical precursor |
|---|---|
| `sisp` | B-spline SumProductKAN precursor |
| `sisp_comp` | B-spline precursor plus the same validation-gated depth-2 composition rescue |
| `sisp_fast` | Gaussian-RBF SumProductKAN precursor |

The precursor supplies numerical initialization and common training infrastructure; it does not restrict the symbolic variable grammar in SISP.

## PowerRuleKAN

`power_rulekan` extends the final model with integer powers of complete symbolic RuleKAN bases. The ordinary `p=1` RuleKAN candidate, transformed-target bases, reciprocal bases, and ratio numerator/denominator bases all inherit the canonical validation-selected RuleKAN effective-support bank.

`power_rulekan_comp` keeps the same powered outer grammar and the same effective-support contract, but each `p=1`, transformed-power, numerator, and denominator base may be replaced by the validation-gated depth-2 composition rescue before powered/ratio polishing. It therefore tests powers and ratios of support-licensed composed symbolic bases rather than only powers of flat bases. See [PowerRuleKAN](../algorithms/power-rulekan.md).

## RuleKAN ablations

The following identifiers remove or alter one or more RuleKAN mechanisms while retaining the same benchmark interface. In the `ablation` profile they inherit the exact `research` RuleKAN baseline configuration and apply only the listed mechanism override.

| Identifier | Change |
|---|---|
| `rulekan_no_product` | maximum numerical/symbolic factor count forced to one |
| `rulekan_no_pruning` | numerical pruning disabled |
| `rulekan_one_shot_prune` | one-shot rather than iterative pruning |
| `rulekan_no_gmp` | GMP preselection disabled |
| `rulekan_no_backfit` | symbolic backfitting disabled |
| `rulekan_no_self_product` | repeated-variable symbolic products disabled |
| `rulekan_no_product_no_pruning` | product rules and pruning disabled |
| `rulekan_no_product_no_gmp` | product rules and GMP disabled |
| `rulekan_no_pruning_no_gmp` | pruning and GMP disabled |
| `rulekan_no_product_no_pruning_no_gmp` | product rules, pruning, and GMP disabled |

## MultKAN symbolic-regression pipelines

These methods use the repository's `MultKAN` implementation and its symbolic-regression procedures. The shallow architecture has multiplication units in the first hidden layer. The deep variants place multiplication after one learned KAN transformation so that nested multiplicative features can be represented.

| Identifier | Symbolic procedure | Numerical edge family |
|---|---|---|
| `autosym` | baseline symbolic regression | B-spline |
| `fastkan_autosym` | baseline symbolic regression | Gaussian RBF |
| `gsr` | greedy symbolic regression | B-spline |
| `fastkan_gsr` | greedy symbolic regression | Gaussian RBF |
| `gmp` | Gated Matching Pursuit | configured symbolic atom bank |
| `multkan_deep_autosym` | baseline symbolic regression | B-spline |
| `fast_multkan_deep_autosym` | baseline symbolic regression | Gaussian RBF |
| `multkan_deep_gsr` | greedy symbolic regression | B-spline |
| `fast_multkan_deep_gsr` | greedy symbolic regression | Gaussian RBF |
| `multkan_deep_gmp` | Gated Matching Pursuit | configured symbolic atom bank |

The shallow and deep methods are regression-only in the benchmark harness.

## External symbolic-regression baselines

| Identifier | System | Benchmark dependency |
|---|---|---|
| `srkan` | SR-KAN, KAN-guided divide-and-conquer symbolic regression | authors' GitHub repository (`benchmarks/requirements-srkan.txt`) |
| `symbolic_kan` | Symbolic-KAN trainable analytic network | authors' `Pub_Symbolic_KANs` checkout pinned by `benchmarks/setup.sh` to commit `9481a82` |
| `pse` | PSE with the official PSRN implementation | `psrn` |
| `rils_rols` | RILS-ROLS iterated-local-search symbolic regression | `rils-rols` |
| `udsr` | unified Deep Symbolic Regression (LINEAR/poly + GP meld) | official DSO PyTorch package |
| `sindy` | SINDy-12 static sparse-library regression using STLSQ (<=12 active non-bias terms) | `pysindy==2.1.0` |
| `sindy_unconstrained` | uncapped SINDy capacity sensitivity | `pysindy==2.1.0` |
| `parfam` | ParFam continuous-global-optimization symbolic regression | `parfam==0.0.2` |
| `eql` | EQL-Div analytic-unit network | built-in PyTorch reproduction of the published architecture |
| `pysr` | PySR / SymbolicRegression.jl evolutionary symbolic regression | `pysr==2.2.1` |
| `operon` | Operon genetic-programming symbolic regression | `pyoperon==0.6.1` |

The harness passes operator sets and compute limits from the selected profile. Under `research` and `research_modern`, direct symbolic primitives are restricted to `core10` wherever the public method API permits; method-native exceptions are recorded in run metadata. `research_modern` schedules `symbolic_kan`, `pse`, `rils_rols`, `sindy`, `parfam`, and `eql` in addition to the `research` models. Model identifier `udsr` is registered but is not selected by that profile. The 23-entry `main_comparison_models` field excludes `pse` and `anfis`. `sindy` is the 12-term controlled variant, while `sindy_unconstrained` is a separate capacity-sensitivity override. PSE, RILS-ROLS, uDSR, PySINDy, and ParFam use their public packages; EQL-Div is implemented in-tree with PyTorch because the released implementations target older Theano/TensorFlow stacks. Symbolic-KAN calls the pinned upstream `train_regression_onehot` routine with benchmark arrays in place of its demo data generator. Dependency checks run before selected external jobs are scheduled. SR-KAN is installed from the authors' repository and checked for the expected `regressor` and `SympyEvaluator` API. Operon is checked through `pyoperon.sklearn`.

## Trainable fuzzy-system baseline

| Identifier | Method |
|---|---|
| `anfis` | First-order Takagi-Sugeno ANFIS with Gaussian product antecedents, normalized firing strengths, and affine consequents. |

The ANFIS adapter uses hybrid learning: premise centers/widths are optimized by Adam while consequent coefficients are refit by ridge least squares. Training data fit the parameters and validation data select the checkpoint. The shared benchmark width controls the number of fuzzy rules. ANFIS is a predictive fuzzy-system comparator; its rule grammar is not the same as RuleKAN's complementary-gate DNF grammar, so the benchmark does not assign RuleKAN-specific fuzzy-rule F1 to ANFIS.

## Numerical predictors

| Identifier | Method |
|---|---|
| `vanilla_kan` | numerical KAN without symbolic extraction |
| `mlp` | multilayer perceptron numerical baseline |

These models provide predictive baselines. Fully symbolic metrics are not attributed to them.
