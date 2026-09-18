# Task catalog

`benchmarks/specs.py` is the canonical task registry. Synthetic inputs are sampled uniformly from the declared raw ranges and standardized from the training split before fitting. Fuzzy gate variables are sampled in `[0,1]`. Real-data preprocessing is described in [Benchmark protocol](protocol.md).

## Synthetic core

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `same_var_exp_sin` | `0.8*exp(-0.7*x0)*sin(2.4*x0)` | 1 | 2 | `in_class` | (0,0) |
| `cross_log_cos` | `0.75*log(1+x0^2)*cos(1.7*x1)` | 2 | 2 | `in_class` | (0,1) |
| `mixed_rank4` | `1.3*exp(-0.9*x0)*sin(2.2*x1)+0.75*log(1+x2^2)*cos(1.6*x3)+0.25*tanh(2*x4)+0.4*exp(-0.65*x4)*sin(2.35*x4)` | 5 | 2 | `in_class` | (0,1), (2,3), (4), (4,4) |
| `three_way_product` | `0.9*sin(2*x0)*exp(0.55*x1)*sqrt(1+(1.2*x2)^2)` | 3 | 3 | `in_class` | (0,1,2) |
| `two_same_var_mechanisms` | `0.3*tanh(2*x0)+0.55*exp(-0.6*x0)*sin(2.5*x0)` | 1 | 2 | `in_class` | (0), (0,0) |
| `rational_product_mix` | `0.9*cos(2*x0)/(1+1.5*x1^2)+0.55*sqrt(1+x2^2)*tanh(1.7*x3)+0.2*sin(3*x4)` | 5 | 2 | `in_class` | (0,1), (2,3), (4) |

## Long-expression / capacity stress

These tasks isolate expression length while remaining inside the configured search spaces. Both compact targets contain **39 binary expression-tree nodes**. This is immediately below the dedicated profile's PySR `maxsize=40` and Operon `max_length=50`; `long_additive_12` additionally uses exactly 12 additive rules, saturating the shared RuleKAN symbolic rule budget. Run them with `--profile long_expression`.

| Task | Formula / source | Variables | Max factors | Terms | Tree nodes | Representability |
|---|---|---:|---:|---:|---:|---|
| `long_additive_12` | 12 unary `sin/cos/tanh/exp` atoms on distinct variables | 12 | 1 | 12 | 39 | `long_expression_in_class` |
| `long_product_6` | six sums of two-factor analytic products | 12 | 2 | 6 | 39 | `long_expression_in_class` |

The two targets deliberately have the same primitive tree size but stress different model resources: rule count versus multiplicative factors. They contain no fitted coefficients or nested affine constants, so the measured difficulty is dominated by structural length/search rather than constant estimation.

## Powered-expression stress

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `power_inv_bilinear` | `(1+0.8*x0*x1)^-1` | 2 | 2 | `expression_power_exact` | (0,1) |
| `power_inv_quadratic_sum` | `(1+0.6*x0^2+0.4*x1^2)^-1` | 2 | 2 | `expression_power_exact` | (0), (1) |
| `power_square_affine_sum` | `(0.8+0.4*x0+0.3*x1)^2` | 2 | 1 | `expression_power_exact` | (0), (1) |
| `power_ratio_product` | `(0.7+0.4*x0)*(1+0.6*x0*x1)^-1` | 2 | 2 | `expression_power_exact` | (0), (0,1) |

## Fuzzy rule recovery

Expected fuzzy rules are declared as gate/branch factor signatures and are scored after expansion into sum-product form. Nested trees use exact DNF expansion.

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `fuzzy_ite_cross` | `(1-x0)*(0.70*sin(2.1*x1)) + x0*(1.10*exp(-0.75*x2))` | 3 | 2 | `fuzzy_in_class` | (0,1), (0,2) |
| `fuzzy_ite_same_variable` | `(1-x0)*(0.65*sin(2.4*x0)) + x0*(0.85*tanh(2*x0))` | 1 | 2 | `fuzzy_same_variable` | (0,0) |
| `fuzzy_ite_shared_branch_variable` | `(1-x0)*(0.80*cos(1.8*x1)) + x0*(0.55*exp(-0.60*x1))` | 2 | 2 | `fuzzy_in_class` | (0,1) |
| `fuzzy_nested_tree` | `(1-x0)*((1-x1)*(0.60*cos(1.7*x2)) + x1*(0.90*exp(-0.80*x3))) + x0*(0.50*tanh(1.9*x4))` | 5 | 3 | `fuzzy_nested_dnf` | (0,1,2), (0,1,3), (0,4) |
| `fuzzy_two_independent_rules` | `0.70*((1-x0)*sin(2*x2)+x0*exp(-0.70*x3)) + 0.40*((1-x1)*cos(1.5*x4)+x1*tanh(1.8*x5))` | 6 | 2 | `fuzzy_multi_rule` | (0,2), (0,3), (1,4), (1,5) |
| `fuzzy_product_branches` | `(1-x0)*(0.80*exp(-0.45*x1)*sin(2.2*x2)) + x0*(0.60*sqrt(1+(1.15*x3)^2)*cos(1.55*x4))` | 5 | 3 | `fuzzy_product_branches` | (0,1,2), (0,3,4) |
| `fuzzy_same_gate_product_branches` | `(1-x0)*(0.70*exp(-0.55*x0)*sin(2*x1)) + x0*(0.65*tanh(2.1*x0)*cos(1.45*x2))` | 3 | 3 | `fuzzy_same_variable_product` | (0,0,1), (0,0,2) |

### Fuzzy structural signatures

A gate signature `xj−` denotes the complement orientation `1-x_j`; `xj+` denotes the direct orientation `x_j`. Branch entries give the required variable/operator family. Sine and cosine are phase-equivalent in the scorer.

| Task | Expected expanded rules |
|---|---|
| `fuzzy_ite_cross` | `x0− · sin(x1)`; `x0+ · exp(x2)` |
| `fuzzy_ite_same_variable` | `x0− · sin(x0)`; `x0+ · tanh(x0)` |
| `fuzzy_ite_shared_branch_variable` | `x0− · cos(x1)`; `x0+ · exp(x1)` |
| `fuzzy_nested_tree` | `x0− · x1− · cos(x2)`; `x0− · x1+ · exp(x3)`; `x0+ · tanh(x4)` |
| `fuzzy_two_independent_rules` | `x0− · sin(x2)`; `x0+ · exp(x3)`; `x1− · cos(x4)`; `x1+ · tanh(x5)` |
| `fuzzy_product_branches` | `x0− · exp(x1) · sin(x2)`; `x0+ · sqrt1p_sq(x3) · cos(x4)` |
| `fuzzy_same_gate_product_branches` | `x0− · exp(x0) · sin(x1)`; `x0+ · tanh(x0) · cos(x2)` |

## Nested-expression stress

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `nested_same_sin_exp` | `sin(exp(0.7*x0))` | 1 | 2 | `nested_unary` | — |
| `nested_same_tanh_sin` | `tanh(1.6*sin(2.2*x0))` | 1 | 2 | `nested_unary` | — |
| `nested_cross_sin_product` | `sin(1.3*x0*x1)+0.2*x2^2` | 3 | 2 | `nested_cross` | — |
| `mixed_nested_products` | `0.7*exp(-0.5*x0)*sin(2.1*x1)+0.4*tanh(1.4*sin(2*x2))/(1+x3^2)+0.35*exp(-0.7*x4)*sin(2.4*x4)` | 5 | 2 | `mixed` | — |
| `nested_cross_oscillator` | `sin(2.5*sin(1.6*x0)+0.7*cos(1.3*x1))` | 2 | 2 | `nested_cross` | — |

## Canonical KAN examples

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `pykan_exp_sin_square` | `exp(sin(pi*x0)+x1^2)` | 2 | 2 | `nested_factorizable` | — |
| `pykan_singularity` | `sin(2*(log(x0)+log(x1)))` | 2 | 2 | `nested_cross` | — |
| `pykan_radial` | `sqrt(x0^2+x1^2)` | 2 | 2 | `nested_cross` | — |

## Feynman-style equations

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `feynman_kinetic` | `0.5*x0*x1^2` | 2 | 2 | `in_class` | (0,1) |
| `feynman_gravity` | `x0*x1/x2^2` | 3 | 3 | `product_rational` | — |
| `feynman_gaussian` | `exp(-x0^2/2)/sqrt(2*pi)` | 1 | 2 | `nested_unary` | — |
| `feynman_lorentz_mix` | `x0*x1+x0*x2*x3` | 4 | 3 | `in_class` | (0,1), (0,2,3) |
| `feynman_resonance` | `1/((x1^2-x0^2)^2+(x2*x0)^2)` | 3 | 3 | `nested_rational` | — |

## Small real datasets

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `sklearn_diabetes` | `scikit-learn diabetes regression` | data-dependent | 2 | `unknown` | — |
| `sklearn_breast_cancer` | `Wisconsin breast-cancer binary classification` | data-dependent | 2 | `unknown` | — |

## UCI binary datasets

| Task | Formula / source | Variables | Max factors | Representability | Expected structures |
|---|---|---:|---:|---|---|
| `uci_adult` | `UCI Adult income binary classification` | data-dependent | 2 | `unknown` | — |
| `uci_spambase` | `UCI Spambase binary classification` | data-dependent | 2 | `unknown` | — |
| `uci_gamma` | `UCI MAGIC Gamma Telescope binary classification` | data-dependent | 2 | `unknown` | — |
| `uci_diabetes_binary` | `UCI CDC diabetes binary classification` | data-dependent | 2 | `unknown` | — |

## Data ranges and sources

Synthetic raw ranges are part of each `TaskSpec`. The most important domain constraints are: powered reciprocal tasks use bounded ranges chosen so their denominators remain safely away from zero; `pykan_singularity` uses positive inputs for logarithms; fuzzy membership variables use `[0,1]`; Feynman-style rational tasks use positive denominator variables. Exact ranges are in `benchmarks/specs.py`.

Real tasks are loaded as follows:

- `sklearn_diabetes`: scikit-learn diabetes regression dataset;
- `sklearn_breast_cancer`: scikit-learn Wisconsin breast-cancer binary dataset;
- `uci_adult`: UCI repository id 2;
- `uci_spambase`: UCI repository id 94;
- `uci_gamma`: UCI repository id 159;
- `uci_diabetes_binary`: UCI repository id 891.

UCI categorical columns are one-hot encoded; rows containing missing values are removed. Binary labels are mapped to `0/1` after preprocessing.

## Representability labels

`in_class` means the declared target is representable by the flat RuleKAN factor vocabulary and the task product order, assuming the needed operator library is enabled. `expression_power_exact` identifies targets that are exact members of the PowerRuleKAN outer-power grammar. `nested_unary`, `nested_cross`, `nested_factorizable`, `nested_rational`, `mixed` and the fuzzy-specific labels describe controlled departures or structured subfamilies. `unknown` is used for real data without a known generating formula.

The representability label is benchmark metadata. It does not participate in RuleKAN or PowerRuleKAN search or validation selection.
