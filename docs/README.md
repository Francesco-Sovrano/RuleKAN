# RuleKAN: Addressing the Symbolic Expressivity Gap in KANs for Symbolic Regression

RuleKAN separates the discovery of variable interactions from the symbolic factorization of those interactions. The separation is required because a flexible one-dimensional KAN edge can absorb symbolic structure that is not recoverable by replacing that edge with one primitive. Typical cases include several analytic factors of the same input, cross-variable interactions, fuzzy gate-and-branch decompositions, operators applied to a complete expression, and nested analytic composition.

Two related symbolic searches are implemented. **RuleKAN** conditions its candidate structures on variable supports learned by a KAN-derived numerical model. **RuleSISP** (Structure-Independent Symbolic Pursuit) uses the same sum-product symbolic language but enumerates all variable multisets through a bounded factor order. Signed integer powers and one additional composition level provide bounded extensions for expression scope and depth.

## Symbolic model

For inputs \(x=(x_1,\ldots,x_d)\), the Stage-2 symbolic language is

\[
f_{\mathrm{SP}}(x)=b+\sum_{r=1}^{R}a_r
\prod_{s=1}^{m_r}\psi_{k_{rs}}(\beta_{rs}x_{j_{rs}}+\gamma_{rs}),
\qquad m_r\le q.
\]

`b` is the global bias, `a_r` is the amplitude of additive term `r`, `m_r` is its number of factors, `j_rs` selects an input variable, and `psi_k` is a discrete analytic primitive. The affine chart `(beta, gamma)` is fitted continuously.

A variable may occur several times in the same product. For example, a numerical Stage-1 edge may approximate \(h(x)\approx e^{-0.7x}\sin(2.4x)\), while Stage 2 receives only support `{x}` and can recover the two factors separately. Several additive symbolic terms may also use the same support.

A two-branch fuzzy rule

\[
(1-g)u+gv
\]

already belongs to the sum-product language. Complementary gate parameterization, joint refitting, and bounded partition rescue retain explicit gate and branch roles when selected by validation.

## Three-stage procedure

### Stage 1 - variable interaction discovery

The numerical model is a sum of product terms

\[
f_{\mathrm{num}}(x)=b+\sum_{r=1}^{W}\rho_r a_r
\prod_{s=1}^{q}\left[1-m_{rs}+m_{rs}h_{rs}(x_{j_{rs}})\right],
\]

where `h_rs` is a learned spline or Gaussian-RBF univariate function, `rho_r` retains or removes a complete numerical term, and `m_rs` retains or removes one factor slot. Training uses relaxed gates followed by hardening, refitting, and iterative pruning.

The primary output of Stage 1 is a support bank containing the distinct variables used by retained numerical terms. Numerical edge shapes are not copied into the symbolic model. High-recall pre-pruning evidence is retained for validation-gated support augmentation.

### Stage 2 - sum-product factorization

For each support, RuleKAN enumerates admissible variable multisets through order `q`. A support `{x,y}` with `q=3`, for example, permits `(x,y)`, `(x,x,y)`, and `(x,y,y)`. This repeated-variable expansion separates numerical interaction discovery from symbolic factor count.

Operator choice is handled by relaxed GMP screening followed by hard operator selection and continuous refitting. Complete candidate terms are evaluated in the current additive context using GSR-style matching pursuit or configured OMP variants. Additive-term expansion permits several symbolic terms with the same variable multiset.

RuleSISP skips Stage-1 support conditioning and enumerates every variable multiset through order `q`:

\[
|\mathcal G_S|=\binom{d+q}{q}-1.
\]

RuleKAN restricts the primary grammar to learned support classes and therefore reduces the structure search before analytic primitives are assigned.

### Stage 3 - complete-expression operators

`PowerRuleKAN` searches

\[
f(x)=b+\sum_{r=1}^{R_o}a_r\prod_t B_{rt}(x)^{p_{rt}},
\qquad p_{rt}\in D\subset\mathbb Z\setminus\{0\},
\]

where each `B_rt` is a complete Stage-2 symbolic base. Negative powers represent reciprocals and ratios; positive powers compactly represent whole-expression powers under bounded term and factor limits.

Composition variants permit one additional analytic function around selected bounded bases,

\[
C(x)=a\,g(\beta B(x)+\gamma)+d.
\]

This covers structures such as `sin(1 + x*y)` or `exp(x*y)` without unrestricted recursive tree search.

## Expressivity boundaries

The method separates four distinct constraints:

| Constraint | Failure mode | Mechanism |
|---|---|---|
| factorization | one learned univariate edge absorbs several analytic factors | repeated-variable Stage-2 factor search |
| interaction coverage | Stage 1 does not retain a required variable support | RuleSISP or validation-gated support augmentation |
| scope | an operator must act on a complete multivariate expression | signed integer powers and ratios |
| depth | a recovered expression must become the input to another analytic function | one bounded composition level |

Predictive error and structural recovery are separate quantities. The fuzzy benchmark therefore scores NRMSE and explicit gate/branch recovery independently.

## Model identifiers

| Identifier | Definition |
|---|---|
| `rulekan` | spline Stage 1 + support-conditioned Stage 2 |
| `rulekan_fast` | Gaussian-RBF Stage 1 + support-conditioned Stage 2 |
| `rulekan_adaptive` | RuleKAN with validation-gated high-recall support augmentation |
| `rulekan_omp_full` | RuleKAN with OMP-style Stage-2 term selection |
| `sisp` | support-independent Stage-2 variable-multiset search |
| `rulekan_comp` | RuleKAN plus one composition level |
| `sisp_comp` | SISP plus one composition level |
| `power_rulekan` | signed integer powers, reciprocals, and ratios of RuleKAN bases |
| `power_rulekan_comp` | PowerRuleKAN with composed bases permitted |

## Controlled analytic benchmark

The analytic benchmark contains 30 noise-free tasks over three seeds: 6 product tasks, 7 fuzzy tasks, 4 power/ratio tasks, 5 nested tasks, 3 canonical KAN targets, and 5 physics-style targets. Synthetic splits contain `1600/400/500` train/validation/test observations. Maximum factor order is `q=3`.

The common controlled vocabulary contains ten elementary primitives:

```text
x, x^2, 1/x, 1/x^2, sqrt, log, exp, sin, cos, tanh
```

Each primitive acts on an affine argument. Compound shortcuts such as Gaussian or `log(1+x^2)` are not members of this vocabulary. The benchmark therefore tests whether the model class can construct required scope and composition rather than receiving compound target fragments as direct atoms.

The 23 main comparison methods are the nine RuleKAN/RuleSISP variants plus 14 external symbolic-regression or KAN-based baselines. The exact runnable model identifiers and suite selection are given in [Benchmark protocol](benchmarks/protocol.md).

## Core terminology

| Term | Definition |
|---|---|
| **numerical term** | one additive product term in Stage 1 |
| **symbolic term / rule** | one additive product term in Stage 2 |
| **factor** | one multiplicative univariate component of a term |
| **support** | set of distinct input variables used by a term |
| **variable multiset** | ordered-independent factor-variable pattern that retains repeated occurrences |
| **multiplicity** | number of occurrences of each input variable in one symbolic term |
| **support rank** | number of independently parameterized additive symbolic terms using one support or multiset |
| **support bank** | collection of support classes available to a support-conditioned symbolic search |
| **effective support bank** | support bank selected after configured validation-gated augmentation |
| **GMP** | differentiable mixture screening used to shortlist analytic primitives |
| **GSR** | greedy symbolic regression / matching pursuit over complete symbolic terms |
| **SISP** | Structure-Independent Symbolic Pursuit over all variable multisets through order `q` |

## Reference index

| Subject | File |
|---|---|
| Installation, tests, examples, and benchmark execution | [Getting started](getting-started.md) |
| Model equations and bounded representability | [Model class](theory/model-class.md) |
| KAN relationship and architectural structure | [RuleKAN architecture](theory/design-rationale.md) |
| Support identifiability, repeated variables, and rank | [Identifiability and support](theory/identifiability-and-support.md) |
| Stage-1 numerical fitting, gates, pruning, and support evidence | [Numerical training](algorithms/numerical-training.md) |
| Stage-2 GMP, hard proposals, GSR/OMP, RuleKAN, and SISP | [Symbolic search](algorithms/symbolic-search.md) |
| Interaction-surface proposals and block pursuit | [Interaction proposals](algorithms/interaction-proposals.md) |
| Complementary fuzzy partitions | [Affine partitions](algorithms/affine-partitions.md) |
| Stage-3 integer powers, reciprocals, and ratios | [PowerRuleKAN](algorithms/power-rulekan.md) |
| Stage-3 bounded composition | [Composition](algorithms/composition-rescue.md) |
| Search-space and runtime scaling | [Complexity](complexity.md) |
| Benchmark task catalogue | [Tasks](benchmarks/tasks.md) |
| Benchmark methods | [Methods](benchmarks/methods.md) |
| Splits, shared controls, profiles, execution, and reproducibility | [Protocol](benchmarks/protocol.md) |
| Predictive and structural metrics | [Metrics and figures](benchmarks/metrics-and-figures.md) |
| Component and OFAT ablations | [Ablations](benchmarks/ablations.md) |
| Hardware and process parallelism | [Hardware and parallelism](benchmarks/hardware-and-parallelism.md) |
| Configuration schema | [Configuration](reference/configuration.md) |
| Numerical RuleKAN configuration | [Numerical configuration](reference/configuration-numerical.md) |
| Symbolic-search configuration | [Symbolic configuration](reference/configuration-symbolic.md) |
| PowerRuleKAN configuration | [Power configuration](reference/configuration-power.md) |
| Baseline configuration | [Baseline configuration](reference/configuration-baselines.md) |
| Symbolic operator libraries | [Symbolic library](reference/symbolic-library.md) |
| Python API | [Python API](reference/python-api.md) |
| Source tree | [Source layout](reference/source-layout.md) |
| Definitions | [Glossary](glossary.md) |
| Literature context | [Related work](related-work.md) |
| Bibliography | [References](references.md) |
