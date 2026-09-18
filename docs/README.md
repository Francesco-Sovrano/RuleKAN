# RuleKAN documentation

## Overview

RuleKAN learns symbolic models in two stages:

1. a numerical sum-product network discovers sparse variable supports and interaction structure;
2. a symbolic search replaces the numerical factors with analytic functions while remaining inside the variable supports evidenced by the numerical stage.

For an input vector \(x=(x_1,\ldots,x_d)\), a symbolic RuleKAN model is

\[
\hat y(x)=b+\sum_{r=1}^{R} a_r\prod_{j=1}^{m_r} g_{rj}(x_{v_{rj}}),
\]

where \(b\) is a global bias, \(a_r\) is a rule amplitude, and each \(g_{rj}\) is a univariate symbolic factor with a fitted affine input chart. A rule may contain several factors, including repeated uses of the same input variable when symbolic multiplicity expansion is enabled.

The numerical precursor uses the same sum-product structure with trainable one-dimensional edge functions. Gates control whether rules and factors remain active. Numerical pruning, validation-based refitting, and pre-pruning support capture produce the structural evidence used by the symbolic stage.

## Relation to KAN

A Kolmogorov-Arnold Network (KAN) represents transformations with learnable univariate functions on edges. MultKAN variants can also contain multiplication nodes. RuleKAN does not define its contribution as adding multiplication to KAN. It changes the unit of structural discovery and symbolic extraction.

| KAN / MultKAN symbolic pipeline | RuleKAN |
|---|---|
| Univariate edges are the primary learned symbolic units. | Complete product rules are the primary structural units. |
| Multiplication may be present as network nodes. | Multiplicative rule structure is explicit in a sparse sum-of-products bank. |
| Edge pruning determines local connectivity. | Rule and factor gates are pruned with validation-aware structural refitting. |
| Symbolic replacement is performed on individual learned functions or edges. | Symbolic candidates are complete products evaluated in the context of the current additive model. |
| A numerical edge corresponds to one numerical univariate function. | A numerical support may expand to repeated symbolic factors of the same variable. |
| Network topology determines available interactions. | Learned variable supports define the admissible symbolic grammar. |

RuleKAN adds the following mechanisms on top of the KAN edge-function substrate:

- **sum-product rule bank:** sparse additive rules containing multiplicative univariate factors;
- **differentiable rule and factor gates:** continuous structural optimization before hardening;
- **pre-pruning support evidence:** high-recall variable-support information retained before destructive pruning;
- **support collapse:** numerical rules are converted to distinct-variable support classes;
- **multiplicity expansion:** symbolic search may use a variable more than once within an evidenced support;
- **support-rank expansion:** several additive symbolic rules may share the same support;
- **GMP operator proposals:** gated matching-pursuit screening proposes symbolic factor families;
- **hard symbolic proposals and learned-support GSR:** complete symbolic rules are evaluated and committed using validation data;
- **symbolic backfitting and cleanup:** continuous parameters of committed rules are jointly refined;
- **validation-gated support rescue:** `RuleKAN Adaptive` may restore supports captured before numerical pruning;
- **depth-2 symbolic composition rescue:** `RuleKAN-Comp` may replace a flat symbolic base with one licensed composition when validation improves;
- **outer integer-power grammar:** `PowerRuleKAN` applies positive or negative integer powers to complete symbolic RuleKAN bases.

The symbolic support restriction is the central contract: symbolic search may change operator identity, factor multiplicity, and additive rank inside an evidenced support, but ordinary RuleKAN does not introduce a new distinct-variable support that was absent from the numerical evidence.

## End-to-end procedure

### Numerical support discovery

`SumProductKAN` trains an overcomplete bank of rules. Each rule has factor slots, variable-selection gates, and spline or Gaussian-RBF univariate functions. Training uses staged regularization, structural hardening, optional iterative pruning, and validation-based checkpointing.

Before pruning, RuleKAN records high-recall support evidence. After pruning and hardening, the active numerical rules define the primary support classes.

### Symbolic grammar construction

Numerical rules are collapsed to supports containing distinct variable indices. Symbolic grammar construction may then expand each support in two ways:

- **multiplicity:** repeated occurrences of a variable, such as support `{x0}` producing factors on `(x0,x0)`;
- **rank:** more than one additive symbolic rule assigned to the same support.

This separation is required because a single numerical univariate function can approximate a product of several analytic functions of the same variable.

### Symbolic proposal and pursuit

Each symbolic factor has the form

\[
g(\beta x+\gamma),
\]

with a discrete operator family \(g\) and fitted affine parameters \(\beta,\gamma\). The configured symbolic library supplies the operator families.

GMP screens operator combinations. Hard exact-operator proposals, learned-support greedy symbolic regression, and optional OMP variants evaluate complete rule candidates against the current residual. Continuous rule scales and affine parameters are refit after candidate selection. Validation error controls selection and cleanup.

### Optional structural extensions

`RuleKAN Adaptive` uses pre-pruning support evidence when the primary support bank underfits validation data. `RuleKAN-Comp` adds one depth-2 symbolic composition under a validation gate. Affine-partition recovery refactors compatible rules into complementary gates for fuzzy conditional structure. `PowerRuleKAN` adds integer powers, reciprocals, and ratios of complete support-conditioned symbolic bases.

## Model families

| Model | Structural source | Symbolic class |
|---|---|---|
| `rulekan` | learned numerical supports | sums of products of univariate symbolic factors |
| `rulekan_fast` | learned supports from Gaussian-RBF numerical edges | same as `rulekan` |
| `rulekan_adaptive` | learned supports plus validation-selected pre-pruning support rescue | same as `rulekan` |
| `rulekan_comp` | learned supports | RuleKAN plus one validation-gated depth-2 composition |
| `sisp` | complete variable-multiset grammar | same flat symbolic factor language without learned-support restriction |
| `sisp_comp` | complete variable-multiset grammar | SISP plus depth-2 composition rescue |
| `power_rulekan` | canonical effective RuleKAN supports | integer powers and ratios of complete RuleKAN bases |
| `power_rulekan_comp` | canonical effective RuleKAN supports | powered/ratio grammar with composed symbolic bases allowed |

## Benchmark configuration

`benchmarks/configs/default.yaml` is the executable benchmark definition. The principal `research` profile has:

- seeds `0,1,2`;
- 32 tasks across fuzzy rules, synthetic core, powered-expression stress, nested-expression stress, canonical KAN examples, Feynman-style equations, and two small real datasets;
- 19 methods;
- synthetic train/validation/test sizes `1600/400/500`;
- per-job timeout `2400` seconds;
- shared numerical width `W=12` and grid `12`;
- task-specific maximum product order;
- the 14-operator `target_core` symbolic library;
- a symbolic rule budget at least as large as the shared width.

The `ablation` profile extends `research`. Its RuleKAN and RuleKAN-RBF controls therefore use exactly the same numerical, symbolic, data, timeout, and shared-capacity settings as `research`; only the selected models and tasks differ.

The `width_sensitivity` profile is a separate sensitivity configuration. It uses the same seeds and data sizes as `research` and sweeps `W={6,8,10,12,16,24}`, but it uses a longer 5400-second timeout and a reduced RuleKAN optimization schedule. Its `W=12` condition is therefore a width-sensitivity condition, not an exact reproduction of the `research` RuleKAN baseline.

## Core terminology

| Term | Meaning |
|---|---|
| **rule** | one additive term in the sum-product model |
| **factor** | one multiplicative univariate term inside a rule |
| **support** | set of distinct variables used by a rule |
| **multiplicity** | number of occurrences of each variable inside one symbolic rule |
| **support rank** | number of independently parameterized additive symbolic rules assigned to one support |
| **support collapse** | conversion of numerical factor assignments to distinct-variable support classes |
| **effective support bank** | validation-selected support source used by downstream RuleKAN symbolic procedures |
| **GMP** | gated matching pursuit used for symbolic operator proposal/screening |
| **GSR** | greedy symbolic regression used to build complete symbolic rules |
| **SISP** | structure-independent symbolic pursuit over the complete variable-multiset grammar |

## Documentation index

| Topic | Document |
|---|---|
| Installation, tests, first benchmark | [Getting started](getting-started.md) |
| Terminology | [Glossary](glossary.md) |
| Model equations and representability | [Model class](theory/model-class.md) |
| Architectural choices and KAN relationship | [Architecture and design](theory/design-rationale.md) |
| Support identifiability, collapse, multiplicity and rank | [Identifiability and support](theory/identifiability-and-support.md) |
| Numerical optimization, gates, pruning and support evidence | [Numerical training](algorithms/numerical-training.md) |
| Symbolic GMP/GSR, proposals, SISP and adaptive rescue | [Symbolic search](algorithms/symbolic-search.md) |
| Interaction-surface proposals | [Interaction proposals](algorithms/interaction-proposals.md) |
| Complementary affine partitions | [Affine partitions](algorithms/affine-partitions.md) |
| Depth-2 symbolic composition | [Composition rescue](algorithms/composition-rescue.md) |
| Integer powers, reciprocals and ratios | [PowerRuleKAN](algorithms/power-rulekan.md) |
| Time and space complexity | [Complexity](complexity.md) |
| Benchmark profiles, data splits and execution | [Benchmark protocol](benchmarks/protocol.md) |
| Component and OFAT ablations | [Ablations](benchmarks/ablations.md) |
| Benchmark methods | [Benchmark methods](benchmarks/methods.md) |
| Benchmark tasks | [Task catalog](benchmarks/tasks.md) |
| Metrics and generated outputs | [Metrics and figures](benchmarks/metrics-and-figures.md) |
| Hardware and parallel execution | [Hardware and parallelism](benchmarks/hardware-and-parallelism.md) |
| Benchmark configuration | [Configuration reference](reference/configuration.md) |
| RuleKAN numerical configuration | [Numerical configuration](reference/configuration-numerical.md) |
| RuleKAN symbolic configuration | [Symbolic configuration](reference/configuration-symbolic.md) |
| PowerRuleKAN configuration | [Power configuration](reference/configuration-power.md) |
| Baseline configuration | [Baseline configuration](reference/configuration-baselines.md) |
| Symbolic operator libraries | [Symbolic library](reference/symbolic-library.md) |
| Python classes and functions | [Python API](reference/python-api.md) |
| Repository modules | [Source layout](reference/source-layout.md) |
| Literature context | [Related work](related-work.md) |
| Bibliographic entries | [References](references.md) |
