# Architecture and design

## Baseline KAN structure

A KAN layer places a trainable univariate function on each active edge. For a standard layer, node outputs are sums of transformed inputs. MultKAN extends this representation with multiplication nodes, allowing products to appear inside the network topology.

RuleKAN uses KAN-style trainable univariate functions as numerical factors but organizes them into an explicit sparse sum-product model. The architectural distinction is the structural unit used for learning, pruning, and symbolic extraction: RuleKAN operates on complete product rules and their variable supports rather than treating individual edges as independent symbolic targets.

## RuleKAN model

For input \(x\in\mathbb R^d\), the numerical model has the form

\[
f_\theta(x)=b+\sum_{r=1}^{R}a_r\,z_r(x),
\qquad
z_r(x)=\prod_{j=1}^{m_r}\phi_{rj}(x_{v_{rj}}),
\]

where each \(\phi_{rj}\) is a learned one-dimensional spline or Gaussian-RBF function. Continuous rule, factor, and variable-selection gates determine which structures remain active during training.

After numerical fitting, the symbolic model uses the same additive rule form but replaces numerical factors with parameterized analytic functions

\[
g(\beta x+\gamma).
\]

Rule amplitudes absorb multiplicative factor scale, leaving operator family and affine input chart as the primary factor parameters.

## Structural additions relative to KAN

RuleKAN introduces the following mechanisms on top of the univariate-function substrate used by KAN:

1. **Explicit sum-product rule bank.** Product rules are represented directly rather than inferred only from general network topology.
2. **Rule- and factor-level sparsity.** Differentiable gates support structural optimization before hard pruning.
3. **Validation-aware structural pruning.** Candidate deletions are followed by local refitting and accepted under configured validation-error budgets.
4. **Pre-pruning support capture.** Variable-support evidence is recorded before destructive pruning.
5. **Support-conditioned symbolic grammar.** Symbolic search is restricted to distinct-variable supports evidenced by the numerical stage.
6. **Multiplicity expansion.** A support may generate symbolic products containing repeated occurrences of the same variable.
7. **Support-rank expansion.** Several additive symbolic rules may be allocated to one support.
8. **Whole-rule symbolic pursuit.** Symbolic candidates are complete products evaluated in the context of the current additive model.
9. **Optional support rescue and composition.** Validation may select a wider support bank or one licensed depth-2 symbolic composition.
10. **Optional outer powers.** PowerRuleKAN applies positive and negative integer powers to complete symbolic bases.

Multiplication itself is not unique to RuleKAN; MultKAN already provides multiplication nodes. RuleKAN's specific restriction is that the numerical rule bank defines admissible symbolic variable supports and that symbolic search operates on complete rules inside those supports.

## Numerical support before symbolic factorization

The numerical precursor is used to identify variable participation and interaction support without requiring each learned edge function to already match a simple analytic primitive. This separation avoids forcing discrete symbolic choices during the early function-fitting phase.

Numerical pruning can remove a support after other rules have adapted around it. RuleKAN therefore stores high-recall support evidence before pruning. The ordinary symbolic path uses the final active support bank. `RuleKAN Adaptive` can use the pre-pruning evidence when validation indicates that the primary symbolic model underfits.

## Same-variable factorization

A numerical univariate edge can approximate a composite one-dimensional function such as

\[
h(x)=e^{-0.7x}\sin(2.4x).
\]

The numerical observation of one function of \(x\) does not identify whether the generating expression contains one symbolic factor or several symbolic factors of the same variable. Numerical same-variable factorization is therefore not structurally identifiable from function values alone.

RuleKAN records the support as `{x}` and handles repeated factors during symbolic grammar construction. A support containing one distinct variable may generate multiplicity candidates such as `(x)`, `(x,x)`, and higher repeated-variable tuples up to the configured order. This preserves the numerical support restriction while allowing the symbolic factorization to be richer than the numerical factor count.

## Support rank

A single variable support can also contain several additive mechanisms. For example,

\[
0.3\tanh(2x)+0.55e^{-0.6x}\sin(2.5x)
\]

contains two additive terms with the same distinct-variable support `{x}`. RuleKAN therefore separates support identity from support rank. Symbolic rank continuation can allocate several independently parameterized rules to the same support when validation improves.

## Affine symbolic factors

Symbolic factors use affine input charts,

\[
g(\beta x+\gamma),
\]

rather than fixed raw operators. This is required for frequency, phase, shift, and scale parameters. It also introduces gauge equivalences: sine and cosine can differ by phase, even operators can identify opposite input slopes, odd operators can exchange sign with rule scale, and exponential shifts can exchange scale with the rule amplitude.

Canonicalization and multi-start affine initialization reduce dependence on equivalent parameterizations. Data-coordinate-aware initializations are included because benchmark inputs are standardized before fitting.

## Whole-rule symbolic pursuit

Symbolic search is performed on complete candidate products. GMP screens operator tuples, after which hard operator proposals and learned-support GSR evaluate candidates against the current residual. Continuous affine parameters and rule scales are refit during selection and backfitting.

This produces a sparse-pursuit problem over a structured dictionary. Each atom is a parameterized product of univariate symbolic functions, and its admissible variable support comes from the numerical support bank.

`SISP` removes the learned-support restriction while retaining the same flat sum-product symbolic language. Its grammar enumerates variable multisets up to the configured product order.

## Effective support contract

RuleKAN exposes one validation-selected support source for downstream symbolic procedures:

```text
symbolic_effective_support_classes
symbolic_effective_support_bank
symbolic_effective_support_source
```

The ordinary primary support bank occupies this contract unless validation selects an adaptive high-recall rescue. PowerRuleKAN, affine-partition recovery, and other support-conditioned symbolic procedures consume the same effective support bank.

## Complementary affine partitions

A conditional expression such as

\[
(1-u)f_0+u f_1
\]

can be algebraically expanded into forms that have the same prediction error but hide the complementary gate pair. The affine-partition procedure searches compatible support-conditioned rule pairs and fits complementary identity charts directly. It does not require a separate `1-x` symbolic operator because both `x` and `1-x` are affine instances of the identity family.

## Depth-2 symbolic composition

Flat RuleKAN represents sums of products of univariate symbolic factors. Some targets require composition of a symbolic outer function with a multivariate inner RuleKAN expression. The composition-rescue variant permits one licensed depth-2 form and accepts it only when validation error improves according to the configured gate.

The composition rescue changes symbolic depth, not the variable-support source. The inner expression remains built from the admissible support bank.

## PowerRuleKAN outer grammar

PowerRuleKAN extends a RuleKAN base expression with integer exponents:

\[
f(x)=b+\sum_r a_r\prod_t B_{rt}(x)^{p_{rt}},
\qquad p_{rt}\in\mathbb Z\setminus\{0\}.
\]

Each base \(B_{rt}\) is a complete support-conditioned RuleKAN expression. Negative powers are admitted only when reciprocal-domain checks pass. Ratio candidates combine positive- and negative-power bases. The ordinary RuleKAN solution is retained as an explicit `p=1` candidate.

## Data separation

Training data fit numerical and continuous symbolic parameters. Validation data control checkpointing, pruning, symbolic candidate selection, support rescue, composition rescue, powered-expression selection, and reciprocal-domain admissibility. Test data are used only after the model has been fixed.

Known generating formulas, representability labels, and expected fuzzy-rule signatures are evaluation metadata. They are not inputs to model search.
