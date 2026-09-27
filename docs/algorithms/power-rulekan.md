# Stage 3: PowerRuleKAN

## Model

PowerRuleKAN represents

\[
f(x)=b+\sum_{r=1}^{R_o} a_r\prod_t B_{rt}(x)^{p_{rt}},
\qquad p_{rt}\in D\subset\mathbb Z\setminus\{0\},
\]

where every `B_rt` is a hard fully symbolic RuleKAN base. In `PowerRuleKAN-Comp`, a base may instead be a validation-selected `ComposedRuleKAN` with one explicit depth-2 symbolic composition level. Positive powers represent repeated powers of a complete symbolic subexpression; negative powers represent reciprocals. Zero is unnecessary because omission already supplies the multiplicative identity.

Examples include

\[
(1+0.8x_0x_1)^{-1},
\qquad
(0.8+0.4x_0+0.3x_1)^2,
\]

and

\[
(0.7+0.4x_0)(1+0.6x_0x_1)^{-1}.
\]

## Canonical support inheritance

PowerRuleKAN does not discover an independent variable-support graph. Its internal RuleKAN precursor is trained without exposing the benchmark test split: validation data occupy the precursor's reporting-only held-out slots, and the true test split is loaded only after the complete powered expression has been selected. It reads RuleKAN's validation-selected

```text
symbolic_effective_support_classes
symbolic_effective_support_bank
symbolic_effective_support_source
```

through a central accessor. Every transformed-power base, ratio numerator and ratio denominator is fitted with `learned_support_symbolic_gsr` against that same support contract. Multiplicity may expand inside the learned supports, and affine-partition refactoring may occur inside them, but an unsupported variable combination remains forbidden.

## Compositional variant

`power_rulekan_comp` enables the depth-2 composition rescue for the ordinary `p=1` base and for each transformed-power or ratio base. Composition candidates are generated only inside the canonical effective RuleKAN support classes; ratio numerator and denominator searches remain narrowed to their pilot-proposed support subsets. If validation does not improve, the flat base remains the incumbent.

After selection, powered and ratio polish freezes the discrete composition topology and optimizes its continuous affine/scalar parameters. This permits expressions such as powers or reciprocals of `g(h(x))` without turning PowerRuleKAN into an unrestricted expression-tree search.

## Ordinary `p=1` candidate

The already-fitted ordinary RuleKAN symbolic model is inserted directly as a `p=1` atom. The powered search therefore contains the ordinary solution explicitly rather than relying on a larger recursive optimizer to rediscover it.

## Transformed-target initialization

For a fixed integer power `p`, PowerRuleKAN first seeks a base `B` satisfying

\[
B(x)^p\approx y.
\]

When a real and numerically safe inverse transformation exists, `inverse_power_target` constructs a transformed target. Examples are

\[
p=-1:\quad B_{target}=1/y,
\]

\[
p=2:\quad B_{target}=\sqrt{y},
\]

and, where positive-domain conditions hold,

\[
p=-2:\quad B_{target}=\sqrt{1/y}.
\]

Transformations that would require unsupported complex values or violate configured epsilon conditions are skipped.

## Hard-before-polish

A transformed-target base is discretized before optimization returns to the original target. Discrete support, variable and operator choices are frozen; continuous symbolic affine parameters and outer linear scale/bias are polished. This separates structural discovery from continuous adjustment under a changed objective.

## Ratio pilot

A ratio term is represented as

\[
A(x)B(x)^{-1}.
\]

To identify plausible numerator/denominator supports cheaply, the benchmark constructs a feature bank `Phi` from RuleKAN contributions and learned-support monomials, fixes the denominator constant to one, and uses

\[
A=a_0+\Phi a,\qquad B=1+\Phi b.
\]

The quotient relation is linearized as

\[
a_0+\Phi a-y\Phi b\approx y.
\]

A ridge solve ranks support evidence for numerator and denominator bases. The pilot proposes structure; it is not exported as the final expression.

## Ratio-aware symbolic initialization

The denominator is fitted symbolically first. With a hard denominator `B`, the numerator target becomes

\[
A_{target}=yB.
\]

After both bases are hard, `hard_ratio_polish` first minimizes the cross-multiplied residual

\[
L_{cross}=\|A-yB\|_2^2+\lambda\,\mathrm{barrier}(B),
\]

then the actual quotient residual

\[
L_{quot}=\|aA/B+b-y\|_2^2+\lambda\,\mathrm{barrier}(B).
\]

This avoids starting direct quotient optimization with an unstable denominator.

## Reciprocal-domain safety

Negative powers use `safe_integer_power`. Reciprocal admissibility is part of model selection, so candidate bases are checked only on the training and validation coordinates. `power_reciprocal_margin` determines the required minimum absolute base value, and barrier terms penalize bases approaching zero during training. The test split is not consulted until the powered expression has been selected.

After selection, the benchmark reports the reciprocal margin on the untouched test coordinates as a generalization diagnostic. A test-set domain violation is therefore recorded as an evaluation outcome; it cannot cause the trainer to replace or reject the selected model.

## Outer powered pursuit

Each complete powered expression is treated as an atom. The bank can contain:

- the ordinary RuleKAN `p=1` atom;
- transformed single-power atoms;
- ratio/product-of-bases atoms.

Validation-scored OMP selects up to `power_outer_rules` atoms and jointly refits their linear outer coefficients. The final model can therefore be a sum of several powered mechanisms rather than a single powered term.

## Formula export and diagnostics

`PowerRuleKAN.symbolic_formula` recursively exports its bases and integer powers. `chosen_powers()` returns the selected exponent pattern, and `reciprocal_domain_margin(x)` reports the smallest absolute value of every negative-power base on supplied data.
