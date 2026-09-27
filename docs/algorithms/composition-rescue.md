# Stage 3: bounded composition

Flat RuleKAN represents a sum of products of univariate symbolic factors,

\[
f(x)=b+\sum_r a_r\prod_s g_{rs}(x_{j_{rs}}),
\]

so increasing the number of rules increases additive rank but does not introduce functional composition. Expressions such as `tanh(sin(x))`, `sqrt(x0^2+x1^2)`, or `exp(sin(x0)+x1^2)` require an outer symbolic operator applied to an inner symbolic expression.

`RuleKAN-Comp`, `SISP-Comp`, and `PowerRuleKAN-Comp` use a validation-gated depth-2 symbolic rescue. Ordinary RuleKAN/SISP is fitted first and remains the incumbent. The rescue is attempted only when the incumbent validation NRMSE exceeds `symbolic_composition_trigger_nrmse`.

## Grammar

One composition atom has the form

\[
C(x)=a\,g\!\left(\beta\left[c_0+\sum_{r=1}^{R_i} c_r
\prod_s h_{rs}(\beta_{rs}x_{j_{rs}}+\gamma_{rs})\right]+\gamma\right)+d.
\]

The current rescue searches three small inner topologies:

- one unary symbolic factor;
- one two-variable product rule;
- a sum of two unary rules.

The outer operator is selected from a small real-valued symbolic family such as `sin`, `cos`, `tanh`, `exp`, and `sqrt`. Inner operators are drawn from the configured RuleKAN library.

For RuleKAN-Comp and PowerRuleKAN-Comp, the union of variables in a composition atom must be one of the validation-selected effective RuleKAN support classes. Ratio numerator/denominator composition searches are further restricted to the support classes proposed for that base. SISP-Comp uses the structure-independent variable grammar.

## Family-diverse search

A coarse screen ranks discrete outer/inner operator families. The refinement beam is not formed by prediction error alone. It also retains:

- all same-variable unary-composition families;
- low-complexity two-variable product families;
- low-complexity sums of unary factors such as `sin(x0)+cos(x1)` and `x0^2+x1^2`.

The beam preserves recursive unary families and low-complexity inner structures during continuous refinement. Affine-partition search uses the same delayed family-pruning scheme.

Each retained family receives Adam refinement followed by LBFGS polishing of continuous affine parameters, inner coefficients, and outer scale/bias. A smaller validation-selected set receives a deeper polish.

## Flat correction rule

A pure composition atom does not cover expressions such as

\[
\sin(x_0x_1)+0.2x_2^2.
\]

For the strongest composition candidates, the rescue can fit one additional flat unary RuleKAN correction to the composition residual and jointly repolish the combined expression. The correction is restricted by the same RuleKAN support contract; SISP-Comp remains structure-independent.

## Validation gate

The incumbent is replaced only when the composed candidate improves validation MSE by at least `symbolic_composition_min_improvement_rel`. Test coordinates and labels do not participate in candidate construction or selection.

The flat model remains available during validation selection. Composition screening and refinement add compute only on cases that cross the trigger.

## Relation to KAN depth

Widening a KAN/MultKAN layer increases the number of units but does not add composition depth. Adding another entry to the width vector adds another functional layer and therefore can represent deeper composition. RuleKAN's `n_rules` is analogous to width: it increases the number of additive product mechanisms, not the depth of the symbolic expression. `RuleKAN-Comp` adds one explicit symbolic composition level without changing the numerical precursor depth.


## PowerRuleKAN-Comp

`PowerRuleKAN-Comp` applies the same rescue inside the powered search rather than only to the final `p=1` fallback. The ordinary RuleKAN base, transformed-target bases, and ratio numerator/denominator bases may each become a `ComposedRuleKAN` when validation improves. Integer-power and ratio polishing then optimizes only continuous parameters of the already fixed symbolic topology. Composition uses the selected effective support bank and does not perform unrestricted support enumeration.
