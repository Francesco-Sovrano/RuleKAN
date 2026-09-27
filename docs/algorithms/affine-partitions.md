# Affine partitions and fuzzy-rule recovery

## Partition form

A two-branch fuzzy conditional with membership `u` can be written

$$
(1-u)f_{0}(x)+u f_{1}(x).
$$

The identity symbolic family already represents both gates. If `u=\beta z+\gamma`, exact complementarity is

$$
1-u=-\beta z+(1-\gamma).
$$

No separate `1-x` operator is required.

## Algebraic non-uniqueness

The same conditional can be written distributively, for example

$$
f_0(x)-u f_0(x)+u f_1(x).
$$

A generic additive symbolic search can fit this expression nearly perfectly while obscuring the two complementary branches. Prediction loss alone therefore does not uniquely determine the partition representation.

Same-variable fuzzy rules introduce an additional identifiability issue: both the gate and branch may depend on the same input, so one flexible numerical edge may absorb several symbolic factors. The support-collapse/multiplicity-expansion mechanism supplies the repeated-variable symbolic structures; the partition rescue supplies the canonical complementary decomposition.

## Structure-conditioned opportunity detection

`_affine_partition_structure_pairs` examines only order-2 symbolic structures already present in the admissible RuleKAN bank. Candidate pairs must share a potential gate variable. Repeated structures such as `(j,j)` are permitted so that two branches can share the same variable and gate.

`_has_affine_partition_refactor_opportunity` can request a refactor even when predictive validation error is already very small. The trigger is structural: the incumbent and learned support bank exhibit a compatible shared-variable product pattern. Fuzzy task labels and target rule annotations are not used by the search.

## Data-aware initialization

For each admissible support pair and gate orientation, the search initializes a canonical data-unit partition chart and evaluates branch operators with both raw and data-unit affine seeds. Branch pairs are screened **jointly** by an exact two-regressor least-squares fit. This is necessary because one branch can be weak in isolation while the pair is exact.

The family screen is operator-family-aware. It keeps one best affine start for each support/gate/operator-family tuple before expensive refinement. The retained beam is a union of:

- families with the lowest coarse prediction error;
- families with low coarse cancellation/description score.

This prevents early prediction-only pruning from eliminating a mechanically simple family whose affine parameters need continuous refinement.

## Tied complementary gate refinement

During continuous refitting, one affine identity gate is free and its paired gate is constrained to remain its exact complement. If the base gate is

$$
u(z)=\beta z+\gamma,
$$

the paired gate is always

$$
1-u(z)=-\beta z+(1-\gamma).
$$

The shared chart can move away from the finite-sample min/max initialization, but optimization cannot destroy complementarity. Branch operators, branch affine parameters, rule scales and bias remain trainable.

## Delayed pruning and polishing

The search first performs a short continuous refinement over a family-diverse beam. A smaller union of predictive leaders and low-cancellation leaders receives deeper polishing. This delays family elimination until affine parameters have had an opportunity to reach an appropriate basin.

## Validation-equivalent selection

Let `M_best` be the lowest validation MSE among the incumbent and partition candidates. A candidate is treated as predictively equivalent when

$$
M\le M_{best}(1+\epsilon_{rel})+
(\epsilon_{nrmse}\,\sigma_y)^2.
$$

Within this validation-defined band, the selector prefers lower internal cancellation and then lower symbolic description complexity. The cancellation term measures total absolute rule contribution plus absolute bias relative to target scale. This discriminates compact partition forms from algebraically equivalent fits that depend on large cancelling contributions.

A partition model may therefore replace the incumbent either because it materially improves validation error or because it is validation-equivalent and has a lower preference score. Test labels and benchmark fuzzy-rule annotations do not enter this decision.

## Support safety

The selected partition rules are checked against the `allowed_supports` supplied to `learned_support_symbolic_gsr`. A candidate whose support is not in the canonical effective RuleKAN support contract raises an error rather than being accepted.

Because PowerRuleKAN base fitting also uses `learned_support_symbolic_gsr`, the same partition mechanism is available inside transformed-power and ratio-base searches without relaxing their RuleKAN support constraints.

## Fuzzy benchmark scoring

The benchmark scores fuzzy recovery separately from predictive error. Expected fuzzy trees are expanded into sum-product rules. Gate factors must be identity-family factors on the correct variable and have the correct orientation after accounting for input standardization and affine gauge. Branch factors are matched by variable and operator family; sine/cosine are treated as phase-equivalent.

PowerRuleKAN receives fuzzy structural credit only for final outer terms consisting of one base at power `+1`. Genuine powered, reciprocal or product-of-bases outer terms are treated as unmatched fuzzy rules. See [Metrics and figures](../benchmarks/metrics-and-figures.md).
