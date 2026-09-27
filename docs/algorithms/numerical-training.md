# Stage 1: numerical interaction discovery

## Numerical objective

Stage 1 fits `SumProductKAN` while learning numerical term presence, optional factor presence, and variable assignments. Its output is a fitted numerical predictor together with retained and pre-pruning variable-support evidence for Stage 2.

The default low-level schedule returned by `default_sum_product_schedule(symbolic=False)` has three phases:

1. **numeric representation warmup** — edge functions fit while structure is held fixed;
2. **continuous rank/order compression** — rule and factor gates become trainable under sparsity penalties;
3. **hard rank/order consolidation** — hardening increases and the compressed graph is consolidated.

Benchmark profiles scale the step counts with `stage_scale` rather than changing the phase ordering.

## Structure initialization

`SumProductKAN` can initialize a diverse collection of variable templates across rules. When self-products are allowed, two-factor models use combinations with replacement. The benchmark RuleKAN training path sets `allow_self_products=False`, so initial numerical two-factor rules cover distinct-variable pairs and spare capacity covers unary templates. Same-variable multiplicity is handled by symbolic support expansion.

The logits remain trainable; the templates are starting points rather than fixed structural assignments.

## Differentiable gates

Variable choice uses categorical logits. Rule and optional factor presence use Hard-Concrete gates. For a gate logit `log_alpha`, temperature `tau`, and stretch endpoints `gamma < 0 < 1 < zeta`, the implementation uses

\[
p_{open}=\sigma\left(\log\alpha-\tau\log\frac{-\gamma}{\zeta}\right)
\]

for the differentiable expected-open probability. In deterministic mode the relaxed value is formed from

\[
s=\sigma(\log\alpha/\tau),\qquad
\tilde s=\operatorname{clip}_{[0,1]}\bigl(s(\zeta-\gamma)+\gamma\bigr).
\]

Stochastic training optionally adds logistic noise before the sigmoid. Hardening interpolates from the relaxed value toward a straight-through binary threshold. The expected-open probabilities enter the L0-style sparsity penalty and the hard values determine the final discrete rule/factor structure. This follows the Hard-Concrete L0-gating construction summarized in [Related work](../related-work.md).

Mandatory prefix factor slots remain active so every rule satisfies `min_order`; optional factor slots can close under the factor gates.

## Exact-forward gradient stabilization

Products of large edge values can create scale-dependent gradients because

\[
\frac{\partial}{\partial f_j}\prod_i f_i=\prod_{i\ne j}f_i.
\]

The implementation contains optional exact-forward/stabilized-backward transforms. `_identity_forward_stable_backward` leaves the forward value unchanged while replacing the backward sensitivity with a scaled `asinh` derivative. `_equalized_product` similarly stabilizes product gradients without changing the represented forward function.

The corresponding constructor/configuration controls are:

```text
symbolic_gradient_scale
symbolic_product_gradient_scale
numeric_factor_gradient_scale
numeric_product_gradient_scale
numeric_product_gradient_power
numeric_product_gradient_max_gain
```

These controls alter optimization dynamics, not the mathematical forward expression.

## High-recall support capture

After the early numerical phases and before destructive pruning, the benchmark records support evidence with `capture_numeric_support_evidence`. The later symbolic search can therefore distinguish a support that disappeared because of numerical redundancy from a support that was never evidenced.

The support evidence is ranked using a mixture of structural probability and observed numerical contribution strength. Final active supports are merged back into the support set even if their pre-pruning score is low.

## Numerical pruning

`prune_numeric_structure_to_stability` tests removable rules and optional factors, refits after tentative deletions, and accepts only changes within configured local/global relative-MSE budgets. Physical masks make accepted deletions persistent. The benchmark supports:

- `iterative` pruning;
- `one_shot` pruning;
- no numerical pruning.

Iterative mode can prune before and after the final hard-consolidation phase. Plateau polishing then optimizes the fixed hard structure. An optional fixed-structure LBFGS precision polish is available through the benchmark configuration.

## Optional numerical symbolic-manifold constraint

With `numeric_symbolic_manifold=true`, compression/consolidation also train auxiliary exact symbolic functions while the actual prediction path remains numerical. The constraint penalizes active edge shapes that remain farther than a configured tolerance from every single symbolic family. The dual coefficient is updated periodically and bounded by `manifold_dual_max`.

The principal controls are:

```text
numeric_symbolic_manifold
manifold_warmup_scale
manifold_hardening_start
manifold_hardening_mid
manifold_hardening_end
manifold_temperature_start
manifold_temperature_mid
manifold_temperature_end
manifold_symbolic_lr_scale
manifold_dual_init
manifold_dual_hard_init
manifold_rho
manifold_hard_rho
manifold_tolerance
manifold_dual_every
manifold_dual_max
manifold_max_samples
```

The resulting symbolic-family parameters are priors for numerical compression; final symbolic discovery still uses the learned-support search described in [Symbolic search](symbolic-search.md).

## Optional numerical logic compression

A hard numerical RuleKAN can contain several large rules whose contributions cancel. `numeric_logic_diagnostics` measures this with the triangle-inequality ratio

\[
C=\frac{\sum_r\lVert c_r\rVert_2}{\lVert\sum_r c_r\rVert_2},
\]

where `c_r` is the validation-set contribution of active rule `r`. Values near one indicate little cancellation; larger values indicate compensating rules. The same diagnostic reports pairwise contribution correlation, span redundancy and the set of active hard variable structures.

When `numeric_logic_compression=true`, `compress_numeric_rule_bank` attempts rollback-safe rule deletion on the discretized numerical model. Each deletion is followed by continuous fixed-structure refitting and is retained only if validation RMSE stays inside a tolerance envelope combining a relative bound with a scale-normalized absolute bound. This compression changes the numerical precursor only; the pre-pruning support snapshot remains available to adaptive symbolic recovery.
