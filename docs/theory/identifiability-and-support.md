# Identifiability and support

## Same-variable factorization is not identifiable in the numerical model

With unrestricted univariate numerical functions,

\[
\phi(x)\psi(x)=h(x)
\]

for another univariate function `h`. A spline or RBF edge can therefore absorb a product such as

\[
e^{-0.7x}\sin(2.4x)
\]

into one numerical edge. The numerical factor count on one variable cannot be interpreted as the number of symbolic factors in the generating expression.

This is why the benchmark numerical RuleKAN stage uses distinct-variable products and treats the numerical network primarily as a **support detector**. Same-variable symbolic multiplicity is recovered after support discovery under the finite symbolic vocabulary.

## Pre-pruning support evidence

`capture_numeric_support_evidence` records one row per numerical rule before destructive pruning. Each row contains:

- the distinct-variable support `tuple(sorted(set(vars)))`;
- the observed multiplicity tuple;
- rule probability;
- joint structure probability;
- numerical contribution strength;
- whether the rule is already active in a hard model.

Pre-pruning capture matters because a flexible numerical rule can become redundant after another rule refits around it even when its variable support remains relevant to a later restricted symbolic decomposition.

## Support collapse

`learned_numeric_support_classes` merges numerical rows by the **set of distinct variables**. Observed multiplicities are retained as diagnostics but do not define separate admissible support classes.

For example, numerical evidence

```text
(0,1)
(0,0,1)
(0,1,1)
```

collapses to the same support class `{0,1}`.

This collapse prevents accidental numerical factorization choices from being treated as symbolic truth. It also merges several numerical rows that describe the same variable interaction.

## Multiplicity expansion

`learned_structure_symbolic_bank` performs the complementary expansion. For each support class, it enumerates every positive multiplicity composition through `max_factors=q`, while requiring each supported variable to occur at least once.

For support `{0,1}` and `q=4`, the symbolic structure bank is

```text
(0,1)
(0,0,1)
(0,1,1)
(0,0,0,1)
(0,0,1,1)
(0,1,1,1)
```

A support of size `s` produces `C(q,s)` such structures. This collapse-then-expansion operation handles both directions of the mismatch between numerical edges and symbolic factors:

- one flexible numerical edge may encode several symbolic factors on the same variable;
- several numerical factors may describe a mechanism that has a simpler symbolic factorization.

The symbolic model is therefore conditioned on variable **support**, not on the numerical precursor's literal slot count.

## Rank expansion

Multiplicity expansion does not address several additive mechanisms that share exactly the same structure. Symbolic pursuit can allocate multiple independently parameterized rules to one support/multiplicity pattern. `symbolic_max_rules_per_structure`, structure-diverse beams and rank continuation govern that capacity.

The combination of support collapse, multiplicity expansion and rank expansion is the structural interface between the flexible numerical precursor and the finite symbolic model.

## Actual real-valued symbolic functions in the symbolic stage

Same-variable factorization becomes meaningful only after the hypothesis class is restricted. The symbolic search evaluates the actual real-valued implementations of symbolic atoms such as `exp`, `sin`, `tanh`, `x^2` and protected rational/square-root forms. It does not replace those atoms with spline or RBF surrogates during symbolic takeover. A candidate `exp(x)*sin(x)` therefore competes against a finite set of unary symbolic candidates rather than against an unrestricted spline `h(x)` that can absorb the whole product. Domain-sensitive operators use the repository's protected real-valued implementations during numerical fitting while formula export retains their symbolic names.

`gmp_symbolic_operator_preselection` is spline-free. Its factor values are evaluations of the configured symbolic operator functions with trainable affine input charts. This gives repeated-variable symbolic products an identifiable role relative to the discrete library.

## Optional numerical-to-symbolic manifold constraint

The numerical stage can optionally use `numeric_symbolic_manifold=true`. This does **not** replace the numerical forward model with symbols. The spline/RBF forward path remains active while auxiliary exact symbolic families are trained alongside it.

For an active numerical factor `h(x)`, `symbolic_manifold_distance` measures normalized distance to a union of single-family manifolds

\[
\{g_k(\beta x+\gamma):k=1,\ldots,K\}.
\]

Operator probabilities provide a differentiable continuation over the union. An augmented-Lagrangian-style penalty is applied when the distance exceeds the configured tolerance. During compression and consolidation, symbolic hardening anneals toward one family. For non-affine functions a temporary amplitude is permitted because factor scale is unidentifiable inside a product and can later be absorbed into the rule coefficient. Nonlinear output offsets are excluded.

This mechanism supplies a real-symbol prior before numerical pruning. It is optional and is not enabled by the standard benchmark profiles unless set in model configuration.

## Affine gauge equivalence

Several operators admit equivalent parameterizations:

- identity: `x`, `1-x` and any affine factor are one family;
- odd operators such as `tanh` and `arctan`: input sign can be traded against output sign;
- sine and cosine differ by phase shifts;
- even operators identify `beta` and `-beta` up to the same curve;
- exponential input shifts can be absorbed partly into rule amplitude.

The implementation canonicalizes the identity family explicitly and uses multi-start affine charts for the broader library. Data-coordinate-aware seeding is described in [Symbolic search](../algorithms/symbolic-search.md).

## Effective support contract

After validation selection, RuleKAN publishes one canonical support payload:

```text
symbolic_effective_support_classes
symbolic_effective_support_bank
symbolic_effective_support_source
```

The payload is initialized from the primary retained Stage-1 supports. Configured validation-gated support construction may add nonempty subsets, selected pre-pruning supports, or bounded gate-aware unions derived from Stage-1 evidence. If `RuleKAN Adaptive` selects its higher-recall rescue, the payload is replaced by the validation-selected augmented bank.

All downstream RuleKAN-derived searches, including affine-partition recovery, composition search, and PowerRuleKAN transformed-power and ratio bases, use this effective support payload. The `symbolic_learned_support_*` fields retain the primary support diagnostics.
