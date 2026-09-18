# Interaction-shape and block proposals

RuleKAN's symbolic target is additive, while each rule is multiplicative. A greedy rule fitted against the full target can therefore be a poor local estimator of one true mechanism when several interactions are present. The symbolic search includes interaction-shape and block proposals to reduce this mismatch without changing the admissible support grammar.

## Two-way interaction surface

For a learned cross-variable support `{i,j}`, `interaction_shape_symbolic_shortlists` bins the training data on `(x_i,x_j)` and estimates a two-dimensional response surface. Additive row and column effects are removed:

\[
I_{ab}=M_{ab}-\bar M_{a\cdot}-\bar M_{\cdot b}+\bar M.
\]

If the interaction is approximately one multiplicative mechanism

\[
f(x_i)g(x_j),
\]

the residual interaction surface is approximately rank one. Its leading singular vectors provide separate one-dimensional shape estimates for the two factors. Each shape is then matched to the symbolic operator library with affine input parameters.

The leading singular-value energy fraction

\[
\rho_1=\frac{\sigma_1^2}{\sum_k\sigma_k^2}
\]

is used as a proposal-quality statistic. The interaction path is enabled only when the configured rank-one criterion is satisfied. These shapes rank or propose operators; they do not force the final symbolic expression.

## Why interaction residualization is useful

For a target

\[
y=f_1(x_i)g_1(x_j)+f_2(x_i)g_2(x_k)+a(x_i)+b(x_j,x_k),
\]

fitting one candidate product directly to `y` mixes the product mechanism with unrelated additive and interaction terms. Removing marginal row/column effects isolates the non-additive part of one two-way surface before operator-family matching.

This is complementary to ordinary GMP. GMP is an end-to-end differentiable proposal mechanism; interaction-shape screening is a shape-specific hard proposal signal for cross-variable products.

## Two-rule initial block pursuit

When several strong interaction supports are present, the search may initialize with two rules jointly instead of committing one rule greedily. Candidate pairs are evaluated with a joint coefficient and bias refit, followed by a small nonlinear continuous polish. The best block can receive an additional consolidation pass before ordinary residual pursuit continues.

Joint initialization addresses an additive decomposition problem: one first rule should not have to approximate two separate multiplicative mechanisms merely because the second rule has not yet been selected.

## Structure restrictions

Interaction-shape proposals do not create variable supports. They are computed only for structures already supplied to the symbolic search. Under `learned_support_symbolic_gsr`, those structures must reduce to RuleKAN's allowed numerical support classes. Under SISP, the caller may instead supply the complete variable-multiset grammar.

## Relation to affine partitions

A complementary partition such as

\[
(1-u)f(x_i)+u g(x_j)
\]

can also be written distributively as

\[
f(x_i)-u f(x_i)+u g(x_j).
\]

Interaction/block proposals can recover the separate product mechanisms in the distributive form. The affine-partition rescue in [Affine partitions](affine-partitions.md) performs a later canonical refactor when a tied `u,1-u` representation is supported and validation-equivalent.
