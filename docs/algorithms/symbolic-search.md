# Symbolic search

## Pipeline

The benchmark RuleKAN symbolic path is:

1. form distinct numerical support classes from pre-pruning evidence plus final active supports;
2. expand each support into all admissible repeated-variable multiplicities through `q`;
3. run symbolic-operator GMP inside those structures;
4. generate additional hard exact-operator proposals;
5. score candidate rules in context against the target or current residual;
6. commit rules by validation-aware matching pursuit/GSR;
7. backfit committed rules and continuously refit symbolic affine parameters;
8. remove negligible/redundant rules within configured validation budgets;
9. optionally apply affine-partition refactoring;
10. optionally run a higher-recall numerical-support rescue when the primary symbolic model underfits validation data.

At every RuleKAN structural stage, candidate supports are checked against numerically learned support classes.

## Symbolic factor parameterization

A candidate factor is evaluated as

\[
g(\beta x_j+\gamma),
\]

with rule amplitude carrying the multiplicative scale. Symbolic search works on actual symbolic operator evaluations. Numerical splines/RBFs do not remain in the final model.

### Raw affine starts

Each operator has deterministic hard affine seeds supplied by `_hard_affine_seed_grid`. Periodic, monotone, polynomial and protected operators receive operator-specific slope/shift starts.

### Data-unit affine starts

Inputs in the benchmark are standardized. A useful symbolic chart may be simple in the original sampled coordinate but shifted/scaled in standardized coordinates. `_data_unit_affine_seed_grid` therefore maps operator seeds through the observed training range.

If

\[
u=\frac{z-z_{\min}}{z_{\max}-z_{\min}}
\]

and an operator seed is `g(beta*u+gamma)`, its equivalent chart on standardized coordinate `z` is

\[
\beta'=\frac{\beta}{z_{\max}-z_{\min}},\qquad
\gamma'=\gamma-\frac{\beta z_{\min}}{z_{\max}-z_{\min}}.
\]

Raw and data-unit starts are both retained. This is important for same-variable factorization and affine membership/complement gates.

## GMP operator preselection

`gmp_symbolic_operator_preselection` jointly optimizes soft operator choices for a supplied set of variable structures. A relaxed factor has the form

\[
S(x_j)=\sum_{k=1}^{K}\pi_k g_k(\beta_k x_j+\gamma_k).
\]

Temperature annealing and optional straight-through/Gumbel modes concentrate the operator distribution. GMP is a proposal mechanism; final symbolic commitment uses hard symbolic operators and end-to-end refitting.

The number of retained operators can differ for unary, repeated-variable and general product structures. `scaled_gmp_screening_sizes` maps library size to practical screening budgets.

### Latent identity charts

GMP supports a `data_dual` identity chart. It uses multiple latent parameterizations of the identity family during optimization, then collapses them back to one canonical `x` operator before top-k extraction. The symbolic grammar therefore does not gain an extra operator merely because initialization uses several identity gauges.

`resolve_gmp_local_policy` selects the effective local policy. Current `auto` behavior uses dual identity charts for cross-variable products and the raw chart for unary and repeated-variable structures. The benchmark configuration defaults to `raw` unless a profile overrides it.

## Hard exact-operator proposals

Soft GMP can suppress a useful operator before hard refitting. `hard_symbolic_tuple_screening` supplies a complementary path: it evaluates exact symbolic atoms from deterministic affine starts and builds hard product tuples with a beam search. At the first factor it retains operator diversity when the beam permits, then limits combinatorics at deeper product order.

The optional hybrid hard screen and hard-proposal union combine these candidates with GMP proposals rather than replacing GMP.

## In-context rule selection

A candidate rule is not accepted from local edge fit alone. Matching pursuit evaluates it in the current additive context, briefly refits continuous parameters, and compares validation behavior. The search can use:

- initial two-rule block pursuit;
- residual-conditioned operator rescue;
- residual-conditioned support prioritization;
- structure-diverse beams;
- joint linear scale refits after each commit;
- backfitting of previously committed rules;
- nonlinear OMP-style extra refinement.

This avoids treating the best local operator on an isolated edge as necessarily the best operator in the final expression.

## Learned-support GSR

`learned_support_symbolic_gsr` requires an explicit `structure_candidates` bank. It computes or receives the allowed numerical support classes and rejects every candidate structure `z` whose distinct-variable set is absent from that set.

It also passes the same allowed-support contract to the affine-partition rescue. This automatic aliasing makes the support restriction a property of the entry point rather than an option each downstream caller must remember to configure.

## SISP

`mandatory_symbolic_matching_pursuit` is the generic whole-rule search engine. If it is supplied the complete `symbolic_structure_bank` rather than a learned-support bank, the result is SISP: Structure-Independent Symbolic Pursuit.

For `d` variables and maximum order `q`, SISP considers

\[
\binom{d+q}{q}-1
\]

variable multisets before operator tuples are considered. This makes SISP a useful control for separating the value of learned numerical structure from the value of the symbolic optimizer.

## Adaptive high-recall support rescue

The `rulekan_adaptive` benchmark model first runs the same primary learned-support path as RuleKAN. A second structure-conditioned search is considered only when the primary symbolic validation error crosses the configured rescue criteria relative to the numerical precursor or an absolute force threshold.

The rescue recomputes support classes from the high-recall evidence captured before numerical pruning and constructs their multiplicity closure. It can therefore restore a support that numerical pruning removed, but it cannot introduce a variable combination absent from the numerical evidence.

The rescue uses a larger symbolic search budget. It replaces the primary symbolic model only when validation MSE improves by the configured relative margin. If selected, its support bank becomes the canonical `symbolic_effective_support_*` payload used by downstream PowerRuleKAN searches.

## Cleanup

Final cleanup can remove tiny-contribution rules and near-redundant rules under relative validation-MSE budgets. Redundancy checks use contribution correlations/span fit, while exact symbolic structure remains hard. Cleanup is bounded by trial count and wall-clock controls.
