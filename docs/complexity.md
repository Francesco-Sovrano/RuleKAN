# Complexity

## Notation

| Symbol | Meaning |
|---|---|
| `N` | samples in a numerical optimization step |
| `N_s` | samples used for symbolic screening |
| `d` | input dimension |
| `q` | maximum factors per rule |
| `W` | numerical RuleKAN width |
| `B` | numerical edge-basis evaluation cost |
| `U` | number of distinct learned supports |
| `s_u` | number of variables in learned support `u` |
| `K` | symbolic library size |
| `F_R` | factor slots instantiated by RuleKAN symbolic screening |
| `T_n` | numerical optimization steps |
| `T_g` | differentiable symbolic/GMP steps |
| `M` | hard symbolic candidates actually trialed |
| `tau` | short in-context refit steps per hard trial |
| `P` | number of integer powers tested by PowerRuleKAN |
| `R_o` | maximum selected outer PowerRuleKAN terms |

## Numerical RuleKAN

A dense numerical step evaluates approximately `W*q*d` candidate one-dimensional edges. With basis cost `B`,

\[
T_{num}=O(T_nNWqdB),
\qquad
S_{num}=O(WqdB).
\]

Hard variable selection avoids this full cost in some later discrete paths, but the expression above is the comparable dense training term.

## SISP structure count

The number of unordered variable multisets of orders one through `q` is

\[
S(d,q)=\binom{d+q}{q}-1.
\]

The total number of factor positions across this grammar is

\[
A_S(d,q)=d\binom{d+q}{q-1}.
\]

A differentiable GMP pass over the full grammar has the comparable screening term

\[
T_{SISP,GMP}=O\!\left(T_gN_sK\,d\binom{d+q}{q-1}\right).
\]

For fixed `q`, structural growth is `Theta(d^q)`.

## RuleKAN learned-support grammar

A learned support of size `s_u` contributes

\[
\binom{q}{s_u}
\]

multiplicity patterns through order `q`. The full learned-support bank has

\[
B_R(q)=\sum_{u=1}^{U}\binom{q}{s_u}
\]

structures.

Its factor-position count is

\[
A_R(q)=\sum_{u=1}^{U}s_u\binom{q+1}{s_u+1}.
\]

SC-GMP generally instantiates only a subset with `F_R` active factor slots, giving

\[
T_{RuleKAN,SC}=O(T_gN_sKF_R).
\]

A wider support rescue approaches the `A_R(q)` term. Bounded nonempty subsets, selected pre-pruning supports, and gate-aware unions may enter the effective bank when derived from Stage-1 evidence; arbitrary support enumeration is reserved for SISP.

## Hard operator tuples

For one `m`-factor structure and a library of size `K`, exhaustive operator tuples scale as `K^m`. GMP local top-k compression to `L` operators per factor reduces the raw tuple universe to roughly `L^m`. Hard tuple beams, family-level pruning and in-context validation reduce the number `M` of candidates that receive continuous refitting.

The hard-refit contribution is approximately

\[
O(M\tau C_{sym}),
\]

where `C_sym` is the cost of evaluating and differentiating the current symbolic model.

## Affine-partition rescue

For each admissible pair of order-2 supports sharing a gate, branch operator families are paired. With `K` operators the family space is `O(K^2)` per orientation before beam selection. The implementation vectorizes the coarse two-regressor fit across affine seed pairs, keeps a bounded family beam, and deeply polishes only a smaller subset. `max_support_pairs`, `family_beam`, `max_samples`, `final_polish_topk` and the seed grid determine practical cost.

## PowerRuleKAN

If `P` integer powers are attempted, transformed-target base fitting contributes up to `P` learned-support symbolic searches. Ratio search adds numerator and denominator base fits plus continuous ratio polishing. Outer OMP works on the resulting atom bank and selects at most `R_o` terms.

The worst-case symbolic cost is therefore dominated by repeated learned-support GSR calls rather than by evaluating integer powers themselves. Negative-power checks are linear in the number of evaluated samples and selected reciprocal bases.

## Interpretation

RuleKAN shifts combinatorial cost from the complete `d`-variable multiset grammar to a support-conditioned grammar whose size depends on `U`, the support sizes and `q`. The numerical precursor has a nontrivial training cost; SISP in this repository uses the same precursor in benchmark comparisons so symbolic-stage comparisons can isolate the effect of structural conditioning.
