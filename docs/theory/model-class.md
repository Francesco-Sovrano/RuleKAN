# Model class

## Notation

Let `x=(x_0,...,x_{d-1})` be the input. `W` denotes numerical rule capacity and `q` the maximum number of factors in one rule.

### Numerical RuleKAN

The numerical model is

\[
f_{\mathrm{num}}(x)=b+\sum_{r=1}^{W}q_r a_r
\prod_{s=1}^{q}\left[(1-m_{rs})+m_{rs}h_{rs}(x_{j_{rs}})\right].
\]

`q_r` is a differentiable rule-presence gate, `m_rs` is a factor-presence gate, `j_rs` is a categorical variable choice, and `h_rs` is a learned univariate numerical edge. Ordinary RuleKAN uses cubic B-spline edge banks. RuleKAN-RBF replaces only the numerical edge bank with Gaussian RBFs.

The multiplicative identity is controlled by `m_rs`; it is not encoded as a pseudo-variable. This allows a product rule to reduce continuously to lower order while variable choice remains a categorical feature selection problem.

The `SumProductKAN` class can represent repeated-variable numerical products when `allow_self_products=True`. The benchmark RuleKAN family intentionally trains the numerical precursor with `allow_self_products=False`; repeated-variable symbolic factorization is recovered later from support closure. The reason is the same-variable non-identifiability described in [Identifiability and support](identifiability-and-support.md).

### Symbolic RuleKAN

The final symbolic model has the form

\[
f_{\mathrm{sym}}(x)=b+\sum_{r=1}^{R}a_r
\prod_{s=1}^{m_r}g_{rs}(\beta_{rs}x_{j_{rs}}+\gamma_{rs}),
\qquad m_r\le q.
\]

Each `g_rs` is a discrete symbolic operator from the configured symbolic library. Discrete support, multiplicity and operator identities are hard by the end of symbolic search. Continuous affine input parameters, rule amplitudes and bias are refitted against data.

## Factor gauge

For a non-affine operator, the canonical factor is

\[
g(\beta x+\gamma).
\]

A separate per-factor output amplitude is redundant inside a product because

\[
\prod_s c_s g_s(\cdot)=\left(\prod_s c_s\right)\prod_s g_s(\cdot),
\]

so those amplitudes can be absorbed into the rule coefficient `a_r`. Nonlinear output offsets are not part of the final factor gauge because multiplying shifted factors introduces lower-order terms and makes a single symbolic factor encode an implicit sum.

The identity family is special. A general affine expression

\[
a(bx+c)+d
\]

is exactly folded into one affine identity chart

\[
\beta x+\gamma,
\quad \beta=ab,\quad \gamma=ac+d.
\]

This makes `x`, `1-x`, signed affine gates and ordinary linear factors members of one canonical operator family.

## Support, multiplicity and support rank

For

\[
g_1(x_0)g_2(x_0)g_3(x_2),
\]

the support is `{0,2}` and the multiplicity tuple is `(0,0,2)`.

Support rank is separate from multiplicity. Several additive rules may have identical support and multiplicity but different symbolic factors or affine parameters. For example,

\[
\sin(x)\cos(x)e^y\cos(y)+\tan(x)\cos(x)e^y\sin(y)
\]

contains two independent rules with support `{x,y}` and multiplicity `(x,x,y,y)`.

This separation is necessary because a flexible numerical rule does not uniquely determine how many symbolic mechanisms should appear after symbolic conversion.

## Learned-support grammar

If the numerical precursor evidences support classes `S_1,...,S_U`, RuleKAN's admissible symbolic structures are

\[
\mathcal G_R=
\bigcup_{u=1}^{U}
\{z:\operatorname{supp}(z)=S_u,\ |z|\le q\}.
\]

All variables in a learned support must remain present in each expanded structure; only their multiplicities may change. For a support of size `s`, the number of positive multiplicity patterns through order `q` is

\[
\sum_{m=s}^{q}\binom{m-1}{s-1}=\binom{q}{s}.
\]

SISP instead uses the complete variable-multiset grammar

\[
\mathcal G_S=
\{z:1\le |z|\le q,\ z_i\in\{0,\ldots,d-1\}\}/\text{permutation},
\]

with

\[
|\mathcal G_S|=\binom{d+q}{q}-1.
\]

RuleKAN and SISP therefore share a symbolic language but differ in where admissible variable structures come from.

## Symbolic rank capacity

The number of numerical rules is not an upper bound on the number of final symbolic rules. A flexible numerical component can absorb several symbolic mechanisms. Benchmark profiles therefore ensure the symbolic rule budget is at least the shared numerical width, and rank continuation can allocate several symbolic rules to one support.

## PowerRuleKAN model class

PowerRuleKAN extends the outer grammar to

\[
f(x)=b+\sum_{r=1}^{R_o}a_r\prod_t B_{rt}(x)^{p_{rt}},
\qquad p_{rt}\in D\subset\mathbb Z\setminus\{0\},
\]

where every `B_rt` is a fully symbolic RuleKAN sum-product expression. A reciprocal is a negative integer power of a complete base; a ratio is a product of positive- and negative-power bases. The ordinary RuleKAN model is the `p=1` subcase.

## Representability

Flat RuleKAN directly represents finite sums of products of configured univariate symbolic functions. PowerRuleKAN additionally represents integer powers and reciprocals of complete RuleKAN subexpressions. Arbitrary functional composition, such as `sin(exp(x))` or `sin(x0*x1)`, is not generally a finite member of those flat grammars unless an equivalent expression exists in the configured vocabulary. `RuleKAN-Comp` and `PowerRuleKAN-Comp` add one validation-gated depth-2 symbolic composition level while preserving the learned effective-support contract. The benchmark includes nested tasks explicitly to measure this boundary.

Finite sums of separable functions also have an approximation-theoretic interpretation. On compact product domains, suitable algebras generated by separable continuous functions satisfy Stone-Weierstrass density conditions. This supports approximation capacity; it does not imply finite exact recovery with a bounded library, bounded product order or finite symbolic rank.
