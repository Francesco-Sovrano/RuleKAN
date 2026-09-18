# Symbolic library

## Factor form

Each symbolic factor is represented as a symbolic operator with an affine input chart,

\[
g(\beta x+\gamma).
\]

Rule coefficients absorb multiplicative factor amplitudes. Identity factors are canonicalized as a single affine expression. Protected operators use numerically safe definitions implemented by `SYMBOLIC_LIB` in `symbolic_kan/sum_product_kan.py`.

## Compact library

The `SumProductKAN` constructor default is:

```text
x
x^2
x^3
exp
sin
cos
tanh
arctan
log1p_sq
sqrt1p_sq
inv1p_sq
```

## Benchmark libraries

### `paper25`

```text
0, 1, x, x^2, x^3, x^4, x^5,
1/x, 1/x^2, 1/x^3, sqrt, 1/sqrt(x),
log, exp, sin, cos, tan, tanh, abs, sgn,
arctan, arcsin, arccos, arctanh, gaussian
```

This is the 25-operator vocabulary selected by the `paper` benchmark profile.

> **Protected `log` semantics.** In the internal KAN library, `log` is numerically
> evaluated as `log(sqrt(x^2 + eps^2))`, i.e. a smooth approximation to
> `log(abs(x))`. SymPy export therefore uses `log(Abs(x))`; it must not be read as
> ordinary real `log(x)` on negative arguments. An ordinary-log benchmark should
> instead enforce a positive argument over the relevant data domain.

### `research26`

The `paper25` operators are used without explicit constants `0` and `1`, because RuleKAN already has a global bias and per-rule amplitude. Three protected squared primitives are added:

```text
log1p_sq
sqrt1p_sq
inv1p_sq
```

The resulting research library has 26 operators.

### `core14` / `target_core`

```text
x
x^2
1/x
1/x^2
sqrt
log
exp
sin
cos
tanh
gaussian
log1p_sq
sqrt1p_sq
inv1p_sq
```

The shared benchmark alias `research` resolves to this target-complete 14-operator vocabulary. Use `research26` to request the larger distractor-rich library.

### `medium20`

`core14` plus:

```text
x^3
x^4
x^5
1/x^3
abs
arctan
```

## Protected squared operators

The protected operators are intended to represent common safe even functions over the full sampled real domain:

```text
log1p_sq     log(1 + x^2)
sqrt1p_sq    sqrt(1 + x^2)
inv1p_sq     1 / (1 + x^2)
```

They avoid the domain restrictions of raw `log`, `sqrt` and reciprocal when the target itself is defined by these protected forms.

## Constants

RuleKAN's expression already contains a global bias `b` and one amplitude `a_r` per rule. Explicit constant factors are therefore usually redundant in the RuleKAN benchmark vocabulary. External symbolic-regression baselines keep their own constant mechanisms according to their native configuration.

## Affine reparameterization

The operator name alone does not determine the symbolic shape. `sin(beta*x+gamma)` and `cos(beta*x+gamma)` can be phase-equivalent; odd/even symmetries can produce equivalent charts; exponential shifts can trade scale with the rule amplitude. Symbolic search therefore uses deterministic multi-start affine seeds plus continuous refitting, and fuzzy gate scoring evaluates identity charts in a gauge-aware way.
