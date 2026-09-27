# Symbolic library

## Factor form

Each symbolic factor is represented as a symbolic operator with an affine input chart,

$$
g(\beta x+\gamma).
$$

Rule coefficients absorb multiplicative factor amplitudes. Identity factors are canonicalized as a single affine expression. Protected operators use numerically safe definitions implemented by `SYMBOLIC_LIB` in `rulekan/sum_product_kan.py`.

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
```

Compound shortcuts are excluded from the compact default.

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

The `paper25` operators are used without explicit constants `0` and `1`, because RuleKAN already has a global bias and per-rule amplitude. The library also contains three protected squared primitives:

```text
log1p_sq
sqrt1p_sq
inv1p_sq
```

The resulting research library has 26 operators.

### `core10` / `target_core`

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
```

The shared benchmark alias `research` resolves to this 10-operator elementary vocabulary. Compound forms such as `gaussian`, `log1p_sq`, `sqrt1p_sq`, and `inv1p_sq` are excluded, so models using this vocabulary must construct them with the available composition or whole-expression operators. `core14` remains accepted as a compatibility alias. Use `research26` to request the larger distractor-rich library.

### `medium16`

`core10` plus:

```text
x^3
x^4
x^5
1/x^3
abs
arctan
```

`medium20` remains accepted as a compatibility alias.

## Protected squared operators

The protected operators are intended to represent common safe even functions over the full sampled real domain:

```text
log1p_sq     log(1 + x^2)
sqrt1p_sq    sqrt(1 + x^2)
inv1p_sq     1 / (1 + x^2)
```

They avoid the domain restrictions of raw `log`, `sqrt` and reciprocal when the target itself is defined by these protected forms.

## Cross-method vocabulary matching

The controlled `research`/`research_modern` profiles use `core10` as the conceptual target vocabulary. RuleKAN-family methods, AutoSym/GSR/GMP/MultKAN controls, and SR-KAN receive an exact ten-atom match (with SR-KAN native aliases such as `linear`, `square`, `inv_x`, and `inv_x2`).

External symbolic-regression systems are matched as closely as their public APIs permit:

- **Symbolic-KAN:** `x, x2, inv, sqrtx, log, exp, sin, cos, tanh`. The official implementation has no one-step inverse-square atom; with two symbolic blocks, `1/x^2` can be composed from `x2` and `inv`. The configured bank does not include duplicate identity tokens or an `x3` shortcut.
- **PySR:** binary `+,-,*,/` plus `square,exp,sin,cos,tanh,sqrt,log,inv`. Extra unary shortcuts `cube`, `atan`, and `abs` are disabled; identity is a variable leaf and inverse-square is compositional.
- **Operon:** arithmetic, constants/variables, `square,exp,sin,cos,tanh,sqrt,log`. Extra `atan` and `abs` shortcuts are disabled; reciprocal and inverse-square are compositional through division and square.
- **PSE/PSRN:** its documented arithmetic/identity plus `sin,cos,exp,log,tanh` grammar is retained. Square and reciprocal can be composed, but the configured public grammar has no dedicated square-root token.
- **uDSR:** the public uDSR function set is retained, including its defining `poly`/LINEAR token; its public grammar does not provide a literal one-to-one `core10` mapping.
- **RILS-ROLS:** the public estimator does not expose an operator-library constructor option, so its method-native grammar is retained.
- **SINDy-12:** the static dictionary contains the exact ten elementary atoms and tensor-product interactions up to the shared task-specific factor order (maximum three). The main-comparison `sindy` condition is capped at 12 active non-bias library terms; a constant/bias is uncharged. `sindy_unconstrained` uses the same dictionary and STLSQ optimizer without the final support cap for a capacity-sensitivity comparison.
- **ParFam:** polynomial and rational structure is native to the parametric family. The configured analytic functions are `sin, cos, exp, log, sqrt, tanh`, which cover the remaining `core10` families as closely as the public wrapper permits.
- **EQL:** the unary bank is `x, x^2, 1/x, sqrt, log, exp, sin, cos, tanh` plus structural multiplication units. Inverse-square is compositional across the two EQL layers.

Each run records `shared_symbolic_native_library`, `shared_symbolic_native_exact_match`, and `shared_symbolic_native_note`, so unavoidable grammar mismatches are explicit in the result metadata rather than silently treated as exact matches.

## Constants

RuleKAN's expression already contains a global bias `b` and one amplitude `a_r` per rule. Explicit constant factors are therefore usually redundant in the RuleKAN benchmark vocabulary. External symbolic-regression baselines retain their native constant mechanisms where those are inseparable from the public method API.

## Affine reparameterization

The operator name alone does not determine the symbolic shape. `sin(beta*x+gamma)` and `cos(beta*x+gamma)` can be phase-equivalent; odd/even symmetries can produce equivalent charts; exponential shifts can trade scale with the rule amplitude. Symbolic search therefore uses deterministic multi-start affine seeds plus continuous refitting, and fuzzy gate scoring evaluates identity charts in a gauge-aware way.
