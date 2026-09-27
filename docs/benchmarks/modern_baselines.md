# Contemporary symbolic-regression baselines

The benchmark harness registers the following additional regression baselines:

- `symbolic_kan`: official `sfaroughi3/Pub_Symbolic_KANs` source pinned by `benchmarks/setup.sh` to commit `9481a82`. The adapter calls `Exp_reaction_diffusion/symKanTraining.py::train_regression_onehot`, supplies benchmark train/validation arrays, evaluates the hardened symbolic network, and serializes the discrete structure for structural scoring.
- `pse`: PSE/PSRN through the public `psrn` package and `PSRN_Regressor` API.
- `rils_rols`: RILS-ROLS through the public `rils-rols` package.
- `udsr`: unified Deep Symbolic Regression through the DSO PyTorch package, with the `poly`/LINEAR token and GP meld enabled by the adapter.
- `sindy`: SINDy-12 through PySINDy's `STLSQ` optimizer. The matched analytic dictionary is used for `y=f(x)` and the selected expression is limited to 12 active non-bias terms. Over-budget supports are ranked by empirical RMS contribution and jointly OLS-refit before validation selection.
- `sindy_unconstrained`: the same SINDy dictionary and optimizer without the final 12-term support cap.
- `parfam`: the ICLR-2025 `parfam` package through `ParFamWrapper`, using the package's `small` configuration by default.
- `eql`: an in-tree PyTorch implementation of the EQL-Div architecture with analytic unary units, multiplication units, regularized division, the three-phase sparsity schedule, and validation-plus-sparsity model selection.

`research_modern` schedules `symbolic_kan`, `pse`, `rils_rols`, `sindy`, `parfam`, and `eql` in addition to the inherited `research` models. `udsr` is registered but is not selected by `research_modern`; it requires a separate compatible environment. The profile schedules 25 models in total, while its `main_comparison_models` field contains 23 models and excludes `pse` and `anfis`.

## Vocabulary policy

`research_modern` inherits the ten-primitive `target_core` / `core10` vocabulary. Exact matching is used where the public method API permits it.

- **RuleKAN, KAN extraction controls, and SR-KAN:** exact ten-family match, using method-native aliases where necessary.
- **Symbolic-KAN:** native bank `x, x2, inv, sqrtx, log, exp, sin, cos, tanh`; inverse square is not a direct primitive and can be constructed across symbolic blocks.
- **PySR:** binary arithmetic plus `square, exp, sin, cos, tanh, sqrt, log, inv`; identity is a variable leaf and inverse square is compositional.
- **Operon:** arithmetic, variables/constants, `square, exp, sin, cos, tanh, sqrt, log`; reciprocals are compositional through division.
- **PSE/PSRN:** public arithmetic/identity grammar plus `sin, cos, exp, log, tanh`; no dedicated square-root token in the configured public grammar.
- **uDSR:** public grammar including the `poly`/LINEAR token; it does not provide a literal one-to-one `core10` bank.
- **RILS-ROLS:** method-native grammar because its estimator does not expose an operator-library constructor.
- **SINDy-12:** exact `core10` atoms in a static feature dictionary with interactions through the task-specific factor order, capped at 12 active non-bias terms.
- **ParFam:** native polynomial/rational families with `sin, cos, exp, log, sqrt, tanh`.
- **EQL:** `x, x^2, 1/x, sqrt, log, exp, sin, cos, tanh` plus structural multiplication units; inverse square is compositional.

Run metadata records the native symbolic bank and whether the requested vocabulary match is exact.

## Installation

Directly pip-installable dependencies are listed in:

```bash
python -m pip install -r benchmarks/requirements-modern-sr.txt
```

The complete baseline environment, including source-based dependencies, is installed by `./benchmarks/setup.sh` and defaults to `.env-baselines`. uDSR requires a separate environment compatible with the upstream DSO dependency pins.

Inspect the profile and vocabulary mapping with:

```bash
python -m benchmarks.run_benchmark --profile research_modern --list
python -m benchmarks.vocabulary_audit --profile research_modern
```

Run the profile with:

```bash
./run_rulekan_benchmark.sh research_modern
```

Run selected additional baselines into a separate result directory with:

```bash
python -m benchmarks.run_missing_baselines \
  --source-results benchmark_results/current \
  --dest-results benchmark_results/current_modern \
  --profile research_modern \
  --models symbolic_kan,pse,rils_rols,sindy,parfam,eql
```

## SINDy term-budget sensitivity

The 23-method analytic comparison uses model identifier `sindy`, the 12-term controlled condition. The uncapped condition is run explicitly:

```bash
python -m benchmarks.run_benchmark \
  --profile research_modern \
  --models sindy_unconstrained \
  --run-dir benchmark_results/sindy_unconstrained
```
