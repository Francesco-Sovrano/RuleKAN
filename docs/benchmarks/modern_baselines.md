# Additional contemporary symbolic-regression baselines

The `research_modern` profile extends `research` with seven regression-only baselines:

- `symbolic_kan`: the authors' official `sfaroughi3/Pub_Symbolic_KANs` implementation, checked out by `setup.sh` at commit `9481a82`. The adapter calls the upstream `Exp_reaction_diffusion/symKanTraining.py::train_regression_onehot` routine directly and only replaces its demo data generator with the benchmark train/validation tensors. Prediction uses the upstream hardened symbolic evaluator; the benchmark additionally serializes the trained discrete network for structural scoring. Run metadata sets `symbolic_kan_exact_official_code=true`.
- `pse`: PSE/PSRN through the official `psrn` Python package, corresponding to Ruan et al. (2026), DOI `10.1038/s43588-025-00904-8`. The adapter uses the public `PSRN_Regressor` API and scores the returned expression directly.
- `rils_rols`: RILS-ROLS through the public `rils-rols` package, corresponding to Kartelj and Dukanovic (2023), DOI `10.1186/s40537-023-00743-2`.
- `udsr`: unified Deep Symbolic Regression through the official DSO PyTorch package. The adapter enables the NeurIPS-2022 `poly`/LINEAR token and GP meld, then exports the best `program_` expression returned by the public `DeepSymbolicRegressor` API.
- `sindy`: **SINDy-12**, a static sparse-library control using PySINDy's official `STLSQ` optimizer. The adapter builds the matched analytic dictionary and applies SINDy's sparse-regression core to `y=f(x)`, but the final expression is capped at 12 active non-bias library terms. Over-budget STLSQ solutions are reduced by RMS contribution strength and jointly OLS-refit before validation selection. `sindy_unconstrained` is excluded from the main comparison model set and is used only as an explicitly requested appendix diagnostic.
- `parfam`: the official ICLR-2025 `parfam` package through `ParFamWrapper`. The adapter passes the benchmark arrays directly, uses the package's `small` configuration by default, and records the returned `formula_reduced`.
- `eql`: a budgeted PyTorch reproduction of the ICML-2018 EQL-Div architecture: native identity/sine/cosine hidden units, multiplication units, final regularized division, the published three-phase sparsity schedule, and validation+sparsity model selection over depth and regularization strength. The authors' released code targets legacy Theano/TensorFlow stacks, so metadata explicitly records that the original source is not executed unchanged.


## Vocabulary policy

`research_modern` inherits the `research` profile's shared `target_core`/`core10` vocabulary. Exact matching is enforced for the RuleKAN/KAN extraction family and SR-KAN. Symbolic-KAN is restricted to its closest official native bank (`x, x2, inv, sqrtx, log, exp, sin, cos, tanh`); its missing one-step inverse-square atom is constructible across its two symbolic blocks. PySR and Operon have non-core unary shortcuts removed under this profile. PSE, uDSR, RILS-ROLS, ParFam, and EQL retain documented method-native grammar constraints where their public APIs or architectures do not permit a literal one-to-one `core10` bank. SINDy-12 uses the exact `core10` atoms in its explicit static dictionary and the same task-specific product-order cap (at most three). Its final support is limited to 12 active non-bias terms; the appendix-only unconstrained variant uses the identical dictionary and optimizer without that support cap. Per-run metadata records the native bank and whether the match is exact. See `docs/reference/symbolic-library.md`.

Install the external packages with:

```bash
python -m pip install -r benchmarks/requirements-modern-sr.txt
```

PSE/PSRN currently supports Python 3.9--3.12. The repository's normal setup uses Python 3.12.

List or run the extended profile with:

```bash
python -m benchmarks.run_benchmark --profile research_modern --list
python -m benchmarks.vocabulary_audit --profile research_modern
./run_rulekan_benchmark.sh research_modern
```

To append only these baselines to an existing result directory without modifying its prior run records:

```bash
python -m benchmarks.run_missing_baselines \
  --source-results benchmark_results/current \
  --dest-results benchmark_results/current_modern \
  --profile research_modern \
  --models symbolic_kan,pse,rils_rols,udsr,sindy,parfam,eql
```


### SINDy term-budget sensitivity

The main 23-method comparison must use `sindy` (SINDy-12). The unconstrained condition is deliberately separated so that a very long fixed-dictionary expansion cannot enter the headline ranking. Run the appendix comparison with:

```bash
python -m benchmarks.run_benchmark --profile research_modern --models sindy_unconstrained --run-dir benchmark_results/current
```

