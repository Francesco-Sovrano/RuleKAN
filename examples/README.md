# Examples

Run examples from the repository root with module syntax so the local `rulekan` package is importable.

```bash
python -m examples.quickstart
python -m examples.example_sum_product_kan
python -m examples.example_rulemask_product
python -m examples.example_simple --help
python -m examples.example_feynman --help
python -m examples.example_logical_features --help
```

`examples.quickstart` is the package-only smoke test. It uses only the dependencies declared by `rulekan` and does not load benchmark baselines.
