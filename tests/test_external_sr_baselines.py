from pathlib import Path
import sys
import types

import numpy as np
import pytest

from benchmarks.models import (
    TRAINERS, train_operon, train_pse, train_pysr, train_rils_rols, train_udsr,
    train_srkan, train_symbolic_kan, train_sindy, train_sindy_unconstrained,
    train_parfam, train_eql,
)
from benchmarks.specs import TASKS, make_synthetic_data


def _tiny_data():
    spec = TASKS["same_var_exp_sin"]
    data = make_synthetic_data(spec, seed=2, train_n=24, val_n=8, test_n=10)
    return spec, data


def test_evolutionary_symbolic_regressors_are_registered():
    assert {"srkan", "symbolic_kan", "pse", "rils_rols", "udsr", "sindy", "sindy_unconstrained", "parfam", "eql", "pysr", "operon"}.issubset(TRAINERS)


def test_pysr_wrapper_uses_benchmark_data_and_reports_formula(monkeypatch):
    seen = {}

    class FakePySRRegressor:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit(self, x, y, variable_names=None):
            seen["fit_shape"] = x.shape
            seen["variable_names"] = variable_names
            return self

        def predict(self, x):
            return np.zeros(x.shape[0], dtype=float)

        def sympy(self):
            return "sin(x0)"

    mod = types.ModuleType("pysr")
    mod.PySRRegressor = FakePySRRegressor
    monkeypatch.setitem(sys.modules, "pysr", mod)

    spec, data = _tiny_data()
    run = train_pysr(spec, data, 5, {"niterations": 3, "populations": 2, "population_size": 8})
    assert run.model_name == "pysr"
    assert run.extras["formula"] == "sin(x0)"
    assert run.extras["symbolic_backend"] == "pysr"
    assert seen["fit_shape"] == (24, 1)
    assert seen["kwargs"]["deterministic"] is True
    assert seen["kwargs"]["parallelism"] == "serial"
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_operon_wrapper_forces_fortran_layout_and_reports_formula(monkeypatch):
    seen = {}

    class FakeOperon:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs
            self.model_ = object()
            self.stats_ = {"model_length": 7}

        def fit(self, x, y):
            seen["fit_fortran"] = bool(x.flags.f_contiguous)
            return self

        def predict(self, x):
            seen["predict_fortran"] = bool(x.flags.f_contiguous)
            return np.zeros(x.shape[0], dtype=float)

        def get_model_string(self, model, precision=8, names=None):
            seen["names"] = names
            return "0.5 * sin(x0)"

    pkg = types.ModuleType("pyoperon")
    skl = types.ModuleType("pyoperon.sklearn")
    skl.SymbolicRegressor = FakeOperon
    pkg.sklearn = skl
    monkeypatch.setitem(sys.modules, "pyoperon", pkg)
    monkeypatch.setitem(sys.modules, "pyoperon.sklearn", skl)

    spec, data = _tiny_data()
    run = train_operon(
        spec,
        data,
        5,
        {
            "population_size": 20,
            "generations": 3,
            "max_evaluations": 100,
            "max_time": 180.0,
        },
    )
    assert run.model_name == "operon"
    assert run.extras["formula"] == "0.5 * sin(x0)"
    assert run.extras["symbolic_backend"] == "operon"
    assert run.extras["symbolic_model_length"] == 7.0
    assert seen["fit_fortran"] and seen["predict_fortran"]
    assert seen["kwargs"]["max_time"] == 180
    assert isinstance(seen["kwargs"]["max_time"], int)
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_srkan_wrapper_uses_official_api_and_reports_formula(monkeypatch):
    seen = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit(self, x, y):
            seen["fit_shape"] = tuple(x.shape)
            return "x0"

    class FakeEvaluator:
        def __init__(self, expression):
            seen["expression"] = expression

        def __call__(self, x, params, mse=False):
            arr = np.asarray(x)
            return arr[:, 0]

    # The core RuleKAN test environment intentionally does not depend on JAX.
    # Stub the small part of the JAX API used by the wrapper so this remains a
    # unit test of the SR-KAN adapter rather than an integration test.
    jax_mod = types.ModuleType("jax")
    jax_mod.__path__ = []
    jax_mod.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)

    jnp_mod = types.ModuleType("jax.numpy")
    jnp_mod.asarray = np.asarray

    jr_mod = types.ModuleType("jax.random")
    jr_mod.key = lambda seed: seed

    jax_mod.numpy = jnp_mod
    jax_mod.random = jr_mod

    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp_mod)
    monkeypatch.setitem(sys.modules, "jax.random", jr_mod)

    mod = types.ModuleType("srkan")
    mod.regressor = FakeRegressor
    mod.SympyEvaluator = FakeEvaluator
    mod.__version__ = "test"
    monkeypatch.setitem(sys.modules, "srkan", mod)

    spec, data = _tiny_data()
    run = train_srkan(
        spec, data, 5,
        {"functions": ["linear"], "brute_force": False, "simplifications": False,
         "rand_constants": False, "backward_elim": False},
    )
    assert run.model_name == "srkan"
    assert run.extras["formula"] == "x0"
    assert run.extras["symbolic_backend"] == "srkan_official_github"
    assert seen["fit_shape"] == (24, 1)
    assert seen["kwargs"]["functions"] == ["linear"]
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_srkan_target_core_contains_only_matched_elementary_atoms(monkeypatch):
    from benchmarks.models import TARGET_CORE_SYMBOLIC_LIBRARY, _resolve_srkan_functions

    native = {
        "linear": object(), "square": object(), "inv_x": object(), "inv_x2": object(),
        "sqrt": object(), "log": object(), "exp": object(), "sin": object(),
        "cos": object(), "tanh": object(),
    }
    function_lib = types.SimpleNamespace(all_expr=dict(native))
    univariate = types.ModuleType("srkan.function_libraries.univariate")
    univariate.function_lib = function_lib
    function_libraries = types.ModuleType("srkan.function_libraries")
    monkeypatch.setitem(sys.modules, "srkan.function_libraries", function_libraries)
    monkeypatch.setitem(sys.modules, "srkan.function_libraries.univariate", univariate)

    out = _resolve_srkan_functions(types.SimpleNamespace(), ["target_core"])
    assert tuple(out) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert len(out) == 10
    assert {"gaussian", "log1p_sq", "sqrt1p_sq", "inv1p_sq"}.isdisjoint(out)



def test_symbolic_kan_wrapper_uses_official_training_and_exports_formula():
    root = Path(__file__).resolve().parents[1]
    upstream = root / "external" / "Pub_Symbolic_KANs" / "Exp_reaction_diffusion" / "symKanTraining.py"
    if not upstream.exists():
        pytest.skip("official Symbolic-KAN checkout is created by benchmarks/setup.sh")
    spec, data = _tiny_data()
    run = train_symbolic_kan(
        spec, data, 5,
        {
            "device": "cpu", "hidden_units": 2, "edges_per_unit": 1,
            "num_blocks": 1, "lib": ["x", "sin"],
            "epochs": 2, "adam_epochs": 1, "lbfgs_steps": 1,
            "lbfgs_max_iter": 1, "use_lbfgs": True,
            "middle_lbfgs_harden": False, "use_unit_gates": False,
            "train_prim_bias": False, "diagnostic_plots": False,
        },
    )
    assert run.model_name == "symbolic_kan"
    assert run.extras["symbolic_backend"] == "symbolic_kan_official_github"
    assert run.extras["symbolic_kan_exact_official_code"] is True
    assert run.extras["symbolic_kan_training_routine"] == "train_regression_onehot"
    assert run.extras["symbolic_kan_official_commit"] == "9481a822e73e5a7520c6c0a425a8a402f2878c03"
    assert isinstance(run.extras["formula"], str) and run.extras["formula"]
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics
    assert run.metrics["symbolic_export_vs_official_rmse"] < 1e-5


def test_pse_wrapper_uses_official_psrn_api_and_scores_exported_formula(monkeypatch):
    seen = {}

    class FakePSRNRegressor:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def fit(self, x, y, **kwargs):
            seen["fit_shape"] = tuple(x.shape)
            seen["fit_kwargs"] = kwargs
            return False, [("x0", 1.0, 0.0, 1)]

        def display_expr_table(self, sort_by="mse"):
            seen["sort_by"] = sort_by
            return [("x0", 1.0, 0.0, 1)]

    mod = types.ModuleType("psrn")
    mod.PSRN_Regressor = FakePSRNRegressor
    monkeypatch.setitem(sys.modules, "psrn", mod)

    spec, data = _tiny_data()
    run = train_pse(
        spec, data, 5,
        {"operators": ["Add", "Mul", "Identity"], "n_down_sample": 12,
         "n_symbol_layers": 2, "extra_input_slots": 1, "use_cpu": True},
    )
    assert run.model_name == "pse"
    assert run.extras["symbolic_backend"] == "pse_psrn_official"
    assert run.extras["formula"] == "x0"
    assert seen["fit_shape"] == (24, 1)
    assert seen["kwargs"]["variables"] == ["x0"]
    assert seen["fit_kwargs"]["n_down_sample"] == 12
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_rils_rols_wrapper_uses_public_sklearn_api(monkeypatch):
    seen = {}

    class FakeRILSROLSRegressor:
        def __init__(self, max_fit_calls=100000, max_seconds=100,
                     complexity_penalty=0.001, max_complexity=200,
                     sample_size=0.1, verbose=False, random_state=0):
            seen["kwargs"] = dict(
                max_fit_calls=max_fit_calls, max_seconds=max_seconds,
                complexity_penalty=complexity_penalty, max_complexity=max_complexity,
                sample_size=sample_size, verbose=verbose, random_state=random_state,
            )

        def fit(self, x, y):
            seen["fit_shape"] = tuple(np.asarray(x).shape)
            return self

        def predict(self, x):
            return np.asarray(x)[:, 0]

        def model_string(self):
            return "x0"

    pkg = types.ModuleType("rils_rols")
    sub = types.ModuleType("rils_rols.rils_rols")
    sub.RILSROLSRegressor = FakeRILSROLSRegressor
    pkg.rils_rols = sub
    monkeypatch.setitem(sys.modules, "rils_rols", pkg)
    monkeypatch.setitem(sys.modules, "rils_rols.rils_rols", sub)

    spec, data = _tiny_data()
    run = train_rils_rols(
        spec, data, 7,
        {"max_fit_calls": 123, "max_seconds": 4, "max_complexity": 17,
         "sample_size": 1.0},
    )
    assert run.model_name == "rils_rols"
    assert run.extras["symbolic_backend"] == "rils_rols_official"
    assert run.extras["formula"] == "x0"
    assert seen["fit_shape"] == (24, 1)
    assert seen["kwargs"]["max_fit_calls"] == 123
    assert seen["kwargs"]["max_seconds"] == 4
    assert seen["kwargs"]["max_complexity"] == 17
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_udsr_wrapper_enables_poly_and_gp_meld(monkeypatch):
    import json
    seen = {}

    class FakeProgram:
        def pretty(self):
            return "sin(x1)"

    class FakeDeepSymbolicRegressor:
        def __init__(self, config=None):
            seen["config_path"] = config
            with open(config, "r", encoding="utf-8") as fh:
                seen["config"] = json.load(fh)
            self.program_ = FakeProgram()

        def fit(self, x, y):
            seen["fit_shape"] = tuple(np.asarray(x).shape)
            return self

        def predict(self, x):
            return np.sin(np.asarray(x)[:, 0])

    mod = types.ModuleType("dso")
    mod.DeepSymbolicRegressor = FakeDeepSymbolicRegressor
    mod.__version__ = "test"
    monkeypatch.setitem(sys.modules, "dso", mod)

    spec, data = _tiny_data()
    run = train_udsr(spec, data, 11, {"n_samples": 1000, "batch_size": 50})
    assert run.model_name == "udsr"
    assert run.extras["symbolic_backend"] == "udsr_dso_official"
    assert run.extras["formula"] == "sin(x0)"
    assert seen["fit_shape"] == (24, 1)
    assert "poly" in seen["config"]["task"]["function_set"]
    assert seen["config"]["gp_meld"]["run_gp_meld"] is True
    assert seen["config"]["training"]["n_samples"] == 1000
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics



def test_sindy_static_dictionary_supports_third_order_interactions():
    from benchmarks.sindy_baseline import static_library
    x = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    theta, names = static_library(
        x, ["x0", "x1", "x2"], primitive_library=["x"], max_interaction_order=3
    )
    assert theta.shape[0] == 2
    assert any(name.count("*") == 2 and "x0" in name and "x1" in name and "x2" in name for name in names)


def test_sindy_wrapper_uses_pysindy_stlsq_and_exports_sparse_formula(monkeypatch):
    seen = {}

    class FakeSTLSQ:
        def __init__(self, threshold=0.1, alpha=0.05, max_iter=20,
                     normalize_columns=False, fit_intercept=False):
            seen.setdefault("thresholds", []).append(float(threshold))
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, theta, y):
            seen["theta_shape"] = tuple(np.asarray(theta).shape)
            coef = np.zeros(theta.shape[1], dtype=float)
            # Column 1 is the first non-constant primitive: x0.
            coef[1] = 1.0
            self.coef_ = coef.reshape(1, -1)
            return self

    mod = types.ModuleType("pysindy")
    mod.STLSQ = FakeSTLSQ
    monkeypatch.setitem(sys.modules, "pysindy", mod)

    spec, data = _tiny_data()
    run = train_sindy(
        spec, data, 5,
        {"primitive_library": ["x", "sin"], "max_interaction_order": 1,
         "thresholds": [0.001, 0.01]},
    )
    assert run.model_name == "sindy"
    assert run.extras["symbolic_backend"] == "pysindy_stlsq_static_library"
    assert run.extras["sindy_optimizer"] == "STLSQ"
    assert run.extras["sindy_variant"] == "main_capped_12"
    assert run.extras["sindy_max_active_terms"] == 12
    assert run.extras["sindy_active_terms"] <= 12
    assert "x0" in run.extras["formula"]
    assert seen["theta_shape"][0] == 24
    assert seen["thresholds"] == [0.001, 0.01]
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics



def test_sindy12_caps_dense_stlsq_support_and_refits(monkeypatch):
    class DenseSTLSQ:
        def __init__(self, **kwargs):
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, theta, y):
            # Deliberately activate every dictionary column so the adapter must
            # enforce the main-paper term budget itself.
            self.coef_ = np.ones((1, theta.shape[1]), dtype=float)
            return self

    mod = types.ModuleType("pysindy")
    mod.STLSQ = DenseSTLSQ
    monkeypatch.setitem(sys.modules, "pysindy", mod)

    # Three variables + x-only primitives through order three gives 20 columns
    # including the bias, so the unconstrained support has 19 non-bias terms.
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 3))
    y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + 0.2 * x[:, 2]

    from benchmarks.sindy_baseline import fit_static_sindy
    fit, pred = fit_static_sindy(
        x[:24], y[:24], x[24:32], y[24:32], x[32:],
        ["x0", "x1", "x2"], primitive_library=["x"],
        max_interaction_order=3, thresholds=[1e-3], max_active_terms=12,
    )
    assert fit.raw_nonzero_terms == 19
    assert fit.active_terms <= 12
    assert fit.max_active_terms == 12
    assert fit.support_capped is True
    assert np.isfinite(pred).all()


def test_unconstrained_sindy_preserves_dense_support(monkeypatch):
    class DenseSTLSQ:
        def __init__(self, **kwargs):
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, theta, y):
            self.coef_ = np.ones((1, theta.shape[1]), dtype=float)
            return self

    mod = types.ModuleType("pysindy")
    mod.STLSQ = DenseSTLSQ
    monkeypatch.setitem(sys.modules, "pysindy", mod)

    from benchmarks.sindy_baseline import fit_static_sindy
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 3))
    y = x[:, 0] + x[:, 1]
    fit, _ = fit_static_sindy(
        x[:24], y[:24], x[24:32], y[24:32], x[32:],
        ["x0", "x1", "x2"], primitive_library=["x"],
        max_interaction_order=3, thresholds=[1e-3], max_active_terms=None,
    )
    assert fit.raw_nonzero_terms == 19
    assert fit.active_terms == 19
    assert fit.max_active_terms is None
    assert fit.support_capped is False


def test_unconstrained_sindy_model_is_appendix_variant(monkeypatch):
    class SparseSTLSQ:
        def __init__(self, **kwargs):
            self.coef_ = None
            self.intercept_ = 0.0
        def fit(self, theta, y):
            coef = np.zeros(theta.shape[1], dtype=float)
            coef[1] = 1.0
            self.coef_ = coef.reshape(1, -1)
            return self

    mod = types.ModuleType("pysindy")
    mod.STLSQ = SparseSTLSQ
    monkeypatch.setitem(sys.modules, "pysindy", mod)
    spec, data = _tiny_data()
    run = train_sindy_unconstrained(
        spec, data, 5,
        {"primitive_library": ["x"], "max_interaction_order": 1, "thresholds": [0.001]},
    )
    assert run.model_name == "sindy_unconstrained"
    assert run.extras["sindy_variant"] == "appendix_unconstrained"
    assert run.extras["sindy_max_active_terms"] is None
    assert run.extras["sindy_formula_characters"] == len(run.extras["formula"])


def test_parfam_wrapper_uses_official_public_api(monkeypatch):
    seen = {}

    class FakeParFamWrapper:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs
            self.formula_reduced = "sin(x0)"

        def fit(self, x, y, **kwargs):
            seen["fit_shape"] = tuple(np.asarray(x).shape)
            seen["fit_kwargs"] = kwargs
            return self

        def predict(self, x):
            return np.sin(np.asarray(x)[:, 0])

    mod = types.ModuleType("parfam")
    mod.ParFamWrapper = FakeParFamWrapper
    monkeypatch.setitem(sys.modules, "parfam", mod)

    spec, data = _tiny_data()
    run = train_parfam(
        spec, data, 5,
        {"config_name": "small", "iterate": True, "functions": ["sin", "cos"], "time_limit": 3},
    )
    assert run.model_name == "parfam"
    assert run.extras["symbolic_backend"] == "parfam_official"
    assert run.extras["formula"] == "sin(x0)"
    assert seen["fit_shape"] == (24, 1)
    assert seen["fit_kwargs"]["time_limit"] == 3.0
    assert len(seen["kwargs"]["functions"]) == 2
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics


def test_eql_reproduction_trains_and_exports_formula():
    spec, data = _tiny_data()
    run = train_eql(
        spec, data, 3,
        {"device": "cpu", "unary_library": ["x", "sin", "cos"],
         "units_per_type": 1, "n_binary": 1, "total_layers": [2],
         "l1_lambdas": [1e-5], "steps_per_hidden_layer": 8,
         "phase1_frac": 0.25, "phase2_frac": 0.75, "penalty_every": 0,
         "batch_size": 8, "lr": 1e-3, "prune_threshold": 1e-6},
    )
    assert run.model_name == "eql"
    assert run.extras["symbolic_backend"] == "eql_div_pytorch_reproduction"
    assert run.extras["eql_official_source_executed"] is False
    assert run.extras["eql_selected_total_layers"] == 2
    assert run.extras["eql_candidate_total_layers"] == [2]
    assert isinstance(run.extras["formula"], str) and run.extras["formula"]
    assert "/" in run.extras["formula"]
    assert "test_rmse" in run.metrics and "symbolic_seconds" in run.metrics
