import sys
import types

import numpy as np

from benchmarks.models import TRAINERS, train_operon, train_pysr, train_srkan
from benchmarks.specs import TASKS, make_synthetic_data


def _tiny_data():
    spec = TASKS["same_var_exp_sin"]
    data = make_synthetic_data(spec, seed=2, train_n=24, val_n=8, test_n=10)
    return spec, data


def test_evolutionary_symbolic_regressors_are_registered():
    assert {"srkan", "pysr", "operon"}.issubset(TRAINERS)


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
    run = train_operon(spec, data, 5, {"population_size": 20, "generations": 3, "max_evaluations": 100})
    assert run.model_name == "operon"
    assert run.extras["formula"] == "0.5 * sin(x0)"
    assert run.extras["symbolic_backend"] == "operon"
    assert run.extras["symbolic_model_length"] == 7.0
    assert seen["fit_fortran"] and seen["predict_fortran"]
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
