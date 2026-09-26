from pathlib import Path
import sys
import types

import numpy as np

from benchmarks.sindy_baseline import fit_static_sindy


def _install_dense_fake_pysindy(monkeypatch):
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


def test_main_sindy_policy_never_exceeds_12_nonbias_terms(monkeypatch):
    _install_dense_fake_pysindy(monkeypatch)
    rng = np.random.default_rng(10)
    x = rng.normal(size=(48, 3))
    y = 0.7 * x[:, 0] - 1.1 * x[:, 1] + 0.3 * x[:, 2]

    fit, pred = fit_static_sindy(
        x[:28], y[:28], x[28:38], y[28:38], x[38:],
        ["x0", "x1", "x2"], primitive_library=["x"],
        max_interaction_order=3, thresholds=[1e-3], max_active_terms=12,
    )
    assert fit.raw_nonzero_terms == 19
    assert fit.active_terms <= 12
    assert fit.max_active_terms == 12
    assert fit.support_capped is True
    assert np.isfinite(pred).all()


def test_unconstrained_sindy_keeps_native_dense_support(monkeypatch):
    _install_dense_fake_pysindy(monkeypatch)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(48, 3))
    y = x[:, 0] + 0.5 * x[:, 1]

    fit, _ = fit_static_sindy(
        x[:28], y[:28], x[28:38], y[28:38], x[38:],
        ["x0", "x1", "x2"], primitive_library=["x"],
        max_interaction_order=3, thresholds=[1e-3], max_active_terms=None,
    )
    assert fit.raw_nonzero_terms == 19
    assert fit.active_terms == 19
    assert fit.max_active_terms is None
    assert fit.support_capped is False
