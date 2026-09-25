from __future__ import annotations

import inspect
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Dict, Sequence

import numpy as np
import torch

from symbolic_kan.utils import SYMBOLIC_LIB


DEFAULT_SINDY_LIBRARY = (
    "x", "x^2", "1/x", "1/x^2", "sqrt", "log", "exp", "sin", "cos", "tanh",
)


@dataclass
class SINDyStaticFit:
    coefficients: np.ndarray
    intercept: float
    threshold: float
    feature_names: list[str]
    library_size: int


def _factor_formula(name: str, var: str) -> str:
    mapping = {
        "x": f"({var})",
        "x^2": f"({var})**2",
        "1/x": f"1/({var})",
        "1/x^2": f"1/(({var})**2)",
        "sqrt": f"sqrt({var})",
        "log": f"log(Abs({var}))",
        "exp": f"exp({var})",
        "sin": f"sin({var})",
        "cos": f"cos({var})",
        "tanh": f"tanh({var})",
    }
    if name not in mapping:
        raise ValueError(f"unsupported SINDy primitive {name!r}")
    return mapping[name]


def static_library(
    x: np.ndarray,
    variable_names: Sequence[str],
    *,
    primitive_library: Sequence[str] = DEFAULT_SINDY_LIBRARY,
    max_interaction_order: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Explicit static dictionary for the SINDy sparse-regression control.

    The dictionary contains a constant, every requested elementary primitive on
    every input coordinate, and (optionally) pairwise products of those factors.
    It deliberately does not perform recursive composition or learn affine
    arguments; this is the controlled sparse-library limitation being measured.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("SINDy static library expects a 2-D design matrix")
    if len(variable_names) != x.shape[1]:
        raise ValueError("variable_names length must equal input dimension")
    lib = tuple(str(p) for p in primitive_library)
    missing = [p for p in lib if p not in SYMBOLIC_LIB]
    if missing:
        raise ValueError(f"SINDy library contains unsupported primitives: {missing}")

    tx = torch.as_tensor(x, dtype=torch.float64)
    factors: list[np.ndarray] = []
    factor_names: list[str] = []
    for j, var in enumerate(variable_names):
        col = tx[:, j]
        for primitive in lib:
            vals = SYMBOLIC_LIB[primitive][0](col).detach().cpu().numpy().astype(np.float64, copy=False)
            if vals.ndim != 1:
                vals = vals.reshape(-1)
            factors.append(vals)
            factor_names.append(_factor_formula(primitive, str(var)))

    cols = [np.ones(x.shape[0], dtype=np.float64)]
    names = ["1"]
    cols.extend(factors)
    names.extend(factor_names)

    if int(max_interaction_order) >= 2:
        # Pairwise products provide the usual SINDy-style tensor interactions
        # without exploding to the unrestricted recursive-expression grammar.
        for i, j in combinations_with_replacement(range(len(factors)), 2):
            prod = factors[i] * factors[j]
            if np.isfinite(prod).all():
                cols.append(prod)
                names.append(f"({factor_names[i]})*({factor_names[j]})")

    theta = np.column_stack(cols)
    finite_cols = np.isfinite(theta).all(axis=0)
    if not finite_cols.all():
        theta = theta[:, finite_cols]
        names = [n for n, keep in zip(names, finite_cols) if keep]
    return theta, names


def _optimizer_kwargs(cls, cfg: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    desired = {
        "threshold": float(threshold),
        "alpha": float(cfg.get("alpha", 1e-6)),
        "max_iter": int(cfg.get("max_iter", 100)),
        "normalize_columns": bool(cfg.get("normalize_columns", True)),
        "fit_intercept": False,
    }
    try:
        params = inspect.signature(cls).parameters
        return {k: v for k, v in desired.items() if k in params}
    except Exception:
        return {"threshold": float(threshold), "alpha": float(cfg.get("alpha", 1e-6))}


def fit_static_sindy(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    variable_names: Sequence[str],
    *,
    primitive_library: Sequence[str] = DEFAULT_SINDY_LIBRARY,
    max_interaction_order: int = 2,
    thresholds: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2),
    optimizer_config: Dict[str, Any] | None = None,
) -> tuple[SINDyStaticFit, np.ndarray]:
    try:
        import pysindy as ps
    except ImportError as exc:
        raise ImportError("SINDy baseline requires pysindy==2.1.0") from exc

    cfg = dict(optimizer_config or {})
    theta_train, feature_names = static_library(
        train_x, variable_names,
        primitive_library=primitive_library,
        max_interaction_order=max_interaction_order,
    )
    theta_val, val_names = static_library(
        val_x, variable_names,
        primitive_library=primitive_library,
        max_interaction_order=max_interaction_order,
    )
    theta_test, test_names = static_library(
        test_x, variable_names,
        primitive_library=primitive_library,
        max_interaction_order=max_interaction_order,
    )
    if feature_names != val_names or feature_names != test_names:
        raise RuntimeError("SINDy dictionary changed across train/validation/test splits")

    best = None
    ytr = np.asarray(train_y, dtype=np.float64).reshape(-1, 1)
    yv = np.asarray(val_y, dtype=np.float64).reshape(-1)
    for threshold in thresholds:
        opt = ps.STLSQ(**_optimizer_kwargs(ps.STLSQ, cfg, float(threshold)))
        opt.fit(theta_train, ytr)
        coef = np.asarray(opt.coef_, dtype=np.float64).reshape(-1)
        if coef.size != theta_train.shape[1]:
            coef = np.asarray(opt.coef_, dtype=np.float64).reshape(1, -1)[0]
        intercept_raw = getattr(opt, "intercept_", 0.0)
        intercept = float(np.asarray(intercept_raw, dtype=np.float64).reshape(-1)[0]) if np.size(intercept_raw) else 0.0
        pred_val = theta_val @ coef + intercept
        if not np.isfinite(pred_val).all():
            continue
        mse = float(np.mean((pred_val - yv) ** 2))
        nnz = int(np.count_nonzero(np.abs(coef) > 0))
        key = (mse, nnz, float(threshold))
        if best is None or key < best[0]:
            best = (key, coef.copy(), intercept, float(threshold))
    if best is None:
        raise RuntimeError("SINDy STLSQ produced no finite validation model")

    _, coef, intercept, threshold = best
    pred_test = theta_test @ coef + intercept
    if not np.isfinite(pred_test).all():
        raise ValueError("SINDy sparse-library model produced non-finite test predictions")
    fit = SINDyStaticFit(
        coefficients=coef,
        intercept=float(intercept),
        threshold=float(threshold),
        feature_names=feature_names,
        library_size=int(len(feature_names)),
    )
    return fit, pred_test


def sindy_formula(fit: SINDyStaticFit, *, coefficient_threshold: float = 1e-12) -> str:
    terms = []
    thr = float(coefficient_threshold)
    if abs(float(fit.intercept)) >= thr:
        terms.append(f"({float(fit.intercept):.16g})")
    for c, name in zip(fit.coefficients, fit.feature_names):
        cf = float(c)
        if abs(cf) < thr:
            continue
        terms.append(f"({cf:.16g})*({name})")
    return " + ".join(terms) if terms else "0"
