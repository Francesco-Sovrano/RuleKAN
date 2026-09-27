from __future__ import annotations

import inspect
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Dict, Sequence

import numpy as np
import torch

from rulekan.utils import SYMBOLIC_LIB


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
    raw_nonzero_terms: int
    active_terms: int
    max_active_terms: int | None
    support_capped: bool


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
    every input coordinate, and products of those factors up to the requested
    interaction order. It deliberately does not perform recursive composition or learn affine
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

    max_order = int(max_interaction_order)
    if max_order < 1 or max_order > 3:
        raise ValueError("SINDy max_interaction_order must be in {1,2,3}")
    for order in range(2, max_order + 1):
        # Tensor-product dictionary interactions up to the benchmark factor
        # order. This is still a fixed feature library, not recursive search.
        for idxs in combinations_with_replacement(range(len(factors)), order):
            prod = np.ones(x.shape[0], dtype=np.float64)
            for idx in idxs:
                prod = prod * factors[idx]
            if np.isfinite(prod).all():
                cols.append(prod)
                names.append("*".join(f"({factor_names[idx]})" for idx in idxs))

    theta = np.column_stack(cols)
    finite_cols = np.isfinite(theta).all(axis=0)
    if not finite_cols.all():
        theta = theta[:, finite_cols]
        names = [n for n, keep in zip(names, finite_cols) if keep]
    return theta, names




def _nonbias_active_indices(coef: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    """Indices of active non-constant dictionary terms.

    Column zero of :func:`static_library` is the constant/bias column and is
    deliberately not charged against the symbolic term budget.
    """
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef.size <= 1:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.abs(coef[1:]) > float(atol)) + 1


def _cap_and_refit_support(
    theta_train: np.ndarray,
    y_train: np.ndarray,
    coef: np.ndarray,
    max_active_terms: int,
) -> tuple[np.ndarray, int, int, bool]:
    """Enforce a non-bias term budget and jointly OLS-refit the retained terms.

    STLSQ coefficients can have very different numerical scales because the
    fixed dictionary mixes powers, reciprocals, exponentials and trigonometric
    columns.  When a candidate exceeds the budget, terms are therefore ranked
    by their RMS contribution ``|coef_j| * rms(theta_j)`` rather than raw
    coefficient magnitude.  The constant column is retained as an uncharged
    bias term, and all retained coefficients are refit jointly by least squares.
    """
    max_active_terms = int(max_active_terms)
    if max_active_terms < 1:
        raise ValueError("SINDy max_active_terms must be >= 1 or None")

    theta = np.asarray(theta_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    raw = np.asarray(coef, dtype=np.float64).reshape(-1)
    if theta.ndim != 2 or theta.shape[1] != raw.size:
        raise ValueError("SINDy support refit received incompatible design/coefficient shapes")

    active = _nonbias_active_indices(raw)
    raw_count = int(active.size)
    capped = raw_count > max_active_terms
    if capped:
        # Stable RMS column scale: direct squaring can overflow for otherwise
        # finite protected exponential/product features.
        scale = np.max(np.abs(theta), axis=0)
        scaled = np.divide(
            theta, scale, out=np.zeros_like(theta), where=scale[None, :] > 0
        )
        rms = scale * np.sqrt(np.mean(scaled * scaled, axis=0))
        score = np.abs(raw) * np.where(np.isfinite(rms), rms, 0.0)
        # Stable deterministic tie-break by feature index.
        ranked = sorted(active.tolist(), key=lambda j: (-float(score[j]), int(j)))
        selected = sorted(ranked[:max_active_terms])
    else:
        selected = sorted(active.tolist())

    # The dictionary's first column is exactly 1.  It acts as a bias and does
    # not count toward the 12-term symbolic budget.  Including it in every
    # refit also absorbs any estimator intercept without creating a 13th term.
    support = [0, *selected]
    refit = np.zeros_like(raw)
    if support:
        sol, *_ = np.linalg.lstsq(theta[:, support], y, rcond=None)
        refit[np.asarray(support, dtype=int)] = np.asarray(sol, dtype=np.float64).reshape(-1)

    active_after = int(_nonbias_active_indices(refit, atol=1e-14).size)
    if active_after > max_active_terms:
        raise RuntimeError("internal SINDy term-budget violation after OLS refit")
    return refit, raw_count, active_after, capped

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
    max_active_terms: int | None = 12,
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
    ytr_flat = ytr.reshape(-1)
    yv = np.asarray(val_y, dtype=np.float64).reshape(-1)
    if max_active_terms is not None and int(max_active_terms) < 1:
        raise ValueError("SINDy max_active_terms must be >= 1 or None")

    for threshold in thresholds:
        opt = ps.STLSQ(**_optimizer_kwargs(ps.STLSQ, cfg, float(threshold)))
        opt.fit(theta_train, ytr)
        raw_coef = np.asarray(opt.coef_, dtype=np.float64).reshape(-1)
        if raw_coef.size != theta_train.shape[1]:
            raw_coef = np.asarray(opt.coef_, dtype=np.float64).reshape(1, -1)[0]
        intercept_raw = getattr(opt, "intercept_", 0.0)
        raw_intercept = float(np.asarray(intercept_raw, dtype=np.float64).reshape(-1)[0]) if np.size(intercept_raw) else 0.0

        if max_active_terms is None:
            # Appendix/native-capacity sensitivity: preserve PySINDy's STLSQ
            # solution exactly rather than silently applying the main-paper cap.
            coef = raw_coef.copy()
            intercept = raw_intercept
            raw_nnz = int(_nonbias_active_indices(raw_coef).size)
            active_nnz = raw_nnz
            capped = False
        else:
            coef, raw_nnz, active_nnz, capped = _cap_and_refit_support(
                theta_train, ytr_flat, raw_coef, int(max_active_terms)
            )
            # The refit always contains the constant column explicitly.
            intercept = 0.0

        pred_val = theta_val @ coef + intercept
        if not np.isfinite(pred_val).all():
            continue
        mse = float(np.mean((pred_val - yv) ** 2))
        key = (mse, active_nnz, float(threshold))
        if best is None or key < best[0]:
            best = (
                key, coef.copy(), intercept, float(threshold),
                int(raw_nnz), int(active_nnz), bool(capped),
            )
    if best is None:
        raise RuntimeError("SINDy STLSQ produced no finite validation model")

    _, coef, intercept, threshold, raw_nnz, active_nnz, capped = best
    pred_test = theta_test @ coef + intercept
    if not np.isfinite(pred_test).all():
        raise ValueError("SINDy sparse-library model produced non-finite test predictions")
    fit = SINDyStaticFit(
        coefficients=coef,
        intercept=float(intercept),
        threshold=float(threshold),
        feature_names=feature_names,
        library_size=int(len(feature_names)),
        raw_nonzero_terms=int(raw_nnz),
        active_terms=int(active_nnz),
        max_active_terms=None if max_active_terms is None else int(max_active_terms),
        support_capped=bool(capped),
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
