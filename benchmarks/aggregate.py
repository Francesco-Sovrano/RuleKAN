from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import sympy as sp
except Exception:  # optional: aggregation still works without semantic formula parsing
    sp = None

try:
    from .specs import TASKS, make_synthetic_data
except Exception:
    try:
        from benchmarks.specs import TASKS, make_synthetic_data
    except Exception:
        TASKS = {}
        make_synthetic_data = None

try:
    from scipy.stats import friedmanchisquare, rankdata, wilcoxon
except Exception:  # aggregation still works without optional statistical output
    friedmanchisquare = rankdata = wilcoxon = None


# Four color families only. Variants are distinguished mainly by marker shape so
# the figures remain readable in print and for common forms of color-vision
# deficiency. Colors are from / close to the Okabe-Ito palette.
FAMILY_COLORS = {
    "rulekan": "#0072B2",   # blue
    "autosym": "#777777",   # neutral gray
    "gsr": "#D55E00",       # vermillion
    "gmp": "#009E73",       # green
    "other": "#555555",
}

MODEL_ORDER = [
    "rulekan",
    "rulekan_comp",
    "rulekan_adaptive",
    "power_rulekan",
    "power_rulekan_comp",
    "rulekan_distill",
    "rulekan_no_product",
    "rulekan_no_pruning",
    "rulekan_no_gmp",
    "rulekan_no_product_no_pruning",
    "rulekan_no_product_no_gmp",
    "rulekan_no_pruning_no_gmp",
    "rulekan_no_product_no_pruning_no_gmp",
    "rulekan_one_shot_prune",
    "rulekan_no_backfit",
    "rulekan_no_self_product",
    "rulekan_fast",
    "rulekan_fast_distill",
    "rulekan_omp_linear",
    "rulekan_omp_nonlinear",
    "rulekan_omp_full",
    "rulekan_graph",
    "rulekan_fast_omp_linear",
    "rulekan_fast_omp_nonlinear",
    "rulekan_fast_omp_full",
    "rulekan_fast_graph",
    "sisp",
    "sisp_comp",
    "sisp_fast",
    "autosym",
    "fastkan_autosym",
    "gsr",
    "fastkan_gsr",
    "gmp",
    "srkan",
    "symbolic_kan",
    "pse",
    "rils_rols",
    "udsr",
    "pysr",
    "operon",
    "anfis",
    "multkan_deep_autosym",
    "fast_multkan_deep_autosym",
    "multkan_deep_gsr",
    "fast_multkan_deep_gsr",
    "multkan_deep_gmp",
]

MODEL_LABELS = {
    "rulekan": "RuleKAN",
    "rulekan_comp": "RuleKAN-Comp",
    "rulekan_adaptive": "RuleKAN Adaptive",
    "power_rulekan": "PowerRuleKAN",
    "power_rulekan_comp": "PowerRuleKAN-Comp",
    "sisp": "SISP",
    "sisp_comp": "SISP-Comp",
    "sisp_fast": "SISP Fast",
    "rulekan_distill": "RuleKAN numeric→symbol distill",
    "rulekan_no_product": "No SumProduct",
    "rulekan_no_pruning": "No iterative pruning",
    "rulekan_no_gmp": "No GMP",
    "rulekan_no_product_no_pruning": "No SumProduct / pruning",
    "rulekan_no_product_no_gmp": "No SumProduct / GMP",
    "rulekan_no_pruning_no_gmp": "No pruning / GMP",
    "rulekan_no_product_no_pruning_no_gmp": "No SumProduct / pruning / GMP",
    "rulekan_one_shot_prune": "One-shot pruning",
    "rulekan_no_backfit": "No symbolic backfit",
    "rulekan_no_self_product": "No same-variable products",
    "rulekan_fast": "RuleKAN-RBF",
    "rulekan_fast_distill": "RuleKAN-RBF numeric→symbol distill",
    "rulekan_omp_linear": "RuleKAN + OMP-L",
    "rulekan_omp_nonlinear": "RuleKAN + OMP-NL",
    "rulekan_omp_full": "RuleKAN + OMP-Full",
    "rulekan_graph": "RuleKAN + graph",
    "rulekan_fast_omp_linear": "RuleKAN-RBF + OMP-L",
    "rulekan_fast_omp_nonlinear": "RuleKAN-RBF + OMP-NL",
    "rulekan_fast_omp_full": "RuleKAN-RBF + OMP-Full",
    "rulekan_fast_graph": "RuleKAN-RBF + graph",
    "autosym": "AutoSym",
    "fastkan_autosym": "FastKAN + AutoSym",
    "gsr": "GSR",
    "fastkan_gsr": "FastKAN + GSR",
    "gmp": "GMP",
    "srkan": "SR-KAN",
    "symbolic_kan": "Symbolic-KAN",
    "pse": "PSE",
    "rils_rols": "RILS-ROLS",
    "udsr": "uDSR",
    "pysr": "PySR",
    "operon": "Operon (GP)",
    "anfis": "ANFIS",
    "multkan_deep_autosym": "Deep MultKAN + AutoSym",
    "fast_multkan_deep_autosym": "Deep FastKAN + AutoSym",
    "multkan_deep_gsr": "Deep MultKAN + GSR",
    "fast_multkan_deep_gsr": "Deep FastKAN + GSR",
    "multkan_deep_gmp": "Deep MultKAN + GMP",
    "vanilla_kan": "KAN (numeric only)",
    "mlp": "MLP (numeric only)",
}

MODEL_MARKERS = {
    "rulekan": "o",
    "rulekan_comp": "*",
    "rulekan_adaptive": "P",
    "power_rulekan": "h",
    "power_rulekan_comp": "*",
    "sisp": "X",
    "sisp_comp": "*",
    "sisp_fast": "D",
    "rulekan_distill": "d",
    "rulekan_fast": "s",
    "rulekan_fast_distill": "D",
    "rulekan_omp_linear": "^",
    "rulekan_omp_nonlinear": "v",
    "rulekan_omp_full": "D",
    "rulekan_graph": "P",
    "rulekan_fast_omp_linear": "X",
    "rulekan_fast_omp_nonlinear": "<",
    "rulekan_fast_omp_full": ">",
    "rulekan_fast_graph": "*",
    "autosym": "o",
    "fastkan_autosym": "s",
    "gsr": "o",
    "fastkan_gsr": "s",
    "gmp": "D",
    "srkan": "8",
    "symbolic_kan": "D",
    "pse": "^",
    "rils_rols": "v",
    "udsr": "<",
    "pysr": "p",
    "operon": "X",
    "anfis": "H",
    "multkan_deep_autosym": "^",
    "fast_multkan_deep_autosym": "v",
    "multkan_deep_gsr": "^",
    "fast_multkan_deep_gsr": "v",
    "multkan_deep_gmp": "^",
}


def _family(model: str) -> str:
    if model.startswith("rulekan") or model.startswith("power_rulekan"):
        return "rulekan"
    if "autosym" in model:
        return "autosym"
    if model in {"gsr", "fastkan_gsr", "multkan_deep_gsr", "fast_multkan_deep_gsr"}:
        return "gsr"
    if model in {"gmp", "multkan_deep_gmp"}:
        return "gmp"
    return "other"


def _model_sort_key(model: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _ordered_tasks(df: pd.DataFrame) -> list[str]:
    """Order fuzzy-rule tasks first, then preserve suite/task appearance order."""
    if df.empty or "task" not in df.columns:
        return []
    rows = df[[c for c in ["suite", "task"] if c in df.columns]].drop_duplicates()
    if "suite" not in rows.columns:
        return rows["task"].astype(str).tolist()
    rows = rows.assign(_priority=rows["suite"].astype(str).ne("fuzzy_rules").astype(int))
    rows = rows.sort_values(["_priority"], kind="stable")
    return rows["task"].astype(str).tolist()


def _load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((run_dir / "runs").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No completed results yet._\n"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in x.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if not np.isfinite(v) else f"{v:.5g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _derive_symbolic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add stage-consistent normalized metrics used by all symbolic figures.

    Paper pipelines store their final symbolic predictor in ``test_*`` and their
    pre-extraction numerical model in ``numeric_test_*``. RuleKAN-family runs
    store the numerical predictor in ``test_*`` and the final formula in
    ``symbolic_test_*``. Canonical columns below reconcile those conventions.
    """
    x = df.copy()
    idx = x.index
    sym_n = _numeric(x.get("symbolic_test_nrmse"), idx)
    sym_r = _numeric(x.get("symbolic_test_rmse"), idx)
    test_n = _numeric(x.get("test_nrmse"), idx)
    test_r = _numeric(x.get("test_rmse"), idx)
    num_n = _numeric(x.get("numeric_test_nrmse"), idx)
    num_r = _numeric(x.get("numeric_test_rmse"), idx)

    model_name = x.get("model", pd.Series("", index=idx)).astype(str)
    symbolic_multkan_names = {
        "autosym", "fastkan_autosym", "gsr", "fastkan_gsr", "gmp",
        "multkan_deep_autosym", "fast_multkan_deep_autosym",
        "multkan_deep_gsr", "fast_multkan_deep_gsr", "multkan_deep_gmp",
        "srkan", "symbolic_kan", "pse", "rils_rols", "udsr", "pysr", "operon",
    }
    paper = x.get("paper_pipeline", pd.Series(np.nan, index=idx)).notna() | model_name.isin(symbolic_multkan_names)
    rulekan = model_name.str.startswith("rulekan") | model_name.str.startswith("power_rulekan")

    # Final symbolic error: explicit RuleKAN symbolic field when present; for
    # paper symbolic pipelines, the final test error already is symbolic.
    x["symbolic_nrmse"] = sym_n.where(sym_n.notna(), test_n.where(paper))
    x["symbolic_rmse"] = sym_r.where(sym_r.notna(), test_r.where(paper))

    # Numerical precursor: explicit paper numeric fields; for RuleKAN the test
    # fields are the numerical stage by construction.
    x["numeric_nrmse"] = num_n.where(num_n.notna(), test_n.where(rulekan))
    x["numeric_rmse"] = num_r.where(num_r.notna(), test_r.where(rulekan))

    x["symbolic_minus_numeric_nrmse"] = x["symbolic_nrmse"] - x["numeric_nrmse"]
    x["symbolic_over_numeric_nrmse"] = x["symbolic_nrmse"] / x["numeric_nrmse"].replace(0.0, np.nan)
    # Final predictive error uses the final symbolic expression when one exists;
    # otherwise it falls back to the method's final numerical predictor. This
    # allows fuzzy-system baselines such as ANFIS to appear in predictive fuzzy
    # comparisons without pretending that they share RuleKAN's symbolic grammar.
    x["final_nrmse"] = x["symbolic_nrmse"].where(x["symbolic_nrmse"].notna(), test_n)
    x["final_rmse"] = x["symbolic_rmse"].where(x["symbolic_rmse"].notna(), test_r)
    return x



def _build_method_redundancy(
    symbolic: pd.DataFrame,
    *,
    threshold_dex: float = 0.05,
    min_pairs: int = 3,
) -> pd.DataFrame:
    """Quantify whether two symbolic methods are empirically redundant.

    Comparisons are paired on task/seed and, when present, shared width/library
    settings.  ``threshold_dex`` is an equivalence margin on log10 RMSE: 0.05
    dex corresponds to an error ratio of about 1.12.  A pair is flagged as
    redundant when its median absolute paired difference is within the margin,
    at least half of paired runs are within the margin, and neither method has a
    >75% rate of wins *outside* the equivalence margin.
    """
    if symbolic.empty or "model" not in symbolic or "symbolic_rmse" not in symbolic:
        return pd.DataFrame()
    x = symbolic.copy()
    x["symbolic_rmse"] = pd.to_numeric(x["symbolic_rmse"], errors="coerce")
    x = x[np.isfinite(x["symbolic_rmse"]) & x["symbolic_rmse"].gt(0)].copy()
    if x.empty:
        return pd.DataFrame()

    pair_keys = [c for c in [
        "suite", "task", "seed", "shared_capacity_width",
        "shared_symbolic_library", "shared_symbolic_library_size",
    ] if c in x.columns]
    # Avoid turning a default all-NaN metadata column into a merge key that
    # prevents otherwise valid task/seed pairings.
    pair_keys = [c for c in pair_keys if x[c].notna().any()]
    for required in ("task", "seed"):
        if required not in pair_keys and required in x.columns:
            pair_keys.append(required)
    if not pair_keys:
        return pd.DataFrame()

    # Collapse accidental duplicate rows for the same model/condition using the
    # median, so reruns do not overweight a condition.
    g = (x.groupby(pair_keys + ["model"], dropna=False)["symbolic_rmse"]
           .median().reset_index())
    pivot = g.pivot_table(index=pair_keys, columns="model", values="symbolic_rmse", aggfunc="median")
    models = sorted([str(c) for c in pivot.columns], key=_model_sort_key)
    rows = []
    eps = float(threshold_dex)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            z = pivot[[a, b]].dropna()
            if z.empty:
                continue
            delta = np.log10(z[a].to_numpy(float)) - np.log10(z[b].to_numpy(float))
            abs_delta = np.abs(delta)
            n = int(len(delta))
            close = abs_delta <= eps
            a_better = delta < -eps
            b_better = delta > eps
            med_abs = float(np.median(abs_delta))
            close_frac = float(np.mean(close))
            a_win = float(np.mean(a_better))
            b_win = float(np.mean(b_better))
            dominant = max(a_win, b_win)
            redundant = bool(
                n >= int(min_pairs)
                and med_abs <= eps
                and close_frac >= 0.50
                and dominant < 0.75
            )
            rows.append({
                "model_a": a,
                "model_b": b,
                "n_paired": n,
                "median_abs_log10_rmse_diff": med_abs,
                "median_abs_rmse_ratio": float(10.0 ** med_abs),
                "equivalent_fraction": close_frac,
                "model_a_win_fraction": a_win,
                "model_b_win_fraction": b_win,
                "dominant_win_fraction": dominant,
                "equivalence_threshold_dex": eps,
                "redundant": redundant,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["redundant", "median_abs_log10_rmse_diff", "n_paired"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _write_method_redundancy_outputs(
    symbolic: pd.DataFrame,
    run_dir: Path,
    *,
    threshold_dex: float = 0.05,
    min_pairs: int = 3,
) -> pd.DataFrame:
    report = _build_method_redundancy(
        symbolic, threshold_dex=threshold_dex, min_pairs=min_pairs,
    )
    if report.empty:
        return report
    report.to_csv(run_dir / "method_redundancy.csv", index=False)
    redundant = report[report["redundant"]].copy()
    redundant.to_csv(run_dir / "redundant_method_pairs.csv", index=False)
    ratio = 10.0 ** float(threshold_dex)
    body = (
        "# Method redundancy report\n\n"
        f"Pairs are matched by task/seed (and shared width/library when present). "
        f"The default equivalence margin is {threshold_dex:.3g} dex, i.e. about a {ratio:.3g}x RMSE ratio. "
        "A pair is flagged only when the median paired difference is inside that margin, at least half "
        "of paired observations are equivalent, and neither method wins outside the margin in 75% or more observations.\n\n"
        "## Flagged redundant pairs\n\n"
    )
    cols = [
        "model_a", "model_b", "n_paired", "median_abs_log10_rmse_diff",
        "median_abs_rmse_ratio", "equivalent_fraction", "dominant_win_fraction",
    ]
    body += _markdown_table(redundant[cols] if not redundant.empty else redundant)
    body += "\n## All method pairs\n\n" + _markdown_table(report[cols], max_rows=300)
    (run_dir / "method_redundancy.md").write_text(body)
    return report


def _q25(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").quantile(0.25))


def _q75(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").quantile(0.75))



def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise error correction."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running = 0.0
    for rank, pos in enumerate(order):
        adj = min(1.0, (m - rank) * float(p[pos]))
        running = max(running, adj)
        out[pos] = running
    return out


def _task_level_metric(
    completed: pd.DataFrame,
    metric: str,
    *,
    regression_only: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Collapse seeds before inference so benchmark tasks are the statistical units."""
    if completed.empty or metric not in completed.columns or "model" not in completed.columns:
        return pd.DataFrame(), []
    x = completed.copy()
    if regression_only and "task_type" in x.columns:
        x = x[x["task_type"].astype(str).eq("regression")]
    x[metric] = pd.to_numeric(x[metric], errors="coerce")
    x = x[np.isfinite(x[metric])].copy()
    if x.empty:
        return pd.DataFrame(), []

    block_cols = [c for c in ("suite", "task") if c in x.columns]
    # Width/library sweeps are distinct benchmark conditions, not extra seeds.
    for c in ("shared_capacity_width", "shared_symbolic_library"):
        if c in x.columns and x[c].notna().any() and x[c].nunique(dropna=True) > 1:
            block_cols.append(c)
    if "task" not in block_cols:
        return pd.DataFrame(), []
    g = x.groupby(block_cols + ["model"], dropna=False)[metric].agg(["median", "count"]).reset_index()
    g = g.rename(columns={"median": metric, "count": "seed_count"})
    return g, block_cols


def _paired_task_tests(
    task_level: pd.DataFrame,
    block_cols: Sequence[str],
    metric: str,
    *,
    lower_better: bool = True,
    min_tasks: int = 5,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests over task-level seed medians."""
    cols = [
        "metric", "method_a", "method_b", "n_tasks", "median_a", "median_b",
        "geomean_a", "geomean_b", "median_ratio_a_over_b", "wins_a", "ties",
        "wins_b", "win_rate_a", "wilcoxon_statistic", "p_value", "p_holm",
        "significant_holm_0_05", "better_method",
    ]
    if task_level.empty or wilcoxon is None:
        return pd.DataFrame(columns=cols)
    models = sorted(task_level["model"].astype(str).unique(), key=_model_sort_key)
    rows = []
    for ia, a in enumerate(models):
        ga = task_level[task_level.model.astype(str).eq(a)][list(block_cols) + [metric]].rename(columns={metric: "a"})
        for b in models[ia + 1:]:
            gb = task_level[task_level.model.astype(str).eq(b)][list(block_cols) + [metric]].rename(columns={metric: "b"})
            m = ga.merge(gb, on=list(block_cols), how="inner")
            av = pd.to_numeric(m["a"], errors="coerce").to_numpy(float)
            bv = pd.to_numeric(m["b"], errors="coerce").to_numpy(float)
            ok = np.isfinite(av) & np.isfinite(bv) & (av >= 0) & (bv >= 0)
            av, bv = av[ok], bv[ok]
            if len(av) < int(min_tasks):
                continue
            # NRMSE/RMSE span orders of magnitude; test paired log error. A tiny
            # deterministic floor handles exact symbolic recovery without dropping it.
            floor = 1e-15
            la = np.log10(np.maximum(av, floor))
            lb = np.log10(np.maximum(bv, floor))
            d = la - lb
            if np.allclose(d, 0.0, rtol=0.0, atol=1e-14):
                stat, pv = 0.0, 1.0
            else:
                try:
                    res = wilcoxon(la, lb, zero_method="pratt", alternative="two-sided", method="auto")
                    stat, pv = float(res.statistic), float(res.pvalue)
                except Exception:
                    stat, pv = np.nan, np.nan
            if lower_better:
                wa = int(np.sum(av < bv)); wb = int(np.sum(av > bv))
            else:
                wa = int(np.sum(av > bv)); wb = int(np.sum(av < bv))
            ties = int(len(av) - wa - wb)
            ratio = float(10.0 ** np.median(d))
            rows.append({
                "metric": metric, "method_a": a, "method_b": b, "n_tasks": int(len(av)),
                "median_a": float(np.median(av)), "median_b": float(np.median(bv)),
                "geomean_a": float(10.0 ** np.mean(la)), "geomean_b": float(10.0 ** np.mean(lb)),
                "median_ratio_a_over_b": ratio, "wins_a": wa, "ties": ties, "wins_b": wb,
                "win_rate_a": float(wa / max(1, wa + wb)), "wilcoxon_statistic": stat,
                "p_value": pv,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["p_holm"] = _holm_adjust(out["p_value"].to_numpy(float))
    out["significant_holm_0_05"] = out["p_holm"].lt(0.05)
    def better(row):
        if not bool(row["significant_holm_0_05"]):
            return ""
        r = float(row["median_ratio_a_over_b"])
        if lower_better:
            return row["method_a"] if r < 1.0 else row["method_b"] if r > 1.0 else ""
        return row["method_a"] if r > 1.0 else row["method_b"] if r < 1.0 else ""
    out["better_method"] = out.apply(better, axis=1)
    return out[cols]


def _friedman_task_test(
    task_level: pd.DataFrame,
    block_cols: Sequence[str],
    metric: str,
    *,
    min_coverage_fraction: float = 0.75,
    min_tasks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman omnibus test on a documented high-coverage complete panel."""
    summary_cols = ["metric", "n_models", "n_tasks", "chi_square", "p_value", "kendall_w", "models"]
    rank_cols = ["metric", "model", "average_rank", "median_metric", "tasks_in_panel"]
    if task_level.empty or friedmanchisquare is None or rankdata is None:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=rank_cols)
    pivot = task_level.pivot_table(index=list(block_cols), columns="model", values=metric, aggfunc="first")
    if pivot.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=rank_cols)
    coverage = pivot.notna().sum(axis=0).sort_values(ascending=False)
    max_cov = int(coverage.max()) if len(coverage) else 0
    cutoff = max(int(min_tasks), int(np.ceil(float(min_coverage_fraction) * max_cov)))
    models = [str(m) for m in coverage.index[coverage >= cutoff].tolist()]
    models = sorted(models, key=_model_sort_key)
    if len(models) < 3:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=rank_cols)
    panel = pivot[models].dropna(axis=0, how="any")
    if len(panel) < int(min_tasks):
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=rank_cols)
    vals = [panel[m].to_numpy(float) for m in models]
    try:
        fr = friedmanchisquare(*vals)
        chi, pv = float(fr.statistic), float(fr.pvalue)
    except Exception:
        chi, pv = np.nan, np.nan
    k, n = len(models), len(panel)
    kw = float(chi / (n * (k - 1))) if np.isfinite(chi) and n > 0 and k > 1 else np.nan
    ranks = np.vstack([rankdata(row, method="average") for row in panel.to_numpy(float)])
    avg = ranks.mean(axis=0)
    rank_rows = [{
        "metric": metric, "model": m, "average_rank": float(avg[j]),
        "median_metric": float(np.median(panel[m].to_numpy(float))), "tasks_in_panel": int(n),
    } for j, m in enumerate(models)]
    summary = pd.DataFrame([{
        "metric": metric, "n_models": int(k), "n_tasks": int(n), "chi_square": chi,
        "p_value": pv, "kendall_w": kw, "models": ";".join(models),
    }])
    return summary[summary_cols], pd.DataFrame(rank_rows)[rank_cols]


def _write_statistical_rank_pdf(ranks: pd.DataFrame, pairwise: pd.DataFrame, path: Path, title: str) -> None:
    if ranks.empty:
        return
    z = ranks.sort_values(["average_rank", "model"], kind="stable").copy()
    best = str(z.iloc[0]["model"])
    significant_vs_best = set()
    if not pairwise.empty:
        for _, row in pairwise.iterrows():
            a, b = str(row["method_a"]), str(row["method_b"])
            if best not in {a, b} or not bool(row["significant_holm_0_05"]):
                continue
            other = b if a == best else a
            if str(row.get("better_method", "")) == best:
                significant_vs_best.add(other)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.42 * len(z) + 1.8)))
    y = np.arange(len(z))[::-1]
    for yi, (_, row) in zip(y, z.iterrows()):
        m = str(row["model"])
        ax.scatter([float(row["average_rank"])], [yi], s=58,
                   marker=MODEL_MARKERS.get(m, "o"), facecolors=FAMILY_COLORS[_family(m)],
                   edgecolors="black", linewidths=0.6)
        if m in significant_vs_best:
            ax.text(float(row["average_rank"]) + 0.12, yi, "*", va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(str(m)) for m in z["model"]])
    ax.set_xlabel("Average rank across common tasks (lower is better)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    fig.text(0.29, 0.035, f"* significantly worse than {_label(best)} by paired Wilcoxon + Holm (α=0.05)",
             fontsize=8, color="#555555", ha="left")
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.16, top=0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_statistical_outputs(completed: pd.DataFrame, run_dir: Path, fig_dir: Path) -> None:
    """Write inferential comparisons without treating the three seeds as independent tasks."""
    specs = [
        ("final_nrmse", "Final predictive NRMSE"),
        ("symbolic_nrmse", "Fully symbolic NRMSE"),
    ]
    sections = [
        "# Statistical comparison of benchmark methods\n",
        "Seeds are collapsed to a median within each task before inference, so the benchmark task—not each seed—is the statistical unit. Pairwise comparisons use two-sided Wilcoxon signed-rank tests on log10 NRMSE across matched tasks with Holm family-wise correction. The omnibus comparison uses a Friedman test on the common-task panel of methods with at least 75% of the maximum task coverage.\n",
    ]
    wrote = False
    for metric, label in specs:
        task_level, block_cols = _task_level_metric(completed, metric, regression_only=True)
        if task_level.empty:
            continue
        prefix = "statistical_" + metric
        task_level.to_csv(run_dir / f"{prefix}_task_medians.csv", index=False)
        pairwise = _paired_task_tests(task_level, block_cols, metric)
        friedman, ranks = _friedman_task_test(task_level, block_cols, metric)
        pairwise.to_csv(run_dir / f"{prefix}_pairwise.csv", index=False)
        friedman.to_csv(run_dir / f"{prefix}_friedman.csv", index=False)
        ranks.to_csv(run_dir / f"{prefix}_ranks.csv", index=False)
        _write_statistical_rank_pdf(ranks, pairwise, fig_dir / f"{prefix}_ranks.pdf", label + " — task-level ranks")
        sections.append(f"## {label}\n")
        if not friedman.empty:
            r = friedman.iloc[0]
            sections.append(
                f"Friedman: χ²={float(r.chi_square):.5g}, p={float(r.p_value):.5g}, "
                f"Kendall W={float(r.kendall_w):.5g}, tasks={int(r.n_tasks)}, models={int(r.n_models)}.\n"
            )
        else:
            sections.append("Friedman: insufficient common completed tasks/models at the current partial-run state.\n")
        if not ranks.empty:
            disp = ranks.sort_values("average_rank").copy()
            disp["model"] = disp["model"].map(_label)
            sections.append(_markdown_table(disp[["model", "average_rank", "median_metric", "tasks_in_panel"]], max_rows=80))
        wrote = True
    if wrote:
        (run_dir / "statistical_tests.md").write_text("\n".join(sections))


def _median_iqr(s: pd.Series) -> tuple[float, float, float, int]:
    z = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    z = z[np.isfinite(z)]
    if z.empty:
        return np.nan, np.nan, np.nan, 0
    return float(z.median()), float(z.quantile(0.25)), float(z.quantile(0.75)), int(len(z))


def _positive_floor(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if np.isfinite(v) and float(v) > 0], dtype=float)
    if vals.size == 0:
        return 1e-12
    return max(float(vals.min()) * 0.5, 1e-12)


def _legend_handles(models: Iterable[str]) -> list[Line2D]:
    handles = []
    for m in sorted(set(models), key=_model_sort_key):
        handles.append(Line2D(
            [0], [0], marker=MODEL_MARKERS.get(m, "o"), linestyle="None",
            markerfacecolor=FAMILY_COLORS[_family(m)], markeredgecolor="black",
            markeredgewidth=0.6, markersize=7, label=_label(m),
        ))
    return handles


def _dot_whisker_page(
    pdf: PdfPages,
    task: str,
    data: pd.DataFrame,
    value_col: str,
    xlabel: str,
    title_prefix: str,
    log_x: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    models = [m for m in sorted(data["model"].dropna().unique(), key=_model_sort_key)]
    stats = []
    for m in models:
        vals = pd.to_numeric(data.loc[data.model.eq(m), value_col], errors="coerce").dropna()
        vals = vals[np.isfinite(vals)]
        if log_x:
            vals = vals[vals > 0]
        if vals.empty:
            continue
        med, q1, q3, n = _median_iqr(vals)
        stats.append((m, vals.to_numpy(float), med, q1, q3, n))
    if not stats:
        return

    h = max(4.4, 0.47 * len(stats) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, h))
    y = np.arange(len(stats))[::-1]
    allvals = []
    for yi, (m, vals, med, q1, q3, n) in zip(y, stats):
        color = FAMILY_COLORS[_family(m)]
        marker = MODEL_MARKERS.get(m, "o")
        allvals.extend(vals.tolist())
        # Min-max thin whisker, IQR thick whisker, median marker, and individual
        # seed observations. This remains informative with only 1-3 partial runs.
        lo, hi = float(np.min(vals)), float(np.max(vals))
        ax.hlines(yi, lo, hi, color=color, linewidth=0.9, alpha=0.55, zorder=1)
        ax.hlines(yi, q1, q3, color=color, linewidth=4.5, alpha=0.9, zorder=2)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(vals, yi + jitter, s=20, facecolors="none", edgecolors=color,
                   linewidths=0.8, alpha=0.75, zorder=3)
        ax.scatter([med], [yi], s=56, marker=marker, facecolors=color,
                   edgecolors="black", linewidths=0.6, zorder=4)
        # n at right in axes coordinates, independent of log range.
        ax.text(1.005, yi, f"n={n}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels([_label(z[0]) for z in stats])
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title_prefix}: {task}")
    if log_x:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", which="both", alpha=0.18, linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.text(0.28, 0.035, "thin = min-max; thick = Q1-Q3; filled marker = median; open points = seeds",
             fontsize=8, color="#555555", ha="left")
    fig.subplots_adjust(left=0.28, right=0.94, bottom=0.20, top=0.9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _write_symbolic_error_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    """One readable horizontal boxplot page per task using raw symbolic RMSE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(symbolic):
            gt = symbolic[symbolic.task.eq(task)].copy()
            models = [m for m in sorted(gt.model.dropna().unique(), key=_model_sort_key)
                      if pd.to_numeric(gt.loc[gt.model.eq(m), "symbolic_rmse"], errors="coerce").notna().any()]
            if not models:
                continue
            data=[]
            labels=[]
            for m in models:
                vals=pd.to_numeric(gt.loc[gt.model.eq(m),"symbolic_rmse"],errors="coerce").dropna().astype(float)
                vals=vals[np.isfinite(vals) & (vals>=0)]
                if vals.empty:
                    continue
                data.append(vals.to_numpy())
                labels.append(m)
            if not data:
                continue
            fig,ax=plt.subplots(figsize=(11.7,max(5.0,0.48*len(data)+2.0)))
            positions=np.arange(1,len(data)+1)
            bp=ax.boxplot(data,orientation="horizontal",positions=positions,widths=.52,patch_artist=False,
                          showfliers=False,whis=(0,100),manage_ticks=False)
            for i,m in enumerate(labels):
                color=FAMILY_COLORS[_family(m)]
                for key in ("boxes","whiskers","caps"):
                    arts=bp[key][i:i+1] if key=="boxes" else bp[key][2*i:2*i+2]
                    for art in arts: art.set_color(color); art.set_linewidth(1.5)
                bp["medians"][i].set_color("black"); bp["medians"][i].set_linewidth(2.0)
                vals=data[i]
                jitter=np.linspace(-.08,.08,len(vals)) if len(vals)>1 else np.array([0.0])
                ax.scatter(vals,positions[i]+jitter,s=22,facecolors="white",edgecolors=color,linewidths=.9,zorder=3)
                med=float(np.median(vals))
                ax.scatter([med],[positions[i]],s=46,marker=MODEL_MARKERS.get(m,"o"),
                           facecolors=color,edgecolors="black",linewidths=.6,zorder=4)
                ax.text(1.003,positions[i],f"n={len(vals)}",transform=ax.get_yaxis_transform(),
                        va="center",ha="left",fontsize=8,color="#555555")
            ax.set_yticks(positions)
            ax.set_yticklabels([_label(m) for m in labels])
            ax.invert_yaxis()
            vals_all=np.concatenate(data)
            if np.all(vals_all>0) and (vals_all.max()/max(vals_all.min(),1e-300)>30):
                ax.set_xscale("log")
            ax.set_xlabel("Fully symbolic RMSE")
            ax.set_title(f"Symbolic regression error — {task}")
            ax.grid(axis="x",which="both",alpha=.18,linewidth=.7)
            fig.text(.29,.035,"box = Q1–Q3; center line/filled marker = median; whiskers = min–max; open points = seeds",
                     fontsize=8,color="#555555",ha="left")
            fig.subplots_adjust(left=.29,right=.94,bottom=.18,top=.9)
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)


def _write_symbolic_runtime_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    if "symbolic_seconds" not in symbolic.columns:
        return
    valid = symbolic[pd.to_numeric(symbolic["symbolic_seconds"], errors="coerce").gt(0)].copy()
    if valid.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(valid):
            _dot_whisker_page(
                pdf, task, valid[valid.task.eq(task)], "symbolic_seconds",
                "Symbolic conversion/search time [s] (lower is better)",
                "Symbolic runtime", log_x=True,
            )


def _visual_group_color(model: str) -> str:
    """Restrained palette for dense comparison figures.

    Method names are printed directly on the y axis, so color only separates
    the proposed RuleKAN family from external paper pipelines.
    """
    return "#1F5A94" if _family(model) == "rulekan" else "#4D4D4D"


def _pareto_models(rows: list[tuple[str, float, float]]) -> set[str]:
    """Return models that are not dominated in (RMSE, runtime)."""
    keep: set[str] = set()
    for m, err, sec in rows:
        dominated = False
        for m2, err2, sec2 in rows:
            if m2 == m:
                continue
            if err2 <= err and sec2 <= sec and (err2 < err or sec2 < sec):
                dominated = True
                break
        if not dominated:
            keep.add(m)
    return keep


def _write_pareto_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    """Readable two-axis dashboard for the accuracy/runtime trade-off.

    The dedicated RMSE and runtime PDFs already show full seed distributions.
    This figure therefore avoids another crowded scatter/seed cloud and aligns
    the two median/IQR summaries on the same method rows. Pareto-efficient
    methods are marked directly in their row labels.
    """
    if "symbolic_seconds" not in symbolic.columns:
        return
    x = symbolic.copy()
    x["symbolic_seconds"] = pd.to_numeric(x["symbolic_seconds"], errors="coerce")
    x["symbolic_rmse"] = pd.to_numeric(x["symbolic_rmse"], errors="coerce")
    x = x[x["symbolic_seconds"].gt(0) & x["symbolic_rmse"].gt(0)]
    if x.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(x):
            gt = x[x.task.eq(task)]
            rows = []
            for m in sorted(gt.model.dropna().unique(), key=_model_sort_key):
                g = gt[gt.model.eq(m)]
                em, e1, e3, en = _median_iqr(g["symbolic_rmse"])
                tm, t1, t3, tn = _median_iqr(g["symbolic_seconds"])
                if np.isfinite(em) and np.isfinite(tm) and em > 0 and tm > 0:
                    rows.append((m, em, e1, e3, tm, t1, t3, min(en, tn)))
            if not rows:
                continue

            # Accuracy first; ties are broken by runtime. This makes the left
            # panel immediately scan from strongest symbolic fit to weakest.
            rows.sort(key=lambda r: (r[1], r[4], _model_sort_key(r[0])))
            pareto = _pareto_models([(r[0], r[1], r[4]) for r in rows])
            n = len(rows)
            fig_h = max(5.2, 0.46 * n + 2.4)
            fig, (ax_e, ax_t) = plt.subplots(
                1, 2, figsize=(11.7, fig_h), sharey=True,
                gridspec_kw={"width_ratios": [1.12, 0.88], "wspace": 0.10},
            )
            ypos = np.arange(n)[::-1]
            labels = []
            for yi, row in zip(ypos, rows):
                m, em, e1, e3, tm, t1, t3, count = row
                color = _visual_group_color(m)
                labels.append(("* " if m in pareto else "   ") + _label(m))
                ax_e.errorbar(
                    em, yi,
                    xerr=np.array([[max(0.0, em-e1)], [max(0.0, e3-em)]]),
                    fmt="o", ms=6.5, mfc=color, mec="white", mew=0.7,
                    ecolor=color, elinewidth=2.0, capsize=3, zorder=3,
                )
                ax_t.errorbar(
                    tm, yi,
                    xerr=np.array([[max(0.0, tm-t1)], [max(0.0, t3-tm)]]),
                    fmt="o", ms=6.5, mfc=color, mec="white", mew=0.7,
                    ecolor=color, elinewidth=2.0, capsize=3, zorder=3,
                )
                ax_t.text(1.01, yi, f"n={count}", transform=ax_t.get_yaxis_transform(),
                          ha="left", va="center", fontsize=8, color="#666666")

            for ax in (ax_e, ax_t):
                ax.set_yticks(ypos)
                ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.7)
                ax.grid(axis="y", visible=False)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0)
            ax_e.set_yticklabels(labels, fontsize=9)
            ax_t.tick_params(labelleft=False)
            ax_e.set_xscale("log")
            ax_t.set_xscale("log")
            ax_e.set_xlabel("Fully symbolic RMSE  <- lower is better")
            ax_t.set_xlabel("Symbolic search time [s]  <- lower is better")
            ax_e.set_title("Accuracy", loc="left", fontsize=11, fontweight="bold")
            ax_t.set_title("Runtime", loc="left", fontsize=11, fontweight="bold")
            fig.suptitle(f"Symbolic accuracy / runtime - {task}", fontsize=14, y=0.97)
            fig.text(
                0.12, 0.035,
                "dot = median; horizontal bar = Q1-Q3 across seeds; * = Pareto-efficient in median RMSE/runtime",
                fontsize=8.5, color="#555555", ha="left",
            )
            fig.subplots_adjust(left=0.27, right=0.95, bottom=0.15, top=0.88)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_numeric_symbolic_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    """Dumbbell chart: numerical precursor versus final symbolic RMSE.

    Stage is encoded by marker fill rather than method color: numerical is an
    open gray marker, symbolic is a filled marker. Each row is one method, so
    no large method legend is necessary. The connector shows the direction and
    size of symbolic-conversion degradation or improvement at a glance.
    """
    both = symbolic[
        pd.to_numeric(symbolic["symbolic_rmse"], errors="coerce").notna()
        & pd.to_numeric(symbolic["numeric_rmse"], errors="coerce").notna()
    ].copy()
    if both.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(both):
            gt = both[both.task.eq(task)]
            rows = []
            for m in sorted(gt.model.dropna().unique(), key=_model_sort_key):
                g = gt[gt.model.eq(m)]
                nm, n1, n3, nn = _median_iqr(g["numeric_rmse"])
                sm, s1, s3, sn = _median_iqr(g["symbolic_rmse"])
                if np.isfinite(nm) and np.isfinite(sm) and nm > 0 and sm > 0:
                    rows.append((m, nm, n1, n3, sm, s1, s3, min(nn, sn)))
            if not rows:
                continue

            # Group RuleKAN first, then paper baselines; within each group sort
            # by final symbolic RMSE so the figure is useful as a ranking too.
            rows.sort(key=lambda r: (0 if _family(r[0]) == "rulekan" else 1, r[4], _model_sort_key(r[0])))
            n = len(rows)
            fig_h = max(5.4, 0.50 * n + 2.2)
            fig, ax = plt.subplots(figsize=(11.7, fig_h))
            ypos = np.arange(n)[::-1]

            for yi, row in zip(ypos, rows):
                m, nm, n1, n3, sm, s1, s3, count = row
                sym_color = "#1F5A94"
                ax.plot([nm, sm], [yi, yi], color="#B8B8B8", linewidth=2.0, zorder=1)
                # numerical precursor: open gray square + IQR
                ax.errorbar(
                    nm, yi-0.08,
                    xerr=np.array([[max(0.0, nm-n1)], [max(0.0, n3-nm)]]),
                    fmt="s", ms=6.2, mfc="white", mec="#777777", mew=1.2,
                    ecolor="#A5A5A5", elinewidth=1.4, capsize=2.5, zorder=3,
                )
                # final symbolic: filled circle + IQR
                ax.errorbar(
                    sm, yi+0.08,
                    xerr=np.array([[max(0.0, sm-s1)], [max(0.0, s3-sm)]]),
                    fmt="o", ms=7.0, mfc=sym_color, mec="white", mew=0.8,
                    ecolor=sym_color, elinewidth=1.8, capsize=2.5, zorder=4,
                )
                ratio = sm / nm
                if ratio >= 1.0:
                    ratio_txt = (f"{ratio:.2g}x higher" if ratio < 1000 else f"{ratio:.1e}x higher")
                else:
                    gain = 1.0 / max(ratio, 1e-300)
                    ratio_txt = (f"{gain:.2g}x lower" if gain < 1000 else f"{gain:.1e}x lower")
                ax.text(1.01, yi, ratio_txt, transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=8.5,
                        color="#333333", fontweight="bold" if ratio > 2 else "normal")

            ax.set_yticks(ypos)
            ax.set_yticklabels([_label(r[0]) for r in rows], fontsize=9)
            ax.set_xscale("log")
            ax.set_xlabel("RMSE (raw target units; log scale)")
            ax.set_title(f"Numerical vs symbolic RMSE - {task}", fontsize=14)
            ax.grid(axis="x", which="both", color="#D9D9D9", linewidth=0.7)
            ax.grid(axis="y", visible=False)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y", length=0)

            stage_handles = [
                Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="white",
                       markeredgecolor="#777777", markeredgewidth=1.2, markersize=7,
                       label="Numerical precursor (median  +/-  IQR)"),
                Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="#1F5A94",
                       markeredgecolor="white", markeredgewidth=0.8, markersize=8,
                       label="Final symbolic model (median  +/-  IQR)"),
            ]
            ax.legend(handles=stage_handles, loc="upper left", frameon=False, fontsize=9, ncol=2)
            fig.text(0.12, 0.035,
                     "gray connector = change after symbolic conversion; right column = symbolic RMSE / numerical RMSE",
                     fontsize=8.5, color="#555555", ha="left")
            fig.subplots_adjust(left=0.28, right=0.86, bottom=0.15, top=0.88)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    den = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt((p*(1-p)/n) + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def _write_redundancy_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    """Per-task paired dot/interval panels for RuleKAN redundancy diagnostics."""
    rk = symbolic[symbolic.model.astype(str).str.startswith("rulekan")].copy()
    if rk.empty or not ({"symbolic_max_abs_rule_corr", "symbolic_max_span_r2"} & set(rk.columns)):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(rk):
            gt = rk[rk.task.eq(task)]
            rows = []
            for m in sorted(gt.model.dropna().unique(), key=_model_sort_key):
                g = gt[gt.model.eq(m)]
                cm, c1, c3, cn = _median_iqr(g.get("symbolic_max_abs_rule_corr", pd.Series(dtype=float)))
                rm, r1, r3, rn = _median_iqr(g.get("symbolic_max_span_r2", pd.Series(dtype=float)))
                if np.isfinite(cm) or np.isfinite(rm):
                    rows.append((m, cm, c1, c3, rm, r1, r3, max(cn, rn)))
            if not rows:
                continue
            rows.sort(key=lambda r: (r[4] if np.isfinite(r[4]) else 2.0,
                                     r[1] if np.isfinite(r[1]) else 2.0,
                                     _model_sort_key(r[0])))
            n = len(rows)
            fig_h = max(4.8, 0.48*n + 2.2)
            fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(11.7, fig_h), sharey=True,
                                             gridspec_kw={"wspace": 0.08})
            ypos = np.arange(n)[::-1]
            for yi, row in zip(ypos, rows):
                m, cm, c1, c3, rm, r1, r3, count = row
                color = "#1F5A94"
                if np.isfinite(cm):
                    ax_c.errorbar(cm, yi, xerr=np.array([[max(0,cm-c1)],[max(0,c3-cm)]]),
                                  fmt="o", ms=6.5, mfc=color, mec="white", mew=.7,
                                  ecolor=color, elinewidth=2.0, capsize=3)
                if np.isfinite(rm):
                    ax_r.errorbar(rm, yi, xerr=np.array([[max(0,rm-r1)],[max(0,r3-rm)]]),
                                  fmt="o", ms=6.5, mfc=color, mec="white", mew=.7,
                                  ecolor=color, elinewidth=2.0, capsize=3)
                ax_r.text(1.01, yi, f"n={count}", transform=ax_r.get_yaxis_transform(),
                          ha="left", va="center", fontsize=8, color="#666666")
            for ax in (ax_c, ax_r):
                ax.set_xlim(-0.02, 1.02)
                ax.set_xticks([0, .25, .5, .75, 1.0])
                ax.grid(axis="x", color="#D9D9D9", linewidth=.7)
                ax.grid(axis="y", visible=False)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0)
            ax_c.set_yticks(ypos)
            ax_c.set_yticklabels([_label(r[0]) for r in rows], fontsize=9)
            ax_r.tick_params(labelleft=False)
            ax_c.set_title("Largest pairwise contribution correlation", loc="left", fontsize=10.5, fontweight="bold")
            ax_r.set_title("Largest span redundancy", loc="left", fontsize=10.5, fontweight="bold")
            ax_c.set_xlabel("max |corr(rule_i, rule_j)|  <- lower")
            ax_r.set_xlabel("max span R²  <- lower")
            fig.suptitle(f"Rule redundancy diagnostics - {task}", fontsize=14, y=.97)
            fig.text(.12,.035,"dot = median; bar = Q1-Q3 across seeds. Values near 1 indicate highly redundant rule contributions.",
                     fontsize=8.5,color="#555555",ha="left")
            fig.subplots_adjust(left=.29,right=.95,bottom=.15,top=.86)
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)



# ---------------------------------------------------------------------------
# Semantic fuzzy-rule backfill for external symbolic-regression formulas
# ---------------------------------------------------------------------------

def _max_boolean_matching_local(matrix: list[list[bool]]) -> int:
    """Maximum bipartite matching for the tiny fuzzy-rule score matrices."""
    if not matrix or not matrix[0]:
        return 0
    n_left, n_right = len(matrix), len(matrix[0])
    match_r = [-1] * n_right
    def dfs(i: int, seen: list[bool]) -> bool:
        for j in range(n_right):
            if not matrix[i][j] or seen[j]:
                continue
            seen[j] = True
            if match_r[j] < 0 or dfs(match_r[j], seen):
                match_r[j] = i
                return True
        return False
    hits = 0
    for i in range(n_left):
        hits += int(dfs(i, [False] * n_right))
    return hits


def _sympy_formula(formula: str, n_var: int):
    if sp is None or not isinstance(formula, str) or not formula.strip():
        return None, ()
    xs = tuple(sp.Symbol(f"x{i}", real=True) for i in range(max(0, int(n_var))))
    local = {f"x{i}": x for i, x in enumerate(xs)}
    local.update({f"x_{i}": x for i, x in enumerate(xs)})
    local.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
        "Abs": sp.Abs, "atan": sp.atan, "arctan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh,
    })
    text = str(formula).replace("^", "**")
    try:
        return sp.sympify(text, locals=local, evaluate=False), xs
    except Exception:
        try:
            return sp.sympify(text, locals=local), xs
        except Exception:
            return None, xs



def _coerce_float_vector(value, n: int):
    """Return a finite float vector from a JSON/pandas cell, if possible."""
    if value is None:
        return None
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = json.loads(text)
        except Exception:
            # CSV round-trips can render Python-style lists.
            try:
                import ast
                raw = ast.literal_eval(text)
            except Exception:
                return None
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except Exception:
        return None
    if len(arr) != int(n) or not np.isfinite(arr).all():
        return None
    return arr


def _row_input_standardization(row, spec):
    """Recover the raw->benchmark input affine map for one stored run.

    New v83+ records persist ``input_mean``/``input_std`` explicitly.  Older
    fuzzy benchmark records can be reconstructed exactly because synthetic data
    generation is deterministic in task, seed and split sizes.
    """
    n = int(getattr(spec, "n_var", 0) or 0)
    if n <= 0:
        return None, None
    mean = _coerce_float_vector(row.get("input_mean"), n)
    std = _coerce_float_vector(row.get("input_std"), n)
    if mean is not None and std is not None and np.all(std > 0):
        return mean, std
    if not bool(getattr(spec, "synthetic", False)) or make_synthetic_data is None:
        return None, None
    try:
        data = make_synthetic_data(
            spec,
            seed=int(row.get("seed", 0)),
            train_n=int(row.get("train_n", 1500)),
            val_n=int(row.get("val_n", 400)),
            test_n=int(row.get("test_n", 500)),
        )
        mean = np.asarray(data.input_mean.detach().cpu(), dtype=float).reshape(-1)
        std = np.asarray(data.input_std.detach().cpu(), dtype=float).reshape(-1)
        if len(mean) == n and len(std) == n and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0):
            return mean, std
    except Exception:
        pass
    return None, None


def _round_sympy_floats(expr, *, sig_digits: int = 7):
    """Quantize fitted floating constants for structural, not numeric, scoring.

    Symbolic engines often emit two copies of the same atom whose fitted
    constants differ only in the last few optimizer digits.  Quantizing to seven
    significant digits is conservative relative to the fuzzy structural
    tolerance and allows exact algebraic cancellation/collection without using
    an unrestricted ``simplify`` pass.
    """
    if sp is None or expr is None:
        return expr
    repl = {}
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Float):
            try:
                v = float(node)
                if math.isfinite(v):
                    # Optimizer residue at this scale is immaterial to a
                    # structural fuzzy-rule metric and otherwise prevents
                    # collection of phase-equivalent atoms.
                    if abs(v) <= 5e-7:
                        repl[node] = sp.Integer(0)
                    else:
                        repl[node] = sp.Float(f"{v:.{int(sig_digits)}g}")
            except Exception:
                pass
    return expr.xreplace(repl) if repl else expr


def _formula_to_raw_coordinates(expr, xs, input_mean, input_std):
    """Map a formula written in benchmark z-coordinates into raw task x."""
    if sp is None or expr is None or input_mean is None or input_std is None:
        return expr
    mean = np.asarray(input_mean, dtype=float).reshape(-1)
    std = np.asarray(input_std, dtype=float).reshape(-1)
    if len(mean) != len(xs) or len(std) != len(xs) or np.any(~np.isfinite(std)) or np.any(std <= 0):
        return expr
    # xreplace is simultaneous: the x on the right-hand side is not recursively
    # replaced again.  Thus z_j -> (x_j-mu_j)/sigma_j is safe with shared symbols.
    repl = {
        x: (x - sp.Float(float(mean[j]))) / sp.Float(float(std[j]))
        for j, x in enumerate(xs)
    }
    return expr.xreplace(repl)


def _formula_is_standardized_space(model: str, row=None) -> bool:
    """Whether the stored formula uses the standardized coordinates fed to SR."""
    if row is not None:
        tag = str(row.get("formula_input_space", "")).strip().lower()
        if tag in {"raw", "original"}:
            return False
        if tag in {"standardized", "normalized", "zscore", "z_score"}:
            return True
    m = str(model or "").lower()
    if m in {"srkan", "symbolic_kan", "pse", "rils_rols", "udsr", "pysr", "operon", "autosym", "fastkan_autosym", "gsr", "fastkan_gsr", "gmp"}:
        return True
    if m.startswith("multkan_deep_") or m.startswith("fast_multkan_deep_"):
        return True
    return False

def _affine_coeffs(expr, xs):
    """Return (variable_index, slope, intercept) for a univariate affine expr."""
    if sp is None:
        return None
    free = [i for i, x in enumerate(xs) if x in getattr(expr, "free_symbols", set())]
    if len(free) != 1:
        return None
    j = free[0]
    x = xs[j]
    try:
        p = sp.Poly(sp.expand(expr), x)
        if p.total_degree() > 1:
            return None
        a = float(p.coeff_monomial(x))
        b = float(p.coeff_monomial(1))
    except Exception:
        return None
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) <= 1e-12:
        return None
    return int(j), float(a), float(b)


def _is_numeric_sympy(expr) -> bool:
    return bool(sp is not None and getattr(expr, "is_number", False) and not getattr(expr, "free_symbols", set()))


def _bounded_dnf(expr, xs, *, max_terms: int = 256, _factor_context: bool = False):
    """Expand only outer sum/product algebra while keeping analytic atoms intact.

    SymPy's general ``expand`` can explode large SR expressions.  This routine
    performs exactly the distributive expansion needed for fuzzy DNF scoring,
    treats univariate affine gates such as ``1-x0`` as atomic factors, and stops
    when the bounded term budget is exceeded.
    """
    if sp is None or expr is None:
        return None
    if isinstance(expr, sp.Add):
        if _factor_context and _affine_coeffs(expr, xs) is not None:
            return [[expr]]
        out = []
        for arg in expr.args:
            zz = _bounded_dnf(arg, xs, max_terms=max_terms, _factor_context=False)
            if zz is None:
                return None
            out.extend(zz)
            if len(out) > int(max_terms):
                return None
        return out
    if isinstance(expr, sp.Mul):
        out = [[]]
        for arg in expr.args:
            zz = _bounded_dnf(arg, xs, max_terms=max_terms, _factor_context=True)
            if zz is None:
                return None
            nxt = []
            for left in out:
                for right in zz:
                    nxt.append(left + right)
                    if len(nxt) > int(max_terms):
                        return None
            out = nxt
        return out
    # Positive integer powers of a genuinely additive expression need outer
    # distribution; powers of a variable/affine atom remain a single analytic
    # factor (x^2, etc.).
    if isinstance(expr, sp.Pow):
        base, exponent = expr.as_base_exp()
        if bool(getattr(exponent, "is_Integer", False)) and int(exponent) > 1 and isinstance(base, sp.Add) and _affine_coeffs(base, xs) is None:
            n = int(exponent)
            if n <= 4:
                out = [[]]
                zz = _bounded_dnf(base, xs, max_terms=max_terms, _factor_context=True)
                if zz is None:
                    return None
                for _ in range(n):
                    nxt=[]
                    for left in out:
                        for right in zz:
                            nxt.append(left+right)
                            if len(nxt)>int(max_terms):
                                return None
                    out=nxt
                return out
    return [[expr]]



def _approximate_gate_factorization(coeff_expr, xs, spec, *, max_gate_vars: int = 3, tol: float = 0.03):
    """Replace a near-product of fuzzy memberships by an exact gate product.

    The fitted coefficient is evaluated only on the task's declared raw domain.
    A scalar multiple of products of x_j / (1-x_j) is accepted when its relative
    RMS residual is small.  This is a semantic canonicalization, not a refit to
    target y values.
    """
    if sp is None or coeff_expr is None:
        return coeff_expr
    free = [j for j, x in enumerate(xs) if x in getattr(coeff_expr, "free_symbols", set())]
    if not free or len(free) > int(max_gate_vars):
        return coeff_expr
    # Only [0,1]-like variables can be fuzzy membership gates in these tasks.
    gate_vars = []
    for j in free:
        try:
            lo, hi = map(float, spec.ranges[j])
        except Exception:
            return coeff_expr
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return coeff_expr
        # The current fuzzy suites use raw membership coordinates on [0,1].
        if abs(lo) > 1e-8 or abs(hi - 1.0) > 1e-8:
            return coeff_expr
        gate_vars.append(j)
    # Affine expressions are already handled perfectly by the ordinary gate
    # classifier; avoid unnecessary numerical work in the common case.
    if len(gate_vars) == 1 and _affine_coeffs(coeff_expr, xs) is not None:
        return coeff_expr
    try:
        f = sp.lambdify([xs[j] for j in gate_vars], coeff_expr, modules="numpy")
        axes = [np.linspace(0.0, 1.0, 7) for _ in gate_vars]
        mesh = np.meshgrid(*axes, indexing="ij")
        y = np.asarray(f(*mesh), dtype=float)
        if y.shape == ():
            y = np.full(mesh[0].shape, float(y), dtype=float)
        if not np.isfinite(y).all():
            return coeff_expr
    except Exception:
        return coeff_expr
    yv = y.reshape(-1)
    scale_ref = max(float(np.sqrt(np.mean(yv * yv))), 1e-12)
    best = None
    import itertools
    for orientations in itertools.product((-1, +1), repeat=len(gate_vars)):
        cand = np.ones_like(y, dtype=float)
        sym = sp.Integer(1)
        for arr, j, orient in zip(mesh, gate_vars, orientations):
            if orient > 0:
                cand = cand * arr
                sym = sym * xs[j]
            else:
                cand = cand * (1.0 - arr)
                sym = sym * (1 - xs[j])
        cv = cand.reshape(-1)
        den = float(np.dot(cv, cv))
        if den <= 1e-18:
            continue
        scl = float(np.dot(cv, yv) / den)
        err = float(np.sqrt(np.mean((scl * cv - yv) ** 2)) / scale_ref)
        if best is None or err < best[0]:
            best = (err, scl, sym)
    if best is not None and best[0] <= float(tol) and math.isfinite(best[1]):
        return sp.Float(best[1]) * best[2]
    return coeff_expr


def _canonicalize_fuzzy_formula(expr, xs, spec, *, max_terms: int = 256, sig_digits: int = 7):
    """Bounded semantic canonicalization for fuzzy-rule structural scoring.

    It intentionally avoids general ``sympy.simplify``.  Instead it quantizes
    insignificant optimizer noise, simplifies multiplicative exponential/power
    cancellations per DNF term, groups terms with the same nonlinear branch
    atoms, and factors the remaining coefficient into raw-space fuzzy gates when
    numerically justified.
    """
    if sp is None or expr is None:
        return expr
    expr = _round_sympy_floats(expr, sig_digits=sig_digits)
    terms = _bounded_dnf(expr, xs, max_terms=max_terms)
    if terms is None:
        return expr
    groups = {}
    for raw_factors in terms:
        try:
            term_expr = sp.powsimp(sp.Mul(*raw_factors), force=True)
            term_expr = _round_sympy_floats(term_expr, sig_digits=sig_digits)
        except Exception:
            term_expr = sp.Mul(*raw_factors)
        factors = list(sp.Mul.make_args(term_expr))
        coeff = sp.Integer(1)
        branch = []
        for fac in factors:
            if _is_numeric_sympy(fac):
                coeff *= fac
                continue
            # Affine factors are potential fuzzy gates and belong in the
            # coefficient to be collected/factored, not in the branch key.
            if _affine_coeffs(fac, xs) is not None:
                coeff *= fac
                continue
            # Polynomial gate products can occur as one factor after powsimp.
            if isinstance(fac, sp.Pow):
                base, exponent = fac.as_base_exp()
                if bool(getattr(exponent, "is_Integer", False)) and int(exponent) > 0 and _affine_coeffs(base, xs) is not None:
                    coeff *= fac
                    continue
            norm = _round_sympy_floats(fac, sig_digits=sig_digits)
            branch.append(norm)
        # SymPy's canonical ordering plus rounded constants makes copies of the
        # same fitted analytic atom share a key while keeping different slopes
        # or phases distinct.
        branch = sorted(branch, key=sp.default_sort_key)
        key = tuple(sp.srepr(z) for z in branch)
        if key not in groups:
            groups[key] = [sp.Integer(0), branch]
        groups[key][0] += coeff

    rebuilt = []
    for coeff, branch in groups.values():
        coeff = _round_sympy_floats(sp.expand(coeff), sig_digits=sig_digits)
        try:
            coeff = sp.factor(coeff)
        except Exception:
            pass
        coeff = _approximate_gate_factorization(coeff, xs, spec)
        coeff = _round_sympy_floats(coeff, sig_digits=sig_digits)
        try:
            if bool(sp.N(coeff) == 0):
                continue
        except Exception:
            pass
        rebuilt.append(coeff * sp.Mul(*branch))
    return sp.Add(*rebuilt) if rebuilt else sp.Integer(0)

def _gate_orientation_from_affine(j: int, a: float, b: float, spec) -> int:
    """Classify an affine gate up to arbitrary nonzero rule scaling."""
    try:
        lo, hi = map(float, spec.ranges[int(j)])
    except Exception:
        lo, hi = 0.0, 1.0
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0
    grid = np.linspace(lo, hi, 33)
    pred = a * grid + b
    targets = [(+1, grid), (-1, 1.0-grid)]
    scored=[]
    for orient, tgt in targets:
        den=float(np.dot(pred,pred))
        if den <= 1e-18:
            err=float('inf')
        else:
            scale=float(np.dot(pred,tgt)/den)
            sd=max(float(np.std(tgt)),1e-8)
            err=float(np.sqrt(np.mean((scale*pred-tgt)**2))/sd)
        scored.append((err,orient))
    scored.sort()
    return int(scored[0][1]) if scored and scored[0][0] <= 0.12 else 0


def _single_variable(expr, xs):
    free=[i for i,x in enumerate(xs) if x in getattr(expr,"free_symbols",set())]
    return int(free[0]) if len(free)==1 else None


def _classify_formula_factor(expr, xs, spec):
    """Return a structural factor dict or an explicit unsupported sentinel."""
    if _is_numeric_sympy(expr):
        return None
    # Any univariate affine factor can be a membership/complement gate.  We only
    # call it a gate when it is proportional (on the task domain) to x or 1-x.
    aff = _affine_coeffs(expr, xs)
    if aff is not None:
        j,a,b=aff
        orient=_gate_orientation_from_affine(j,a,b,spec)
        if orient:
            return {"variable":j,"operator":"x","role":"gate","orientation":orient}
        return {"variable":j,"operator":"x","role":"branch","orientation":0}

    # sqrt(1 + affine(x)^2) is RuleKAN's sqrt1p_sq family.  SymPy represents
    # sqrt as Pow(base, 1/2), so handle it before generic powers.
    if isinstance(expr, sp.Pow) and expr.exp == sp.Rational(1,2):
        base=expr.base
        j=_single_variable(base,xs)
        if j is not None:
            x=xs[j]
            try:
                poly=sp.Poly(sp.expand(base),x)
                # Accept c0 + c2*x^2 + c1*x after an affine square expansion.
                if poly.total_degree() <= 2 and abs(float(poly.coeff_monomial(x**2))) > 1e-12:
                    return {"variable":j,"operator":"sqrt1p_sq","role":"branch","orientation":0}
            except Exception:
                pass
        return {"variable":-1,"operator":"__unsupported__","role":"branch","orientation":0}

    if isinstance(expr, sp.Function):
        name=str(expr.func).lower()
        opmap={"sin":"sin","cos":"cos","exp":"exp","tanh":"tanh","atan":"arctan","log":"log","abs":"abs"}
        op=opmap.get(name)
        if op is not None and len(expr.args)==1:
            j=_single_variable(expr.args[0],xs)
            if j is not None:
                return {"variable":j,"operator":op,"role":"branch","orientation":0}

    if isinstance(expr, sp.Pow):
        base, exponent=expr.as_base_exp()
        aff=_affine_coeffs(base,xs)
        if aff is not None and getattr(exponent,"is_Integer",False):
            n=int(exponent); j=aff[0]
            if 2 <= n <= 5:
                return {"variable":j,"operator":f"x^{n}","role":"branch","orientation":0}
            if -5 <= n <= -1:
                return {"variable":j,"operator":f"1/x^{abs(n)}" if n != -1 else "1/x","role":"branch","orientation":0}

    return {"variable":-1,"operator":"__unsupported__","role":"branch","orientation":0}


def _formula_structural_rules(
    formula: str, spec, *, max_terms: int = 256,
    input_mean=None, input_std=None, formula_input_space: str = "raw",
):
    expr,xs=_sympy_formula(formula,int(spec.n_var or 0))
    if expr is None:
        return None
    if str(formula_input_space).lower() in {"standardized", "normalized", "zscore", "z_score"}:
        expr = _formula_to_raw_coordinates(expr, xs, input_mean, input_std)
    expr = _canonicalize_fuzzy_formula(expr, xs, spec, max_terms=max_terms)
    terms=_bounded_dnf(expr,xs,max_terms=max_terms)
    if terms is None:
        return None
    rules=[]
    for term in terms:
        factors=[]
        for fac in term:
            item=_classify_formula_factor(fac,xs,spec)
            if item is not None:
                factors.append(item)
        # A pure constant term is a readout bias, not a fuzzy rule.
        if not factors:
            continue
        rules.append(factors)
    return rules


def _formula_factor_matches(expected, learned) -> bool:
    if int(learned.get("variable",-1)) != int(expected.variable):
        return False
    lop=str(learned.get("operator","")); eop=str(expected.operator)
    if expected.role == "gate":
        return lop == "x" and str(learned.get("role")) == "gate" and int(learned.get("orientation",0)) == int(expected.orientation)
    if str(learned.get("role")) == "gate":
        return False
    return lop == eop or {lop,eop} <= {"sin","cos"}


def _formula_rule_matches(expected_rule, learned_rule) -> bool:
    exp=list(expected_rule.factors)
    if len(exp) != len(learned_rule):
        return False
    matrix=[[_formula_factor_matches(e,l) for l in learned_rule] for e in exp]
    return _max_boolean_matching_local(matrix) == len(exp)


def fuzzy_formula_recovery_scores(
    formula: str, task_name: str, *, input_mean=None, input_std=None,
    formula_input_space: str = "raw",
) -> dict[str,float]:
    """Structural fuzzy scores from a stored symbolic formula.

    This is intentionally a conservative parser: it scores only structure that
    is explicit in a bounded outer DNF expansion.  It never uses test targets or
    refits the external method.  Unsupported atoms remain unmatched and hurt
    precision rather than being silently discarded.
    """
    spec=TASKS.get(str(task_name)) if isinstance(TASKS,dict) else None
    if spec is None or not getattr(spec,"fuzzy_rules",()):
        return {}
    learned=_formula_structural_rules(
        formula, spec, input_mean=input_mean, input_std=input_std,
        formula_input_space=formula_input_space,
    )
    if learned is None:
        return {}
    expected=list(spec.fuzzy_rules)
    matrix=[[_formula_rule_matches(e,l) for l in learned] for e in expected]
    matched=_max_boolean_matching_local(matrix)
    p=matched/max(1,len(learned)); r=matched/max(1,len(expected)); f1=2*p*r/max(1e-12,p+r)
    exp_gate=[f for rr in expected for f in rr.factors if f.role=="gate"]
    exp_branch=[f for rr in expected for f in rr.factors if f.role=="branch"]
    learned_f=[f for rr in learned for f in rr]
    learned_gate=[f for f in learned_f if f.get("role")=="gate"]
    learned_branch=[f for f in learned_f if f.get("role")!="gate"]
    gm=[[_formula_factor_matches(e,l) for l in learned_gate] for e in exp_gate]
    bm=[[_formula_factor_matches(e,l) for l in learned_branch] for e in exp_branch]
    gh=_max_boolean_matching_local(gm); bh=_max_boolean_matching_local(bm)
    return {
        "fuzzy_expected_rules":float(len(expected)),
        "fuzzy_found_rules":float(len(learned)),
        "fuzzy_rule_precision":float(p),"fuzzy_rule_recall":float(r),"fuzzy_rule_f1":float(f1),
        "fuzzy_gate_precision":float(gh/max(1,len(learned_gate))),"fuzzy_gate_recall":float(gh/max(1,len(exp_gate))),
        "fuzzy_branch_precision":float(bh/max(1,len(learned_branch))),"fuzzy_branch_recall":float(bh/max(1,len(exp_branch))),
        "fuzzy_exact_structure_recovery":float(matched==len(expected)==len(learned)),
        "fuzzy_rule_count_error":float(len(learned)-len(expected)),
        "fuzzy_rule_count_abs_error":float(abs(len(learned)-len(expected))),
        "fuzzy_formula_semantic_backfill":1.0,
    }


def _backfill_fuzzy_formula_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing fuzzy structural metrics from stored symbolic formulas.

    Existing native RuleKAN metrics always win.  This makes historical SR-KAN,
    PySR, Operon and MultKAN-extractor result JSONs eligible for the same fuzzy
    recovery plots without rerunning them.
    """
    if df.empty or "suite" not in df.columns or "formula" not in df.columns:
        return df
    x=df.copy()
    fuzzy_mask=x["suite"].astype(str).eq("fuzzy_rules")
    if not fuzzy_mask.any():
        return x
    metric_names=[
        "fuzzy_expected_rules","fuzzy_found_rules","fuzzy_rule_precision","fuzzy_rule_recall","fuzzy_rule_f1",
        "fuzzy_gate_precision","fuzzy_gate_recall","fuzzy_branch_precision","fuzzy_branch_recall",
        "fuzzy_exact_structure_recovery","fuzzy_rule_count_error","fuzzy_rule_count_abs_error",
        "fuzzy_formula_semantic_backfill",
    ]
    for c in metric_names:
        if c not in x.columns:
            x[c]=np.nan
    for idx,row in x[fuzzy_mask].iterrows():
        if pd.notna(row.get("fuzzy_rule_f1")):
            continue
        formula=row.get("formula")
        if not isinstance(formula,str) or not formula.strip():
            continue
        task_name = str(row.get("task", ""))
        spec = TASKS.get(task_name) if isinstance(TASKS, dict) else None
        input_mean = input_std = None
        input_space = "raw"
        if spec is not None and _formula_is_standardized_space(str(row.get("model", "")), row):
            input_space = "standardized"
            input_mean, input_std = _row_input_standardization(row, spec)
            # Without the affine map, scoring standardized coordinates against
            # raw fuzzy gates is known to be invalid; leave metrics missing.
            if input_mean is None or input_std is None:
                continue
        scores=fuzzy_formula_recovery_scores(
            formula, task_name, input_mean=input_mean, input_std=input_std,
            formula_input_space=input_space,
        )
        for k,v in scores.items():
            if k in x.columns:
                x.at[idx,k]=v
    return x

def _write_fuzzy_recovery_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    """One task per page with aligned structural-recovery dot plots.

    Heatmaps are avoided because they hide uncertainty and become unreadable as
    the method/task matrix grows. Continuous scores show median and IQR. Exact
    recovery is a success fraction with a Wilson 95% interval.
    """
    fuzzy = symbolic[symbolic.get("suite", pd.Series("", index=symbolic.index)).eq("fuzzy_rules")].copy()
    if fuzzy.empty:
        return
    structural = fuzzy[pd.to_numeric(fuzzy.get("fuzzy_rule_f1"), errors="coerce").notna()].copy()
    if structural.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(structural):
            gt = structural[structural.task.eq(task)]
            rows = []
            for m in sorted(gt.model.dropna().unique(), key=_model_sort_key):
                g = gt[gt.model.eq(m)]
                rf = _median_iqr(g.get("fuzzy_rule_f1", pd.Series(dtype=float)))
                gr = _median_iqr(g.get("fuzzy_gate_recall", pd.Series(dtype=float)))
                br = _median_iqr(g.get("fuzzy_branch_recall", pd.Series(dtype=float)))
                ex = pd.to_numeric(g.get("fuzzy_exact_structure_recovery", pd.Series(dtype=float)), errors="coerce").dropna()
                ex = ex[np.isfinite(ex)]
                if rf[3] == 0 and gr[3] == 0 and br[3] == 0 and len(ex) == 0:
                    continue
                succ = int(np.sum(ex >= 0.5))
                exn = int(len(ex))
                exrate = succ/exn if exn else np.nan
                exlo, exhi = _wilson_interval(succ, exn)
                rows.append((m, rf, gr, br, exrate, exlo, exhi, succ, exn))
            if not rows:
                continue
            rows.sort(key=lambda r: (-(r[1][0] if np.isfinite(r[1][0]) else -1), _model_sort_key(r[0])))
            n = len(rows)
            fig_h = max(5.0, 0.50*n + 2.3)
            fig, axes = plt.subplots(1, 4, figsize=(13.5, fig_h), sharey=True,
                                     gridspec_kw={"wspace": 0.18})
            ypos = np.arange(n)[::-1]
            metrics = [
                (1, "Expanded-rule F1"),
                (2, "Gate / complement recall"),
                (3, "Branch-function recall"),
            ]
            for ax, (idx, title) in zip(axes[:3], metrics):
                for yi, row in zip(ypos, rows):
                    m = row[0]
                    med, q1, q3, count = row[idx]
                    if not np.isfinite(med):
                        continue
                    color = _visual_group_color(m)
                    ax.errorbar(med, yi, xerr=np.array([[max(0,med-q1)],[max(0,q3-med)]]),
                                fmt="o", ms=6.5, mfc=color, mec="white", mew=.7,
                                ecolor=color, elinewidth=2.0, capsize=3)
                ax.set_title(title, fontsize=9.5, fontweight="bold")
            ax_ex = axes[3]
            for yi, row in zip(ypos, rows):
                m, _, _, _, rate, lo, hi, succ, exn = row
                if not np.isfinite(rate):
                    continue
                color = _visual_group_color(m)
                ax_ex.errorbar(rate, yi,
                               xerr=np.array([[max(0,rate-lo)],[max(0,hi-rate)]]),
                               fmt="D", ms=6.3, mfc=color, mec="white", mew=.7,
                               ecolor=color, elinewidth=1.8, capsize=3)
                ax_ex.text(1.03, yi, f"{succ}/{exn}", transform=ax_ex.get_yaxis_transform(),
                           ha="left", va="center", fontsize=8, color="#555555")
            ax_ex.set_title("Exact structure recovery", fontsize=9.5, fontweight="bold")

            for ax in axes:
                ax.set_xlim(-0.03, 1.03)
                ax.set_xticks([0, .5, 1.0])
                ax.grid(axis="x", color="#D9D9D9", linewidth=.7)
                ax.grid(axis="y", visible=False)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.tick_params(axis="x", labelsize=8.5)
                ax.set_xlabel("score")
            axes[0].set_yticks(ypos)
            axes[0].set_yticklabels([_label(r[0]) for r in rows], fontsize=9)
            for ax in axes[1:]:
                ax.tick_params(labelleft=False)
            fig.suptitle(f"Fuzzy if/then structural recovery - {task}", fontsize=14, y=.97)
            fig.text(.12,.035,
                     "first 3 panels: median  +/-  Q1-Q3 across seeds; exact recovery: success fraction  +/-  Wilson 95% interval; right labels = successes/runs",
                     fontsize=8.3,color="#555555",ha="left")
            fig.subplots_adjust(left=.23,right=.95,bottom=.15,top=.86)
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)



DEEP_MULTKAN_PAIRS = {
    "autosym": "multkan_deep_autosym",
    "fastkan_autosym": "fast_multkan_deep_autosym",
    "gsr": "multkan_deep_gsr",
    "fastkan_gsr": "fast_multkan_deep_gsr",
    "gmp": "multkan_deep_gmp",
}

def _build_multkan_depth_pairs(symbolic: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for shallow,deep in DEEP_MULTKAN_PAIRS.items():
        key_cols=["suite","task","seed"]
        if "shared_capacity_width" in symbolic.columns:
            key_cols.append("shared_capacity_width")
        a=symbolic[symbolic.model.eq(shallow)][key_cols+["symbolic_rmse","symbolic_seconds"]].copy()
        b=symbolic[symbolic.model.eq(deep)][key_cols+["symbolic_rmse","symbolic_seconds"]].copy()
        if a.empty or b.empty:
            continue
        a=a.rename(columns={"symbolic_rmse":"shallow_symbolic_rmse","symbolic_seconds":"shallow_symbolic_seconds"})
        b=b.rename(columns={"symbolic_rmse":"deep_symbolic_rmse","symbolic_seconds":"deep_symbolic_seconds"})
        z=a.merge(b,on=key_cols,how="inner")
        for _,r in z.iterrows():
            sr=float(r.shallow_symbolic_rmse); dr=float(r.deep_symbolic_rmse)
            st=float(r.shallow_symbolic_seconds) if pd.notna(r.shallow_symbolic_seconds) else np.nan
            dt=float(r.deep_symbolic_seconds) if pd.notna(r.deep_symbolic_seconds) else np.nan
            rows.append({
                "suite":r.suite,"task":r.task,"seed":int(r.seed),
                **({"shared_capacity_width": int(r.shared_capacity_width)} if "shared_capacity_width" in z.columns and pd.notna(r.shared_capacity_width) else {}),
                "shallow_model":shallow,"deep_model":deep,
                "shallow_symbolic_rmse":sr,"deep_symbolic_rmse":dr,
                "deep_over_shallow_rmse":dr/sr if np.isfinite(sr) and sr>0 and np.isfinite(dr) else np.nan,
                "shallow_symbolic_seconds":st,"deep_symbolic_seconds":dt,
                "deep_over_shallow_runtime":dt/st if np.isfinite(st) and st>0 and np.isfinite(dt) else np.nan,
            })
    return pd.DataFrame(rows)

def _write_multkan_depth_pdf(symbolic: pd.DataFrame, path: Path) -> None:
    pairs=_build_multkan_depth_pairs(symbolic)
    if pairs.empty:
        return
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(pairs):
            g=pairs[pairs.task.eq(task)]
            rows=[]
            for shallow,deep in DEEP_MULTKAN_PAIRS.items():
                h=g[g.shallow_model.eq(shallow)]
                if h.empty: continue
                sm,sq1,sq3,_=_median_iqr(h.shallow_symbolic_rmse)
                dm,dq1,dq3,_=_median_iqr(h.deep_symbolic_rmse)
                rows.append((shallow,deep,sm,sq1,sq3,dm,dq1,dq3,len(h)))
            if not rows: continue
            fig,ax=plt.subplots(figsize=(10.5,max(4.2,0.62*len(rows)+2.2)))
            y=np.arange(len(rows))[::-1]
            for yi,(shallow,deep,sm,sq1,sq3,dm,dq1,dq3,n) in zip(y,rows):
                color=FAMILY_COLORS[_family(shallow)]
                if np.isfinite(sm) and np.isfinite(dm):
                    ax.plot([sm,dm],[yi,yi],color="#B8B8B8",lw=1.2,zorder=1)
                if np.isfinite(sm):
                    ax.errorbar(sm,yi,xerr=np.array([[max(0,sm-sq1)],[max(0,sq3-sm)]]),
                                fmt="o",mfc="white",mec=color,mew=1.4,ms=6.5,ecolor=color,elinewidth=1.4,capsize=3,zorder=3)
                if np.isfinite(dm):
                    ax.errorbar(dm,yi,xerr=np.array([[max(0,dm-dq1)],[max(0,dq3-dm)]]),
                                fmt="^",mfc=color,mec="white",mew=.7,ms=7.3,ecolor=color,elinewidth=1.4,capsize=3,zorder=4)
                ratio=dm/sm if np.isfinite(sm) and sm>0 and np.isfinite(dm) else np.nan
                if np.isfinite(ratio):
                    ax.text(1.01,yi,f"deep/shallow={ratio:.3g}  n={n}",transform=ax.get_yaxis_transform(),va="center",fontsize=8,color="#555555")
            positive=[v for row in rows for v in (row[2],row[5]) if np.isfinite(v) and v>0]
            if positive and max(positive)/min(positive)>50:
                ax.set_xscale("log")
            ax.set_yticks(y)
            ax.set_yticklabels([_label(r[0]).replace("FastKAN + ","FastKAN ") for r in rows],fontsize=9)
            ax.set_xlabel("Final symbolic RMSE")
            ax.set_title(f"Shallow vs late-multiplication MultKAN - {task}",fontsize=13)
            ax.grid(axis="x",alpha=.18)
            ax.grid(axis="y",visible=False)
            for spine in ("top","right","left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y",length=0)
            handles=[
                Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor='#444444',label='shallow multiplication',markersize=7),
                Line2D([0],[0],marker='^',color='none',markerfacecolor='#444444',markeredgecolor='white',label='late hidden-layer multiplication',markersize=8),
            ]
            ax.legend(handles=handles,loc="upper right",frameon=False,fontsize=8.5)
            fig.subplots_adjust(left=.27,right=.80,bottom=.14,top=.88)
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)

def _build_fuzzy_predictive_summary(fuzzy: pd.DataFrame) -> pd.DataFrame:
    """Predictive comparison for all completed fuzzy-task methods.

    ``final_nrmse`` is symbolic when a final symbolic model exists and numerical
    otherwise. Structural fuzzy-rule metrics remain in ``fuzzy_summary`` and are
    intentionally not fabricated for methods with a different fuzzy grammar.
    """
    if fuzzy.empty or "final_nrmse" not in fuzzy.columns:
        return pd.DataFrame()
    rows=[]
    group_cols=["task","model"]
    if "shared_capacity_width" in fuzzy.columns and pd.to_numeric(fuzzy["shared_capacity_width"],errors="coerce").notna().any():
        group_cols.append("shared_capacity_width")
    for keys,g in fuzzy.groupby(group_cols,dropna=False):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(group_cols,keys))
        med,q1,q3,n=_median_iqr(g["final_nrmse"])
        tt,t1,t3,tn=_median_iqr(g.get("elapsed_seconds",pd.Series(dtype=float)))
        row.update({"final_nrmse_median":med,"final_nrmse_q25":q1,"final_nrmse_q75":q3,
                    "elapsed_seconds_median":tt,"elapsed_seconds_q25":t1,"elapsed_seconds_q75":t3,"n":n})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out=pd.DataFrame(rows)
    sort_cols=["task","model"] + (["shared_capacity_width"] if "shared_capacity_width" in out.columns else [])
    return out.sort_values(sort_cols,key=lambda z:z.map(lambda m:_model_sort_key(m)[0]) if z.name=="model" else z)


def _write_fuzzy_predictive_pdf(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    tasks=list(dict.fromkeys(summary["task"].astype(str)))
    with PdfPages(path) as pdf:
        for task in tasks:
            g=summary[summary.task.astype(str).eq(task)].copy()
            g=g[np.isfinite(pd.to_numeric(g.final_nrmse_median,errors="coerce"))]
            if g.empty: continue
            g=g.sort_values("model",key=lambda z:z.map(lambda m:_model_sort_key(m)[0]))
            fig_h=max(3.8,0.42*len(g)+1.6)
            fig,ax=plt.subplots(figsize=(10.2,fig_h))
            y=np.arange(len(g))
            for yi,(_,r) in enumerate(g.iterrows()):
                m=float(r.final_nrmse_median); q1=float(r.final_nrmse_q25); q3=float(r.final_nrmse_q75)
                color=FAMILY_COLORS[_family(str(r.model))]; marker=MODEL_MARKERS.get(str(r.model),"o")
                ax.errorbar(m,yi,xerr=np.array([[max(0,m-q1)],[max(0,q3-m)]]),fmt=marker,
                            mfc=color,mec="white",mew=.7,ms=7,ecolor=color,elinewidth=1.4,capsize=3)
                ax.text(1.01,yi,f"n={int(r.n)}",transform=ax.get_yaxis_transform(),va="center",fontsize=8,color="#555555")
            vals=pd.to_numeric(g.final_nrmse_median,errors="coerce")
            if (vals>0).all() and vals.max()/max(vals.min(),1e-300)>50: ax.set_xscale("log")
            ax.set_yticks(y); ax.set_yticklabels([_label(m) for m in g.model])
            ax.invert_yaxis(); ax.set_xlabel("Final test NRMSE")
            ax.set_title(f"Fuzzy-task predictive error — {task}")
            ax.grid(axis="x",alpha=.18); ax.grid(axis="y",visible=False)
            for spine in ("top","right","left"): ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y",length=0)
            fig.subplots_adjust(left=.30,right=.89,bottom=.14,top=.88)
            pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def _build_fuzzy_summary(fuzzy: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in [
        "symbolic_rmse", "fuzzy_rule_f1", "fuzzy_rule_precision", "fuzzy_rule_recall",
        "fuzzy_gate_precision", "fuzzy_gate_recall", "fuzzy_branch_precision", "fuzzy_branch_recall",
        "fuzzy_exact_structure_recovery", "fuzzy_gate_rmse_median", "fuzzy_gate_rmse_max", "fuzzy_gate_nrmse_median", "fuzzy_gate_nrmse_max",
        "fuzzy_expected_rules", "fuzzy_found_rules", "fuzzy_rule_count_error", "fuzzy_rule_count_abs_error",
        "symbolic_seconds", "elapsed_seconds",
    ] if c in fuzzy.columns]
    group_cols=["task","model"]
    if "shared_capacity_width" in fuzzy.columns and pd.to_numeric(fuzzy["shared_capacity_width"],errors="coerce").notna().any():
        group_cols.append("shared_capacity_width")
    rows = []
    for keys, g in fuzzy.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        row = dict(zip(group_cols, keys)); row["n"] = int(len(g))
        for metric in metrics:
            med, q1, q3, n = _median_iqr(g[metric])
            row[f"{metric}_median"] = med
            row[f"{metric}_q25"] = q1
            row[f"{metric}_q75"] = q3
            row[f"{metric}_n"] = n
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    sort_cols=["task","model"] + (["shared_capacity_width"] if "shared_capacity_width" in out.columns else [])
    return out.sort_values(sort_cols, key=lambda z: z.map(lambda m: _model_sort_key(m)[0]) if z.name == "model" else z)


def _build_symbolic_summary(symbolic: pd.DataFrame) -> pd.DataFrame:
    group_cols=["suite","task","model"]
    if "shared_capacity_width" in symbolic.columns and pd.to_numeric(symbolic["shared_capacity_width"],errors="coerce").notna().any():
        group_cols.append("shared_capacity_width")
    rows=[]
    for keys,g in symbolic.groupby(group_cols,dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        keymap=dict(zip(group_cols,keys))
        sm,s1,s3,sn=_median_iqr(g["symbolic_rmse"])
        nm,n1,n3,nn=_median_iqr(g["numeric_rmse"])
        rt,r1,r3,rn=_median_iqr(g.get("symbolic_seconds",pd.Series(dtype=float)))
        tt,t1,t3,tn=_median_iqr(g.get("elapsed_seconds",pd.Series(dtype=float)))
        row={
            **keymap,
            "symbolic_rmse_median":sm,"symbolic_rmse_q25":s1,"symbolic_rmse_q75":s3,"n":sn,
            "numeric_rmse_median":nm,"numeric_rmse_q25":n1,"numeric_rmse_q75":n3,
            "symbolic_over_numeric_median":float(pd.to_numeric(g["symbolic_over_numeric_nrmse"],errors="coerce").median()),
            "symbolic_seconds_median":rt,"symbolic_seconds_q25":r1,"symbolic_seconds_q75":r3,
            "elapsed_seconds_median":tt,"elapsed_seconds_q25":t1,"elapsed_seconds_q75":t3,
        }
        rows.append(row)
    out=pd.DataFrame(rows)
    sort_cols=["suite","task","model"] + (["shared_capacity_width"] if "shared_capacity_width" in out.columns else [])
    return out.sort_values(sort_cols,key=lambda z: z.map(lambda m:_model_sort_key(m)[0]) if z.name=="model" else z)


def _fmt_interval(m: float,q1: float,q3: float) -> str:
    if not np.isfinite(m): return ""
    if not (np.isfinite(q1) and np.isfinite(q3)): return f"{m:.4g}"
    return f"{m:.4g} [{q1:.4g}, {q3:.4g}]"



FACTORIAL_MODELS = {
    "rulekan": (1, 1, 1),
    "rulekan_no_product": (0, 1, 1),
    "rulekan_no_pruning": (1, 0, 1),
    "rulekan_no_gmp": (1, 1, 0),
    "rulekan_no_product_no_pruning": (0, 0, 1),
    "rulekan_no_product_no_gmp": (0, 1, 0),
    "rulekan_no_pruning_no_gmp": (1, 0, 0),
    "rulekan_no_product_no_pruning_no_gmp": (0, 0, 0),
}

SINGLE_ABLATION_MODELS = [
    "rulekan_no_product", "rulekan_no_pruning", "rulekan_no_gmp",
    "rulekan_one_shot_prune", "rulekan_no_backfit", "rulekan_no_self_product",
    "rulekan_fast", "rulekan_omp_linear", "rulekan_graph",
]


def _build_ablation_pairs(symbolic: pd.DataFrame) -> pd.DataFrame:
    x = symbolic[symbolic["model"].isin(["rulekan"] + SINGLE_ABLATION_MODELS)].copy()
    if x.empty:
        return pd.DataFrame()
    for col in ["symbolic_seconds", "structure_f1", "fuzzy_rule_f1"]:
        if col not in x.columns:
            x[col] = np.nan
    base = x[x.model.eq("rulekan")][["task","seed","suite","symbolic_rmse","symbolic_seconds","structure_f1","fuzzy_rule_f1"]].copy()
    base = base.rename(columns={c:f"base_{c}" for c in ["symbolic_rmse","symbolic_seconds","structure_f1","fuzzy_rule_f1"]})
    rows=[]
    for m in SINGLE_ABLATION_MODELS:
        g=x[x.model.eq(m)].copy()
        if g.empty: continue
        z=g.merge(base,on=["task","seed","suite"],how="inner")
        for _,r in z.iterrows():
            br=float(r.get("base_symbolic_rmse",np.nan)); vr=float(r.get("symbolic_rmse",np.nan))
            bs=float(r.get("base_symbolic_seconds",np.nan)); vs=float(r.get("symbolic_seconds",np.nan))
            rows.append({
                "suite":r["suite"],"task":r["task"],"seed":r["seed"],"model":m,
                "symbolic_rmse_ratio": vr/br if np.isfinite(vr) and np.isfinite(br) and br>0 else np.nan,
                "symbolic_runtime_ratio": vs/bs if np.isfinite(vs) and np.isfinite(bs) and bs>0 else np.nan,
                "structure_f1_delta": float(r.get("structure_f1",np.nan))-float(r.get("base_structure_f1",np.nan)),
                "fuzzy_rule_f1_delta": float(r.get("fuzzy_rule_f1",np.nan))-float(r.get("base_fuzzy_rule_f1",np.nan)),
            })
    return pd.DataFrame(rows)


def _build_factorial_effects(symbolic: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    x=symbolic[symbolic.model.isin(FACTORIAL_MODELS)].copy()
    rows=[]
    for (suite,task,seed),g in x.groupby(["suite","task","seed"]):
        vals={}
        for _,r in g.iterrows():
            tup=FACTORIAL_MODELS.get(str(r.model)); rm=float(r.symbolic_rmse)
            if tup is not None and np.isfinite(rm) and rm>0: vals[tup]=np.log10(rm)
        if len(vals)<8: continue
        def mean_where(pred):
            return float(np.mean([v for k,v in vals.items() if pred(k)]))
        s_eff=mean_where(lambda k:k[0]==1)-mean_where(lambda k:k[0]==0)
        p_eff=mean_where(lambda k:k[1]==1)-mean_where(lambda k:k[1]==0)
        g_eff=mean_where(lambda k:k[2]==1)-mean_where(lambda k:k[2]==0)
        # Difference-in-differences, averaged over the third factor.
        def sp_at(gv):
            return (vals[(1,1,gv)]-vals[(0,1,gv)])-(vals[(1,0,gv)]-vals[(0,0,gv)])
        def sg_at(pv):
            return (vals[(1,pv,1)]-vals[(0,pv,1)])-(vals[(1,pv,0)]-vals[(0,pv,0)])
        def pg_at(sv):
            return (vals[(sv,1,1)]-vals[(sv,0,1)])-(vals[(sv,1,0)]-vals[(sv,0,0)])
        rows.append({"suite":suite,"task":task,"seed":seed,
                     "sumproduct_log10_effect":s_eff,"pruning_log10_effect":p_eff,"gmp_log10_effect":g_eff,
                     "sumproduct_pruning_interaction":0.5*(sp_at(0)+sp_at(1)),
                     "sumproduct_gmp_interaction":0.5*(sg_at(0)+sg_at(1)),
                     "pruning_gmp_interaction":0.5*(pg_at(0)+pg_at(1))})
    raw=pd.DataFrame(rows)
    if raw.empty: return raw,pd.DataFrame()
    out=[]
    for c,label in [("sumproduct_log10_effect","SumProduct"),("pruning_log10_effect","Iterative pruning"),("gmp_log10_effect","GMP")]:
        z=raw[c].dropna();
        if z.empty: continue
        # Convert ON-vs-OFF log effect to an RMSE multiplier. <1 means enabling helps.
        ratios=np.power(10.0,z.to_numpy(float))
        out.append({"component":label,"rmse_multiplier_median":float(np.median(ratios)),
                    "rmse_multiplier_q25":float(np.quantile(ratios,.25)),"rmse_multiplier_q75":float(np.quantile(ratios,.75)),"n":int(len(ratios))})
    return raw,pd.DataFrame(out)


def _factorial_summary_by_suite(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty: return pd.DataFrame()
    rows=[]
    cols=[("sumproduct_log10_effect","SumProduct"),("pruning_log10_effect","Iterative pruning"),("gmp_log10_effect","GMP")]
    for suite,g in raw.groupby("suite"):
        for c,label in cols:
            z=pd.to_numeric(g[c],errors="coerce").dropna(); z=z[np.isfinite(z)]
            if z.empty: continue
            rr=np.power(10.0,z.to_numpy(float))
            rows.append({"suite":suite,"component":label,"rmse_multiplier_median":float(np.median(rr)),
                         "rmse_multiplier_q25":float(np.quantile(rr,.25)),"rmse_multiplier_q75":float(np.quantile(rr,.75)),"n":int(len(rr))})
    return pd.DataFrame(rows)


def _write_ablation_pdf(pair_df: pd.DataFrame, factorial_summary: pd.DataFrame, path: Path, factorial_by_suite: pd.DataFrame | None = None) -> None:
    if pair_df.empty and factorial_summary.empty: return
    with PdfPages(path) as pdf:
        if not factorial_summary.empty:
            fig,ax=plt.subplots(figsize=(9.5,4.8))
            d=factorial_summary.copy(); y=np.arange(len(d))[::-1]
            med=d.rmse_multiplier_median.to_numpy(float); q1=d.rmse_multiplier_q25.to_numpy(float); q3=d.rmse_multiplier_q75.to_numpy(float)
            ax.errorbar(med,y,xerr=[med-q1,q3-med],fmt='o',color=FAMILY_COLORS['rulekan'],ecolor=FAMILY_COLORS['rulekan'],capsize=3,lw=1.5)
            ax.axvline(1.0,color='#777777',lw=1,ls='--')
            ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels(d.component)
            ax.set_xlabel('Symbolic RMSE multiplier when component is enabled (median, Q1-Q3)')
            ax.set_title('Core 2x2x2 factorial: main effects')
            ax.text(.01,-.16,'< 1 improves symbolic RMSE; > 1 worsens it. Matched task/seed cells only.',transform=ax.transAxes,fontsize=8,color='#555555')
            ax.grid(axis='x',which='both',alpha=.18); fig.tight_layout(); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
        if factorial_by_suite is not None and not factorial_by_suite.empty:
            for suite,d in factorial_by_suite.groupby("suite"):
                d=d.copy(); y=np.arange(len(d))[::-1]
                fig,ax=plt.subplots(figsize=(9.5,4.8))
                med=d.rmse_multiplier_median.to_numpy(float); q1=d.rmse_multiplier_q25.to_numpy(float); q3=d.rmse_multiplier_q75.to_numpy(float)
                ax.errorbar(med,y,xerr=[med-q1,q3-med],fmt='o',color=FAMILY_COLORS['rulekan'],ecolor=FAMILY_COLORS['rulekan'],capsize=3,lw=1.5)
                ax.axvline(1.0,color='#777777',lw=1,ls='--'); ax.set_xscale('log')
                ax.set_yticks(y); ax.set_yticklabels(d.component)
                ax.set_xlabel('Symbolic RMSE multiplier when component is enabled (median, Q1-Q3)')
                ax.set_title(f'Core factorial main effects: {suite}')
                ax.grid(axis='x',which='both',alpha=.18); fig.tight_layout(); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)
        if not pair_df.empty:
            rows=[]
            for m,g in pair_df.groupby('model'):
                z=pd.to_numeric(g.symbolic_rmse_ratio,errors='coerce').dropna(); z=z[np.isfinite(z)&(z>0)]
                if z.empty: continue
                rows.append((m,float(z.median()),float(z.quantile(.25)),float(z.quantile(.75)),len(z)))
            rows.sort(key=lambda x:_model_sort_key(x[0]))
            if rows:
                fig,ax=plt.subplots(figsize=(10.5,max(5,0.52*len(rows)+1.8)))
                y=np.arange(len(rows))[::-1]
                for yi,(m,med,q1,q3,n) in zip(y,rows):
                    ax.errorbar([med],[yi],xerr=[[med-q1],[q3-med]],fmt=MODEL_MARKERS.get(m,'o'),color=FAMILY_COLORS['rulekan'],ecolor=FAMILY_COLORS['rulekan'],capsize=3,lw=1.2)
                    ax.text(1.01,yi,f'n={n}',transform=ax.get_yaxis_transform(),va='center',fontsize=8,color='#555555')
                ax.axvline(1.0,color='#777777',lw=1,ls='--'); ax.set_xscale('log')
                ax.set_yticks(y); ax.set_yticklabels([_label(r[0]) for r in rows])
                ax.set_xlabel('Symbolic RMSE / full RuleKAN symbolic RMSE (matched runs)')
                ax.set_title('Single-component and algorithm ablations')
                ax.grid(axis='x',which='both',alpha=.18); fig.tight_layout(); pdf.savefig(fig,bbox_inches='tight'); plt.close(fig)


def _build_library_sensitivity(symbolic: pd.DataFrame) -> pd.DataFrame:
    """Summarize nested symbolic-library sweeps by task/model/library size."""
    if symbolic.empty or "shared_symbolic_library_size" not in symbolic.columns:
        return pd.DataFrame()
    x = symbolic.copy()
    x["shared_symbolic_library_size"] = pd.to_numeric(x["shared_symbolic_library_size"], errors="coerce")
    x["symbolic_rmse"] = pd.to_numeric(x["symbolic_rmse"], errors="coerce")
    x["symbolic_seconds"] = pd.to_numeric(x.get("symbolic_seconds"), errors="coerce")
    x = x[x["shared_symbolic_library_size"].notna() & x["symbolic_rmse"].notna()]
    if x.empty or x["shared_symbolic_library_size"].nunique() < 2:
        return pd.DataFrame()
    rows=[]
    keys=["suite","task","model","shared_symbolic_library_size"]
    for (suite,task,model,size),g in x.groupby(keys,dropna=False):
        rm,r1,r3,rn=_median_iqr(g["symbolic_rmse"])
        tm,t1,t3,tn=_median_iqr(g["symbolic_seconds"])
        names=[]
        if "shared_symbolic_library" in g.columns:
            names=[str(v) for v in g["shared_symbolic_library"].dropna().unique()]
        rows.append({
            "suite":suite,"task":task,"model":model,"shared_symbolic_library_size":int(size),
            "shared_symbolic_library":"/".join(sorted(names)) if names else "",
            "symbolic_rmse_median":rm,"symbolic_rmse_q25":r1,"symbolic_rmse_q75":r3,
            "symbolic_seconds_median":tm,"symbolic_seconds_q25":t1,"symbolic_seconds_q75":t3,
            "n":rn,
        })
    return pd.DataFrame(rows).sort_values(["suite","task","model","shared_symbolic_library_size"])


def _write_library_sensitivity_pdf(summary: pd.DataFrame, path: Path, *, metric: str) -> None:
    if summary.empty:
        return
    if metric == "symbolic_rmse":
        med,q1,q3="symbolic_rmse_median","symbolic_rmse_q25","symbolic_rmse_q75"
        ylabel="Final symbolic RMSE"
    else:
        med,q1,q3="symbolic_seconds_median","symbolic_seconds_q25","symbolic_seconds_q75"
        ylabel="Symbolic-search time (s)"
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(summary):
            gt=summary[summary.task.eq(task)].copy()
            if gt.empty: continue
            fig,ax=plt.subplots(figsize=(10.2,5.4))
            for j,m in enumerate(sorted(gt.model.unique(),key=_model_sort_key)):
                h=gt[gt.model.eq(m)].sort_values("shared_symbolic_library_size")
                xv=h.shared_symbolic_library_size.to_numpy(float)
                yv=pd.to_numeric(h[med],errors="coerce").to_numpy(float)
                lo=pd.to_numeric(h[q1],errors="coerce").to_numpy(float)
                hi=pd.to_numeric(h[q3],errors="coerce").to_numpy(float)
                good=np.isfinite(xv)&np.isfinite(yv)&(yv>0)
                if not good.any(): continue
                xv=xv[good]; yv=yv[good]; lo=lo[good]; hi=hi[good]
                marker=MODEL_MARKERS.get(m,["o","s","^","D","v"][j%5])
                color=FAMILY_COLORS[_family(m)]
                ax.plot(xv,yv,marker=marker,lw=1.4,label=_label(m),color=color)
                if np.isfinite(lo).all() and np.isfinite(hi).all():
                    ax.fill_between(xv,lo,hi,alpha=.12,color=color)
            ax.set_xlabel("Symbolic library size")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{task}: symbolic-library sensitivity")
            ax.set_xticks(sorted(gt.shared_symbolic_library_size.dropna().unique()))
            if metric == "symbolic_rmse": ax.set_yscale("log")
            ax.grid(alpha=.18); ax.legend(fontsize=8); fig.tight_layout()
            pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def _build_width_sensitivity(symbolic: pd.DataFrame) -> pd.DataFrame:
    if symbolic.empty or "shared_capacity_width" not in symbolic.columns:
        return pd.DataFrame()
    x = symbolic.copy()
    x["shared_capacity_width"] = pd.to_numeric(x["shared_capacity_width"], errors="coerce")
    x["symbolic_rmse"] = pd.to_numeric(x["symbolic_rmse"], errors="coerce")
    x["symbolic_seconds"] = pd.to_numeric(x.get("symbolic_seconds"), errors="coerce")
    x = x[x["shared_capacity_width"].notna() & x["symbolic_rmse"].notna()]
    if x.empty or x["shared_capacity_width"].nunique() < 2:
        return pd.DataFrame()
    rows=[]
    for (suite,task,model,width),g in x.groupby(["suite","task","model","shared_capacity_width"],dropna=False):
        rm,r1,r3,rn=_median_iqr(g["symbolic_rmse"])
        tm,t1,t3,tn=_median_iqr(g["symbolic_seconds"])
        rows.append({
            "suite":suite,"task":task,"model":model,"shared_capacity_width":int(width),
            "symbolic_rmse_median":rm,"symbolic_rmse_q25":r1,"symbolic_rmse_q75":r3,
            "symbolic_seconds_median":tm,"symbolic_seconds_q25":t1,"symbolic_seconds_q75":t3,
            "n":rn,
        })
    return pd.DataFrame(rows).sort_values(["suite","task","model","shared_capacity_width"])


def _build_width_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows=[]
    for (suite,task,model),g in summary.groupby(["suite","task","model"],dropna=False):
        g=g.sort_values("shared_capacity_width")
        vals=pd.to_numeric(g["symbolic_rmse_median"],errors="coerce")
        good=np.isfinite(vals)&(vals>0)
        if not good.any():
            continue
        h=g.loc[good].copy(); v=pd.to_numeric(h["symbolic_rmse_median"],errors="coerce").astype(float)
        best_i=v.idxmin(); worst_i=v.idxmax()
        rows.append({
            "suite":suite,"task":task,"model":model,
            "width_count":int(len(h)),
            "best_width":int(h.loc[best_i,"shared_capacity_width"]),
            "best_symbolic_rmse":float(v.loc[best_i]),
            "worst_width":int(h.loc[worst_i,"shared_capacity_width"]),
            "worst_symbolic_rmse":float(v.loc[worst_i]),
            "worst_over_best_rmse":float(v.loc[worst_i]/v.loc[best_i]),
            "log10_rmse_range":float(np.log10(v.max())-np.log10(v.min())),
            "median_symbolic_rmse_across_widths":float(v.median()),
            "q25_symbolic_rmse_across_widths":float(v.quantile(.25)),
            "q75_symbolic_rmse_across_widths":float(v.quantile(.75)),
        })
    return pd.DataFrame(rows)


def _write_width_sensitivity_pdf(summary: pd.DataFrame, path: Path, *, metric: str = "symbolic_rmse") -> None:
    if summary.empty:
        return
    if metric == "symbolic_rmse":
        med,q1,q3 = "symbolic_rmse_median","symbolic_rmse_q25","symbolic_rmse_q75"
        ylabel = "Final symbolic RMSE"
        title_metric = "symbolic RMSE"
    else:
        med,q1,q3 = "symbolic_seconds_median","symbolic_seconds_q25","symbolic_seconds_q75"
        ylabel = "Symbolic-search time (s)"
        title_metric = "symbolic-search runtime"
    groups = [
        ("RuleKAN", lambda m: str(m).startswith("rulekan")),
        ("Shallow MultKAN", lambda m: str(m) in PAPER_PIPELINES_FOR_PLOTS),
        ("Deep MultKAN", lambda m: str(m).startswith("multkan_deep_") or str(m).startswith("fast_multkan_deep_")),
    ]
    linestyles=["-","--","-.",":"]
    with PdfPages(path) as pdf:
        for task in _ordered_tasks(summary):
            gt=summary[summary.task.eq(task)].copy()
            present=[]
            for label,pred in groups:
                gm=gt[gt.model.map(pred)]
                if not gm.empty:
                    present.append((label,gm))
            if not present:
                continue
            fig,axes=plt.subplots(len(present),1,figsize=(10.8,3.5*len(present)+1.2),squeeze=False)
            for ax,(group_label,gm) in zip(axes[:,0],present):
                models=sorted(gm.model.unique(),key=_model_sort_key)
                for j,m in enumerate(models):
                    h=gm[gm.model.eq(m)].sort_values("shared_capacity_width")
                    xv=h.shared_capacity_width.to_numpy(float)
                    yv=pd.to_numeric(h[med],errors="coerce").to_numpy(float)
                    lo=pd.to_numeric(h[q1],errors="coerce").to_numpy(float)
                    hi=pd.to_numeric(h[q3],errors="coerce").to_numpy(float)
                    good=np.isfinite(xv)&np.isfinite(yv)&(xv>0)&(yv>0)
                    if not good.any():
                        continue
                    xv=xv[good]; yv=yv[good]; lo=lo[good]; hi=hi[good]
                    color=FAMILY_COLORS[_family(m)] if group_label=="RuleKAN" else "#555555"
                    marker=MODEL_MARKERS.get(m,["o","s","^","D","v"][j%5])
                    ls=linestyles[j%len(linestyles)]
                    ax.plot(xv,yv,marker=marker,linestyle=ls,color=color,lw=1.5,ms=5.5,label=_label(m))
                    if np.isfinite(lo).any() and np.isfinite(hi).any():
                        ax.fill_between(xv,lo,hi,color=color,alpha=.08,linewidth=0)
                ax.set_xscale("log")
                widths=sorted(gt.shared_capacity_width.dropna().unique())
                ax.set_xticks(widths)
                ax.set_xticklabels([str(int(w)) for w in widths])
                ax.set_yscale("log")
                ax.set_ylabel(ylabel)
                ax.set_title(group_label,loc="left",fontsize=10.5,fontweight="bold")
                ax.grid(axis="both",which="major",alpha=.16)
                ax.legend(loc="best",frameon=False,fontsize=8,ncol=min(3,max(1,len(models))))
                for spine in ("top","right"):
                    ax.spines[spine].set_visible(False)
            axes[-1,0].set_xlabel("Shared capacity width W")
            fig.suptitle(f"Width sensitivity - {task} - {title_metric}",fontsize=13.5,y=.995)
            fig.tight_layout(rect=[0,0,1,.97])
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)


PAPER_PIPELINES_FOR_PLOTS = {"autosym","fastkan_autosym","gsr","fastkan_gsr","gmp"}

def aggregate(run_dir: Path, quiet: bool = False) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir=run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    # Benchmark outputs are vector PDF only. Remove stale raster outputs and old
    # mixed-stage plots whose semantics were inconsistent across model families.
    for pat in ("*.png","*.jpg","*.jpeg"):
        for p in fig_dir.glob(pat):
            p.unlink(missing_ok=True)
    for old in [
        "predictive_error_by_task.pdf","runtime_by_task.pdf","rulekan_symbolic_gap.pdf",
        "accuracy_runtime_pareto.pdf","seed_variability.pdf",
    ]:
        (fig_dir/old).unlink(missing_ok=True)

    df = _load(run_dir)
    if df.empty:
        skipped_manifest_count = 0
        skipped_manifest_path = run_dir / "skipped_incompatible_jobs.json"
        if skipped_manifest_path.exists():
            try:
                skipped_manifest_count = len(json.loads(skipped_manifest_path.read_text()))
            except Exception:
                skipped_manifest_count = 0
        status_lines = [
            "completed: 0",
            "symbolic regression completed: 0",
            "failed: 0",
            "skipped records: 0",
            "running: 0",
            "other status: 0",
            f"incompatible jobs not scheduled: {skipped_manifest_count}",
            "total run records: 0",
        ]
        (run_dir / "STATUS.txt").write_text("\n".join(status_lines) + "\n")
        if not quiet:
            print("[aggregate] " + " | ".join(status_lines))
            if skipped_manifest_count:
                print(f"[aggregate] incompatible jobs: {run_dir / 'skipped_incompatible_jobs.csv'}")
        return
    df.to_csv(run_dir / "runs.csv", index=False)
    completed = df[df["status"] == "completed"].copy() if "status" in df else pd.DataFrame()
    if "status" in df:
        status_norm = df["status"].fillna("unknown").astype(str).str.lower()
        failed = df[status_norm.eq("failed")].copy()
        skipped = df[status_norm.eq("skipped")].copy()
        running = df[status_norm.eq("running")].copy()
        other = df[~status_norm.isin(["completed", "failed", "skipped", "running"])].copy()
        incomplete = df[~status_norm.eq("completed")].copy()
    else:
        failed = skipped = running = other = incomplete = pd.DataFrame()

    # Keep the status audit machine-readable.  A stale ``running`` row usually
    # means the orchestrator or host was interrupted after run_one wrote its
    # initial record; failures retain their exact ``error`` and traceback.
    for path in (run_dir / "failures.csv", run_dir / "incomplete_runs.csv", run_dir / "failure_summary.csv"):
        path.unlink(missing_ok=True)
    if not failed.empty:
        failed.to_csv(run_dir / "failures.csv", index=False)
    if not incomplete.empty:
        cols = [c for c in [
            "suite", "task", "model", "seed", "shared_capacity_width", "shared_symbolic_library",
            "status", "error", "elapsed_seconds", "started_unix", "finished_unix",
            "code_version", "benchmark_build_fingerprint", "device",
        ] if c in incomplete.columns]
        incomplete[cols].to_csv(run_dir / "incomplete_runs.csv", index=False)
        if "error" in incomplete.columns:
            fs = incomplete.copy()
            fs["error"] = fs["error"].fillna("").astype(str)
            group_cols = [c for c in ["status", "error"] if c in fs.columns]
            if group_cols:
                fs.groupby(group_cols, dropna=False).size().reset_index(name="count").sort_values(
                    ["count"], ascending=False
                ).to_csv(run_dir / "failure_summary.csv", index=False)

    summary = pd.DataFrame()
    symbolic_summary = pd.DataFrame()
    if not completed.empty:
        completed = _derive_symbolic_metrics(completed)
        completed = _backfill_fuzzy_formula_metrics(completed)
        metric_cols = [c for c in [
            "test_rmse", "test_nrmse", "test_r2", "test_accuracy", "test_f1", "test_roc_auc",
            "symbolic_rmse", "symbolic_nrmse", "numeric_rmse", "numeric_nrmse", "final_rmse", "final_nrmse",
            "symbolic_minus_numeric_nrmse", "symbolic_over_numeric_nrmse",
            "structure_precision", "structure_recall", "structure_f1",
            "fuzzy_rule_precision", "fuzzy_rule_recall", "fuzzy_rule_f1",
            "fuzzy_gate_precision", "fuzzy_gate_recall", "fuzzy_branch_precision", "fuzzy_branch_recall",
            "fuzzy_exact_structure_recovery", "fuzzy_gate_rmse_median", "fuzzy_gate_rmse_max", "fuzzy_gate_nrmse_median", "fuzzy_gate_nrmse_max",
            "fuzzy_expected_rules", "fuzzy_found_rules", "fuzzy_rule_count_error", "fuzzy_rule_count_abs_error",
            "numeric_seconds", "symbolic_seconds", "elapsed_seconds", "parameters",
            "active_numeric_rules", "active_symbolic_rules",
            "symbolic_max_abs_rule_corr", "symbolic_max_span_r2",
        ] if c in completed.columns]

        summary_group_cols=["suite","task","task_type","model"]
        if "shared_capacity_width" in completed.columns and pd.to_numeric(completed["shared_capacity_width"],errors="coerce").notna().any():
            summary_group_cols.append("shared_capacity_width")
        grouped=completed.groupby(summary_group_cols,dropna=False)
        agg=grouped[metric_cols].agg(["mean","std","median","min","max","count",_q25,_q75])
        agg.columns=[f"{a}_{b if isinstance(b,str) else getattr(b,'__name__',str(b))}" for a,b in agg.columns]
        # pandas names callable columns after function __name__ already; normalize.
        agg.columns=[c.replace("__q25","_q25").replace("__q75","_q75")
                     .replace("_<lambda_0>","_q25").replace("_<lambda_1>","_q75") for c in agg.columns]
        summary=agg.reset_index()
        summary.to_csv(run_dir/"summary.csv",index=False)

        _write_statistical_outputs(completed, run_dir, fig_dir)

        # Symbolic-only rows: regression tasks with an actual final symbolic model.
        symbolic=completed[
            completed["task_type"].eq("regression")
            & pd.to_numeric(completed["symbolic_rmse"],errors="coerce").notna()
        ].copy()
        symbolic= symbolic[np.isfinite(pd.to_numeric(symbolic["symbolic_rmse"],errors="coerce"))]
        symbolic.to_csv(run_dir/"symbolic_runs.csv",index=False)

        if not symbolic.empty:
            symbolic_summary=_build_symbolic_summary(symbolic)
            symbolic_summary.to_csv(run_dir/"symbolic_summary.csv",index=False)
            disp=symbolic_summary.copy()
            disp["symbolic RMSE median [Q1,Q3]"]=[_fmt_interval(a,b,c) for a,b,c in zip(disp.symbolic_rmse_median,disp.symbolic_rmse_q25,disp.symbolic_rmse_q75)]
            disp["numeric RMSE median [Q1,Q3]"]=[_fmt_interval(a,b,c) for a,b,c in zip(disp.numeric_rmse_median,disp.numeric_rmse_q25,disp.numeric_rmse_q75)]
            disp["symbolic time s median [Q1,Q3]"]=[_fmt_interval(a,b,c) for a,b,c in zip(disp.symbolic_seconds_median,disp.symbolic_seconds_q25,disp.symbolic_seconds_q75)]
            mdcols=["suite","task","model"] + (["shared_capacity_width"] if "shared_capacity_width" in disp.columns else []) + ["symbolic RMSE median [Q1,Q3]","numeric RMSE median [Q1,Q3]","symbolic time s median [Q1,Q3]","n"]
            (run_dir/"summary.md").write_text(
                "# Partial symbolic-regression benchmark summary\n\n"
                "The primary error is raw RMSE. Symbolic RMSE and numerical-precursor RMSE are reported separately. "
                "Intervals are Q1-Q3 across currently completed seeds.\n\n"+
                _markdown_table(disp[mdcols])
            )

            recent_cols=[c for c in [
                "suite","task","model","seed","shared_capacity_width","status","code_version","benchmark_build_fingerprint",
                "symbolic_gmp_identity_chart","symbolic_gmp_atom_backward_normalization","symbolic_gmp_effective_policy",
                "symbolic_rmse","numeric_rmse","symbolic_over_numeric_nrmse","symbolic_seconds","elapsed_seconds",
                "structure_f1","fuzzy_rule_f1","fuzzy_gate_recall","fuzzy_branch_recall",
                "fuzzy_gate_nrmse_median","symbolic_max_abs_rule_corr","symbolic_max_span_r2",
            ] if c in symbolic.columns]
            recent=symbolic.sort_values("finished_unix" if "finished_unix" in symbolic else "seed",ascending=False)[recent_cols]
            (run_dir/"latest_results.md").write_text(
                "# Most recent completed symbolic-regression runs\n\n"+_markdown_table(recent,60)
            )

            _write_method_redundancy_outputs(symbolic, run_dir)

            library_summary = _build_library_sensitivity(symbolic)
            if not library_summary.empty:
                library_summary.to_csv(run_dir/"library_sensitivity.csv", index=False)
                (run_dir/"library_sensitivity_summary.md").write_text(
                    "# Symbolic-library sensitivity\n\n"
                    "Nested libraries retain the target-core primitives and add distractor operators. "
                    "The table reports median final symbolic RMSE and symbolic-search runtime.\n\n"
                    + _markdown_table(library_summary)
                )
                _write_library_sensitivity_pdf(library_summary, fig_dir/"library_sensitivity_rmse.pdf", metric="symbolic_rmse")
                _write_library_sensitivity_pdf(library_summary, fig_dir/"library_sensitivity_runtime.pdf", metric="symbolic_seconds")

            width_summary = _build_width_sensitivity(symbolic)
            width_sweep = not width_summary.empty
            if width_sweep:
                width_summary.to_csv(run_dir/"width_sensitivity.csv", index=False)
                width_robustness = _build_width_robustness(width_summary)
                if not width_robustness.empty:
                    width_robustness.to_csv(run_dir/"width_sensitivity_robustness.csv", index=False)
                    (run_dir/"width_sensitivity_summary.md").write_text(
                        "# Shared-capacity width sensitivity\n\n"
                        "`worst_over_best_rmse` is the ratio between the worst and best median symbolic RMSE across completed widths for the same task/model. Values near 1 indicate low width sensitivity.\n\n"
                        + _markdown_table(width_robustness)
                    )
                _write_width_sensitivity_pdf(width_summary, fig_dir/"width_sensitivity_rmse.pdf", metric="symbolic_rmse")
                _write_width_sensitivity_pdf(width_summary, fig_dir/"width_sensitivity_runtime.pdf", metric="symbolic_seconds")
            else:
                _write_fuzzy_recovery_pdf(symbolic,fig_dir/"fuzzy_rule_recovery.pdf")
                _write_symbolic_error_pdf(symbolic,fig_dir/"symbolic_rmse_by_task.pdf")
                _write_symbolic_runtime_pdf(symbolic,fig_dir/"symbolic_runtime_by_task.pdf")
                _write_pareto_pdf(symbolic,fig_dir/"symbolic_accuracy_runtime.pdf")
                _write_numeric_symbolic_pdf(symbolic,fig_dir/"numeric_vs_symbolic_rmse.pdf")
                _write_redundancy_pdf(symbolic,fig_dir/"rule_redundancy.pdf")
                depth_pairs = _build_multkan_depth_pairs(symbolic)
                if not depth_pairs.empty:
                    depth_pairs.to_csv(run_dir/"multkan_depth_pairs.csv", index=False)
                _write_multkan_depth_pdf(symbolic,fig_dir/"multkan_depth_comparison.pdf")
            ab_pairs = _build_ablation_pairs(symbolic)
            if not ab_pairs.empty:
                ab_pairs.to_csv(run_dir/"ablation_pairs.csv", index=False)
            fact_raw, fact_summary = _build_factorial_effects(symbolic)
            fact_by_suite = _factorial_summary_by_suite(fact_raw)
            if not fact_raw.empty:
                fact_raw.to_csv(run_dir/"ablation_factorial_effects.csv", index=False)
            if not fact_summary.empty:
                fact_summary.to_csv(run_dir/"ablation_component_summary.csv", index=False)
            if not fact_by_suite.empty:
                fact_by_suite.to_csv(run_dir/"ablation_component_by_suite.csv", index=False)
            _write_ablation_pdf(ab_pairs, fact_summary, fig_dir/"ablation_component_effects.pdf", fact_by_suite)
            if (not ab_pairs.empty) or (not fact_summary.empty):
                sections=["# RuleKAN component ablation summary\n"]
                if not fact_summary.empty:
                    sections.append("## Core factorial main effects\n\nThe multiplier compares symbolic RMSE with a component enabled versus disabled across matched 2x2x2 cells. Values below 1 indicate lower symbolic RMSE when enabled.\n\n"+_markdown_table(fact_summary))
                if not ab_pairs.empty:
                    rr=[]
                    for m,g in ab_pairs.groupby("model"):
                        z=pd.to_numeric(g.symbolic_rmse_ratio,errors="coerce").dropna(); z=z[np.isfinite(z)&(z>0)]
                        t=pd.to_numeric(g.symbolic_runtime_ratio,errors="coerce").dropna(); t=t[np.isfinite(t)&(t>0)]
                        rr.append({"variant":_label(m),"RMSE ratio median":float(z.median()) if len(z) else np.nan,"runtime ratio median":float(t.median()) if len(t) else np.nan,"matched runs":int(len(z))})
                    sections.append("\n## Paired variants versus full RuleKAN\n\n"+_markdown_table(pd.DataFrame(rr)))
                (run_dir/"ablation_summary.md").write_text("\n".join(sections))
            fuzzy_predictive = completed[completed["suite"].eq("fuzzy_rules")].copy()
            fuzzy_predictive = fuzzy_predictive[np.isfinite(pd.to_numeric(fuzzy_predictive["final_nrmse"],errors="coerce"))]
            if not fuzzy_predictive.empty:
                fps = _build_fuzzy_predictive_summary(fuzzy_predictive)
                fps.to_csv(run_dir/"fuzzy_predictive_summary.csv",index=False)
                fpd=fps.copy()
                fpd["final NRMSE median [Q1,Q3]"]=[_fmt_interval(a,b,c) for a,b,c in zip(fpd.final_nrmse_median,fpd.final_nrmse_q25,fpd.final_nrmse_q75)]
                (run_dir/"fuzzy_predictive_summary.md").write_text(
                    "# Fuzzy-task predictive comparison\n\n"
                    "Final NRMSE uses the final symbolic model when a method produces one and the final numerical predictor otherwise. "
                    "ANFIS is included here as a trainable fuzzy-system baseline; exact RuleKAN/DNF rule-recovery metrics are not assigned to a different fuzzy grammar.\n\n"
                    + _markdown_table(fpd[[c for c in ["task","model","final NRMSE median [Q1,Q3]","elapsed_seconds_median","n"] if c in fpd.columns]])
                )
                _write_fuzzy_predictive_pdf(fps,fig_dir/"fuzzy_predictive_nrmse.pdf")

            fuzzy = symbolic[symbolic["suite"].eq("fuzzy_rules")].copy()
            if not fuzzy.empty:
                fuzzy.to_csv(run_dir/"fuzzy_runs.csv", index=False)
                fuzzy_summary = _build_fuzzy_summary(fuzzy)
                fuzzy_summary.to_csv(run_dir/"fuzzy_summary.csv", index=False)
                fdisp = fuzzy_summary.copy()
                if "fuzzy_rule_f1_median" in fdisp:
                    fdisp["rule F1 median [Q1,Q3]"] = [
                        _fmt_interval(a,b,c) for a,b,c in zip(
                            fdisp.fuzzy_rule_f1_median, fdisp.fuzzy_rule_f1_q25, fdisp.fuzzy_rule_f1_q75
                        )
                    ]
                if "fuzzy_gate_rmse_median_median" in fdisp:
                    fdisp["gate RMSE median [Q1,Q3]"] = [
                        _fmt_interval(a,b,c) for a,b,c in zip(
                            fdisp.fuzzy_gate_rmse_median_median, fdisp.fuzzy_gate_rmse_median_q25,
                            fdisp.fuzzy_gate_rmse_median_q75
                        )
                    ]
                keep = [c for c in [
                    "task","model","symbolic_rmse_median","rule F1 median [Q1,Q3]",
                    "fuzzy_gate_recall_median","fuzzy_branch_recall_median",
                    "gate RMSE median [Q1,Q3]","fuzzy_exact_structure_recovery_median",
                    "fuzzy_rule_count_abs_error_median","n"
                ] if c in fdisp.columns]
                (run_dir/"fuzzy_summary.md").write_text(
                    "# Partial fuzzy if/then rule-recovery summary\n\n"
                    "Rule F1 scores the expanded sum-product/DNF rule signatures. Gate RMSE is computed "
                    "only on observed held-out points; no off-domain samples are generated.\n\n"
                    + _markdown_table(fdisp[keep])
                )
        else:
            (run_dir/"summary.md").write_text("# Partial symbolic-regression benchmark summary\n\n_No completed symbolic regression runs yet._\n")
            (run_dir/"latest_results.md").write_text("# Most recent completed symbolic-regression runs\n\n_No completed symbolic regression runs yet._\n")

    skipped_manifest_count = 0
    skipped_manifest_path = run_dir / "skipped_incompatible_jobs.json"
    if skipped_manifest_path.exists():
        try:
            skipped_manifest_count = len(json.loads(skipped_manifest_path.read_text()))
        except Exception:
            skipped_manifest_count = 0
    status_lines=[
        f"completed: {len(completed)}",
        f"symbolic regression completed: {0 if symbolic_summary.empty else int(symbolic_summary['n'].sum())}",
        f"failed: {len(failed)}",
        f"skipped records: {len(skipped)}",
        f"running: {len(running)}",
        f"other status: {len(other)}",
        f"incompatible jobs not scheduled: {skipped_manifest_count}",
        f"total run records: {len(df)}",
    ]
    (run_dir/"STATUS.txt").write_text("\n".join(status_lines)+"\n")
    if not quiet:
        print("[aggregate] "+" | ".join(status_lines))
        if not incomplete.empty:
            print(f"[aggregate] incomplete details: {run_dir / 'incomplete_runs.csv'}")
            if not failed.empty:
                print(f"[aggregate] failure summary:   {run_dir / 'failure_summary.csv'}")
        if skipped_manifest_count:
            print(f"[aggregate] incompatible jobs: {run_dir / 'skipped_incompatible_jobs.csv'}")
        if not summary.empty:
            print(f"[aggregate] symbolic tables:   {run_dir / 'symbolic_summary.csv'}")
            print(f"[aggregate] PDFs:              {fig_dir}")


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    ap.add_argument("--quiet",action="store_true")
    args=ap.parse_args()
    aggregate(Path(args.run_dir),quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
