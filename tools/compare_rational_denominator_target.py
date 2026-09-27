#!/usr/bin/env python3
"""Focused symbolic ablation: fit actual D(x) vs fit latent H(x).

This deliberately isolates denominator symbolization from numerical rational
training.  The numerator target is shared between both conditions; the only
change is whether symbolic pursuit targets D directly or targets
H=(-1+sqrt(4D-3))/2 and reconstructs D=1+H+H^2.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import torch

from benchmarks.models import _symbolicize_rational_branch
from rulekan.rational_sum_product_kan import RationalSumProductKAN


def _cfg():
    # Intentionally small so this is a diagnostic, not a replacement for the
    # research profile. Both conditions receive exactly the same search budget.
    return dict(
        symbolic_max_rules=3, symbolic_min_rules=1,
        allow_symbolic_self_products=True, use_gmp_preselection=True,
        symbolic_gmp_steps=3, symbolic_gmp_topk=2,
        symbolic_gmp_unary_topk=3, symbolic_gmp_self_topk=3,
        symbolic_tuple_steps=1, symbolic_hybrid_hard_screening=False,
        symbolic_beam=1, symbolic_residual_structure_topk=1,
        symbolic_joint_scale_refit=False, symbolic_trial_steps=3,
        symbolic_commit_steps=3, symbolic_commit_lbfgs=0,
        symbolic_backfit=False, symbolic_final_steps=3,
        symbolic_final_lbfgs=0, symbolic_target_rmse=1e-5,
        redundancy_cleanup=False, symbolic_cleanup_seconds=0.2,
        symbolic_cleanup_trials=1,
    )


def _sample(g: torch.Generator, n: int) -> torch.Tensor:
    return -1.0 + 2.0 * torch.rand(n, 2, generator=g)


def _targets(case: str):
    if case == "additiveD":
        D = lambda x: 2.0 + 0.30 * x[:, 0:1] + 0.20 * x[:, 1:2]
    elif case == "separableQuadraticD":
        D = lambda x: 1.2 + 0.8 * x[:, 1:2].square()
    elif case == "quadraticNative":
        H0 = lambda x: 0.35 * x[:, 0:1] - 0.25 * x[:, 1:2]
        D = lambda x: 1.0 + H0(x) + H0(x).square()
    else:
        raise ValueError(case)
    N = lambda x: 0.7 * torch.sin(1.4 * x[:, 0:1]) + 0.35 * x[:, 1:2].square() + 0.2
    H = lambda x: (-1.0 + torch.sqrt((4.0 * D(x) - 3.0).clamp_min(1e-8))) / 2.0
    Y = lambda x: N(x) / D(x)
    return N, D, H, Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output", default="V57_DIRECT_D_SYMBOLIC_ABLATION.csv")
    args = ap.parse_args()
    cfg = _cfg()
    rows = []
    for case in ("additiveD", "separableQuadraticD", "quadraticNative"):
        for seed in range(args.seeds):
            gen = torch.Generator().manual_seed(seed)
            tr, va, te = _sample(gen, 80), _sample(gen, 40), _sample(gen, 100)
            N, D, H, Y = _targets(case)
            shell = RationalSumProductKAN(
                in_dim=2, n_rules=3, denominator_n_rules=3,
                max_factors=2, grid=3, seed=seed,
            )
            shell.discretize(force_symbolic=False, freeze_gates=True)
            n_sym = _symbolicize_rational_branch(
                shell.numerator, tr, N(tr), va, N(va), cfg,
                symbolic_seed=1000 + seed,
            )
            d_sym = _symbolicize_rational_branch(
                shell.denominator_latent, tr, D(tr), va, D(va), cfg,
                symbolic_seed=2000 + seed,
            )
            h_sym = _symbolicize_rational_branch(
                shell.denominator_latent, tr, H(tr), va, H(va), cfg,
                symbolic_seed=2000 + seed,
            )
            with torch.no_grad():
                n_hat = n_sym(te)
                d_direct = d_sym(te)
                h_hat = h_sym(te)
                d_from_h = 1.0 + h_hat + h_hat.square()
                y = Y(te)
                direct_rmse = float(torch.sqrt(torch.mean((n_hat / d_direct - y) ** 2)))
                latent_rmse = float(torch.sqrt(torch.mean((n_hat / d_from_h - y) ** 2)))
                direct_d_rmse = float(torch.sqrt(torch.mean((d_direct - D(te)) ** 2)))
                latent_d_rmse = float(torch.sqrt(torch.mean((d_from_h - D(te)) ** 2)))
            rows.append({
                "case": case, "seed": seed,
                "direct_symbolic_rmse": direct_rmse,
                "latent_H_symbolic_rmse": latent_rmse,
                "direct_D_rmse": direct_d_rmse,
                "reconstructed_D_from_H_rmse": latent_d_rmse,
                "direct_D_min": float(d_direct.min()),
                "H_D_min": float(d_from_h.min()),
            })
    out = Path(args.output)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    for case in sorted({r["case"] for r in rows}):
        rr = [r for r in rows if r["case"] == case]
        dm = statistics.mean(r["direct_symbolic_rmse"] for r in rr)
        hm = statistics.mean(r["latent_H_symbolic_rmse"] for r in rr)
        wins = sum(r["direct_symbolic_rmse"] < r["latent_H_symbolic_rmse"] for r in rr)
        print(f"{case}: direct={dm:.6g}, latent-H={hm:.6g}, H/direct={hm/dm:.3f}, direct wins={wins}/{len(rr)}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
