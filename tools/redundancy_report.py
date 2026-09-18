#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmarks.aggregate import _derive_symbolic_metrics, _write_method_redundancy_outputs


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantify redundant symbolic-regression methods from a benchmark run directory.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--threshold-dex", type=float, default=0.05,
                    help="equivalence margin on |log10(RMSE_A/RMSE_B)|; 0.05 ~= 1.12x")
    ap.add_argument("--min-pairs", type=int, default=3)
    args = ap.parse_args()

    path = args.run_dir / "symbolic_runs.csv"
    if path.exists():
        symbolic = pd.read_csv(path)
        if "symbolic_rmse" not in symbolic.columns:
            symbolic = _derive_symbolic_metrics(symbolic)
    else:
        runs = args.run_dir / "runs.csv"
        if not runs.exists():
            raise SystemExit(f"no {path} or {runs} found")
        df = pd.read_csv(runs)
        if "status" in df.columns:
            df = df[df["status"].eq("completed")]
        symbolic = _derive_symbolic_metrics(df)
        symbolic = symbolic[pd.to_numeric(symbolic["symbolic_rmse"], errors="coerce").notna()]

    report = _write_method_redundancy_outputs(
        symbolic, args.run_dir,
        threshold_dex=args.threshold_dex,
        min_pairs=args.min_pairs,
    )
    if report.empty:
        print("No method pairs had overlapping completed symbolic runs.")
        return 0
    flagged = report[report["redundant"]]
    print(f"Compared {len(report)} method pairs; flagged {len(flagged)} as redundant.")
    if not flagged.empty:
        print(flagged[["model_a", "model_b", "n_paired", "median_abs_rmse_ratio", "equivalent_fraction"]].to_string(index=False))
    print(f"Wrote {args.run_dir / 'method_redundancy.csv'}")
    print(f"Wrote {args.run_dir / 'method_redundancy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
