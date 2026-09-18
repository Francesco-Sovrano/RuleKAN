#!/usr/bin/env python3
"""Numeric-only RuleKAN spline-grid sensitivity check.

Example:
    python tools/grid_sensitivity_rulekan.py \
      --profile quick --task fuzzy_ite_cross --seeds 0,1,2 --grids 3,4,5,6,8,12 \
      --output grid_sensitivity.csv

The symbolic stage is disabled deliberately: `grid` controls only the numerical
KAN edge resolution, so this isolates whether spline resolution is a numerical
bottleneck before symbolic extraction.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import yaml

from benchmarks.models import resolve_shared_benchmark_config, run_model
from benchmarks.specs import TASKS, load_task_data


def ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmarks/configs/default.yaml")
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--grids", default="3,4,5,6,8,12")
    ap.add_argument("--output", type=Path, default=Path("grid_sensitivity.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    profile = cfg["profiles"][args.profile]
    spec = TASKS[args.task]
    dcfg = profile.get("data", {})
    rows = []

    for seed in ints(args.seeds):
        data = load_task_data(
            spec,
            seed=seed,
            train_n=int(dcfg.get("train_n", 1500)),
            val_n=int(dcfg.get("val_n", 400)),
            test_n=int(dcfg.get("test_n", 500)),
            max_real_samples=int(dcfg.get("max_real_samples", 5000)),
        )
        for grid in ints(args.grids):
            mcfg = dict(profile.get("model_config", {}).get("rulekan", {}))
            mcfg["symbolic"] = False
            shared = dict(profile.get("shared_settings", {}))
            shared["grid"] = int(grid)
            mcfg, _ = resolve_shared_benchmark_config("rulekan", spec, mcfg, shared)
            t0 = time.perf_counter()
            run = run_model("rulekan", spec, data, seed, mcfg)
            rows.append({
                "task": args.task,
                "seed": seed,
                "grid": grid,
                "test_rmse": run.metrics.get("test_rmse"),
                "test_nrmse": run.metrics.get("test_nrmse"),
                "parameters": run.extras.get("parameters"),
                "wall_seconds": time.perf_counter() - t0,
            })
            print(rows[-1], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
