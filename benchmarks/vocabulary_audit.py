"""Print the resolved symbolic vocabulary for a benchmark profile.

This audit is dependency-free: it resolves configuration only and does not
import/train external symbolic-regression packages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config_utils import resolve_profile
from .models import resolve_shared_benchmark_config
from .specs import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/configs/default.yaml")
    parser.add_argument("--profile", default="research_modern")
    parser.add_argument("--task", default="same_var_exp_sin")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    profile = resolve_profile(cfg, args.profile)
    if args.task not in TASKS:
        raise KeyError(f"unknown task {args.task!r}")
    spec = TASKS[args.task]
    shared = profile.get("shared_settings", {})
    model_cfgs = profile.get("model_config", {})

    rows = []
    for model in profile.get("models", []):
        base = dict(model_cfgs.get(model, {}))
        resolved, meta = resolve_shared_benchmark_config(model, spec, base, shared)
        native = meta.get("shared_symbolic_native_library")
        if native is None:
            # No symbolic vocabulary applies (e.g. ANFIS) or the selected
            # profile does not control this model's vocabulary.
            continue
        rows.append({
            "model": model,
            "conceptual_library": meta.get("shared_symbolic_library"),
            "conceptual_size": meta.get("shared_symbolic_library_size"),
            "native_library": native,
            "native_size": meta.get("shared_symbolic_native_library_size"),
            "exact_match": meta.get("shared_symbolic_native_exact_match"),
            "note": meta.get("shared_symbolic_native_note"),
        })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"profile={args.profile} task={args.task}")
    for row in rows:
        native = ", ".join(str(x) for x in row["native_library"])
        status = "exact" if row["exact_match"] else "native/constructive"
        print(f"{row['model']:<24} {status:<19} [{native}]")
        if row["note"]:
            print(f"{'':<44}{row['note']}")


if __name__ == "__main__":
    main()
