from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
import traceback
from pathlib import Path

# Harmless on CPU/CUDA; required before torch import for direct MPS run_one use.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml

from .models import run_model, resolve_shared_benchmark_config
from .config_utils import resolve_profile
from .device_utils import configure_torch_threads, resolve_device
from .build_info import benchmark_build_fingerprint, code_version
from .specs import TASKS, load_task_data


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cpu-threads", default=None, type=int, help="PyTorch CPU threads for this experiment process")
    ap.add_argument("--edge-policy", default=None)
    ap.add_argument("--gmp-refinement-policy", default=None)
    ap.add_argument("--shared-width", default=None, type=int, help="override shared total width W for sensitivity sweeps")
    ap.add_argument("--shared-library", default=None, help="override shared symbolic library (core10, medium16, research26)")
    args = ap.parse_args()

    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as exc:
        ap.error(str(exc))
    configure_torch_threads(device, args.cpu_threads)

    config_path = Path(args.config).resolve()
    root_dir = Path(__file__).resolve().parents[1]
    build_fp = benchmark_build_fingerprint(root_dir, config_path)
    build_version = code_version(root_dir)
    cfg = yaml.safe_load(config_path.read_text())
    profile = resolve_profile(cfg, args.profile)
    spec = TASKS[args.task]
    out = Path(args.output)
    started = time.time()
    payload = {
        "profile": args.profile,
        "task": spec.name,
        "suite": spec.suite,
        "task_type": spec.task_type,
        "description": spec.description,
        "representability": spec.representability,
        "source": spec.source,
        "formula_target": spec.formula or "",
        "target_terms": spec.target_terms,
        "target_tree_nodes": spec.target_tree_nodes,
        "model": args.model,
        "seed": int(args.seed),
        "status": "running",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": device,
        "cpu_threads": int(torch.get_num_threads()) if device == "cpu" else None,
        "code_version": build_version,
        "benchmark_build_fingerprint": build_fp,
        "started_unix": started,
    }
    _atomic_json(out, payload)
    try:
        dcfg = profile.get("data", {})
        data = load_task_data(
            spec, seed=args.seed,
            train_n=int(dcfg.get("train_n", 1500)),
            val_n=int(dcfg.get("val_n", 400)),
            test_n=int(dcfg.get("test_n", 500)),
            max_real_samples=int(dcfg.get("max_real_samples", 5000)),
        )
        model_cfgs = profile.get("model_config", {})
        if args.model in model_cfgs:
            mcfg = dict(model_cfgs[args.model])
        elif args.model.startswith("fast_multkan_deep_"):
            base = "fastkan_autosym" if args.model.endswith("autosym") else "fastkan_gsr"
            mcfg = dict(model_cfgs.get(base, model_cfgs.get("gsr", {})))
        elif args.model.startswith("multkan_deep_"):
            if args.model.endswith("autosym"):
                base = "autosym"
            elif args.model.endswith("gmp"):
                base = "gmp"
            else:
                base = "gsr"
            mcfg = dict(model_cfgs.get(base, {}))
        elif args.model.startswith("power_rulekan") and "rulekan" in model_cfgs:
            # PowerRuleKAN differs from RuleKAN only in the final powered-base
            # symbolic search.  All shared numerical/support-discovery settings
            # therefore inherit the ordinary RuleKAN configuration.
            mcfg = dict(model_cfgs["rulekan"])
        elif args.model.startswith("sisp_fast") and "rulekan_fast" in model_cfgs:
            mcfg = dict(model_cfgs["rulekan_fast"])
        elif args.model.startswith("sisp") and "rulekan" in model_cfgs:
            mcfg = dict(model_cfgs["rulekan"])
        elif args.model.startswith("rulekan_fast") and "rulekan_fast" in model_cfgs:
            mcfg = dict(model_cfgs["rulekan_fast"])
        elif args.model.startswith("rulekan") and "rulekan" in model_cfgs:
            mcfg = dict(model_cfgs["rulekan"])
        else:
            mcfg = {}
        mcfg["device"] = device
        if args.edge_policy is not None:
            mcfg["edge_policy"] = args.edge_policy
        if args.gmp_refinement_policy is not None:
            mcfg["gmp_refinement_policy"] = args.gmp_refinement_policy
        mcfg, shared_meta = resolve_shared_benchmark_config(
            args.model, spec, mcfg, profile.get("shared_settings", {}),
            width_override=args.shared_width, library_override=args.shared_library,
        )
        if shared_meta.get("shared_capacity_width") is not None:
            mcfg["_resolved_shared_width"] = int(shared_meta["shared_capacity_width"])
        run = run_model(args.model, spec, data, args.seed, mcfg)
        payload.update(shared_meta)
        payload.update(run.metrics)
        payload.update(run.extras)
        payload.update({
            "input_dim": int(data.train_x.shape[1]),
            # Persist the benchmark coordinate transform so stored symbolic
            # formulas can always be mapped back to the task's raw variables.
            # External SR engines see ``train_x`` (standardized coordinates),
            # whereas RuleKAN's exported formula is already unstandardized.
            "input_mean": [float(v) for v in data.input_mean.detach().cpu().reshape(-1)],
            "input_std": [float(v) for v in data.input_std.detach().cpu().reshape(-1)],
            "train_n": int(data.train_x.shape[0]),
            "val_n": int(data.val_x.shape[0]),
            "test_n": int(data.test_x.shape[0]),
            "status": "completed",
            "elapsed_seconds": float(time.time() - started),
            "finished_unix": time.time(),
        })
    except Exception as exc:
        payload.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": float(time.time() - started),
            "finished_unix": time.time(),
        })
    _atomic_json(out, payload)
    return 0 if payload["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
