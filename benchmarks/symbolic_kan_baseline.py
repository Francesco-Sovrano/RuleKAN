from __future__ import annotations

"""Adapter for the authors' official Symbolic-KAN implementation.

The benchmark uses a pinned checkout of
``sfaroughi3/Pub_Symbolic_KANs`` under ``external/Pub_Symbolic_KANs``. The
checkout is reconstructed by ``setup.sh`` and is intentionally gitignored.
This module does *not* reimplement Symbolic-KAN. It loads the
upstream supervised-regression code from ``Exp_reaction_diffusion`` and calls
its ``train_regression_onehot`` routine directly.

The only benchmark-side adaptations are:

* provide the benchmark's already-created train/validation tensors instead of
  the upstream demo data generator;
* force the device requested by the benchmark rather than the upstream module's
  import-time device choice;
* suppress diagnostic plots during benchmark jobs (training/selection/hardening
  are unchanged);
* serialize the trained discrete upstream model to a SymPy expression using
  the upstream model parameters.  The upstream text exporter is not used
  because it assumes at most four named inputs and omits learned readout
  weights/residual additions; those are required for faithful benchmark
  scoring of arbitrary-dimensional tasks.
"""

from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import sympy as sp
import torch


_UPSTREAM_URL = "https://github.com/sfaroughi3/Pub_Symbolic_KANs"
_DEFAULT_SUBDIR = "Exp_reaction_diffusion"


@dataclass(frozen=True)
class OfficialSymbolicKANAPI:
    repo_root: Path
    source_dir: Path
    commit: str
    training: Any
    classes: Any
    data_utils: Any
    vis_utils: Any
    primitive_utils: Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_official_repo_root() -> Path:
    env = os.environ.get("SYMBOLIC_KAN_OFFICIAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (_project_root() / "external" / "Pub_Symbolic_KANs").resolve()


def _read_commit(repo_root: Path) -> str:
    marker = repo_root / "UPSTREAM_COMMIT.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    git_dir = repo_root / ".git"
    if git_dir.exists():
        try:
            import subprocess
            return subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
    return "unknown"


def load_official_symbolic_kan(
    repo_root: str | os.PathLike[str] | None = None,
    *,
    subdir: str = _DEFAULT_SUBDIR,
    device: str | torch.device = "cpu",
) -> OfficialSymbolicKANAPI:
    """Load the pinned upstream Symbolic-KAN modules without rewriting them."""
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else default_official_repo_root()
    src = root / str(subdir)
    required = [
        src / "symKanTraining.py",
        src / "symKanClass.py",
        src / "genDataUtil.py",
        src / "visUtil.py",
        src / "primitiveUtil.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "official Symbolic-KAN source is missing; expected the authors' "
            f"Pub_Symbolic_KANs repository at {root}. Missing: {missing}"
        )

    src_s = str(src)
    if src_s not in sys.path:
        # Upstream modules use absolute sibling imports such as
        # ``from symKanClass import *``; adding only this source directory is
        # therefore the least invasive way to execute the repository verbatim.
        sys.path.insert(0, src_s)

    primitive_utils = importlib.import_module("primitiveUtil")
    classes = importlib.import_module("symKanClass")
    data_utils = importlib.import_module("genDataUtil")
    vis_utils = importlib.import_module("visUtil")
    training = importlib.import_module("symKanTraining")

    # Protect against accidentally resolving same-named modules from some other
    # checkout already on PYTHONPATH.
    for module in (primitive_utils, classes, data_utils, vis_utils, training):
        mod_file = Path(module.__file__).resolve()
        if src not in mod_file.parents:
            raise ImportError(
                f"loaded {module.__name__} from {mod_file}, not official Symbolic-KAN source {src}"
            )

    dev = torch.device(device)
    # Upstream files choose MPS/CUDA/CPU at import time.  The benchmark already
    # has an explicit --device contract, so update only those module globals.
    for module in (training, classes, data_utils, vis_utils):
        if hasattr(module, "DEVICE"):
            module.DEVICE = dev

    return OfficialSymbolicKANAPI(
        repo_root=root,
        source_dir=src,
        commit=_read_commit(root),
        training=training,
        classes=classes,
        data_utils=data_utils,
        vis_utils=vis_utils,
        primitive_utils=primitive_utils,
    )


def _official_primitive_expr(kind: str, z: sp.Expr, eps: float = 1e-6) -> sp.Expr:
    """Symbolic counterpart of the upstream exporter's intended primitives.

    Numerical clamps used by the upstream PyTorch implementation for stability
    are intentionally not emitted here, matching its own text/LaTeX exporter.
    Where the upstream exporter explicitly includes ``eps``, use its numeric
    constant so the result is directly evaluable by the benchmark.
    """
    k = str(kind)
    e = sp.Float(float(eps))
    if k in {"id", "x"}:
        return z
    if k == "zero":
        return sp.Integer(0)
    if k == "const":
        return sp.Integer(1)
    if k == "x2":
        return z**2
    if k == "x3":
        return z**3
    if k == "x4":
        return z**4
    if k == "x5":
        return z**5
    if k == "sqrtx":
        return sp.sqrt(sp.Abs(z))
    if k == "abs":
        return sp.Abs(z)
    if k == "sign":
        return sp.sign(z)
    if k == "relu":
        return sp.Max(sp.Integer(0), z)
    if k == "softplus":
        return sp.log(1 + sp.exp(z))
    if k == "sigmoid":
        return 1 / (1 + sp.exp(-z))
    if k == "swish":
        return z / (1 + sp.exp(-z))
    if k == "sin":
        return sp.sin(z)
    if k == "cos":
        return sp.cos(z)
    if k == "tan":
        return sp.tan(z)
    if k == "tanh":
        return sp.tanh(z)
    if k == "sinh":
        return sp.sinh(z)
    if k == "cosh":
        return sp.cosh(z)
    if k == "sech":
        return 1 / sp.cosh(z)
    if k == "exp":
        return sp.exp(z)
    if k == "exp_m1":
        return sp.exp(z) - 1
    if k == "log":
        return sp.log(sp.Abs(z) + e)
    if k == "log1p":
        return sp.log(1 + sp.sqrt(z**2 + e))
    if k == "inv":
        return 1 / (z + e * sp.sign(z))
    if k == "rsqrt":
        return 1 / sp.sqrt(z**2 + e)
    if k == "gauss":
        return sp.exp(-(z**2))
    if k == "lorentz":
        return 1 / (1 + z**2)
    if k == "elu":
        return sp.Piecewise((z, z >= 0), (sp.exp(z) - 1, True))
    if k == "softsign":
        return z / (1 + sp.Abs(z))
    raise ValueError(f"unsupported official Symbolic-KAN primitive for export: {k!r}")


def _selected_edge(block: Any, unit_idx: int) -> int:
    if getattr(block, "edge_mask", None) is not None:
        mask = block.edge_mask[unit_idx].detach().cpu().numpy()
        active = np.flatnonzero(mask >= 0.5)
        if len(active):
            return int(active[0])
    base = int(unit_idx) * int(block.edges_per_unit)
    confs = []
    for e in range(int(block.edges_per_unit)):
        gates = block.phi[base + e].gates.detach()
        confs.append(float(torch.softmax(gates, dim=0).max().cpu()))
    return int(np.argmax(confs))


def official_model_to_sympy(model: torch.nn.Module, variable_names: Sequence[str]) -> sp.Expr:
    """Serialize the *trained upstream model* without changing its structure."""
    exprs: list[sp.Expr] = [sp.Symbol(str(v), real=True) for v in variable_names]
    hidden: list[sp.Expr] | None = None

    for block in model.blocks:
        block_input = exprs if hidden is None else hidden
        unit_probs = None
        if getattr(block, "unit_logit", None) is not None:
            unit_probs = torch.sigmoid(block.unit_logit.detach()).cpu().numpy()
        fresh: list[sp.Expr] = []
        for j in range(int(block.hidden_units)):
            if unit_probs is not None and float(unit_probs[j]) <= 0.5:
                fresh.append(sp.Integer(0))
                continue
            e = _selected_edge(block, j)
            phi = block.phi[j * int(block.edges_per_unit) + e]
            p = int(torch.argmax(phi.gates.detach()).item())
            prim = phi.prims[p]

            w = block.proj_w[j, e].detach().cpu().numpy().astype(float)
            b_edge = float(block.proj_b[j, e].detach().cpu())
            s: sp.Expr = sp.Float(b_edge)
            for coeff, previous in zip(w, block_input):
                if abs(float(coeff)) > 1e-14:
                    s += sp.Float(float(coeff)) * previous

            gamma = float(3.0 * torch.tanh(prim.gamma_raw.detach()).cpu())
            beta = float(torch.tanh(prim.beta_raw.detach()).cpu())
            amp = float(3.0 * torch.tanh(prim.amp_raw.detach()).cpu())
            bias = float(torch.tanh(prim.bias_raw.detach()).cpu())
            z = sp.Float(gamma) * s + sp.Float(beta)
            out = sp.Float(amp) * _official_primitive_expr(prim.kind, z) + sp.Float(bias)
            fresh.append(out)

        # Upstream SymbolicKANBlock adds a residual after the first block when
        # hidden widths match.  Its own text exporter currently omits this; the
        # benchmark serializer must retain it so the formula represents the
        # actual trained discrete model.
        if hidden is not None and bool(getattr(block, "residual", True)) and len(hidden) == len(fresh):
            hidden = [a + b for a, b in zip(hidden, fresh)]
        else:
            hidden = fresh

    if hidden is None:
        raise RuntimeError("official Symbolic-KAN model has no blocks")
    weight = model.readout.weight.detach().cpu().numpy().reshape(-1).astype(float)
    expr: sp.Expr = sp.Integer(0)
    for coeff, h in zip(weight, hidden):
        if abs(float(coeff)) > 1e-14:
            expr += sp.Float(float(coeff)) * h
    if model.readout.bias is not None:
        expr += sp.Float(float(model.readout.bias.detach().cpu().reshape(-1)[0]))
    return expr


def official_formula(model: torch.nn.Module, variable_names: Sequence[str]) -> str:
    return sp.sstr(official_model_to_sympy(model, variable_names))


@torch.no_grad()
def official_hardened_predict(api: OfficialSymbolicKANAPI, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    p = next(model.parameters())
    xx = x.to(device=p.device, dtype=p.dtype)
    model.eval()
    return api.vis_utils.eval_symbolic_selected(model, xx).detach().cpu().reshape(-1, 1)


def fit_official_symbolic_kan(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    seed: int,
    device: str = "cpu",
    repo_root: str | os.PathLike[str] | None = None,
    source_subdir: str = _DEFAULT_SUBDIR,
    config: dict[str, Any] | None = None,
) -> tuple[OfficialSymbolicKANAPI, torch.nn.Module, dict[str, Any]]:
    """Fit through the authors' ``train_regression_onehot`` routine itself."""
    cfg = dict(config or {})
    api = load_official_symbolic_kan(repo_root, subdir=source_subdir, device=device)
    tr = api.training

    dev = torch.device(device)
    tx = train_x.detach().to(device=dev, dtype=torch.float32)
    ty = train_y.detach().to(device=dev, dtype=torch.float32).reshape(-1, 1)
    vx = val_x.detach().to(device=dev, dtype=torch.float32)
    vy = val_y.detach().to(device=dev, dtype=torch.float32).reshape(-1, 1)
    in_dim = int(tx.shape[1])

    # The upstream training routine owns its data generation.  Swap only that
    # function for the duration of this call so every optimizer, schedule,
    # selection, hardening and LBFGS step remains upstream code.
    original_make_dataset = getattr(tr, "make_dataset", None)
    original_plot = getattr(tr, "plot_alpha_grid_all_units_with_edge_conf", None)

    def _benchmark_dataset(**_kwargs):
        # Upstream returns both network-space and raw-space arrays.  Our model
        # is intentionally trained in the benchmark's standardized input space,
        # exactly like the other external symbolic regressors.
        return tx, ty, vx, vy, tx.clone(), vx.clone()

    tr.make_dataset = _benchmark_dataset
    if not bool(cfg.get("diagnostic_plots", False)) and original_plot is not None:
        tr.plot_alpha_grid_all_units_with_edge_conf = lambda *args, **kwargs: None

    # Use the authors' generic SymbolicKAN defaults unless the benchmark profile
    # explicitly controls them.  Primitive names here are upstream names, not
    # RuleKAN's local symbolic-library aliases.
    lib = tuple(str(v) for v in cfg.get(
        "lib", ("id", "x", "x2", "x3", "sin", "cos", "exp", "tanh")
    ))
    epochs = int(cfg.get("epochs", 1200))
    adam_epochs = int(cfg.get("adam_epochs", min(1000, max(1, epochs - 1))))
    if bool(cfg.get("use_lbfgs", True)) and adam_epochs >= epochs:
        adam_epochs = max(1, epochs - 1)

    kwargs = dict(
        in_dim=in_dim,
        hidden_units=int(cfg.get("hidden_units", 12)),
        edges_per_unit=int(cfg.get("edges_per_unit", cfg.get("n_edges", 3))),
        num_blocks=int(cfg.get("num_blocks", 2)),
        lib=lib,
        tau_start=float(cfg.get("tau_start", 4.0)),
        tau_end=float(cfg.get("tau_end", 0.2)),
        drop_tau_to=float(cfg.get("drop_tau_to", 0.1)),
        hard_gumbel=bool(cfg.get("hard_gumbel", False)),
        residual=bool(cfg.get("residual", True)),
        epochs=epochs,
        lr=float(cfg.get("lr", 5e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        use_lbfgs=bool(cfg.get("use_lbfgs", True)),
        adam_epochs=adam_epochs,
        lbfgs_max_iter=int(cfg.get("lbfgs_max_iter", 15)),
        lbfgs_steps=int(cfg.get("lbfgs_steps", 40)),
        sel_w_start=float(cfg.get("sel_w_start", 0.0)),
        sel_w_end=float(cfg.get("sel_w_end", 5e-3)),
        entropy_weight=float(cfg.get("entropy_weight", 0.2)),
        nms_weight=float(cfg.get("nms_weight", 0.1)),
        sel_start_frac=float(cfg.get("sel_start_frac", 0.6)),
        unit_gate_w=float(cfg.get("unit_gate_w", 1e-4)),
        Middle_LBFGS_harden=bool(cfg.get("middle_lbfgs_harden", False)),
        do_logit_snap=bool(cfg.get("do_logit_snap", True)),
        print_every=int(cfg.get("print_every", 100)),
        ckpt_every=int(cfg.get("ckpt_every", max(epochs + 1, 1000000))),
        seed=int(seed),
        train_proj_b=bool(cfg.get("train_proj_b", True)),
        train_prim_beta=bool(cfg.get("train_prim_beta", False)),
        train_prim_bias=bool(cfg.get("train_prim_bias", False)),
        train_readout_bias=bool(cfg.get("train_readout_bias", False)),
        use_unit_gates=bool(cfg.get("use_unit_gates", True)),
        freeze_edges_for_lbfgs=bool(cfg.get("freeze_edges_for_lbfgs", True)),
        if_pde_problem=False,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="rulekan_symbolic_kan_official_") as run_dir:
            model, _returned_data = tr.train_regression_onehot(RUN_DIR=run_dir, **kwargs)
    finally:
        if original_make_dataset is not None:
            tr.make_dataset = original_make_dataset
        if original_plot is not None:
            tr.plot_alpha_grid_all_units_with_edge_conf = original_plot

    model.eval()
    active = 0
    total = 0
    for block in model.blocks:
        if getattr(block, "unit_logit", None) is None:
            active += int(block.hidden_units)
            total += int(block.hidden_units)
        else:
            probs = torch.sigmoid(block.unit_logit.detach())
            active += int((probs > 0.5).sum().cpu())
            total += int(probs.numel())

    diag: dict[str, Any] = {
        "symbolic_kan_upstream_url": _UPSTREAM_URL,
        "symbolic_kan_upstream_commit": api.commit,
        "symbolic_kan_upstream_subdir": str(source_subdir),
        "symbolic_kan_hidden_units": int(kwargs["hidden_units"]),
        "symbolic_kan_edges_per_unit": int(kwargs["edges_per_unit"]),
        "symbolic_kan_num_blocks": int(kwargs["num_blocks"]),
        "symbolic_kan_primitives": list(lib),
        "symbolic_kan_epochs": int(kwargs["epochs"]),
        "symbolic_kan_adam_epochs": int(kwargs["adam_epochs"]),
        "symbolic_kan_lbfgs_steps": int(kwargs["lbfgs_steps"]),
        "symbolic_kan_active_units": int(active),
        "symbolic_kan_total_units": int(total),
    }
    return api, model, diag
