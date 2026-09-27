#!/usr/bin/env python3
"""RuleMask-KAN with in-context structural and symbolic refinement.

The numerical path intentionally preserves the strong v3 KAN workflow:

    overcomplete fit -> KAN prune/refit -> KAN prune/refit -> final polish

The paper-inspired pieces are applied *after* or *around* that reliable numeric
path rather than replacing it:

* RuleMask-GSR: ambiguous product masks are proposed from gate probabilities,
  but a mask edit is committed only when a short end-to-end refit improves an
  independent validation set.
* Symbolic GSR: spline-to-symbol choices are made by end-to-end validation loss
  after brief full-model refits, not by isolated edge R^2 alone.
* Optional GMP mode: soft categorical gates, entropy regularisation, scaled
  asinh variance compression, progressive top-k candidate pruning, then hard
  discretisation and restricted GSR refinement.

For these synthetic examples an independent validation sample is generated from
exactly the same target/ranges and normalized with the training-set statistics.
The provided test split stays untouched until final reporting.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from rulekan import KAN, create_dataset
from rulekan.rule_mask import RuleMaskProduct
from rulekan.utils import SYMBOLIC_LIB

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ExampleCase:
    name: str
    description: str
    variable_names: List[str]
    ranges: List[List[float]]
    fn: TensorFn
    rule_orders: List[int]

    @property
    def max_order(self) -> int:
        return max(self.rule_orders)

    @property
    def hidden_rules(self) -> int:
        return len(self.rule_orders)


def damped_nonlinear_response(x: torch.Tensor) -> torch.Tensor:
    t, phase, concentration, angle, control = [x[:, [i]] for i in range(5)]
    return (
        1.3 * torch.exp(-0.9 * t) * torch.sin(2.2 * phase)
        + 0.75 * torch.log1p(concentration.square()) * torch.cos(1.6 * angle)
        + 0.25 * torch.tanh(2.0 * control)
    )


def three_way_interaction(x: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4, x5 = [x[:, [i]] for i in range(5)]
    return (
        1.1 * torch.sin(math.pi * x1) * torch.exp(0.7 * x2) * (1.0 + 0.5 * x3.square())
        - 0.6 * torch.tanh(2.0 * x4) * torch.cos(1.8 * x5)
    )


def rational_smooth_mix(x: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4, x5 = [x[:, [i]] for i in range(5)]
    return (
        0.9 * torch.cos(2.0 * x1) / (1.0 + 1.5 * x2.square())
        + 0.55 * torch.sqrt(1.0 + x3.square()) * torch.tanh(1.7 * x4)
        + 0.2 * torch.sin(3.0 * x5)
    )


CASES: Dict[str, ExampleCase] = {
    "damped": ExampleCase(
        "damped",
        "damped oscillation + nonlinear response + unary control",
        ["t", "phase", "concentration", "angle", "control"],
        [[0.0, 2.0], [-math.pi, math.pi], [-2.0, 2.0], [-math.pi, math.pi], [-1.5, 1.5]],
        damped_nonlinear_response,
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
    "three_way": ExampleCase(
        "three_way",
        "one 3-way nonlinear product plus one pairwise product",
        ["x1", "x2", "x3", "x4", "x5"],
        [[-1.0, 1.0]] * 5,
        three_way_interaction,
        [2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
    ),
    "rational": ExampleCase(
        "rational",
        "rational, square-root, trigonometric and saturating factors",
        ["x1", "x2", "x3", "x4", "x5"],
        [[-1.5, 1.5]] * 5,
        rational_smooth_mix,
        [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    ),
}


def build_model(case: ExampleCase, seed: int, gate_mode: str = "hard_st") -> KAN:
    # hard_st deliberately matches the numerically successful v3 settings.
    # GMP gets a warmer distribution because it is trained as a soft mixture.
    if gate_mode == "hard_st":
        temperature = 0.70
        surrogate_clip = 6.0
    else:
        temperature = 1.0
        surrogate_clip = 6.0

    return KAN(
        width=[len(case.variable_names), [case.hidden_rules, 0], 1],
        grid=8,
        k=3,
        grid_range=[-2.0, 2.0],
        seed=seed,
        auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={
            "init_prob": 0.50,
            "min_order": 1,
            "max_order": case.max_order,
            "rule_orders": case.rule_orders,
            "temperature": temperature,
            "stochastic": False,
            "surrogate_clip": surrogate_clip,
            "gate_mode": gate_mode,
            "asinh_scale": 4.0,
            "learnable_asinh_scale": True,
        },
    )


def _make_external_validation(case: ExampleCase, dataset, n: int, seed: int):
    """Independent validation sample, normalized by *training* statistics."""
    g = torch.Generator(device=dataset["train_input"].device)
    g.manual_seed(seed + 982451653)
    ranges = torch.as_tensor(case.ranges, dtype=dataset["train_input"].dtype,
                             device=dataset["train_input"].device)
    lo, hi = ranges[:, 0], ranges[:, 1]
    raw = torch.rand((n, len(case.variable_names)), generator=g,
                     device=dataset["train_input"].device,
                     dtype=dataset["train_input"].dtype)
    raw = raw * (hi - lo) + lo
    y = case.fn(raw)
    mean = dataset["input_mean"].to(raw)
    std = dataset["input_std"].to(raw)
    x = (raw - mean) / std
    return x, y


def _mse(model: KAN, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


def _rmse(model: KAN, x: torch.Tensor, y: torch.Tensor) -> float:
    return math.sqrt(_mse(model, x, y))


def _metrics(model: KAN, dataset) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred, y = model(dataset["test_input"]), dataset["test_label"]
        e = pred - y
        return {
            "rmse": float(e.square().mean().sqrt()),
            "mae": float(e.abs().mean()),
            "max_abs": float(e.abs().max()),
            "baseline": float(((y - y.mean()).square().mean()).sqrt()),
        }


def _fit(model: KAN, data, *, steps: int, lr: float, lamb: float = 0.0,
         entropy: float = 0.0, log: int = 100000, validation=None,
         restore_best: bool = False, lr_schedule=None, min_lr: float = 0.0,
         grad_clip: float = 1.0, rule_mask_lr_scale: float = 1.0) -> None:
    model.fit(
        data,
        optimizer="Adam",
        lr=float(lr),
        steps=int(steps),
        lamb=float(lamb),
        rule_mask_l1=0.0,
        rule_mask_entropy=float(entropy),
        log=max(1, int(log)),
        validation_data=validation,
        restore_best=bool(restore_best),
        validation_check_every=max(1, min(10, int(steps))),
        lr_schedule=lr_schedule,
        min_lr=float(min_lr),
        grad_clip=float(grad_clip),
        rule_mask_lr_scale=float(rule_mask_lr_scale),
    )


def _stable_polish(model: KAN, data, val_x, val_y, *, steps: int, lr: float,
                   min_lr: float, log: int, label: str) -> KAN:
    """Long continuous fit with frozen discrete structure and best-val restore.

    Once RuleMask is discretised there is no reason to keep training with a
    constant learning rate.  A cosine decay removes the late-stage Adam jitter,
    while restoring the best independent-validation checkpoint prevents the
    final iterate from being worse than an earlier one.
    """
    before = _rmse(model, val_x, val_y)
    _fit(
        model, data, steps=steps, lr=lr, lamb=0.0, entropy=0.0, log=log,
        validation=(val_x, val_y), restore_best=True, lr_schedule="cosine",
        min_lr=min_lr, grad_clip=0.5, rule_mask_lr_scale=1.0,
    )
    after = _rmse(model, val_x, val_y)
    print(f"  {label}: validation RMSE {before:.6g} -> {after:.6g}")
    return model


def _fit_if_not_worse(model: KAN, data, val_x, val_y, *, steps: int, lr: float,
                      lamb: float = 0.0, entropy: float = 0.0,
                      rel_budget: float = 0.0, log: int = 100000,
                      label: str = "refit") -> KAN:
    """Train a copy and keep it only if independent validation does not worsen."""
    base = _mse(model, val_x, val_y)
    trial = model.copy()
    _fit(trial, data, steps=steps, lr=lr, lamb=lamb, entropy=entropy, log=log)
    score = _mse(trial, val_x, val_y)
    allowed = base * (1.0 + max(0.0, float(rel_budget)))
    keep = score <= allowed
    print(
        f"  {label}: {math.sqrt(base):.6g} -> {math.sqrt(score):.6g} "
        f"({'accepted' if keep else 'rolled back'})"
    )
    return trial if keep else model


def _rulemask(model: KAN, layer: int = 0) -> RuleMaskProduct:
    rm = model.rule_masks[layer]
    if not isinstance(rm, RuleMaskProduct):
        raise TypeError(f"layer {layer} is not a RuleMask layer")
    return rm


def _print_rules(model: KAN, names: Sequence[str], title: str) -> None:
    print(f"\n{title}")
    print(f"  width: {model.width}")
    for layer, rules in model.get_rule_mask_rules(list(names)).items():
        for r, rule in enumerate(rules):
            print(f"  layer {layer} rule {r}: {rule}")


def _print_gate_diag(model: KAN, prefix: str = "") -> None:
    d = _rulemask(model).gate_diagnostics()
    print(
        f"{prefix}gate entropy={d['entropy']:.3f}, "
        f"margin={d['margin']:.3f}, mean shortlist={d['shortlist']:.2f}"
    )


def _kan_prune_refit(model: KAN, train_data, val_x, val_y, *, lr: float,
                     lamb: float, node_th: float, edge_th: float,
                     refit_steps: int, max_cycles: int, rel_budget: float,
                     entropy: float, log: int, guard: bool = False) -> KAN:
    """Run the native KAN prune/refit schedule.

    The default deliberately mirrors the numerically strong v3/paper workflow:
    every requested prune/refit cycle is completed, even when the threshold
    removes no nodes and even when an intermediate validation checkpoint is
    temporarily worse.  A strict validation rollback is available only as an
    opt-in diagnostic because rejecting an intermediate stage can prevent later
    refits from recovering/converging.
    """
    print("\n=== KAN prune/refit ===")
    reference = _mse(model, val_x, val_y)
    allowed = reference * (1.0 + max(0.0, rel_budget))
    if guard:
        print(f"  guarded reference RMSE={math.sqrt(reference):.6g}; cap={math.sqrt(allowed):.6g}")
    else:
        print(f"  reference RMSE={math.sqrt(reference):.6g}; fixed prune/refit schedule enabled")

    for cycle in range(int(max_cycles)):
        model.get_act(train_data)
        before = copy.deepcopy(model.width)
        cand = model.prune(node_th=float(node_th), edge_th=float(edge_th))
        changed = cand.width != before

        # A prune/refit cycle is an optimization stage even when no node is
        # removed.  Never skip the refit merely because width is unchanged.
        _fit(cand, train_data, steps=refit_steps, lr=lr, lamb=lamb,
             entropy=entropy, log=log)
        score = _mse(cand, val_x, val_y)
        tag = "pruned" if changed else "no deletion; refit only"
        print(
            f"  cycle {cycle + 1} ({tag}): {before} -> {cand.width}, "
            f"val RMSE={math.sqrt(score):.6g}"
        )
        if guard and score > allowed:
            print("  guarded rollback: candidate exceeded the original validation budget")
            continue
        model = cand
    return model


def _gmp_train(model: KAN, train_data, val_x, val_y, *, lr: float,
               entropy: float, warmup_steps: int, topk_steps: int,
               topk_schedule: Sequence[int], log: int) -> Tuple[KAN, list]:
    """Optional paper-style GMP proposal stage with delayed top-k pruning."""
    print("\n=== GMP: soft gated warmup ===")
    _fit(model, train_data, steps=warmup_steps, lr=lr, entropy=entropy, log=log)
    print(f"  validation RMSE={_rmse(model, val_x, val_y):.6g}")
    _print_gate_diag(model, "  ")

    rm = _rulemask(model)
    for k in topk_schedule:
        k = int(k)
        if k >= rm.in_dim:
            continue
        print(f"\n=== GMP: keep top-{k} feature candidates per factor slot ===")
        rm.prune_candidates_topk(k)
        rm.temperature = max(0.55, rm.temperature * 0.8)
        _fit(model, train_data, steps=topk_steps, lr=lr, entropy=entropy, log=log)
        print(f"  validation RMSE={_rmse(model, val_x, val_y):.6g}")
        _print_gate_diag(model, "  ")

    return model, rm.retained_feature_candidates(topk=2)


def _discretize_without_damage(model: KAN, train_data, val_x, val_y, *,
                               lr: float, refit_steps: int, log: int) -> KAN:
    """Commit current slot argmax without sacrificing the numeric checkpoint.

    Hard-ST already evaluates the hard product during training, so discretising it
    should be function preserving and requires no extra optimization.  GMP uses a
    soft mixture during training, therefore only GMP receives a short guarded hard
    refit after argmax discretisation.
    """
    print("\n=== gate discretisation ===")
    base_rmse = _rmse(model, val_x, val_y)
    trial = model.copy()
    rm = _rulemask(trial)
    rm.discretize(freeze=True)
    hard_rmse = _rmse(trial, val_x, val_y)
    print(f"  validation RMSE before={base_rmse:.6g}, hard={hard_rmse:.6g}")

    if rm.gate_mode == "hard_st" or int(refit_steps) <= 0:
        return trial

    adapted = trial.copy()
    _fit(adapted, train_data, steps=refit_steps, lr=lr, entropy=0.0, log=log)
    refit_rmse = _rmse(adapted, val_x, val_y)
    print(f"  GMP hard refit RMSE={refit_rmse:.6g}")
    return adapted if refit_rmse <= hard_rmse else trial


def _force_rule_features(model: KAN, rule_idx: int, features: Sequence[int]) -> None:
    rm = _rulemask(model)
    hard = rm.hard_mask(stochastic=False).to(rm.logits.device)
    hard[rule_idx].zero_()
    hard[rule_idx, torch.as_tensor(features, dtype=torch.long, device=hard.device)] = 1.0
    rm.set_hard_mask(hard, strength=18.0, freeze=True)


def _candidate_feature_sets(shortlist_row: list, order: int, current: Sequence[int], limit: int = 10):
    pool = sorted({j for slot in shortlist_row[:order] for j in slot})
    cur = tuple(sorted(int(i) for i in current))
    combos = list(itertools.combinations(pool, order)) if len(pool) >= order else []
    if cur not in combos:
        combos.append(cur)
    combos = sorted(set(combos), key=lambda c: (len(set(c) ^ set(cur)), c))
    return combos[: int(limit)]


def _rule_importance_order(model: KAN, data) -> List[int]:
    rm = _rulemask(model)
    try:
        model.get_act(data)
        model.attribute()
        vals = model.edge_scores[1].reshape(-1).detach().cpu().tolist()
        return sorted(range(rm.out_dim), key=lambda r: vals[r], reverse=True)
    except Exception:
        return list(range(rm.out_dim))


def _structural_gsr(model: KAN, train_data, val_x, val_y, shortlist: list, *,
                    lr: float, trial_steps: int, max_rules: int,
                    min_rel_improvement: float, log: int) -> Tuple[KAN, int]:
    """Paper-style in-context scoring for ambiguous product factor identities.

    Every edit is compared with the *untouched current model*. Equal-complexity
    mask changes therefore need to improve validation loss; they cannot silently
    degrade an already good numeric solution.
    """
    print("\n=== RuleMask-GSR: in-context factor refinement ===")
    order_rules = _rule_importance_order(model, train_data)
    changed = 0

    for r in order_rules[: int(max_rules)]:
        rm = _rulemask(model)
        current = torch.nonzero(rm.hard_mask()[r] > 0, as_tuple=False).squeeze(-1).tolist()
        q = len(current)
        if q == 0 or r >= len(shortlist):
            continue
        candidates = _candidate_feature_sets(shortlist[r], q, current)
        if len(candidates) <= 1:
            continue

        base_mse = _mse(model, val_x, val_y)
        best_score = base_mse
        best_model = None
        best_set = tuple(sorted(current))

        for cand_set in candidates:
            if tuple(sorted(cand_set)) == best_set:
                continue
            trial = model.copy()
            _force_rule_features(trial, r, cand_set)
            _fit(
                trial, train_data, steps=trial_steps, lr=lr, entropy=0.0,
                log=max(log, trial_steps + 1), validation=(val_x, val_y),
                restore_best=True, lr_schedule="cosine", min_lr=max(lr * 0.1, 1e-5),
                grad_clip=0.5,
            )
            score = _mse(trial, val_x, val_y)
            if score < best_score:
                best_score, best_model, best_set = score, trial, tuple(cand_set)

        required = base_mse * (1.0 - max(0.0, min_rel_improvement))
        old = tuple(sorted(current))
        if best_model is not None and best_score < required:
            model = best_model
            changed += 1
            print(f"  rule {r}: {old} -> {best_set}, val RMSE {math.sqrt(base_mse):.6g} -> {math.sqrt(best_score):.6g}")
        else:
            print(f"  rule {r}: keep {old}; no in-context improvement")

    print(f"  accepted {changed} structural edit(s)")
    return model, changed


def _compact_for_symbolic(model: KAN, train_data, val_x, val_y, *,
                          lr: float, min_rules: int, refit_steps: int,
                          rel_budget: float, log: int) -> KAN:
    """Find a compact hidden-rule subset using end-to-end validation loss.

    This is intentionally applied only to the *symbolic copy*. The high-accuracy
    numerical model is preserved. Rules are ranked by downstream KAN importance,
    then we test increasingly large top-k subsets with a brief whole-model refit.
    The smallest subset inside one global validation budget is selected.
    """
    rm = _rulemask(model)
    total = rm.out_dim
    if total <= min_rules:
        return model

    ranking = _rule_importance_order(model, train_data)
    base = _mse(model, val_x, val_y)
    cap = base * (1.0 + max(0.0, float(rel_budget)))
    print("\n=== in-context symbolic compaction ===")
    print(f"  baseline rules={total}, RMSE={math.sqrt(base):.6g}, cap={math.sqrt(cap):.6g}")

    best = model
    # Complexity-first search. Stop at the first subset that meets the global
    # budget; if none does, retain the unpruned symbolic copy.
    for k in range(max(1, int(min_rules)), total):
        keep = sorted(ranking[:k])
        try:
            cand = model.prune_node(
                mode="manual",
                active_neurons_id=[keep],
                log_history=False,
            )
        except Exception:
            continue
        _fit(
            cand, train_data, steps=refit_steps, lr=lr, lamb=0.0, entropy=0.0,
            log=max(log, refit_steps + 1), validation=(val_x, val_y),
            restore_best=True, lr_schedule="cosine", min_lr=max(lr * 0.1, 1e-5),
            grad_clip=0.5,
        )
        score = _mse(cand, val_x, val_y)
        print(f"  try {k:2d}/{total} rules: val RMSE={math.sqrt(score):.6g}")
        if score <= cap:
            print(f"  accepted compact symbolic bank: {total} -> {k} rules")
            best = cand
            break
    return best

def _selected_active_edges(model: KAN, data) -> List[Tuple[int, int, int, float]]:
    model.get_act(data)
    try:
        model.attribute()
    except Exception:
        pass
    edges = []
    for l in range(model.depth):
        selected = None
        if isinstance(model.rule_masks[l], RuleMaskProduct):
            active = model.act_fun[l].mask.T > 0
            if hasattr(model.symbolic_fun[l], "mask"):
                active = active | (model.symbolic_fun[l].mask > 0)
            selected = model.rule_masks[l].hard_mask(active_mask=active, stochastic=False)
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                if selected is not None and selected[j, i] <= 0:
                    continue
                # convert only still-numeric edges
                if not bool(model.act_fun[l].mask[i, j] > 0):
                    continue
                imp = 1.0
                try:
                    imp = float(model.edge_scores[l][j, i].detach().cpu())
                except Exception:
                    pass
                edges.append((l, i, j, imp))
    return sorted(edges, key=lambda e: e[3], reverse=True)


def _fast_symbolic_shortlist(model: KAN, data, edge, lib: Sequence[str], topk: int):
    """Fast local *screen* for GSR candidates.

    The old implementation called ``old_fix_symbolic`` for every library atom
    just to build a shortlist. That performs a non-convex affine fit per atom and
    made GSR prohibitively slow. Here we only sample the learned numeric edge,
    evaluate a small coarse grid of input scale/shift values, and solve the
    output affine fit analytically. The expensive KAN symbolic fitter is then
    called only for the top-k candidates inside the end-to-end GSR trial.

    This screen is deliberately not the final decision criterion; global
    validation loss after short refitting remains the decision criterion.
    """
    l, i, j, _ = edge
    model.get_act(data)
    x = model.acts[l][:, i].detach().reshape(-1)
    y = model.spline_postacts[l][:, j, i].detach().reshape(-1)

    # Limit screening samples; ranking primitive shapes does not need the full
    # training set and this keeps conversion cost nearly independent of N.
    if x.numel() > 512:
        ids = torch.linspace(0, x.numel() - 1, 512, device=x.device).long()
        x, y = x[ids], y[ids]

    y_mean = y.mean()
    y_center = y - y_mean
    sst = (y_center.square().sum()).clamp_min(1e-12)

    # Coarse affine-input search. Sign is included because exp/log/tanh need it;
    # periodic atoms also get phase-shift candidates.
    bvals = torch.tensor(
        [-4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
          0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        device=x.device, dtype=x.dtype,
    )
    cvals_general = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=x.device, dtype=x.dtype)
    cvals_periodic = torch.tensor(
        [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi],
        device=x.device, dtype=x.dtype,
    )

    scores = []
    for name in lib:
        if name not in SYMBOLIC_LIB:
            continue
        fun = SYMBOLIC_LIB[name][0]
        cvals = cvals_periodic if name in {"sin", "cos"} else cvals_general
        z = bvals[:, None, None] * x[None, None, :] + cvals[None, :, None]
        try:
            g = fun(z)
        except Exception:
            continue
        g = torch.nan_to_num(g, nan=0.0, posinf=1e6, neginf=-1e6)
        g = g.clamp(-1e6, 1e6).reshape(-1, x.numel())
        gm = g.mean(dim=1, keepdim=True)
        gc = g - gm
        var = gc.square().sum(dim=1).clamp_min(1e-12)
        # Least-squares y ~= a*g + d for every coarse (b,c) candidate.
        a = (gc * y_center.unsqueeze(0)).sum(dim=1) / var
        d = y_mean - a * gm.squeeze(1)
        pred = a[:, None] * g + d[:, None]
        sse = (pred - y.unsqueeze(0)).square().sum(dim=1)
        best_sse = torch.min(sse)
        r2 = 1.0 - best_sse / sst
        scores.append((float(r2.detach().cpu()), name))

    scores.sort(reverse=True)
    return [name for _, name in scores[: max(1, int(topk))]]


def _symbolic_gsr(model: KAN, train_data, val_x, val_y, *, lib: Sequence[str],
                  local_topk: int, trial_steps: int, lr: float, log: int) -> KAN:
    """Greedy in-context symbolic regression from the uploaded paper."""
    print("\n=== Symbolic GSR: end-to-end operator selection ===")
    edges = _selected_active_edges(model, train_data)
    print(f"  active numeric edges to convert: {len(edges)}")

    for n, edge in enumerate(edges, 1):
        l, i, j, imp = edge
        shortlist = _fast_symbolic_shortlist(model, train_data, edge, lib, local_topk)
        # The readout of a sum-of-products model should preferably be affine.
        # Do not force it before numerical training (v4 did and regressed), but
        # always let in-context GSR test identity first at conversion time.
        if l == model.depth - 1 and "x" not in shortlist:
            shortlist = ["x"] + shortlist
        if not shortlist:
            print(f"  edge {n}/{len(edges)} ({l},{i}->{j}): no valid candidates")
            continue

        best_model, best_name, best_score = None, None, float("inf")
        for name in shortlist:
            trial = model.copy()
            trial.get_act(train_data)
            try:
                trial.old_fix_symbolic(l, i, j, name, verbose=False, log_history=False)
                _fit(
                    trial, train_data, steps=trial_steps, lr=lr, entropy=0.0,
                    log=max(log, trial_steps + 1), validation=(val_x, val_y),
                    restore_best=True, lr_schedule="cosine", min_lr=max(lr * 0.1, 1e-5),
                    grad_clip=0.5,
                )
                score = _mse(trial, val_x, val_y)
            except Exception:
                continue
            if score < best_score:
                best_score, best_model, best_name = score, trial, name

        if best_model is None:
            print(f"  edge {n}/{len(edges)} ({l},{i}->{j}): conversion failed")
            continue
        model = best_model
        print(
            f"  edge {n}/{len(edges)} ({l},{i}->{j}), importance={imp:.3g}: "
            f"{best_name}, val RMSE={math.sqrt(best_score):.6g}"
        )
    return model


def _print_formula(model: KAN, dataset, case: ExampleCase, simplify: bool) -> None:
    normalizer = [dataset["input_mean"].detach().cpu(), dataset["input_std"].detach().cpu()]
    formulas, _ = model.symbolic_formula(
        var=case.variable_names,
        normalizer=normalizer,
        simplify=bool(simplify),
        compact=True,
    )
    print("\nFINAL SYMBOLIC FORMULA (original input coordinates)")
    for i, expr in enumerate(formulas):
        print(f"  y[{i}] = {expr}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=sorted(CASES), default="damped")
    p.add_argument("--train-num", type=int, default=1800)
    p.add_argument("--test-num", type=int, default=700)
    p.add_argument("--val-num", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--log", type=int, default=10)

    # Numerically proven v3 schedule. Do not change these defaults casually.
    p.add_argument("--steps", type=int, default=350, help="steps per warmup/prune-refit stage")
    p.add_argument("--final-steps", type=int, default=1200, help="frozen-structure numerical convergence steps")
    p.add_argument("--final-lr", type=float, default=1.0e-3)
    p.add_argument("--final-min-lr", type=float, default=5.0e-5)
    p.add_argument("--prune-iters", type=int, default=2)
    p.add_argument("--node-th", type=float, default=0.03)
    p.add_argument("--edge-th", type=float, default=0.0)
    p.add_argument("--lamb", type=float, default=1e-5)
    p.add_argument("--rule-mask-entropy", type=float, default=1e-5)
    p.add_argument("--prune-rel-budget", type=float, default=0.05,
                   help="global independent-validation MSE budget for KAN pruning")
    p.add_argument("--guard-kan-pruning", action=argparse.BooleanOptionalAction, default=False,
                   help="optionally roll back prune/refit stages that exceed the initial validation budget; off by default to preserve the native KAN schedule")

    # Paper-inspired gate path. Hard-ST remains default because the paper itself
    # reports that gated selection can separate more slowly and be schedule-sensitive.
    p.add_argument("--gate-training", choices=["hard_st", "gmp"], default="hard_st")
    p.add_argument("--gmp-warmup-steps", type=int, default=700)
    p.add_argument("--gmp-topk-steps", type=int, default=220)
    p.add_argument("--gate-entropy", type=float, default=1e-3)
    p.add_argument("--hard-refit-steps", type=int, default=120)

    # Restricted in-context structure correction.
    p.add_argument("--structure-gsr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--structure-gsr-rules", type=int, default=6)
    p.add_argument("--structure-gsr-steps", type=int, default=80)
    p.add_argument("--structure-gsr-lr", type=float, default=5.0e-4)
    p.add_argument("--structure-gsr-min-improvement", type=float, default=0.0)

    # End-to-end symbolic extraction. Smaller shortlist/step defaults keep GSR practical.
    p.add_argument("--symbolic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--symbolic-min-rules", type=int, default=3)
    p.add_argument("--symbolic-compact-steps", type=int, default=120)
    p.add_argument("--symbolic-lr", type=float, default=7.5e-4)
    p.add_argument("--symbolic-structure-rel-budget", type=float, default=0.10,
                   help="MSE budget for simplifying only the symbolic copy")
    p.add_argument("--symbolic-local-topk", type=int, default=2)
    p.add_argument("--symbolic-gsr-steps", type=int, default=30)
    p.add_argument("--symbolic-polish-steps", type=int, default=300)
    p.add_argument("--formula-simplify", action=argparse.BooleanOptionalAction, default=False,
                   help="SymPy simplification can be very slow on large recovered formulas")
    args = p.parse_args()

    if args.prune_iters < 0:
        raise ValueError("--prune-iters must be >= 0")

    case = CASES[args.case]
    torch.manual_seed(args.seed)
    dataset = create_dataset(
        case.fn,
        n_var=len(case.variable_names),
        ranges=case.ranges,
        train_num=args.train_num,
        test_num=args.test_num,
        normalize_input=True,
        seed=args.seed,
    )
    val_x, val_y = _make_external_validation(case, dataset, args.val_num, args.seed)
    model = build_model(case, args.seed, gate_mode=args.gate_training)

    print(f"case: {case.name}")
    print(f"target: {case.description}")
    print(f"variables: {case.variable_names}")
    print(f"rule-order bank: {case.rule_orders}")
    print(f"gate training: {args.gate_training}")
    print("initial hard masks:")
    print(model.get_rule_masks(hard=True))

    if args.gate_training == "gmp":
        model, shortlist = _gmp_train(
            model, dataset, val_x, val_y,
            lr=args.lr,
            entropy=args.gate_entropy,
            warmup_steps=args.gmp_warmup_steps,
            topk_steps=args.gmp_topk_steps,
            topk_schedule=(3, 2),
            log=args.log,
        )
        model = _discretize_without_damage(
            model, dataset, val_x, val_y,
            lr=args.lr, refit_steps=args.hard_refit_steps, log=args.log,
        )
        # KAN pruning only after the gated factor candidates have separated.
        model = _kan_prune_refit(
            model, dataset, val_x, val_y,
            lr=args.lr, lamb=args.lamb,
            node_th=args.node_th, edge_th=args.edge_th,
            refit_steps=args.steps, max_cycles=args.prune_iters,
            rel_budget=args.prune_rel_budget,
            entropy=0.0, log=args.log, guard=args.guard_kan_pruning,
        )
    else:
        # Exact v3-style numeric schedule, now guarded by independent validation
        # only at the stage boundaries. The training data itself is not reduced.
        print("\n=== stage 1: v3-compatible overcomplete numerical warmup ===")
        _fit(model, dataset, steps=args.steps, lr=args.lr, lamb=0.0,
             entropy=args.rule_mask_entropy, log=args.log)
        print(f"  independent validation RMSE={_rmse(model, val_x, val_y):.6g}")
        _print_rules(model, case.variable_names, "structure before pruning")

        model = _kan_prune_refit(
            model, dataset, val_x, val_y,
            lr=args.lr, lamb=args.lamb,
            node_th=args.node_th, edge_th=args.edge_th,
            refit_steps=args.steps, max_cycles=args.prune_iters,
            rel_budget=args.prune_rel_budget,
            entropy=args.rule_mask_entropy, log=args.log, guard=args.guard_kan_pruning,
        )

        # Preserve the pre-freeze top-2 candidates for later restricted GSR,
        # then freeze RuleMask *before* the long final fit.  The loss spikes in
        # earlier versions came mostly from discrete argmax switches, not from
        # floating-point underflow of pairwise products.
        shortlist = _rulemask(model).retained_feature_candidates(topk=2)
        model = _discretize_without_damage(
            model, dataset, val_x, val_y,
            lr=args.lr, refit_steps=0, log=args.log,
        )

        print("\n=== final frozen-structure numerical convergence ===")
        model = _stable_polish(
            model, dataset, val_x, val_y, steps=args.final_steps,
            lr=args.final_lr, min_lr=args.final_min_lr, log=args.log,
            label="frozen-mask cosine polish",
        )

    _print_rules(model, case.variable_names, "hard product structure")

    structure_changes = 0
    if args.structure_gsr:
        model, structure_changes = _structural_gsr(
            model, dataset, val_x, val_y, shortlist,
            lr=args.structure_gsr_lr,
            trial_steps=args.structure_gsr_steps,
            max_rules=args.structure_gsr_rules,
            min_rel_improvement=args.structure_gsr_min_improvement,
            log=args.log,
        )
        _print_rules(model, case.variable_names, "structure after RuleMask-GSR")

    # Preserve the proven numerical checkpoint when GSR makes no structural edit.
    # If GSR does edit a rule, allow a guarded adaptation of the continuous KAN
    # functions, but roll it back unless validation improves.
    if structure_changes > 0:
        model = _stable_polish(
            model, dataset, val_x, val_y, steps=max(300, args.final_steps // 2),
            lr=args.final_lr, min_lr=args.final_min_lr, log=args.log,
            label="post-structure frozen polish",
        )

    numeric_test = _metrics(model, dataset)
    print("\nNUMERICAL TEST METRICS (untouched test split)")
    print(f"  RMSE:          {numeric_test['rmse']:.6g}")
    print(f"  MAE:           {numeric_test['mae']:.6g}")
    print(f"  max |e|:       {numeric_test['max_abs']:.6g}")
    print(f"  mean-baseline: {numeric_test['baseline']:.6g}")
    print(f"  independent validation RMSE: {_rmse(model, val_x, val_y):.6g}")

    if not args.symbolic:
        return

    # Keep the numerical model untouched. Symbolic GSR gets its own copy, so a
    # poor symbolic library can never degrade the numerical result reported above.
    lib = [
        "x", "x^2", "x^3", "x^4", "1/x", "sqrt", "exp", "log",
        "sin", "cos", "tanh", "arctan", "gaussian",
        "log1p_sq", "sqrt1p_sq", "inv1p_sq",
    ]
    symbolic_model = _compact_for_symbolic(
        model.copy(), dataset, val_x, val_y,
        lr=args.symbolic_lr, min_rules=args.symbolic_min_rules,
        refit_steps=args.symbolic_compact_steps,
        rel_budget=args.symbolic_structure_rel_budget, log=args.log,
    )
    _print_rules(symbolic_model, case.variable_names, "symbolic-copy structure after in-context compaction")
    symbolic_model = _symbolic_gsr(
        symbolic_model, dataset, val_x, val_y,
        lib=lib,
        local_topk=args.symbolic_local_topk,
        trial_steps=args.symbolic_gsr_steps,
        lr=args.symbolic_lr,
        log=args.log,
    )
    symbolic_model = _stable_polish(
        symbolic_model, dataset, val_x, val_y, steps=args.symbolic_polish_steps,
        lr=args.symbolic_lr, min_lr=max(args.symbolic_lr * 0.05, 1e-5),
        log=args.log, label="symbolic cosine polish",
    )

    symbolic_test = _metrics(symbolic_model, dataset)
    print("\nSYMBOLIC TEST METRICS")
    print(f"  RMSE:          {symbolic_test['rmse']:.6g}")
    print(f"  MAE:           {symbolic_test['mae']:.6g}")
    print(f"  numerical RMSE:{numeric_test['rmse']:.6g}")
    _print_formula(symbolic_model, dataset, case, simplify=args.formula_simplify)


if __name__ == "__main__":
    main()
