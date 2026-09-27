#!/usr/bin/env python3
"""End-to-end Sum-Product KAN demo on the damped nonlinear response.

Ground truth:

    1.3 exp(-0.9 t) sin(2.2 phase)
  + 0.75 log(1 + concentration^2) cos(1.6 angle)
  + 0.25 tanh(2 control)
  + 0.40 exp(-0.65 control) sin(2.35 control)

The final term is deliberately a product of two different symbolic functions of
the *same scalar input*.  The numerical spline phase is forbidden from using a
same-input product because one expressive spline h(control) is sufficient and
the factorization is non-identifiable there.  The fully-symbolic GMP/GSR phase
is allowed to discover exp(control)*sin(control), where the decomposition has a
meaning because factors are restricted to the symbolic library.

Unlike the RuleMask example, multiplication is part of the predictive layer
itself.  A diverse unary/pairwise initialization is used only to cover the local
search space; factor-presence and whole-rule gates remain differentiable, so
interaction order and effective functional rank are learned end-to-end.
"""

from __future__ import annotations

import argparse
import copy
import math

import torch
from tqdm.auto import tqdm

from rulekan import (
    SumProductKAN,
    SumProductRegularization,
    SumProductTrainingStage,
    create_dataset,
    default_sum_product_schedule,
    fit_sum_product_kan,
    prune_numeric_structure_to_stability,
    audit_sumproduct_symbolic_library,
    conformal_prune_sum_product_rules,
    conformal_refit_prune_sum_product_rules,
    in_context_symbolic_rule_gsr,
    hard_numeric_plateau_polish,
    hard_symbolic_plateau_polish,
    compact_sumproduct_for_symbolic,
    hard_numeric_precision_polish,
    mandatory_symbolic_matching_pursuit,
)


def target(x: torch.Tensor) -> torch.Tensor:
    t, phase, concentration, angle, control = [x[:, [i]] for i in range(5)]
    return (
        1.3 * torch.exp(-0.9 * t) * torch.sin(2.2 * phase)
        + 0.75 * torch.log1p(concentration.square()) * torch.cos(1.6 * angle)
        + 0.25 * torch.tanh(2.0 * control)
        + 0.40 * torch.exp(-0.65 * control) * torch.sin(2.35 * control)
    )


RANGES = [
    [0.0, 2.0],
    [-math.pi, math.pi],
    [-2.0, 2.0],
    [-math.pi, math.pi],
    [-1.5, 1.5],
]
NAMES = ["t", "phase", "concentration", "angle", "control"]


def external_sample(dataset, n: int, seed: int):
    g = torch.Generator(device=dataset["train_input"].device)
    g.manual_seed(int(seed))
    ranges = torch.tensor(
        RANGES,
        dtype=dataset["train_input"].dtype,
        device=dataset["train_input"].device,
    )
    raw = torch.rand(
        (int(n), 5),
        generator=g,
        dtype=dataset["train_input"].dtype,
        device=dataset["train_input"].device,
    )
    raw = raw * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
    y = target(raw)
    x = (raw - dataset["input_mean"].to(raw)) / dataset["input_std"].to(raw)
    return x, y


@torch.no_grad()
def rmse(model, x, y):
    model.eval()
    return float(torch.sqrt(torch.mean((model(x) - y) ** 2)).cpu())


def scaled_schedule(scale: float, symbolic: bool, base_lr: float):
    stages = default_sum_product_schedule(base_lr=base_lr, symbolic=symbolic)
    for stage in stages:
        stage.steps = max(1, int(round(stage.steps * float(scale))))
    return stages


def print_structure(model: SumProductKAN):
    print("\nHARD STRUCTURE")
    for rule in model.hard_structure(NAMES):
        if not rule["active"]:
            continue
        pieces = []
        for f in rule["factors"]:
            if f.get("identity"):
                continue
            pieces.append(f"{f['expert']}({f['variable']})")
        body = " * ".join(pieces) if pieces else "1"
        print(f"  rule {rule['rule']:2d}: {rule['scale']:+.5f} * {body}")


def assert_no_numeric_same_input_products(model: SumProductKAN) -> None:
    """The numerical spline phase must not use phi(x_j)*psi(x_j)."""
    violations=[]
    for r in torch.nonzero(model.hard_rule_choice,as_tuple=False).squeeze(-1).tolist():
        vars_=[]
        for slot in range(model.max_factors):
            j=int(model.hard_variable_choice[r,slot].item())
            if j!=model.in_dim:
                vars_.append(j)
        if len(vars_)!=len(set(vars_)):
            violations.append((int(r),tuple(vars_)))
    if violations:
        raise RuntimeError(
            "numerical same-input products are forbidden in this benchmark; "
            f"found {violations}"
        )



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rules", type=int, default=15,
                   help="15 covers every unary and pairwise structure once for 5 inputs")
    p.add_argument("--max-factors", type=int, default=2)
    p.add_argument("--allow-self-products", action=argparse.BooleanOptionalAction, default=True,
                   help="allow repeated variables only in the fully-symbolic search; numerical spline products always forbid them")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--numeric-basis", choices=["spline", "rbf"], default="spline",
                   help="numerical univariate basis: cubic B-spline RuleKAN or Gaussian-RBF RuleKAN-RBF")
    p.add_argument("--rbf-width-scale", type=float, default=1.0,
                   help="RuleKAN-RBF Gaussian width relative to uniform centre spacing")
    p.add_argument("--train-num", type=int, default=2200)
    p.add_argument("--test-num", type=int, default=700)
    p.add_argument("--val-num", type=int, default=600)
    p.add_argument("--calib-num", type=int, default=700)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--steps-scale", type=float, default=1.0,
                   help="multiply the default continuation-stage lengths")
    p.add_argument("--hard-consolidation-steps", type=int, default=1800,
                   help="explicit step budget for the hard rank/order consolidation stage; applied after --steps-scale")
    p.add_argument("--early-prune", action=argparse.BooleanOptionalAction, default=True,
                   help="rollback-safe numerical rule/factor pruning to a fixed point before hard consolidation")
    p.add_argument("--early-prune-after-consolidation", action=argparse.BooleanOptionalAction, default=True,
                   help="repeat prune-to-stability after hard consolidation")
    p.add_argument("--early-prune-refit-steps", type=int, default=80)
    p.add_argument("--early-prune-stabilize-rounds", type=int, default=2,
                   help="additional short fitting rounds after a plausible deletion")
    p.add_argument("--early-prune-stabilize-steps", type=int, default=60)
    p.add_argument("--early-prune-recovery-probes", type=int, default=2,
                   help="also refit this many best immediate destructive candidates before declaring a pruning fixed point")
    p.add_argument("--early-prune-max-candidates-per-pass", type=int, default=6,
                   help="maximum least-destructive deletion candidates receiving refits in one pruning pass")
    p.add_argument("--early-prune-stabilize-tol", type=float, default=1e-3,
                   help="stop post-prune stabilization when relative validation-MSE improvement falls below this")
    p.add_argument("--early-prune-local-rel-mse-budget", type=float, default=0.01)
    p.add_argument("--early-prune-global-rel-mse-budget", type=float, default=0.03)
    p.add_argument("--numeric-plateau-rounds", type=int, default=10)
    p.add_argument("--numeric-plateau-steps", type=int, default=600)
    p.add_argument("--numeric-plateau-tol", type=float, default=5e-4)
    p.add_argument("--numeric-target-rmse", type=float, default=5e-5,
                   help="precision target before symbolic takeover; lower is stricter")
    p.add_argument("--numeric-refine-grids", type=str, default="24,32",
                   help="comma-separated KAN grid resolutions used after the fixed-grid plateau")
    p.add_argument("--numeric-refine-steps", type=int, default=1200)
    p.add_argument("--numeric-lbfgs-steps", type=int, default=400)
    p.add_argument("--numeric-extra-rounds", type=int, default=4)
    p.add_argument("--symbolic", action=argparse.BooleanOptionalAction, default=True,
                   help="build a mandatory 100%% symbolic model after numerical convergence")
    p.add_argument("--force-symbolic", action=argparse.BooleanOptionalAction, default=True,
                   help="compatibility flag; symbolic output is always fully symbolic when --symbolic is enabled")
    p.add_argument("--symbolic-local-topk", type=int, default=7)
    p.add_argument("--symbolic-gmp", action=argparse.BooleanOptionalAction, default=True,
                   help="run spline-free GMP operator gating before expensive symbolic matching pursuit")
    p.add_argument("--symbolic-gmp-topk", type=int, default=3,
                   help="operators retained per ordinary pair factor after GMP")
    p.add_argument("--symbolic-gmp-unary-topk", type=int, default=5,
                   help="wider GMP shortlist for unary structures to preserve operator recall")
    p.add_argument("--symbolic-gmp-self-topk", type=int, default=4,
                   help="wider GMP shortlist for repeated-variable symbolic products")
    p.add_argument("--symbolic-gmp-steps", type=int, default=140)
    p.add_argument("--symbolic-gmp-lr", type=float, default=2e-2)
    p.add_argument("--symbolic-gmp-temperature-start", type=float, default=1.5)
    p.add_argument("--symbolic-gmp-temperature-end", type=float, default=0.35)
    p.add_argument("--symbolic-gmp-entropy", type=float, default=2e-4)
    p.add_argument("--symbolic-gmp-tuple-refine-steps", type=int, default=50,
                   help="cheap hard top-k tuple polish after soft GMP and before full GSR")
    p.add_argument("--symbolic-gmp-tuple-refine-lr", type=float, default=1.5e-2)
    p.add_argument("--symbolic-rule-max-candidates", type=int, default=32)
    p.add_argument("--symbolic-include-inactive-structures", action=argparse.BooleanOptionalAction, default=True,
                   help="let mandatory symbolic pursuit reuse structures from numerically pruned rules, including self-products")
    p.add_argument("--symbolic-beam-width", type=int, default=4)
    p.add_argument("--symbolic-gsr-steps", type=int, default=180)
    p.add_argument("--symbolic-lr", type=float, default=3e-4)
    p.add_argument("--symbolic-commit-refit-steps", type=int, default=300)
    p.add_argument("--symbolic-commit-lbfgs-steps", type=int, default=80)
    p.add_argument("--symbolic-backfit", action=argparse.BooleanOptionalAction, default=True,
                   help="reconsider overlapping committed operator choices after each new symbolic rule")
    p.add_argument("--symbolic-backfit-beam-width", type=int, default=3)
    p.add_argument("--symbolic-backfit-steps", type=int, default=120)
    p.add_argument("--symbolic-backfit-lbfgs-steps", type=int, default=40)
    p.add_argument("--symbolic-pursuit-mode", choices=("gsr","omp_linear","omp_nonlinear","omp_full"), default="gsr",
                   help="symbolic pursuit ablation: reference GSR or progressively stronger orthogonal refitting")
    p.add_argument("--symbolic-omp-extra-steps", type=int, default=120)
    p.add_argument("--symbolic-redundancy-cleanup", action=argparse.BooleanOptionalAction, default=True,
                   help="rollback-safe post-fit deletion of contribution-redundant symbolic rules")
    p.add_argument("--symbolic-redundancy-span-r2", type=float, default=0.995)
    p.add_argument("--symbolic-redundancy-corr", type=float, default=0.995)
    p.add_argument("--symbolic-min-rule-improvement-rel", type=float, default=5e-4,
                   help="minimum post-refit relative MSE gain required to keep another symbolic rule")
    p.add_argument("--symbolic-debug-topk", type=int, default=5,
                   help="print top-k GMP, GSR, and backfit rankings; <=0 disables")
    p.add_argument("--symbolic-log-style", choices=("compact", "verbose"), default="compact",
                   help="compact prints one-line top-k summaries; verbose also prints affine constants and multi-line rankings")
    p.add_argument("--symbolic-tiny-rule-scale", type=float, default=1e-4,
                   help="rules below this absolute coefficient are tested for final removal")
    p.add_argument("--symbolic-tiny-rule-contribution-rmse", type=float, default=1e-5,
                   help="rules below this validation contribution RMSE are tested for final removal")
    p.add_argument("--symbolic-tiny-rule-rel-mse-budget", type=float, default=2e-3,
                   help="maximum relative validation-MSE increase allowed when removing a negligible rule")
    p.add_argument("--symbolic-cleanup-max-seconds", type=float, default=120.0,
                   help="wall-clock budget for symbolic rescue/reduction cleanup; <=0 disables")
    p.add_argument("--symbolic-cleanup-max-trials", type=int, default=32,
                   help="maximum expensive symbolic cleanup trials; <=0 disables")
    p.add_argument("--symbolic-cleanup-min-rules", type=int, default=1,
                   help="minimum rule count only for final cleanup; forward pursuit still uses --symbolic-min-rules")
    p.add_argument("--symbolic-residual-operator-rescue", action=argparse.BooleanOptionalAction, default=True,
                   help="refresh operator candidates against leave-one-rule-out residuals after symbolic convergence")
    p.add_argument("--symbolic-residual-rescue-gmp-topk", type=int, default=4,
                   help="residual-conditioned GMP top-k per factor for product-rule operator rescue")
    p.add_argument("--symbolic-residual-rescue-gmp-steps", type=int, default=80)
    p.add_argument("--symbolic-residual-rescue-beam-width", type=int, default=3,
                   help="full-model trials per active rule after cheap residual-conditioned screening")
    p.add_argument("--symbolic-residual-rescue-steps", type=int, default=80)
    p.add_argument("--symbolic-residual-rescue-lbfgs-steps", type=int, default=30)
    p.add_argument("--symbolic-elimination-rel-mse-budget", type=float, default=1e-2,
                   help="maximum relative validation-MSE increase allowed when deleting a converged symbolic cleanup rule")
    p.add_argument("--formula-simplify", action=argparse.BooleanOptionalAction, default=False,
                   help="run bounded SymPy simplify on the final formula; disabled by default")
    p.add_argument("--formula-simplify-timeout", type=float, default=5.0,
                   help="maximum seconds spent simplifying the final formula")
    p.add_argument("--symbolic-min-rules", type=int, default=4,
                   help="benchmark prior: four ground-truth mechanisms; final cleanup may remove unnecessary rules")
    p.add_argument("--symbolic-max-rules", type=int, default=6)
    p.add_argument("--symbolic-target-rmse", type=float, default=1e-4)
    p.add_argument("--symbolic-final-steps", type=int, default=2400)
    p.add_argument("--symbolic-final-lbfgs-steps", type=int, default=500)
    p.add_argument("--conformal-prune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--conformal-alpha", type=float, default=0.10)
    p.add_argument("--conformal-rel-tol", type=float, default=0.05)
    p.add_argument("--conformal-refit-steps", type=int, default=160)
    p.add_argument("--conformal-validation-budget", type=float, default=0.15)
    p.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=True,
                   help="show tqdm progress bars for long optimization phases")
    p.add_argument("--log", type=int, default=100)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dataset = create_dataset(
        target,
        n_var=5,
        ranges=RANGES,
        train_num=args.train_num,
        test_num=args.test_num,
        normalize_input=True,
        seed=args.seed,
    )
    val_x, val_y = external_sample(dataset, args.val_num, args.seed + 100003)
    calib_x, calib_y = external_sample(dataset, args.calib_num, args.seed + 200003)

    model = SumProductKAN(
        in_dim=5,
        n_rules=args.rules,
        max_factors=args.max_factors,
        # Same-input numerical products are intentionally forbidden:
        # phi(x)*psi(x) is just another unconstrained univariate h(x).  Repeated
        # variables are opened later only in the symbolic GMP/GSR structure bank.
        allow_self_products=False,
        grid=args.grid,
        k=3,
        grid_range=(-2.1, 2.1),
        numeric_basis=args.numeric_basis,
        rbf_width_scale=args.rbf_width_scale,
        symbolic_library=(
            "x", "x^2", "exp", "sin", "cos", "tanh",
            "arctan", "log1p_sq", "sqrt1p_sq", "inv1p_sq",
        ),
        min_order=1,
        init_rule_open_prob=0.95,
        init_factor_open_prob=0.95,
        init_spline_prob=0.995,
        seed=args.seed,
        device=str(dataset["train_input"].device),
    )

    print("SUM-PRODUCT KAN")
    print(f"  variables: {NAMES}")
    print(f"  max rules: {args.rules}")
    print(f"  max factors/rule: {args.max_factors}")
    print(f"  numerical basis: {args.numeric_basis}{' (RuleKAN-RBF)' if args.numeric_basis == 'rbf' else ''}")
    print("  numerical same-input products: False")
    print(f"  symbolic same-input products: {args.allow_self_products}")
    print("  target functional rank: 4 (two cross-variable products + unary + one same-variable symbolic product)")
    print("  same-variable benchmark term: 0.40*exp(-0.65*control)*sin(2.35*control)")

    # Symbolic operators are not co-trained with the numerical KAN in this
    # benchmark.  First fit and freeze the numerical dependency structure; then
    # build a separate fully symbolic model by in-context search.
    stages = scaled_schedule(args.steps_scale, False, args.lr)
    for stage in stages:
        if stage.name == "hard rank/order consolidation":
            stage.steps = max(1, int(args.hard_consolidation_steps))

    pre_hard = [stage for stage in stages if stage.name != "hard rank/order consolidation"]
    hard_stages = [stage for stage in stages if stage.name == "hard rank/order consolidation"]
    fit_sum_product_kan(
        model, dataset["train_input"], dataset["train_label"], val_x, val_y,
        stages=pre_hard, log_every=args.log, restore_best_each_stage=True,
        show_progress=args.tqdm,
    )

    if args.early_prune:
        model, early_info = prune_numeric_structure_to_stability(
            model, dataset["train_input"], dataset["train_label"], val_x, val_y,
            refit_steps=args.early_prune_refit_steps,
            refit_lr=max(args.lr * 0.15, 1e-4),
            stabilize_rounds=args.early_prune_stabilize_rounds,
            stabilize_steps=args.early_prune_stabilize_steps,
            stabilize_min_relative_improvement=args.early_prune_stabilize_tol,
            recovery_probe_count=args.early_prune_recovery_probes,
            max_candidates_per_pass=args.early_prune_max_candidates_per_pass,
            local_rel_mse_budget=args.early_prune_local_rel_mse_budget,
            global_rel_mse_budget=args.early_prune_global_rel_mse_budget,
            min_rules=1, prune_optional_factors=True, verbose=True,
            show_progress=args.tqdm,
        )
        print(
            f"  early pruning accepted {len(early_info['accepted'])} deletions; "
            f"rules={early_info['active_rules']}, optional factors={early_info['active_optional_factors']}"
        )

    if hard_stages:
        fit_sum_product_kan(
            model, dataset["train_input"], dataset["train_label"], val_x, val_y,
            stages=hard_stages, log_every=args.log, restore_best_each_stage=True,
            show_progress=args.tqdm,
        )

    if args.early_prune and args.early_prune_after_consolidation:
        model, hard_info = prune_numeric_structure_to_stability(
            model, dataset["train_input"], dataset["train_label"], val_x, val_y,
            refit_steps=args.early_prune_refit_steps,
            refit_lr=max(args.lr * 0.12, 1e-4),
            stabilize_rounds=args.early_prune_stabilize_rounds,
            stabilize_steps=args.early_prune_stabilize_steps,
            stabilize_min_relative_improvement=args.early_prune_stabilize_tol,
            recovery_probe_count=args.early_prune_recovery_probes,
            max_candidates_per_pass=args.early_prune_max_candidates_per_pass,
            local_rel_mse_budget=args.early_prune_local_rel_mse_budget,
            global_rel_mse_budget=args.early_prune_global_rel_mse_budget,
            min_rules=1, prune_optional_factors=True, verbose=True,
            show_progress=args.tqdm,
        )
        print(
            f"  post-consolidation pruning accepted {len(hard_info['accepted'])} deletions; "
            f"rules={hard_info['active_rules']}, optional factors={hard_info['active_optional_factors']}"
        )

    # Keep learning the exact hard numerical structure until validation stops
    # improving. Structural gates are frozen, so this cannot escape by changing
    # into a new overcomplete series expansion.
    model, _ = hard_numeric_plateau_polish(
        model, dataset["train_input"], dataset["train_label"], val_x, val_y,
        rounds=args.numeric_plateau_rounds,
        steps_per_round=args.numeric_plateau_steps,
        lr=max(args.lr * 0.15, 2e-4),
        min_relative_improvement=args.numeric_plateau_tol,
        verbose=True,
        show_progress=args.tqdm,
    )

    soft_val = rmse(model, val_x, val_y)
    soft_test = rmse(model, dataset["test_input"], dataset["test_label"])
    print("\nSOFT MODEL")
    print(f"  validation RMSE: {soft_val:.8g}")
    print(f"  test RMSE:       {soft_test:.8g}")
    print(f"  diagnostics: {model.diagnostics()}")

    # Evaluate the exact hard endpoint before committing it.
    hard_probe = copy.deepcopy(model)
    hard_probe.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    hard_val = rmse(hard_probe, val_x, val_y)
    hard_test = rmse(hard_probe, dataset["test_input"], dataset["test_label"])
    print("\nHARD-ENDPOINT PROBE")
    print(f"  validation RMSE: {hard_val:.8g}")
    print(f"  test RMSE:       {hard_test:.8g}")
    print(f"  soft-hard gap:   {hard_val - soft_val:+.8g}")

    # Commit the categorical architecture.  If force_symbolic=False, edges that
    # still need splines are honestly retained as spline factors in the formula.
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    model.discretize(force_symbolic=False, freeze_gates=True)
    assert_no_numeric_same_input_products(model)
    print("  numerical same-input product violations: 0")

    # Continuous-only final polish after architecture is fixed.
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=max(args.lr * 0.15, 1e-4)
    )
    best = copy.deepcopy(model.state_dict())
    best_val = rmse(model, val_x, val_y)
    polish_steps = max(50, int(round(500 * args.steps_scale)))
    for it in tqdm(
        range(polish_steps), total=polish_steps, desc="Fixed hard numerical polish",
        disable=not args.tqdm, leave=True, dynamic_ncols=True, mininterval=0.25,
    ):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(dataset["train_input"])
        loss = torch.mean((pred - dataset["train_label"]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        if it % 20 == 0 or it == polish_steps - 1:
            v = rmse(model, val_x, val_y)
            if v < best_val:
                best_val = v
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)

    if args.conformal_prune:
        cp = conformal_refit_prune_sum_product_rules(
            model,
            dataset["train_input"], dataset["train_label"],
            val_x, val_y, calib_x, calib_y,
            alpha=args.conformal_alpha,
            relative_tolerance=args.conformal_rel_tol,
            validation_rel_budget=args.conformal_validation_budget,
            refit_steps=args.conformal_refit_steps,
            refit_lr=max(args.lr * 0.10, 1e-4),
            min_rules=1,
            show_progress=args.tqdm,
        )
        model = cp["model"]
        print("\nCONFORMAL IN-CONTEXT HARD-RULE PRUNING")
        print(f"  accepted removals: {cp['accepted']}")
        print(f"  rejected trials:   {cp['rejected']}")
        print(f"  active rules:      {cp['active_rules']}")
        if cp.get("certificates"):
            last = cp["certificates"][-1]
            print(
                f"  last certificate: upper={last['upper_excess']:.6g}, "
                f"tol={last['tolerance']:.6g}, accepted={last['accepted']}"
            )

    # Conformal pruning can slightly perturb continuous parameters during its
    # delete/refit trials. Give the accepted hard numerical topology one final
    # plateau fit before freezing the numerical benchmark.
    model, _ = hard_numeric_plateau_polish(
        model, dataset["train_input"], dataset["train_label"], val_x, val_y,
        rounds=max(1, args.numeric_plateau_rounds // 2),
        steps_per_round=args.numeric_plateau_steps,
        lr=max(args.lr * 0.10, 1e-4),
        min_relative_improvement=args.numeric_plateau_tol,
        verbose=True,
        show_progress=args.tqdm,
    )

    refine_grids = tuple(
        int(z.strip()) for z in str(args.numeric_refine_grids).split(",") if z.strip()
    )
    model, precision_history = hard_numeric_precision_polish(
        model, dataset["train_input"], dataset["train_label"], val_x, val_y,
        target_rmse=args.numeric_target_rmse,
        grid_schedule=refine_grids,
        adam_steps_per_grid=args.numeric_refine_steps,
        adam_lr=max(args.lr * 0.12, 1e-4),
        lbfgs_steps=args.numeric_lbfgs_steps,
        extra_final_rounds=args.numeric_extra_rounds,
        min_relative_improvement=max(args.numeric_plateau_tol * 0.5, 1e-5),
        verbose=True,
        show_progress=args.tqdm,
    )

    print_structure(model)
    final_val = rmse(model, val_x, val_y)
    final_test = rmse(model, dataset["test_input"], dataset["test_label"])
    print("\nFINAL NUMERICAL HARD MODEL")
    print(f"  validation RMSE: {final_val:.8g}")
    print(f"  test RMSE:       {final_test:.8g}")
    print(f"  active rules:    {int(model.hard_rule_choice.sum())}/{model.n_rules}")
    print(f"  precision target: {args.numeric_target_rmse:.3g}; reached={final_val <= args.numeric_target_rmse}")

    if not args.symbolic:
        formula = model.symbolic_formula(
            NAMES, input_mean=dataset["input_mean"], input_std=dataset["input_std"],
            digits=5, simplify=False,
        )
        print("\nNUMERICAL SUM-PRODUCT STRUCTURE")
        print(f"  y = {formula}")
        return

    if not args.force_symbolic:
        print("\nNOTE: --no-force-symbolic is retained for compatibility; symbolic output is always fully symbolic.")

    audit = audit_sumproduct_symbolic_library(model.symbolic_library)
    bad_symbols = {name: info for name, info in audit.items() if not bool(info.get("ok", False))}
    print(f"\n[SYMBOL AUDIT] {len(audit) - len(bad_symbols)}/{len(audit)} passed")
    if bad_symbols:
        for name, info in bad_symbols.items():
            print(f"  FAIL {name}: {info}")
        raise RuntimeError("symbolic function audit failed before GMP/GSR")

    symbolic_model, mp_history = mandatory_symbolic_matching_pursuit(
        model,
        dataset["train_input"], dataset["train_label"],
        val_x, val_y,
        library=model.symbolic_library,
        local_topk=args.symbolic_local_topk,
        max_rule_candidates=args.symbolic_rule_max_candidates,
        max_symbolic_rules=args.symbolic_max_rules,
        min_symbolic_rules=args.symbolic_min_rules,
        beam_width=args.symbolic_beam_width,
        trial_steps=args.symbolic_gsr_steps,
        trial_lr=args.symbolic_lr,
        final_steps=args.symbolic_final_steps,
        final_lr=max(args.symbolic_lr * 0.75, 1e-4),
        final_lbfgs_steps=args.symbolic_final_lbfgs_steps,
        target_val_rmse=args.symbolic_target_rmse,
        include_inactive_structures=args.symbolic_include_inactive_structures,
        allow_symbolic_self_products=args.allow_self_products,
        use_gmp_preselection=args.symbolic_gmp,
        gmp_topk=args.symbolic_gmp_topk,
        gmp_unary_topk=args.symbolic_gmp_unary_topk,
        gmp_self_product_topk=args.symbolic_gmp_self_topk,
        gmp_steps=args.symbolic_gmp_steps,
        gmp_lr=args.symbolic_gmp_lr,
        gmp_temperature_start=args.symbolic_gmp_temperature_start,
        gmp_temperature_end=args.symbolic_gmp_temperature_end,
        gmp_entropy_weight=args.symbolic_gmp_entropy,
        gmp_tuple_refine_steps=args.symbolic_gmp_tuple_refine_steps,
        gmp_tuple_refine_lr=args.symbolic_gmp_tuple_refine_lr,
        commit_refit_steps=args.symbolic_commit_refit_steps,
        commit_lbfgs_steps=args.symbolic_commit_lbfgs_steps,
        backfit=args.symbolic_backfit,
        backfit_beam_width=args.symbolic_backfit_beam_width,
        backfit_steps=args.symbolic_backfit_steps,
        backfit_lbfgs_steps=args.symbolic_backfit_lbfgs_steps,
        pursuit_mode=args.symbolic_pursuit_mode,
        omp_extra_steps=args.symbolic_omp_extra_steps,
        min_rule_improvement_rel=args.symbolic_min_rule_improvement_rel,
        residual_operator_rescue=args.symbolic_residual_operator_rescue,
        residual_rescue_gmp_topk=args.symbolic_residual_rescue_gmp_topk,
        residual_rescue_gmp_steps=args.symbolic_residual_rescue_gmp_steps,
        residual_rescue_beam_width=args.symbolic_residual_rescue_beam_width,
        residual_rescue_steps=args.symbolic_residual_rescue_steps,
        residual_rescue_lbfgs_steps=args.symbolic_residual_rescue_lbfgs_steps,
        elimination_rel_mse_budget=args.symbolic_elimination_rel_mse_budget,
        debug_topk=args.symbolic_debug_topk,
        debug_log_style=args.symbolic_log_style,
        tiny_rule_scale_threshold=args.symbolic_tiny_rule_scale,
        tiny_rule_contribution_rmse=args.symbolic_tiny_rule_contribution_rmse,
        tiny_rule_rel_mse_budget=args.symbolic_tiny_rule_rel_mse_budget,
        cleanup_max_seconds=args.symbolic_cleanup_max_seconds,
        cleanup_max_trials=args.symbolic_cleanup_max_trials,
        cleanup_min_rules=args.symbolic_cleanup_min_rules,
        redundancy_cleanup=args.symbolic_redundancy_cleanup,
        redundancy_span_r2_threshold=args.symbolic_redundancy_span_r2,
        redundancy_corr_threshold=args.symbolic_redundancy_corr,
        verbose=True,
        show_progress=args.tqdm,
    )

    symbolic_val = rmse(symbolic_model, val_x, val_y)
    symbolic_test = rmse(symbolic_model, dataset["test_input"], dataset["test_label"])

    # Absolute invariant: a successful symbolic run may not contain any spline.
    active_rules = torch.nonzero(symbolic_model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
    violations = []
    for r in active_rules:
        for slot in range(symbolic_model.max_factors):
            j = int(symbolic_model.hard_variable_choice[r, slot].item())
            if j != symbolic_model.in_dim and bool(symbolic_model.hard_spline_choice[r, slot, j].item()):
                violations.append((r, slot, j))
    if violations:
        raise RuntimeError(f"FINAL SYMBOLIC MODEL CONTAINS SPLINES: {violations}")

    print("\nFINAL FULLY SYMBOLIC MODEL")
    print(f"  validation RMSE: {symbolic_val:.8g}")
    print(f"  test RMSE:       {symbolic_test:.8g}")
    print(f"  active symbolic rules: {len(active_rules)}")
    print(f"  numerical test RMSE:   {final_test:.8g}")
    print(f"  symbolic minus numerical: {symbolic_test - final_test:+.8g}")
    print("  spline factors remaining: 0")

    print_structure(symbolic_model)
    formula = symbolic_model.symbolic_formula(
        NAMES,
        input_mean=dataset["input_mean"],
        input_std=dataset["input_std"],
        digits=7,
        simplify=args.formula_simplify,
        simplify_timeout=args.formula_simplify_timeout,
    )
    formula_text = str(formula)
    if "Spline_" in formula_text:
        raise RuntimeError("symbolic_formula emitted a spline placeholder in mandatory-symbolic mode")
    print("\nFINAL FULLY SYMBOLIC SUM-PRODUCT FORMULA (original input coordinates)")
    print(f"  y = {formula}")


if __name__ == "__main__":
    main()
