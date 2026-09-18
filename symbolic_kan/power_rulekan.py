"""Powered-expression extension of RuleKAN.

PowerRuleKAN composes already symbolic sum-product blocks with non-zero integer
powers.  The benchmark fitter uses discrete power search and transformed-target
warm starts; the final object is fully symbolic and contains no numerical spline
path.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .sum_product_kan import SumProductKAN
from .composition_rulekan import ComposedRuleKAN


def inverse_power_target(y: torch.Tensor, p: int, *, eps: float = 1e-8) -> Optional[torch.Tensor]:
    """Return a real-valued base target z satisfying z**p = y when possible."""
    pp = int(p)
    if pp == 0:
        return None
    yy = y
    if pp < 0:
        if bool((yy.abs() <= float(eps)).any()):
            return None
        yy = 1.0 / yy
        pp = -pp
    if pp == 1:
        return yy.detach().clone()
    if pp % 2 == 0:
        if bool((yy < 0).any()):
            return None
        return yy.clamp_min(0.0).pow(1.0 / float(pp)).detach()
    return (torch.sign(yy) * yy.abs().pow(1.0 / float(pp))).detach()


def safe_integer_power(z: torch.Tensor, p: int, *, eps: float = 1e-8) -> torch.Tensor:
    pp = int(p)
    if pp == 0:
        return torch.ones_like(z)
    if pp > 0:
        return z.pow(pp)
    sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    zsafe = sign * z.abs().clamp_min(float(eps))
    return zsafe.pow(pp)


def fit_affine_atom(h: torch.Tensor, y: torch.Tensor, *, ridge: float = 1e-10) -> Tuple[float, float]:
    """Least-squares fit y ~= a*h+b."""
    hh = h.reshape(h.shape[0], -1).mean(dim=1)
    yy = y.reshape(y.shape[0], -1).mean(dim=1)
    X = torch.stack([hh, torch.ones_like(hh)], dim=1)
    eye = torch.eye(2, device=X.device, dtype=X.dtype)
    beta = torch.linalg.solve(X.T @ X + float(ridge) * eye, X.T @ yy)
    return float(beta[0].detach().cpu()), float(beta[1].detach().cpu())


def harden_symbolic_base(base: nn.Module) -> nn.Module:
    """Force a base onto a fully symbolic hard architecture before polishing.

    Power objectives are especially prone to soft-gate drift because several
    operator mixtures can produce similar powered values on a bounded domain.
    Once a transformed-target or ratio pilot has found a useful base, the
    discrete structure/operator choices are frozen and only continuous symbolic
    parameters are allowed to move.
    """
    if isinstance(base, ComposedRuleKAN):
        # The composition topology and operator identities are already discrete.
        # Only continuous affine/scalar parameters remain trainable.  A possible
        # flat correction is an ordinary symbolic RuleKAN block and must be
        # hardened recursively before powered/ratio polishing.
        if base.correction is not None:
            harden_symbolic_base(base.correction)
        base.eval()
        return base
    if not isinstance(base, SumProductKAN):
        raise TypeError(f"unsupported PowerRuleKAN base type {type(base).__name__}")
    if not bool(base.discretized.item()):
        base.discretize(force_symbolic=True, freeze_gates=True)
    else:
        base.hard_spline_choice.zero_()
        base.variable_logits.requires_grad_(False)
        base.operator_logits.requires_grad_(False)
        base.spline_logits.requires_grad_(False)
        base.rule_gate.log_alpha.requires_grad_(False)
        base.factor_gate.log_alpha.requires_grad_(False)
    base.symbolic_enabled = True
    base.structure_hardening = 1.0
    base.factor_hardening = 1.0
    base.rule_hardening = 1.0
    base.symbolic_hardening = 1.0
    return base


def _hard_continuous_parameters(base: nn.Module):
    """Return continuous parameters of a fixed symbolic base topology."""
    harden_symbolic_base(base)
    for p in base.parameters():
        p.requires_grad_(False)
    if isinstance(base, SumProductKAN):
        params = []
        for p in (base.rule_scale, base.bias, base.symbolic_affine):
            p.requires_grad_(True)
            params.append(p)
        return params
    if isinstance(base, ComposedRuleKAN):
        params = []
        for p in base.composition.parameters():
            p.requires_grad_(True)
            params.append(p)
        if base.correction is not None:
            params.extend(_hard_continuous_parameters(base.correction))
        return params
    raise TypeError(f"unsupported PowerRuleKAN base type {type(base).__name__}")


def hard_power_polish(
    base: nn.Module,
    power: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    steps: int = 40,
    lr: float = 2e-3,
    reciprocal_epsilon: float = 1e-8,
    reciprocal_margin: float = 2e-2,
    reciprocal_barrier: float = 2e-2,
) -> Tuple[float, float, float]:
    """Polish a transformed-target solution only after hard discretisation.

    The base is optimized on the *original* powered target, with an outer affine
    readout fitted jointly.  Discrete support/operator choices cannot drift.
    Returns ``(validation_rmse, outer_scale, outer_bias)``.
    """
    harden_symbolic_base(base)
    params = _hard_continuous_parameters(base)
    with torch.no_grad():
        h0 = safe_integer_power(base(train_x), int(power), eps=reciprocal_epsilon)
    aa0, bb0 = fit_affine_atom(h0, train_y)
    aa = nn.Parameter(torch.tensor(float(aa0), device=train_x.device, dtype=train_x.dtype))
    bb = nn.Parameter(torch.tensor(float(bb0), device=train_x.device, dtype=train_x.dtype))
    opt = torch.optim.Adam(params + [aa, bb], lr=float(lr))
    best = None
    best_val = float('inf')
    for _ in range(max(0, int(steps))):
        base.train(); opt.zero_grad(set_to_none=True)
        z = base(train_x)
        h = safe_integer_power(z, int(power), eps=reciprocal_epsilon)
        pred = aa * h + bb
        loss = torch.mean((pred-train_y)**2)
        if int(power) < 0 and float(reciprocal_barrier) > 0:
            deficit = torch.relu(float(reciprocal_margin) - z.abs())
            loss = loss + float(reciprocal_barrier) * torch.mean(deficit.square())
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + [aa, bb], 1.0)
        opt.step()
        base.eval()
        with torch.no_grad():
            zv = base(val_x)
            pv = aa * safe_integer_power(zv, int(power), eps=reciprocal_epsilon) + bb
            vmse = float(torch.mean((pv-val_y)**2).cpu())
            margin = float(zv.abs().min().cpu()) if int(power) < 0 else float('inf')
        if math.isfinite(vmse) and (int(power) >= 0 or margin >= float(reciprocal_margin)) and vmse < best_val:
            best_val = vmse
            best = ({k:v.detach().clone() for k,v in base.state_dict().items()}, float(aa.detach().cpu()), float(bb.detach().cpu()))
    if best is not None:
        base.load_state_dict(best[0]); aa_val, bb_val = best[1], best[2]
    else:
        base.eval()
        with torch.no_grad():
            hv = safe_integer_power(base(val_x), int(power), eps=reciprocal_epsilon)
        aa_val, bb_val = fit_affine_atom(hv, val_y)
        with torch.no_grad():
            best_val = float(torch.mean((aa_val*hv + bb_val - val_y)**2).cpu())
    base.eval()
    return math.sqrt(max(best_val,0.0)), float(aa_val), float(bb_val)


def hard_ratio_polish(
    numerator: nn.Module,
    denominator: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    cross_steps: int = 35,
    quotient_steps: int = 35,
    lr: float = 2e-3,
    reciprocal_epsilon: float = 1e-8,
    reciprocal_margin: float = 2e-2,
    reciprocal_barrier: float = 2e-2,
) -> Tuple[float, float, float, float]:
    """Ratio-aware hard polish using ``A - y B`` before quotient loss.

    Stage 1 optimizes the stable cross-multiplied relationship ``A-yB≈0``.
    Stage 2 optimizes the actual quotient with a denominator-domain barrier.
    Both bases are hard-discretized before either stage.  Returns validation
    RMSE, outer scale, outer bias, and minimum denominator margin.
    """
    harden_symbolic_base(numerator); harden_symbolic_base(denominator)
    params = _hard_continuous_parameters(numerator) + _hard_continuous_parameters(denominator)
    opt = torch.optim.Adam(params, lr=float(lr))
    for _ in range(max(0,int(cross_steps))):
        numerator.train(); denominator.train(); opt.zero_grad(set_to_none=True)
        A = numerator(train_x); B = denominator(train_x)
        loss = torch.mean((A-train_y*B)**2)
        deficit = torch.relu(float(reciprocal_margin)-B.abs())
        loss = loss + float(reciprocal_barrier)*torch.mean(deficit.square())
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()

    # Fit the affine quotient readout before direct quotient polish.
    numerator.eval(); denominator.eval()
    with torch.no_grad():
        ratio0 = numerator(train_x) * safe_integer_power(denominator(train_x), -1, eps=reciprocal_epsilon)
    aa0, bb0 = fit_affine_atom(ratio0, train_y)
    aa = nn.Parameter(torch.tensor(float(aa0),device=train_x.device,dtype=train_x.dtype))
    bb = nn.Parameter(torch.tensor(float(bb0),device=train_x.device,dtype=train_x.dtype))
    opt = torch.optim.Adam(params+[aa,bb],lr=float(lr))
    best=None; best_val=float('inf'); best_margin=0.0
    for _ in range(max(0,int(quotient_steps))):
        numerator.train(); denominator.train(); opt.zero_grad(set_to_none=True)
        A=numerator(train_x); B=denominator(train_x)
        ratio=A*safe_integer_power(B,-1,eps=reciprocal_epsilon)
        pred=aa*ratio+bb
        deficit=torch.relu(float(reciprocal_margin)-B.abs())
        loss=torch.mean((pred-train_y)**2)+float(reciprocal_barrier)*torch.mean(deficit.square())
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(params+[aa,bb],1.0); opt.step()
        numerator.eval(); denominator.eval()
        with torch.no_grad():
            Bv=denominator(val_x); Av=numerator(val_x)
            pv=aa*Av*safe_integer_power(Bv,-1,eps=reciprocal_epsilon)+bb
            vmse=float(torch.mean((pv-val_y)**2).cpu()); margin=float(Bv.abs().min().cpu())
        if math.isfinite(vmse) and margin>=float(reciprocal_margin) and vmse<best_val:
            best_val=vmse; best_margin=margin
            best=({k:v.detach().clone() for k,v in numerator.state_dict().items()},
                  {k:v.detach().clone() for k,v in denominator.state_dict().items()},
                  float(aa.detach().cpu()),float(bb.detach().cpu()))
    if best is not None:
        numerator.load_state_dict(best[0]); denominator.load_state_dict(best[1]); aa_val,bb_val=best[2],best[3]
    else:
        numerator.eval(); denominator.eval()
        with torch.no_grad():
            Bv=denominator(val_x); Av=numerator(val_x)
            rv=Av*safe_integer_power(Bv,-1,eps=reciprocal_epsilon)
            best_margin=float(Bv.abs().min().cpu())
        aa_val,bb_val=fit_affine_atom(rv,val_y)
        with torch.no_grad(): best_val=float(torch.mean((aa_val*rv+bb_val-val_y)**2).cpu())
    numerator.eval(); denominator.eval()
    return math.sqrt(max(best_val,0.0)),float(aa_val),float(bb_val),float(best_margin)



class PowerRuleKAN(nn.Module):
    """A sum of products of powered symbolic RuleKAN blocks.

    Each term is represented by a list of ``(base_index, integer_power)`` pairs:

        f(x) = bias + sum_r scale_r * prod_t base[idx_rt](x)**p_rt.

    The class is deliberately generic even though the default benchmark search
    currently starts with one-block powered terms and an ordinary RuleKAN
    fallback.  This leaves room for multi-block ratio/product search without a
    separate rational architecture.
    """

    def __init__(
        self,
        bases: Sequence[nn.Module],
        terms: Sequence[Sequence[Tuple[int, int]]],
        *,
        scales: Optional[Sequence[float]] = None,
        bias: float = 0.0,
        reciprocal_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not bases:
            raise ValueError("PowerRuleKAN requires at least one base block")
        self.bases = nn.ModuleList(list(bases))
        self.terms = [tuple((int(i), int(p)) for i, p in term) for term in terms]
        if any(p == 0 for term in self.terms for _, p in term):
            raise ValueError("zero powers are represented by factor omission, not p=0")
        self.term_scale = nn.Parameter(
            torch.as_tensor(list(scales) if scales is not None else [1.0] * len(self.terms), dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(torch.tensor([float(bias)], dtype=torch.float32), requires_grad=False)
        self.reciprocal_epsilon = float(reciprocal_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_values = [b(x).reshape(x.shape[0], -1).mean(dim=1, keepdim=True) for b in self.bases]
        y = self.bias.to(x) * torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        for r, term in enumerate(self.terms):
            v = torch.ones_like(y)
            for idx, p in term:
                v = v * safe_integer_power(base_values[idx], p, eps=self.reciprocal_epsilon)
            y = y + self.term_scale[r].to(x) * v
        return y

    def chosen_powers(self) -> List[List[int]]:
        return [[int(p) for _, p in term] for term in self.terms]

    def reciprocal_domain_margin(self, x: torch.Tensor) -> float:
        vals: List[float] = []
        with torch.no_grad():
            for term in self.terms:
                for idx, p in term:
                    if int(p) < 0:
                        vals.append(float(self.bases[idx](x).abs().min().cpu()))
        return min(vals) if vals else float("inf")

    def symbolic_formula(
        self,
        variable_names: Optional[Sequence[str]] = None,
        *,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        digits: int = 8,
        simplify: bool = False,
    ):
        import sympy as sp
        expr = sp.Float(float(self.bias.detach().cpu()), digits)
        base_expr = [
            b.symbolic_formula(
                variable_names=variable_names,
                input_mean=input_mean,
                input_std=input_std,
                digits=digits,
                simplify=False,
            )
            for b in self.bases
        ]
        for r, term in enumerate(self.terms):
            v = sp.Float(float(self.term_scale[r].detach().cpu()), digits)
            for idx, p in term:
                v *= base_expr[idx] ** int(p)
            expr += v
        return sp.simplify(expr) if simplify else expr
