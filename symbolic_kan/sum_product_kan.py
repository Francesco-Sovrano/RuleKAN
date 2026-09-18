"""Sum-Product KAN for numerical dependency discovery and symbolic factorization.

The numerical model represents a scalar response as a sum of low-order product
rules whose factors are ordinary univariate KAN edges.  Rule survival, optional
factor presence, and variable choice are differentiable and can be hardened to
an exact discrete architecture.

The recommended symbolic workflow is deliberately separate from numerical
training.  After the numerical architecture is fixed and polished, a new model
is built from restricted symbolic factors of the form

    g(beta*x + gamma)

where ``g`` belongs to a finite operator library.  Output amplitude is carried
by the rule coefficient and factor output offsets are fixed to zero.  GMP-style gates are used to
prune operator candidates, and in-context matching pursuit makes the final
fully-symbolic rule selections by validation loss after refitting.

Repeated-variable products such as ``exp(x) * sin(x)`` are meaningful in the
symbolic stage because the factors belong to a restricted operator family.
They are normally excluded from the numerical spline stage because the product
of two flexible functions of the same scalar is itself another flexible
univariate function and its factorization is non-identifiable.
"""

from __future__ import annotations

import copy
import math
import itertools
import time
import signal
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .KANLayer import KANLayer
from .gated_kan import RadialBasisFunction
from .utils import SYMBOLIC_LIB



def _progress_range(n: int, *, desc: str, enabled: bool = False, leave: bool = False):
    n = max(0, int(n))
    return tqdm(
        range(n), total=n, desc=str(desc), disable=not bool(enabled),
        leave=bool(leave), dynamic_ncols=True, mininterval=0.25, smoothing=0.08,
    )

_DEFAULT_SYMBOLIC_LIBRARY = (
    "x",
    "x^2",
    "x^3",
    "exp",
    "sin",
    "cos",
    "tanh",
    "arctan",
    "log1p_sq",
    "sqrt1p_sq",
    "inv1p_sq",
)


def _canonical_symbolic_affine(values, operator_name: Optional[str] = None) -> Tuple[float, float, float, float]:
    """Return the canonical Sum-Product factor parameterization.

    General symbolic factors keep only the input chart ``g(beta*x+gamma)``;
    output amplitude is absorbed by the rule coefficient and output offsets are
    forbidden because they create lower-order leakage inside products.

    The affine/identity family is special: ``a*(b*x+c)+d`` is *exactly* another
    identity chart, ``x((a*b)*x + (a*c+d))``.  Folding ``a,d`` into ``beta,gamma``
    preserves the fitted gate shape instead of silently discarding it.
    """
    vals = list(values)
    a = float(vals[0]) if len(vals) > 0 else 1.0
    b = float(vals[1]) if len(vals) > 1 else 1.0
    c = float(vals[2]) if len(vals) > 2 else 0.0
    d = float(vals[3]) if len(vals) > 3 else 0.0
    if str(operator_name) == "x":
        return (1.0, a*b, a*c+d, 0.0)
    return (1.0, b, c, 0.0)




def audit_sumproduct_symbolic_library(
    library: Optional[Sequence[str]] = None,
    *,
    atol: float = 2e-10,
    rtol: float = 2e-9,
) -> Dict[str, Dict[str, object]]:
    """Numerically verify the real analytic atoms used by SumProduct symbolic search.

    The audit compares each torch implementation against its explicit real-valued
    mathematical definition on a safe test interval and checks finite outputs and
    gradients.  It intentionally covers the SumProduct default library rather
    than legacy singular operators whose runtime implementations use protective
    extensions outside their mathematical domains.
    """
    names = tuple(_DEFAULT_SYMBOLIC_LIBRARY if library is None else library)
    x = torch.linspace(-3.0, 3.0, 257, dtype=torch.float64, requires_grad=True)
    refs = {
        "x": lambda z: z,
        "x^2": lambda z: z**2,
        "x^3": lambda z: z**3,
        "exp": lambda z: torch.exp(z),
        "sin": torch.sin,
        "cos": torch.cos,
        "tanh": torch.tanh,
        "arctan": torch.atan,
        "log1p_sq": lambda z: torch.log1p(z**2),
        "sqrt1p_sq": lambda z: torch.sqrt(1.0 + z**2),
        "inv1p_sq": lambda z: 1.0 / (1.0 + z**2),
    }
    report: Dict[str, Dict[str, object]] = {}
    for name in names:
        if name not in refs:
            report[name] = {"ok": False, "reason": "no exact SumProduct audit reference"}
            continue
        if name not in SYMBOLIC_LIB:
            report[name] = {"ok": False, "reason": "missing from SYMBOLIC_LIB"}
            continue
        actual = SYMBOLIC_LIB[name][0](x)
        expected = refs[name](x)
        finite = bool(torch.isfinite(actual).all())
        max_abs = float(torch.max(torch.abs(actual - expected)).detach().cpu()) if finite else float("inf")
        scale = float(torch.max(torch.abs(expected)).detach().cpu())
        ok_value = finite and max_abs <= float(atol) + float(rtol) * max(1.0, scale)
        grad_ok = False
        if finite:
            grad = torch.autograd.grad(actual.sum(), x, retain_graph=True, allow_unused=False)[0]
            grad_ok = bool(torch.isfinite(grad).all())
        report[name] = {"ok": bool(ok_value and grad_ok), "max_abs_error": max_abs, "finite_gradient": grad_ok}

    # Explicitly stress the fixed exponential overflow guard.  This is a
    # numerical-stability property, not part of the equality check above.
    stress = SYMBOLIC_LIB["exp"][0](torch.tensor([100.0], dtype=torch.float32))
    report["exp"]["overflow_guard_finite"] = bool(torch.isfinite(stress).all())
    report["exp"]["ok"] = bool(report["exp"]["ok"] and report["exp"]["overflow_guard_finite"])
    return report


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=dim)


def _st_hard_categorical(
    logits: torch.Tensor,
    temperature: float,
    hardening: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softmax -> straight-through one-hot continuation.

    Forward values interpolate continuously between soft and hard categorical
    choices.  The hard endpoint uses a one-hot forward value but preserves the
    softmax gradient, avoiding the zero-gradient argmax problem.
    """
    t = max(float(temperature), 1e-4)
    z = logits / t
    if mask is not None:
        z = z.masked_fill(~mask.bool(), -1e9)
    soft = torch.softmax(z, dim=-1)
    idx = torch.argmax(soft, dim=-1)
    hard = F.one_hot(idx, num_classes=soft.shape[-1]).to(dtype=soft.dtype)
    hard_st = hard + soft - soft.detach()
    h = float(max(0.0, min(1.0, hardening)))
    return (1.0 - h) * soft + h * hard_st


def _st_hard_sigmoid(logits: torch.Tensor, temperature: float, hardening: float) -> torch.Tensor:
    t = max(float(temperature), 1e-4)
    soft = torch.sigmoid(logits / t)
    hard = (soft >= 0.5).to(soft.dtype)
    hard_st = hard + soft - soft.detach()
    h = float(max(0.0, min(1.0, hardening)))
    return (1.0 - h) * soft + h * hard_st


class HardConcreteRuleGate(nn.Module):
    """Deterministic/stochastic Hard-Concrete gate with differentiable L0 cost."""

    def __init__(
        self,
        n_rules: int,
        init_open_prob: float = 0.97,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        if not (gamma < 0.0 < 1.0 < zeta):
            raise ValueError("Hard-Concrete requires gamma < 0 < 1 < zeta")
        self.n_rules = int(n_rules)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.stochastic = bool(stochastic)

        shift = self.temperature * math.log(-self.gamma / self.zeta)
        init_alpha = _logit(init_open_prob) + shift
        self.log_alpha = nn.Parameter(torch.full((self.n_rules,), init_alpha))

    def open_probability(self) -> torch.Tensor:
        shift = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift)

    def soft_value(self) -> torch.Tensor:
        if self.stochastic and self.training:
            u = torch.rand_like(self.log_alpha).clamp_(1e-6, 1.0 - 1e-6)
            logistic = torch.log(u) - torch.log1p(-u)
            s = torch.sigmoid((logistic + self.log_alpha) / self.temperature)
        else:
            s = torch.sigmoid(self.log_alpha / self.temperature)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def forward(self, hardening: float = 0.0) -> torch.Tensor:
        soft = self.soft_value()
        hard = (soft >= 0.5).to(soft.dtype)
        hard_st = hard + soft - soft.detach()
        h = float(max(0.0, min(1.0, hardening)))
        return (1.0 - h) * soft + h * hard_st


@dataclass
class SumProductRegularization:
    rule_l0: float = 0.0
    factor_l0: float = 0.0
    variable_entropy: float = 0.0
    operator_entropy: float = 0.0
    spline_penalty: float = 0.0
    contribution_group_lasso: float = 0.0
    diversity: float = 0.0
    contribution_correlation: float = 0.0
    contribution_correlation_threshold: float = 0.95
    graph_redundancy: float = 0.0


@dataclass
class SumProductTrainingStage:
    name: str
    steps: int
    lr: float
    structure_hardening_start: float = 0.0
    structure_hardening_end: float = 0.0
    # Factor-presence and whole-rule gates are separated from categorical
    # variable choice.  This lets order/rank change continuously while variable
    # selection remains an exact hard-forward KAN structure.
    factor_hardening_start: Optional[float] = None
    factor_hardening_end: Optional[float] = None
    rule_hardening_start: Optional[float] = None
    rule_hardening_end: Optional[float] = None
    symbolic_hardening_start: float = 0.0
    symbolic_hardening_end: float = 0.0
    variable_temperature_start: float = 1.0
    variable_temperature_end: float = 1.0
    operator_temperature_start: float = 1.0
    operator_temperature_end: float = 1.0
    rule_temperature_start: float = 2.0 / 3.0
    rule_temperature_end: float = 2.0 / 3.0
    regularization: SumProductRegularization = None
    grad_clip: float = 0.5
    variable_lr_scale: float = 1.0
    rank_lr_scale: float = 1.0
    symbolic_lr_scale: float = 1.0
    train_structure: bool = True
    train_variable_gates: Optional[bool] = None
    train_rank_gates: Optional[bool] = None
    train_symbolic: bool = True
    symbolic_enabled: bool = True
    # Optional augmented-Lagrangian constraint that keeps each selected
    # numerical univariate factor close to one member of the analytic symbolic
    # library while the SumProduct structure is still compressible.  The
    # symbolic branch is auxiliary here: symbolic_enabled may remain False.
    symbolic_manifold_dual_init: float = 0.0
    symbolic_manifold_rho: float = 0.0
    symbolic_manifold_tolerance: float = 0.02
    symbolic_manifold_dual_update_every: int = 50
    symbolic_manifold_dual_max: float = 0.05
    symbolic_manifold_max_samples: int = 256
    checkpoint_hard_endpoint: bool = False

    def __post_init__(self):
        if self.regularization is None:
            self.regularization = SumProductRegularization()



class _ScaledAsinhBackward(torch.autograd.Function):
    """Identity forward with a bounded, scale-aware backward Jacobian.

    For an activation ``z`` and positive scale ``s`` the forward value is exactly
    ``z``.  The backward multiplier is ``1/sqrt(1 + (z/s)^2)``, i.e. the
    derivative of ``s*asinh(z/s)``.  Large exponential/product activations thus
    cannot amplify upstream parameter gradients in proportion to their raw
    magnitude, while ordinary small activations retain an almost-unit gradient.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        s = float(scale)
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError("scaled-asinh backward scale must be finite and positive")
        ctx.scale = s
        ctx.save_for_backward(value)
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        s = ctx.scale
        multiplier = torch.rsqrt(1.0 + (value / s).square())
        return grad_output * multiplier, None


def _identity_forward_stable_backward(value: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply :class:`_ScaledAsinhBackward` when ``scale`` is positive."""
    if float(scale) <= 0.0:
        return value
    return _ScaledAsinhBackward.apply(value, float(scale))


class _EqualizedProductBackward(torch.autograd.Function):
    """Exact product forward with log-compressed leave-one-out Jacobians.

    True product derivatives contain ``prod_{i!=j} f_i``. Their magnitude can
    vanish exponentially when several factors are below one or explode when
    factors are large. In backward only, raise that magnitude to ``power``
    (0 < power <= 1) relative to unit scale and cap the multiplicative change.
    ``power=1`` is the exact ordinary product gradient.
    """

    @staticmethod
    def forward(ctx, factors: torch.Tensor, power: float, max_gain: float):
        p = float(power)
        g = float(max_gain)
        if not (0.0 < p <= 1.0):
            raise ValueError("product gradient power must satisfy 0 < power <= 1")
        if not math.isfinite(g) or g < 1.0:
            raise ValueError("product gradient max_gain must be finite and >= 1")
        ctx.power = p
        ctx.max_gain = g
        ctx.save_for_backward(factors)
        return torch.prod(factors, dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (factors,) = ctx.saved_tensors
        k = int(factors.shape[-1])
        grads = []
        eps = torch.finfo(factors.dtype).tiny
        for j in range(k):
            if k == 1:
                other = torch.ones_like(factors[..., j])
            else:
                other = torch.prod(
                    torch.cat((factors[..., :j], factors[..., j + 1 :]), dim=-1),
                    dim=-1,
                )
            if ctx.power < 1.0:
                mag = other.abs()
                # gain = |J|^(power-1): >1 for weak Jacobians and <1 for
                # oversized Jacobians. Clamp symmetrically in multiplicative space.
                gain = mag.clamp_min(eps).pow(ctx.power - 1.0)
                gain = gain.clamp(min=1.0 / ctx.max_gain, max=ctx.max_gain)
                # Preserve exact zero derivative when any required factor is exactly zero.
                gain = torch.where(mag > 0, gain, torch.ones_like(gain))
                other = other * gain
            grads.append(grad_output * other)
        return torch.stack(grads, dim=-1), None, None


def _equalized_product(factors: torch.Tensor, power: float, max_gain: float) -> torch.Tensor:
    if float(power) >= 1.0:
        return torch.prod(factors, dim=-1)
    return _EqualizedProductBackward.apply(factors, float(power), float(max_gain))


def _safe_sympy_simplify(expr, timeout: float = 5.0):
    """Best-effort SymPy simplify with a POSIX wall-clock bound.

    Formula export must never become an unbounded post-training phase.  On
    platforms without SIGALRM, return the unsimplified expression instead of
    risking an indefinite simplify call.
    """
    import sympy as sp
    timeout = float(timeout)
    if timeout <= 0:
        return expr
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return expr
    old_handler = signal.getsignal(signal.SIGALRM)
    def _handler(signum, frame):
        raise TimeoutError("SymPy simplify timed out")
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return sp.simplify(expr)
        except TimeoutError:
            return expr
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


class FastRBFEdgeBank(nn.Module):
    """FastKAN-style Gaussian-RBF univariate edge bank.

    The module mirrors the subset of :class:`KANLayer`'s API used by
    :class:`SumProductKAN`: every input-output pair is an independent
    univariate function, but the nonlinear component is expanded in Gaussian
    radial basis functions instead of cubic B-splines.  Rule/factor gates,
    variable selection, pruning, symbolic takeover, and the readout are
    unchanged.

    The forward signature is KAN-compatible and returns
    ``(output, preacts, postacts, postbasis)``.  ``postacts`` has shape
    ``[batch, out_dim, in_dim]`` and is what SumProductKAN consumes.
    """

    basis_name = "rbf"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num: int = 16,
        *,
        grid_range: Sequence[float] = (-2.1, 2.1),
        base_fun: Optional[nn.Module] = None,
        noise_scale: float = 0.05,
        scale_base_mu: float = 0.55,
        scale_base_sigma: float = 0.08,
        scale_sp: float = 0.20,
        train_grid: bool = True,
        train_width: bool = True,
        width_scale: float = 1.0,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if int(num) < 2:
            raise ValueError("RBF edge bank requires num >= 2")
        lo, hi = float(grid_range[0]), float(grid_range[1])
        if not hi > lo:
            raise ValueError("grid_range must satisfy high > low")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num = int(num)
        # k is retained only for lightweight KAN API compatibility; RBFs have
        # no polynomial degree.
        self.k = 0
        self.grid_eps = 0.0
        self.grid_range = (lo, hi)
        self.base_fun = base_fun if base_fun is not None else nn.Identity()
        spacing = (hi - lo) / max(1, self.num - 1)
        self.basis = RadialBasisFunction(
            grid_min=lo,
            grid_max=hi,
            num_grids=self.num,
            denominator=max(1e-6, float(width_scale) * spacing),
            train_grid=bool(train_grid),
            train_denominator=bool(train_width),
        )
        # [input, output, basis].  The small nonlinear initialization is paired
        # with an identity-like base path, matching the spline RuleKAN startup.
        coef_std = float(noise_scale) / math.sqrt(max(1, self.num))
        self.coef = nn.Parameter(torch.empty(self.in_dim, self.out_dim, self.num))
        nn.init.normal_(self.coef, mean=0.0, std=coef_std)
        self.mask = nn.Parameter(torch.ones(self.in_dim, self.out_dim), requires_grad=False)
        base0 = (
            float(scale_base_mu) / math.sqrt(self.in_dim)
            + float(scale_base_sigma)
            * (torch.rand(self.in_dim, self.out_dim) * 2.0 - 1.0)
            / math.sqrt(self.in_dim)
        )
        self.scale_base = nn.Parameter(base0, requires_grad=bool(sb_trainable))
        self.scale_sp = nn.Parameter(
            torch.ones(self.in_dim, self.out_dim)
            * float(scale_sp)
            / math.sqrt(self.in_dim),
            requires_grad=bool(sp_trainable),
        )
        self.to(device)

    @property
    def grid(self) -> torch.Tensor:
        """KAN-like grid view, one copy of the shared RBF centres per input."""
        return self.basis.grid[None, :].expand(self.in_dim, -1)

    @property
    def denominator(self) -> torch.Tensor:
        return self.basis.denominator

    def raw_basis_component(self, x: torch.Tensor) -> torch.Tensor:
        """Return the coefficient-weighted RBF component ``[B,I,O]``."""
        phi = self.basis(x)  # [B,I,G]
        return torch.einsum("big,iog->bio", phi, self.coef)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        preacts = x[:, None, :].expand(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x)
        rbf = self.raw_basis_component(x)  # [B,I,O]
        edge = (
            self.scale_base[None, :, :] * base[:, :, None]
            + self.scale_sp[None, :, :] * rbf
        )
        edge = self.mask[None, :, :] * edge
        postacts = edge.permute(0, 2, 1)
        postbasis = rbf.permute(0, 2, 1)
        y = edge.sum(dim=1)
        return y, preacts, postacts, postbasis

    @torch.no_grad()
    def update_grid_from_samples(self, x: torch.Tensor):
        # Keep a stable global RBF range during ordinary training.  Dedicated
        # refinement uses least-squares transfer in refine_sumproduct_numeric_grid.
        return

    @torch.no_grad()
    def initialize_grid_from_parent(self, parent_layer, parent_acts: torch.Tensor):
        return


class SumProductKAN(nn.Module):
    """Low-rank sum-product KAN with differentiable symbolic migration."""

    def __init__(
        self,
        in_dim: int,
        n_rules: int = 10,
        max_factors: int = 2,
        grid: int = 16,
        k: int = 3,
        grid_range: Sequence[float] = (-2.1, 2.1),
        numeric_basis: str = "spline",
        rbf_train_grid: bool = True,
        rbf_train_width: bool = True,
        rbf_width_scale: float = 1.0,
        symbolic_library: Sequence[str] = _DEFAULT_SYMBOLIC_LIBRARY,
        min_order: int = 1,
        init_rule_open_prob: float = 0.95,
        init_factor_open_prob: float = 0.95,
        init_spline_prob: float = 0.995,
        variable_temperature: float = 1.2,
        operator_temperature: float = 1.2,
        rule_temperature: float = 2.0 / 3.0,
        factor_temperature: float = 2.0 / 3.0,
        symbolic_gradient_scale: float = 4.0,
        symbolic_product_gradient_scale: float = 0.0,
        numeric_factor_gradient_scale: float = 0.0,
        numeric_product_gradient_scale: float = 0.0,
        numeric_product_gradient_power: float = 1.0,
        numeric_product_gradient_max_gain: float = 8.0,
        stochastic_rule_gates: bool = False,
        diverse_structure_init: bool = True,
        allow_self_products: bool = True,
        structure_init_margin: float = 0.75,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if in_dim < 1 or n_rules < 1 or max_factors < 1:
            raise ValueError("in_dim, n_rules and max_factors must be positive")
        if not (1 <= min_order <= max_factors):
            raise ValueError("min_order must satisfy 1 <= min_order <= max_factors")
        for name in symbolic_library:
            if name not in SYMBOLIC_LIB:
                raise KeyError(f"unknown symbolic operator: {name}")

        torch.manual_seed(int(seed))
        self.in_dim = int(in_dim)
        self.n_rules = int(n_rules)
        self.max_factors = int(max_factors)
        self.min_order = int(min_order)
        self.symbolic_library = tuple(symbolic_library)
        self.n_ops = len(self.symbolic_library)
        self.symbolic_gradient_scale = float(symbolic_gradient_scale)
        self.symbolic_product_gradient_scale = float(symbolic_product_gradient_scale)
        # Numerical SumProduct stabilization is exact in the forward pass and
        # changes only the backward derivative. Large spline/RBF factors and
        # products otherwise create scale-dependent gradients: d(prod f_i)/df_j
        # contains every other factor. This is especially harmful when one rule
        # dominates the global gradient-clipping budget. A scaled-asinh surrogate
        # bounds that sensitivity without changing the represented function.
        self.numeric_factor_gradient_scale = float(numeric_factor_gradient_scale)
        self.numeric_product_gradient_scale = float(numeric_product_gradient_scale)
        self.numeric_product_gradient_power = float(numeric_product_gradient_power)
        self.numeric_product_gradient_max_gain = float(numeric_product_gradient_max_gain)
        if not (0.0 < self.numeric_product_gradient_power <= 1.0):
            raise ValueError("numeric_product_gradient_power must satisfy 0 < power <= 1")
        if self.numeric_product_gradient_max_gain < 1.0 or not math.isfinite(self.numeric_product_gradient_max_gain):
            raise ValueError("numeric_product_gradient_max_gain must be finite and >= 1")
        self.allow_self_products = bool(allow_self_products)
        numeric_basis = str(numeric_basis).strip().lower()
        if numeric_basis in {"fast", "fastkan", "gaussian", "radial", "radial_bf"}:
            numeric_basis = "rbf"
        if numeric_basis not in {"spline", "rbf"}:
            raise ValueError("numeric_basis must be 'spline' or 'rbf'")
        self.numeric_basis = numeric_basis
        self.rbf_train_grid = bool(rbf_train_grid)
        self.rbf_train_width = bool(rbf_train_width)
        self.rbf_width_scale = float(rbf_width_scale)

        # Persistent physical-pruning masks. Accepted deletions are exact and
        # cannot be resurrected by later gate optimization. Optional factors may
        # collapse product order; mandatory prefix factors are removed only by
        # deleting their whole rule.
        self.register_buffer("rule_alive_mask", torch.ones(self.n_rules, dtype=torch.bool))
        self.register_buffer(
            "factor_alive_mask",
            torch.ones(self.n_rules, self.max_factors, dtype=torch.bool),
        )

        # One independent univariate numerical expert for every
        # (rule, slot, input) tuple. RuleKAN uses cubic B-splines by default;
        # RuleKAN-RBF swaps only this expert bank for Gaussian RBFs.
        edge_kwargs = dict(
            in_dim=self.in_dim,
            out_dim=self.n_rules * self.max_factors,
            num=int(grid),
            grid_range=list(grid_range),
            base_fun=nn.Identity(),
            noise_scale=0.05,
            scale_base_mu=0.55,
            scale_base_sigma=0.08,
            scale_sp=0.20,
            device=device,
        )
        if self.numeric_basis == "spline":
            self.numeric_edges = KANLayer(k=int(k), **edge_kwargs)
        else:
            self.numeric_edges = FastRBFEdgeBank(
                train_grid=self.rbf_train_grid,
                train_width=self.rbf_train_width,
                width_scale=self.rbf_width_scale,
                **edge_kwargs,
            )

        # Variable categorical gates choose *features only*.  Multiplicative
        # identity is controlled by a separate factor-presence gate below.  This
        # avoids the discontinuous "identity wins the argmax" failure mode and
        # gives a smooth differentiable path pairwise -> unary.
        self.variable_logits = nn.Parameter(
            0.04 * torch.randn(self.n_rules, self.max_factors, self.in_dim + 1)
        )
        allowed = torch.ones(self.n_rules, self.max_factors, self.in_dim + 1, dtype=torch.bool)
        allowed[..., self.in_dim] = False
        self.register_buffer("variable_allowed", allowed)

        if diverse_structure_init:
            # Cover the combinatorial search space without fixing it. For two
            # factors we seed different rules with different unordered pairs,
            # then a few unary/identity rules. The logits stay trainable and can
            # swap variables immediately through the ST gradient.
            with torch.no_grad():
                margin = float(structure_init_margin)
                templates = []
                if self.max_factors == 1:
                    templates = [(j,) for j in range(self.in_dim)]
                elif self.max_factors == 2:
                    # Cover pair structures directly.  With replacement means
                    # rules such as phi(x_j)*psi(x_j) are first-class citizens,
                    # not an accidental duplicate-variable state that training
                    # must discover from a distinct-variable initialization.
                    if self.allow_self_products:
                        templates.extend(list(itertools.combinations_with_replacement(range(self.in_dim), 2)))
                    else:
                        templates.extend(list(itertools.combinations(range(self.in_dim), 2)))
                        # When self products are disabled, use the spare capacity
                        # for explicit unary seeds.  With self products enabled,
                        # unary structure is obtained smoothly by m_{r,1}->0.
                        templates.extend((j, self.in_dim) for j in range(self.in_dim))
                else:
                    # General fallback: deterministic cyclic tuples plus unary
                    # identity-padded templates.
                    for r in range(max(self.n_rules, self.in_dim)):
                        templates.append(tuple((r + s) % self.in_dim for s in range(self.max_factors)))
                    templates.extend(tuple([j] + [self.in_dim] * (self.max_factors - 1)) for j in range(self.in_dim))
                # Generate the template permutation on CPU. This avoids backend-
                # specific generator limitations (notably MPS) without changing
                # the resulting discrete initialization.
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed) + 7919)
                perm = torch.randperm(len(templates), generator=g, device="cpu").tolist()
                templates = [templates[i] for i in perm]
                self._initial_factor_active = torch.ones(
                    self.n_rules, self.max_factors, dtype=torch.bool,
                    device=self.variable_logits.device,
                )
                for r in range(self.n_rules):
                    tpl = templates[r % len(templates)]
                    for s_idx in range(self.max_factors):
                        choice = tpl[s_idx] if s_idx < len(tpl) else self.in_dim
                        active = choice != self.in_dim or s_idx < self.min_order
                        if not active:
                            # Identity is represented by the factor gate. Seed a
                            # harmless feature here; it is multiplied by m≈0.
                            choice = r % self.in_dim
                        self._initial_factor_active[r, s_idx] = active
                        # Absolute winner margin, rather than +=, so every
                        # template is guaranteed to be the hard-forward start.
                        row = self.variable_logits[r, s_idx, :self.in_dim]
                        self.variable_logits[r, s_idx, choice] = row.max() + margin

        if not hasattr(self, "_initial_factor_active"):
            self._initial_factor_active = torch.ones(
                self.n_rules, self.max_factors, dtype=torch.bool,
                device=self.variable_logits.device,
            )

        # Separate optional factor gate m_rs. Mandatory prefix slots are forced
        # to one; later slots use Hard-Concrete so interaction order is a smooth
        # differentiable quantity rather than part of a categorical argmax.
        self.factor_gate = HardConcreteRuleGate(
            self.n_rules * self.max_factors,
            init_open_prob=init_factor_open_prob,
            temperature=factor_temperature,
            stochastic=False,
        )
        with torch.no_grad():
            beta = self.factor_gate.temperature
            shift = beta * math.log(-self.factor_gate.gamma / self.factor_gate.zeta)
            open_alpha = _logit(init_factor_open_prob) + shift
            closed_alpha = _logit(1.0 - init_factor_open_prob) + shift
            active = self._initial_factor_active.reshape(-1)
            self.factor_gate.log_alpha[active] = open_alpha
            self.factor_gate.log_alpha[~active] = closed_alpha

        # Symbolic operator gate for each possible univariate edge.
        self.operator_logits = nn.Parameter(
            0.02 * torch.randn(
                self.n_rules, self.max_factors, self.in_dim, self.n_ops
            )
        )
        if "x" in self.symbolic_library:
            with torch.no_grad():
                self.operator_logits[..., self.symbolic_library.index("x")] += 0.25

        # a * g(b*x + c) + d for each symbolic candidate.
        affine = torch.zeros(
            self.n_rules, self.max_factors, self.in_dim, self.n_ops, 4
        )
        affine[..., 0] = 1.0  # amplitude a
        affine[..., 1] = 1.0  # frequency/scale b
        affine[..., 2] = 0.0  # shift c
        affine[..., 3] = 0.0  # bias d
        affine[..., 0] += 0.02 * torch.randn_like(affine[..., 0])
        affine[..., 1] += 0.02 * torch.randn_like(affine[..., 1])
        self.symbolic_affine = nn.Parameter(affine)

        # rho=1 -> numerical spline; rho=0 -> symbolic expert mixture.
        self.spline_logits = nn.Parameter(
            torch.full(
                (self.n_rules, self.max_factors, self.in_dim),
                _logit(init_spline_prob),
            )
        )

        self.rule_gate = HardConcreteRuleGate(
            self.n_rules,
            init_open_prob=init_rule_open_prob,
            temperature=rule_temperature,
            stochastic=stochastic_rule_gates,
        )
        self.rule_scale = nn.Parameter(torch.randn(self.n_rules) * 0.30)
        self.bias = nn.Parameter(torch.zeros(1))

        self.variable_temperature = float(variable_temperature)
        self.operator_temperature = float(operator_temperature)
        self.structure_hardening = 0.0  # categorical variable choice
        self.factor_hardening = 0.0
        self.rule_hardening = 0.0
        self.symbolic_hardening = 0.0
        self.symbolic_enabled = True

        # Optional exact hard masks produced by discretize().
        self.register_buffer("discretized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "hard_variable_choice",
            torch.zeros(self.n_rules, self.max_factors, dtype=torch.long),
        )
        self.register_buffer(
            "hard_operator_choice",
            torch.zeros(self.n_rules, self.max_factors, self.in_dim, dtype=torch.long),
        )
        self.register_buffer(
            "hard_spline_choice",
            torch.ones(self.n_rules, self.max_factors, self.in_dim, dtype=torch.bool),
        )
        self.register_buffer(
            "hard_rule_choice",
            torch.ones(self.n_rules, dtype=torch.bool),
        )

        self.to(device)

    # ------------------------------------------------------------------
    # gate distributions
    # ------------------------------------------------------------------
    def variable_probabilities(self, hardening: Optional[float] = None) -> torch.Tensor:
        h = self.structure_hardening if hardening is None else float(hardening)
        if bool(self.discretized.item()):
            return F.one_hot(
                self.hard_variable_choice,
                num_classes=self.in_dim + 1,
            ).to(self.variable_logits.dtype)
        return _st_hard_categorical(
            self.variable_logits,
            self.variable_temperature,
            h,
            mask=self.variable_allowed,
        )

    def operator_probabilities(self, hardening: Optional[float] = None) -> torch.Tensor:
        h = self.symbolic_hardening if hardening is None else float(hardening)
        if bool(self.discretized.item()):
            return F.one_hot(
                self.hard_operator_choice,
                num_classes=self.n_ops,
            ).to(self.operator_logits.dtype)
        return _st_hard_categorical(
            self.operator_logits,
            self.operator_temperature,
            h,
        )

    def spline_probabilities(self, hardening: Optional[float] = None) -> torch.Tensor:
        h = self.symbolic_hardening if hardening is None else float(hardening)
        if bool(self.discretized.item()):
            return self.hard_spline_choice.to(self.spline_logits.dtype)
        return _st_hard_sigmoid(self.spline_logits, 1.0, h)

    def factor_values(self, hardening: Optional[float] = None) -> torch.Tensor:
        h = self.factor_hardening if hardening is None else float(hardening)
        alive = self.factor_alive_mask.to(self.rule_scale.dtype)
        if bool(self.discretized.item()):
            return (self.hard_variable_choice != self.in_dim).to(self.rule_scale.dtype) * alive
        m = self.factor_gate(hardening=h).reshape(self.n_rules, self.max_factors)
        if self.min_order > 0:
            mandatory = torch.zeros_like(m, dtype=torch.bool)
            mandatory[:, : self.min_order] = True
            m = torch.where(mandatory, torch.ones_like(m), m)
        return m * alive

    def factor_open_probabilities(self) -> torch.Tensor:
        p = self.factor_gate.open_probability().reshape(self.n_rules, self.max_factors)
        if self.min_order > 0:
            mandatory = torch.zeros_like(p, dtype=torch.bool)
            mandatory[:, : self.min_order] = True
            p = torch.where(mandatory, torch.ones_like(p), p)
        return p * self.factor_alive_mask.to(p.dtype)

    def rule_values(self, hardening: Optional[float] = None) -> torch.Tensor:
        h = self.rule_hardening if hardening is None else float(hardening)
        alive = self.rule_alive_mask.to(self.rule_scale.dtype)
        if bool(self.discretized.item()):
            return self.hard_rule_choice.to(self.rule_scale.dtype) * alive
        return self.rule_gate(hardening=h) * alive

    # ------------------------------------------------------------------
    # experts and forward
    # ------------------------------------------------------------------
    def _numeric_edge_values(self, x: torch.Tensor) -> torch.Tensor:
        _, _, postacts, _ = self.numeric_edges(x)
        # [B, R*S, D] -> [B, R, S, D]
        return postacts.reshape(x.shape[0], self.n_rules, self.max_factors, self.in_dim)

    def _symbolic_edge_values(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return gated symbolic edge and all candidate values.

        Forward candidate values are exact safe-library evaluations.  Backward
        gradients are compressed by scaled-asinh without changing the forward
        value; this is useful for exp/log/rational candidates behind soft gates.
        """
        # [B, 1, 1, D, 1]
        xb = x[:, None, None, :, None]
        aff = self.symbolic_affine
        a = aff[..., 0]
        b = aff[..., 1]
        c = aff[..., 2]
        d = aff[..., 3]
        z = b[None, ...] * xb + c[None, ...]

        candidates: List[torch.Tensor] = []
        for k, name in enumerate(self.symbolic_library):
            fun = SYMBOLIC_LIB[name][0]
            raw = fun(z[..., k])
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
            raw = a[..., k][None, ...] * raw + d[..., k][None, ...]
            if self.symbolic_gradient_scale > 0:
                raw = _identity_forward_stable_backward(raw, self.symbolic_gradient_scale)
            candidates.append(raw)
        all_values = torch.stack(candidates, dim=-1)  # [B,R,S,D,K]
        op = self.operator_probabilities()
        symbolic = torch.einsum("brsdk,rsdk->brsd", all_values, op)
        return symbolic, all_values

    @torch.no_grad()
    def initialize_symbolic_manifold_seeds_(self) -> None:
        """Initialize analytic-family parameters for numerical shape constraints.

        These seeds mirror the robust symbolic/GMP initialization rather than
        starting every atom at the same local linearization.  They do not alter
        the numerical forward path because the symbolic branch can remain off.
        """
        for k, name in enumerate(self.symbolic_library):
            self.symbolic_affine[..., k, 0].fill_(1.0)
            self.symbolic_affine[..., k, 2].zero_()
            self.symbolic_affine[..., k, 3].zero_()
            if name in {"sin", "cos"}:
                self.symbolic_affine[..., k, 1].fill_(2.0)
            elif name == "exp":
                self.symbolic_affine[..., k, 1].fill_(-0.5)
            else:
                self.symbolic_affine[..., k, 1].fill_(1.0)
        self.operator_logits.zero_()
        if "x" in self.symbolic_library:
            self.operator_logits[..., self.symbolic_library.index("x")].fill_(0.15)

    def symbolic_manifold_distance(
        self,
        x: torch.Tensor,
        details: Optional[Dict[str, torch.Tensor]] = None,
        *,
        eps: float = 1e-4,
        max_normalized_error: float = 50.0,
        max_samples: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Distance of active numerical factors to a union of analytic manifolds.

        A selected numerical factor is compared with one *single* symbolic
        family ``g(beta*x+gamma)``.  For non-affine atoms a temporary output
        amplitude is allowed because products make factor amplitude
        non-identifiable; that amplitude is later absorbed into ``rule_scale``.
        Output offsets are not allowed for nonlinear atoms, preventing hidden
        lower-order terms.  The affine ``x`` atom naturally represents both
        fuzzy gates ``x`` and ``1-x`` through beta/gamma.

        Operator probabilities define a continuation over the union of
        manifolds.  Annealing ``symbolic_hardening`` to one yields an exact
        single-family forward choice with straight-through gradients.
        """
        if details is None:
            _, details = self(x, return_details=True)
        numeric = details["numeric_edges"]
        if int(max_samples) > 0 and x.shape[0] > int(max_samples):
            ids = torch.linspace(0, x.shape[0] - 1, int(max_samples), device=x.device).long()
            x = x[ids]
            numeric = numeric[ids]
        pi = details["variable_probabilities"][..., : self.in_dim]
        selected_numeric = torch.einsum("brsd,rsd->brs", numeric, pi)

        xb = x[:, None, None, :, None]
        aff = self.symbolic_affine
        a = aff[..., 0]
        b = aff[..., 1]
        c = aff[..., 2]
        symbolic_candidates: List[torch.Tensor] = []
        for k, name in enumerate(self.symbolic_library):
            z = b[..., k][None, ...] * xb[..., 0] + c[..., k][None, ...]
            if name == "x":
                raw = z
            else:
                raw = SYMBOLIC_LIB[name][0](z)
                raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)
                raw = a[..., k][None, ...] * raw
            symbolic_candidates.append(raw)
        all_candidates = torch.stack(symbolic_candidates, dim=-1)  # [B,R,S,D,K]
        selected_candidates = torch.einsum("brsdk,rsd->brsk", all_candidates, pi)

        centered = selected_numeric - selected_numeric.mean(dim=0, keepdim=True)
        denom = centered.square().mean(dim=0).clamp_min(float(eps))
        errors = (selected_numeric[..., None] - selected_candidates).square().mean(dim=0)
        errors = errors / denom[..., None]
        errors = torch.clamp(errors, max=float(max_normalized_error))

        op = self.operator_probabilities()
        selected_op = torch.einsum("rsdk,rsd->rsk", op, pi)
        per_factor = (selected_op * errors).sum(dim=-1)

        rule_weight = self.rule_gate.open_probability() * self.rule_alive_mask.to(self.rule_scale.dtype)
        factor_weight = self.factor_open_probabilities()
        weight = rule_weight[:, None] * factor_weight
        total_weight = weight.sum().clamp_min(1e-8)
        distance = (weight * per_factor).sum() / total_weight
        return distance, {
            "per_factor": per_factor,
            "operator_errors": errors,
            "selected_operator_probabilities": selected_op,
            "weight": weight,
        }

    def _can_fast_discrete_symbolic_forward(self) -> bool:
        if not bool(self.discretized.item()) or not bool(self.symbolic_enabled):
            return False
        active = torch.nonzero(
            self.hard_rule_choice & self.rule_alive_mask,
            as_tuple=False,
        ).squeeze(-1).tolist()
        for r in active:
            for slot in range(self.max_factors):
                if not bool(self.factor_alive_mask[r, slot]):
                    continue
                j = int(self.hard_variable_choice[r, slot].item())
                if j == self.in_dim:
                    continue
                if bool(self.hard_spline_choice[r, slot, j].item()):
                    return False
        return True

    def _forward_discrete_symbolic_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate only selected active symbolic factors in a hard model.

        Full symbolic GSR repeatedly refits a model with only a handful of active
        rules.  Evaluating every operator for every inactive rule/slot/input is
        unnecessary and dominated symbolic runtime.  This path preserves the
        exact forward function and the same asinh-compressed backward derivative
        used by the dense symbolic evaluator.
        """
        y = self.bias.expand(x.shape[0], 1)
        active = torch.nonzero(
            self.hard_rule_choice & self.rule_alive_mask,
            as_tuple=False,
        ).squeeze(-1).tolist()
        for r in active:
            prod = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            for slot in range(self.max_factors):
                if not bool(self.factor_alive_mask[r, slot]):
                    continue
                j = int(self.hard_variable_choice[r, slot].item())
                if j == self.in_dim:
                    continue
                k = int(self.hard_operator_choice[r, slot, j].item())
                name = self.symbolic_library[k]
                a, b, c, d = self.symbolic_affine[r, slot, j, k].unbind()
                raw = SYMBOLIC_LIB[name][0](b * x[:, j] + c)
                if not torch.isfinite(raw).all():
                    return torch.full((x.shape[0], 1), float("nan"), device=x.device, dtype=x.dtype)
                raw = a * raw + d
                if self.symbolic_gradient_scale > 0:
                    gs = torch.as_tensor(
                        self.symbolic_gradient_scale, device=raw.device, dtype=raw.dtype
                    )
                    stable = gs * torch.asinh(raw / gs)
                    raw = raw.detach() + stable - stable.detach()
                prod = prod * raw
            y = y + self.rule_scale[r] * prod[:, None]
        return y

    def forward(self, x: torch.Tensor, return_details: bool = False):
        if not return_details and self._can_fast_discrete_symbolic_forward():
            return self._forward_discrete_symbolic_fast(x)
        numeric = self._numeric_edge_values(x)
        rho = self.spline_probabilities()
        if self.symbolic_enabled:
            symbolic, symbolic_candidates = self._symbolic_edge_values(x)
            edge = rho[None, ...] * numeric + (1.0 - rho[None, ...]) * symbolic
        else:
            symbolic = torch.zeros_like(numeric)
            symbolic_candidates = None
            edge = numeric

        pi = self.variable_probabilities()
        # Identity category is masked; retain the D+1 shape only for API/formula
        # compatibility.  Feature selection is exact-ST while factor presence is
        # a separate continuous gate: F=(1-m)+m*phi(x).
        selected = torch.einsum("brsd,rsd->brs", edge, pi[..., : self.in_dim])
        m = self.factor_values()
        factors = (1.0 - m[None, ...]) + m[None, ...] * selected

        # Exact-forward / stabilized-backward numerical products. The custom
        # autograd identity returns the original values exactly, while backward
        # follows the bounded derivative of scale*asinh(x/scale). We only use this during
        # the numerical phase; symbolic extraction already has its own protected
        # operator gradients and should optimize the exact symbolic geometry.
        product_factors = factors
        if self.training and not self.symbolic_enabled and self.numeric_factor_gradient_scale > 0:
            product_factors = _identity_forward_stable_backward(
                factors, self.numeric_factor_gradient_scale
            )

        if self.training and not self.symbolic_enabled and self.numeric_product_gradient_power < 1.0:
            raw_rules = _equalized_product(
                product_factors,
                self.numeric_product_gradient_power,
                self.numeric_product_gradient_max_gain,
            )
        else:
            raw_rules = torch.prod(product_factors, dim=-1)
        q = self.rule_values()
        contributions = q[None, :] * self.rule_scale[None, :] * raw_rules
        if self.training:
            if self.symbolic_enabled and self.symbolic_product_gradient_scale > 0:
                contributions = _identity_forward_stable_backward(
                    contributions, self.symbolic_product_gradient_scale
                )
            elif (not self.symbolic_enabled) and self.numeric_product_gradient_scale > 0:
                contributions = _identity_forward_stable_backward(
                    contributions, self.numeric_product_gradient_scale
                )
        y = contributions.sum(dim=-1, keepdim=True) + self.bias

        if not return_details:
            return y
        return y, {
            "numeric_edges": numeric,
            "symbolic_edges": symbolic,
            "symbolic_candidates": symbolic_candidates,
            "edge_values": edge,
            "variable_probabilities": pi,
            "operator_probabilities": self.operator_probabilities(),
            "spline_probabilities": rho,
            "factor_values": factors,
            "factor_gate_values": m,
            "raw_rules": raw_rules,
            "rule_values": q,
            "contributions": contributions,
        }

    # ------------------------------------------------------------------
    # structural objective
    # ------------------------------------------------------------------
    def expected_rule_count(self) -> torch.Tensor:
        if bool(self.discretized.item()):
            return self.hard_rule_choice.to(self.rule_scale.dtype).sum()
        return (
            self.rule_gate.open_probability() * self.rule_alive_mask.to(self.rule_scale.dtype)
        ).sum()

    def expected_factor_count(self) -> torch.Tensor:
        active = self.factor_open_probabilities().sum(dim=-1)
        q = self.rule_gate.open_probability() * self.rule_alive_mask.to(self.rule_scale.dtype)
        return (q * active).sum()

    def regularization(
        self,
        details: Dict[str, torch.Tensor],
        weights: SumProductRegularization,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.rule_scale.device
        dtype = self.rule_scale.dtype
        total = torch.zeros((), device=device, dtype=dtype)
        parts: Dict[str, torch.Tensor] = {}

        q_open = self.rule_gate.open_probability() * self.rule_alive_mask.to(self.rule_scale.dtype)
        pi_soft = _st_hard_categorical(
            self.variable_logits,
            self.variable_temperature,
            0.0,
            mask=self.variable_allowed,
        )
        op_soft = _st_hard_categorical(
            self.operator_logits,
            self.operator_temperature,
            0.0,
        )
        rho_soft = torch.sigmoid(self.spline_logits)

        if weights.rule_l0:
            parts["rule_l0"] = float(weights.rule_l0) * q_open.sum()
            total = total + parts["rule_l0"]

        if weights.factor_l0:
            # Penalize the actual optional Hard-Concrete factor-presence gates.
            # The identity category is masked from variable selection, so using
            # that categorical channel here is constant and cannot reduce order.
            factor_open = self.factor_open_probabilities()
            optional = torch.ones_like(factor_open)
            if self.min_order > 0:
                optional[:, : self.min_order] = 0.0
            parts["factor_l0"] = float(weights.factor_l0) * (
                q_open[:, None] * factor_open * optional
            ).sum()
            total = total + parts["factor_l0"]

        if weights.variable_entropy:
            ent = _entropy(pi_soft, dim=-1)
            parts["variable_entropy"] = float(weights.variable_entropy) * (
                q_open[:, None] * ent
            ).mean()
            total = total + parts["variable_entropy"]

        if weights.operator_entropy:
            ent_op = _entropy(op_soft, dim=-1)
            factor_open = self.factor_open_probabilities().detach()
            symbolic_use = 1.0 - rho_soft
            parts["operator_entropy"] = float(weights.operator_entropy) * (
                q_open[:, None, None] * factor_open[:, :, None] * symbolic_use * ent_op
            ).mean()
            total = total + parts["operator_entropy"]

        if weights.spline_penalty:
            feature_probs = pi_soft[..., : self.in_dim]
            # Penalize using the numerical spline only when that input edge is
            # structurally selected. This drives spline -> symbol migration.
            factor_open = self.factor_open_probabilities().detach()
            parts["spline_penalty"] = float(weights.spline_penalty) * (
                q_open[:, None, None] * factor_open[:, :, None] * feature_probs * rho_soft
            ).sum()
            total = total + parts["spline_penalty"]

        if weights.contribution_group_lasso:
            c = details["contributions"]
            rms = torch.sqrt(c.square().mean(dim=0) + 1e-8)
            parts["contribution_group_lasso"] = float(weights.contribution_group_lasso) * rms.sum()
            total = total + parts["contribution_group_lasso"]

        if weights.diversity and self.n_rules > 1:
            v = details["raw_rules"]
            vc = v - v.mean(dim=0, keepdim=True)
            vn = vc / torch.sqrt(vc.square().mean(dim=0, keepdim=True) + 1e-6)
            corr = (vn.T @ vn) / float(max(1, vn.shape[0]))
            eye = torch.eye(self.n_rules, device=v.device, dtype=v.dtype)
            pair_weight = q_open[:, None] * q_open[None, :]
            diversity = ((1.0 - eye) * pair_weight * corr.square()).sum()
            diversity = diversity / (1.0 - eye).sum().clamp_min(1.0)
            parts["diversity"] = float(weights.diversity) * diversity
            total = total + parts["diversity"]

        # Contribution-space redundancy is different from raw-rule diversity:
        # it measures what each rule actually contributes after its learned
        # amplitude/gate.  The thresholded form does not punish ordinary
        # correlations; it only acts on nearly-collinear compensating rules.
        if (weights.contribution_correlation or weights.graph_redundancy) and self.n_rules > 1:
            c = details["contributions"]
            cc = c - c.mean(dim=0, keepdim=True)
            cn = cc / torch.sqrt(cc.square().mean(dim=0, keepdim=True) + 1e-8)
            corr = (cn.T @ cn) / float(max(1, cn.shape[0]))
            eye = torch.eye(self.n_rules, device=c.device, dtype=c.dtype)
            abs_corr = corr.abs() * (1.0 - eye)
            thr = float(max(0.0, min(0.999999, weights.contribution_correlation_threshold)))
            excess = torch.relu(abs_corr - thr) / max(1e-6, 1.0 - thr)
            pair_weight = q_open[:, None] * q_open[None, :]
            if weights.contribution_correlation:
                pen = (pair_weight * excess.square()).sum() / (1.0 - eye).sum().clamp_min(1.0)
                parts["contribution_correlation"] = float(weights.contribution_correlation) * pen
                total = total + parts["contribution_correlation"]
            if weights.graph_redundancy:
                # Interpretable graph-message precursor to a learned GNN: rules
                # are nodes; thresholded |corr| is adjacency; each node receives
                # a redundancy message from its neighbours.  Weighting the L0
                # pressure by that message makes duplicate rules easier to prune
                # without forcing all useful rules to be orthogonal.
                deg = excess.sum(dim=1).clamp_min(1e-8)
                message = (excess @ q_open) / deg
                node_pen = (q_open * message).mean()
                parts["graph_redundancy"] = float(weights.graph_redundancy) * node_pen
                total = total + parts["graph_redundancy"]

        parts["total"] = total
        return total, parts

    # ------------------------------------------------------------------
    # hardening / diagnostics / formula
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_hardening(
        self,
        structure: Optional[float] = None,
        symbolic: Optional[float] = None,
        factor: Optional[float] = None,
        rule: Optional[float] = None,
    ) -> None:
        if structure is not None:
            h = float(max(0.0, min(1.0, structure)))
            self.structure_hardening = h
            if factor is None:
                self.factor_hardening = h
            if rule is None:
                self.rule_hardening = h
        if factor is not None:
            self.factor_hardening = float(max(0.0, min(1.0, factor)))
        if rule is not None:
            self.rule_hardening = float(max(0.0, min(1.0, rule)))
        if symbolic is not None:
            self.symbolic_hardening = float(max(0.0, min(1.0, symbolic)))

    @torch.no_grad()
    def discretize(
        self,
        rule_threshold: float = 0.5,
        force_symbolic: bool = False,
        freeze_gates: bool = True,
    ) -> None:
        pi = _st_hard_categorical(
            self.variable_logits,
            self.variable_temperature,
            0.0,
            mask=self.variable_allowed,
        )
        op = _st_hard_categorical(
            self.operator_logits,
            self.operator_temperature,
            0.0,
        )
        rho = torch.sigmoid(self.spline_logits)
        feature_choice = torch.argmax(pi[..., : self.in_dim], dim=-1)
        factor_open = self.factor_values(hardening=1.0) >= 0.5
        hard_choice = feature_choice.clone()
        hard_choice[~factor_open] = self.in_dim
        self.hard_variable_choice.copy_(hard_choice)
        self.hard_operator_choice.copy_(torch.argmax(op, dim=-1))
        if force_symbolic:
            self.hard_spline_choice.fill_(False)
        else:
            self.hard_spline_choice.copy_(rho >= 0.5)
        self.hard_rule_choice.copy_(self.rule_values(hardening=1.0) >= float(rule_threshold))
        self.discretized.fill_(True)
        self.structure_hardening = 1.0
        self.factor_hardening = 1.0
        self.rule_hardening = 1.0
        self.symbolic_hardening = 1.0
        if freeze_gates:
            self.variable_logits.requires_grad_(False)
            self.operator_logits.requires_grad_(False)
            self.spline_logits.requires_grad_(False)
            self.rule_gate.log_alpha.requires_grad_(False)
            self.factor_gate.log_alpha.requires_grad_(False)

    @torch.no_grad()
    def undiscretize(self) -> None:
        self.discretized.fill_(False)
        self.variable_logits.requires_grad_(True)
        self.operator_logits.requires_grad_(True)
        self.spline_logits.requires_grad_(True)
        self.rule_gate.log_alpha.requires_grad_(True)
        self.factor_gate.log_alpha.requires_grad_(True)

    def diagnostics(self) -> Dict[str, object]:
        pi = _st_hard_categorical(
            self.variable_logits.detach(), self.variable_temperature, 0.0, mask=self.variable_allowed
        )
        op = _st_hard_categorical(
            self.operator_logits.detach(), self.operator_temperature, 0.0
        )
        rho = torch.sigmoid(self.spline_logits.detach())
        q = (
            self.rule_gate.open_probability() * self.rule_alive_mask.to(self.rule_scale.dtype)
        ).detach()
        pvar = pi[..., : self.in_dim].clamp_min(1e-8)
        pvar = pvar / pvar.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        var_ent = float(_entropy(pvar).mean().cpu())
        op_ent = float(_entropy(op).mean().cpu())
        top2 = torch.topk(pi, k=min(2, pi.shape[-1]), dim=-1).values
        margin = float((top2[..., 0] - top2[..., 1]).mean().cpu()) if top2.shape[-1] > 1 else 1.0
        return {
            "expected_rules": float(q.sum().cpu()),
            "expected_factors": float(self.expected_factor_count().detach().cpu()),
            "rule_open_probabilities": q.cpu().tolist(),
            "factor_open_probabilities": self.factor_open_probabilities().detach().cpu().tolist(),
            "variable_entropy": var_ent,
            "variable_margin": margin,
            "operator_entropy": op_ent,
            "mean_spline_probability": float(rho.mean().cpu()),
            "structure_hardening": float(self.structure_hardening),
            "factor_hardening": float(self.factor_hardening),
            "rule_hardening": float(self.rule_hardening),
            "symbolic_hardening": float(self.symbolic_hardening),
        }

    def hard_structure(self, variable_names: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
        names = list(variable_names) if variable_names is not None else [f"x{i}" for i in range(self.in_dim)]
        if len(names) != self.in_dim:
            raise ValueError("variable_names length mismatch")

        if bool(self.discretized.item()):
            vchoice = self.hard_variable_choice.detach().cpu()
            ochoice = self.hard_operator_choice.detach().cpu()
            spline = self.hard_spline_choice.detach().cpu()
            active = self.hard_rule_choice.detach().cpu()
        else:
            pi = _st_hard_categorical(self.variable_logits.detach(), self.variable_temperature, 0.0, mask=self.variable_allowed)
            op = _st_hard_categorical(self.operator_logits.detach(), self.operator_temperature, 0.0)
            vchoice = torch.argmax(pi[..., : self.in_dim], dim=-1).cpu()
            fopen = (self.factor_values(hardening=1.0).detach() >= 0.5).cpu()
            vchoice[~fopen] = self.in_dim
            ochoice = torch.argmax(op, dim=-1).cpu()
            spline = (torch.sigmoid(self.spline_logits.detach()) >= 0.5).cpu()
            active = (self.rule_values(hardening=1.0).detach() >= 0.5).cpu()

        out: List[Dict[str, object]] = []
        for r in range(self.n_rules):
            factors = []
            for s in range(self.max_factors):
                j = int(vchoice[r, s])
                if j == self.in_dim:
                    factors.append({"slot": s, "identity": True})
                else:
                    use_spline = bool(spline[r, s, j])
                    factors.append({
                        "slot": s,
                        "identity": False,
                        "variable": names[j],
                        "variable_index": j,
                        "expert": "spline" if use_spline else self.symbolic_library[int(ochoice[r, s, j])],
                    })
            out.append({
                "rule": r,
                "active": bool(active[r]),
                "scale": float(self.rule_scale.detach().cpu()[r]),
                "factors": factors,
            })
        return out

    def symbolic_formula(
        self,
        variable_names: Optional[Sequence[str]] = None,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        digits: int = 5,
        simplify: bool = False,
        simplify_timeout: float = 5.0,
    ):
        """Return a SymPy expression for the hard symbolic part of the model.

        If an active selected edge is still a numerical spline, it is represented
        by an uninterpreted SymPy function ``Spline_r_s_var`` rather than silently
        pretending that a symbolic operator was recovered.
        """
        import sympy as sp

        names = list(variable_names) if variable_names is not None else [f"x{i}" for i in range(self.in_dim)]
        xs = sp.symbols(" ".join(names))
        if self.in_dim == 1:
            xs = (xs,)
        if input_mean is None:
            mean = [0.0] * self.in_dim
        else:
            mean = [float(v) for v in input_mean.detach().cpu().reshape(-1)]
        if input_std is None:
            std = [1.0] * self.in_dim
        else:
            std = [float(v) for v in input_std.detach().cpu().reshape(-1)]

        if not bool(self.discretized.item()):
            # Formula should describe an actual discrete architecture.
            tmp = copy.deepcopy(self)
            tmp.discretize(freeze_gates=False)
            return tmp.symbolic_formula(
                names, input_mean, input_std, digits=digits, simplify=simplify,
                simplify_timeout=simplify_timeout,
            )

        expr = sp.Float(round(float(self.bias.detach().cpu()), digits))
        for r in range(self.n_rules):
            if not bool(self.hard_rule_choice[r]):
                continue
            prod = sp.Integer(1)
            for s in range(self.max_factors):
                j = int(self.hard_variable_choice[r, s])
                if j == self.in_dim:
                    continue
                xnorm = (xs[j] - sp.Float(mean[j])) / sp.Float(std[j])
                if bool(self.hard_spline_choice[r, s, j]):
                    factor = sp.Function(f"Spline_r{r}_s{s}_{names[j]}")(xnorm)
                else:
                    k = int(self.hard_operator_choice[r, s, j])
                    name = self.symbolic_library[k]
                    aff = self.symbolic_affine.detach().cpu()[r, s, j, k]
                    a, b, c, d = [sp.Float(round(float(v), digits)) for v in aff]
                    z = b * xnorm + c
                    symfun = SYMBOLIC_LIB[name][1]
                    try:
                        g = symfun(z)
                    except Exception:
                        # Some custom entries use Python arithmetic lambdas that
                        # still work with SymPy, but keep a readable fallback.
                        g = sp.Function(name)(z)
                    factor = a * g + d
                prod = prod * factor
            scale = sp.Float(round(float(self.rule_scale.detach().cpu()[r]), digits))
            expr = expr + scale * prod
        return _safe_sympy_simplify(expr, simplify_timeout) if simplify else expr


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------

def default_sum_product_schedule(
    base_lr: float = 2e-3,
    symbolic: bool = True,
) -> List[SumProductTrainingStage]:
    """Staged low-rank discovery: fit functions first, then rank/order, then symbols."""
    stages = [
        SumProductTrainingStage(
            name="numeric representation warmup",
            steps=1200,
            lr=base_lr,
            structure_hardening_start=1.0,
            structure_hardening_end=1.0,
            factor_hardening_start=1.0,
            factor_hardening_end=1.0,
            rule_hardening_start=1.0,
            rule_hardening_end=1.0,
            symbolic_hardening_start=0.0,
            symbolic_hardening_end=0.0,
            variable_temperature_start=0.7,
            variable_temperature_end=0.7,
            regularization=SumProductRegularization(),
            train_structure=False,
            train_variable_gates=False,
            train_rank_gates=False,
            train_symbolic=False,
            symbolic_enabled=False,
            grad_clip=0.5,
        ),
        SumProductTrainingStage(
            name="continuous rank/order compression",
            steps=900,
            lr=base_lr * 0.55,
            structure_hardening_start=1.0,
            structure_hardening_end=1.0,
            factor_hardening_start=0.0,
            factor_hardening_end=0.80,
            rule_hardening_start=0.0,
            rule_hardening_end=0.80,
            symbolic_hardening_start=0.0,
            symbolic_hardening_end=0.0,
            variable_temperature_start=0.7,
            variable_temperature_end=0.7,
            regularization=SumProductRegularization(
                rule_l0=8.0e-5,
                factor_l0=3.0e-5,
                contribution_group_lasso=1.0e-5,
                diversity=1.0e-5,
            ),
            train_structure=True,
            train_variable_gates=False,
            train_rank_gates=True,
            train_symbolic=False,
            symbolic_enabled=False,
            rank_lr_scale=6.0,
            grad_clip=0.35,
        ),
        SumProductTrainingStage(
            name="hard rank/order consolidation",
            steps=1800,
            lr=base_lr * 0.25,
            structure_hardening_start=1.0,
            structure_hardening_end=1.0,
            factor_hardening_start=0.80,
            factor_hardening_end=1.0,
            rule_hardening_start=0.80,
            rule_hardening_end=1.0,
            symbolic_hardening_start=0.0,
            symbolic_hardening_end=0.0,
            variable_temperature_start=0.7,
            variable_temperature_end=0.55,
            regularization=SumProductRegularization(
                rule_l0=3.0e-4,
                factor_l0=8.0e-5,
                contribution_group_lasso=3.0e-5,
                diversity=3.0e-5,
            ),
            train_structure=True,
            train_variable_gates=False,
            train_rank_gates=True,
            train_symbolic=False,
            symbolic_enabled=False,
            rank_lr_scale=8.0,
            grad_clip=0.25,
            checkpoint_hard_endpoint=True,
        ),
    ]
    if symbolic:
        stages.extend([
            SumProductTrainingStage(
                name="spline to symbol migration",
                steps=900,
                lr=base_lr * 0.30,
                structure_hardening_start=1.0,
                structure_hardening_end=1.0,
                factor_hardening_start=1.0,
                factor_hardening_end=1.0,
                rule_hardening_start=1.0,
                rule_hardening_end=1.0,
                symbolic_hardening_start=0.0,
                symbolic_hardening_end=0.75,
                variable_temperature_start=0.55,
                variable_temperature_end=0.45,
                operator_temperature_start=0.9,
                operator_temperature_end=0.45,
                regularization=SumProductRegularization(
                    rule_l0=6e-5,
                    factor_l0=2e-5,
                    operator_entropy=5e-5,
                    spline_penalty=2e-5,
                    contribution_group_lasso=1.5e-5,
                ),
                train_structure=False,
                train_variable_gates=False,
                train_rank_gates=False,
                train_symbolic=True,
                symbolic_enabled=True,
                grad_clip=0.25,
            ),
            SumProductTrainingStage(
                name="symbolic hardening",
                steps=400,
                lr=base_lr * 0.12,
                structure_hardening_start=1.0,
                structure_hardening_end=1.0,
                factor_hardening_start=1.0,
                factor_hardening_end=1.0,
                rule_hardening_start=1.0,
                rule_hardening_end=1.0,
                symbolic_hardening_start=0.75,
                symbolic_hardening_end=1.0,
                variable_temperature_start=0.45,
                variable_temperature_end=0.40,
                operator_temperature_start=0.45,
                operator_temperature_end=0.30,
                regularization=SumProductRegularization(
                    operator_entropy=7e-5,
                    spline_penalty=3e-5,
                ),
                train_structure=False,
                train_variable_gates=False,
                train_rank_gates=False,
                train_symbolic=True,
                symbolic_enabled=True,
                grad_clip=0.18,
            ),
        ])
    return stages


def _interpolate(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def fit_sum_product_kan(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: Optional[torch.Tensor] = None,
    val_y: Optional[torch.Tensor] = None,
    stages: Optional[Sequence[SumProductTrainingStage]] = None,
    log_every: int = 100,
    weight_decay: float = 0.0,
    restore_best_each_stage: bool = True,
    show_progress: bool = False,
) -> List[Dict[str, object]]:
    """Train a SumProductKAN with continuation and independent validation."""
    if stages is None:
        stages = default_sum_product_schedule()
    history: List[Dict[str, object]] = []

    for stage in stages:
        # Stage-wise alternating optimization: learn useful univariate factors
        # before allowing the combinatorial structure to move, then open the
        # symbolic branch only after the numerical sum-product model is useful.
        model.symbolic_enabled = bool(stage.symbolic_enabled)
        train_var = stage.train_structure if stage.train_variable_gates is None else stage.train_variable_gates
        train_rank = stage.train_structure if stage.train_rank_gates is None else stage.train_rank_gates
        model.variable_logits.requires_grad_(bool(train_var))
        model.rule_gate.log_alpha.requires_grad_(bool(train_rank))
        model.factor_gate.log_alpha.requires_grad_(bool(train_rank))
        for p in (model.operator_logits, model.symbolic_affine, model.spline_logits):
            p.requires_grad_(bool(stage.train_symbolic))
        # Different structures need different timescales. Rank/order logits must
        # move several logit units to cross a hard pruning boundary, while KAN
        # spline coefficients need much smaller updates.
        rank_ids = {id(model.rule_gate.log_alpha), id(model.factor_gate.log_alpha)}
        variable_ids = {id(model.variable_logits)}
        symbolic_ids = {id(model.operator_logits), id(model.symbolic_affine), id(model.spline_logits)}
        base_params, rank_params, variable_params, symbolic_params = [], [], [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in rank_ids:
                rank_params.append(p)
            elif id(p) in variable_ids:
                variable_params.append(p)
            elif id(p) in symbolic_ids:
                symbolic_params.append(p)
            else:
                base_params.append(p)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": stage.lr})
        if rank_params:
            groups.append({"params": rank_params, "lr": stage.lr * float(stage.rank_lr_scale)})
        if variable_params:
            groups.append({"params": variable_params, "lr": stage.lr * float(stage.variable_lr_scale)})
        if symbolic_params:
            groups.append({"params": symbolic_params, "lr": stage.lr * float(stage.symbolic_lr_scale)})
        params = [p for g in groups for p in g["params"]]
        optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(stage.steps)), eta_min=max(stage.lr * 0.05, 1e-6)
        )
        best_state = None
        best_meta = None
        best_val = float("inf")
        manifold_dual = float(max(0.0, stage.symbolic_manifold_dual_init))
        print(f"\n=== SumProductKAN: {stage.name} ===")

        stage_bar = _progress_range(
            stage.steps, desc=f"SumProductKAN · {stage.name}",
            enabled=show_progress, leave=True,
        )
        for it in stage_bar:
            u = 1.0 if stage.steps <= 1 else it / float(stage.steps - 1)
            model.structure_hardening = _interpolate(
                stage.structure_hardening_start, stage.structure_hardening_end, u
            )
            fh0 = stage.structure_hardening_start if stage.factor_hardening_start is None else stage.factor_hardening_start
            fh1 = stage.structure_hardening_end if stage.factor_hardening_end is None else stage.factor_hardening_end
            rh0 = stage.structure_hardening_start if stage.rule_hardening_start is None else stage.rule_hardening_start
            rh1 = stage.structure_hardening_end if stage.rule_hardening_end is None else stage.rule_hardening_end
            model.factor_hardening = _interpolate(fh0, fh1, u)
            model.rule_hardening = _interpolate(rh0, rh1, u)
            model.symbolic_hardening = _interpolate(
                stage.symbolic_hardening_start, stage.symbolic_hardening_end, u
            )
            model.variable_temperature = _interpolate(
                stage.variable_temperature_start, stage.variable_temperature_end, u
            )
            model.operator_temperature = _interpolate(
                stage.operator_temperature_start, stage.operator_temperature_end, u
            )
            model.rule_gate.temperature = _interpolate(
                stage.rule_temperature_start, stage.rule_temperature_end, u
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, details = model(train_x, return_details=True)
            task = torch.mean((pred - train_y) ** 2)
            reg, parts = model.regularization(details, stage.regularization)
            loss = task + reg
            manifold_distance = None
            manifold_violation = None
            if float(stage.symbolic_manifold_dual_init) > 0.0 or float(stage.symbolic_manifold_rho) > 0.0:
                manifold_distance, _ = model.symbolic_manifold_distance(
                    train_x, details, max_samples=int(stage.symbolic_manifold_max_samples)
                )
                manifold_violation = torch.relu(
                    manifold_distance - float(stage.symbolic_manifold_tolerance)
                )
                loss = loss + manifold_dual * manifold_violation + 0.5 * float(stage.symbolic_manifold_rho) * manifold_violation.square()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in stage {stage.name}, iter {it}")
            loss.backward()
            if stage.grad_clip and stage.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(stage.grad_clip))
            optimizer.step()
            scheduler.step()
            if manifold_violation is not None and float(stage.symbolic_manifold_rho) > 0.0:
                every = max(1, int(stage.symbolic_manifold_dual_update_every))
                if (it + 1) % every == 0:
                    manifold_dual = min(
                        float(stage.symbolic_manifold_dual_max),
                        max(0.0, manifold_dual + float(stage.symbolic_manifold_rho) * float(manifold_violation.detach().cpu())),
                    )

            if val_x is not None and val_y is not None and (it % max(1, log_every) == 0 or it == stage.steps - 1):
                model.eval()
                with torch.no_grad():
                    vp = model(val_x)
                    vmse = torch.mean((vp - val_y) ** 2)
                    vrmse = float(torch.sqrt(vmse).cpu())
                hard_vrmse = None
                checkpoint_score = vrmse
                if bool(stage.checkpoint_hard_endpoint):
                    # Structural consolidation must be selected by the function
                    # we will actually commit.  Earlier versions restored a
                    # partially hardened checkpoint with excellent soft error,
                    # then alpha=1 caused a large discontinuity.
                    old_fh, old_rh = model.factor_hardening, model.rule_hardening
                    model.factor_hardening = 1.0
                    model.rule_hardening = 1.0
                    with torch.no_grad():
                        hp = model(val_x)
                        hard_vrmse = float(torch.sqrt(torch.mean((hp - val_y) ** 2)).cpu())
                    model.factor_hardening, model.rule_hardening = old_fh, old_rh
                    checkpoint_score = hard_vrmse
                if checkpoint_score < best_val:
                    best_val = checkpoint_score
                    best_state = copy.deepcopy(model.state_dict())
                    best_meta = (
                        float(model.structure_hardening),
                        float(model.factor_hardening),
                        float(model.rule_hardening),
                        float(model.symbolic_hardening),
                        float(model.variable_temperature),
                        float(model.operator_temperature),
                        float(model.rule_gate.temperature),
                    )
                d = model.diagnostics()
                train_rmse = float(torch.sqrt(task).detach().cpu())
                detail_line = (
                    f"  {it+1:4d}/{stage.steps}: train RMSE={train_rmse:.6g}, "
                    f"val RMSE={vrmse:.6g}, E[rules]={d['expected_rules']:.2f}, "
                    f"E[factors]={d['expected_factors']:.2f}, spline={d['mean_spline_probability']:.3f}, "
                    f"hard=({d['structure_hardening']:.2f},{d['symbolic_hardening']:.2f})"
                    + (f", hard-endpoint RMSE={hard_vrmse:.6g}" if hard_vrmse is not None else "")
                )
                if show_progress:
                    # Keep the compact live bar, but also preserve the exact
                    # detailed checkpoint line from the pre-tqdm implementation.
                    # tqdm.write() leaves it in terminal history without
                    # overwriting/corrupting the active progress bar.
                    stage_bar.set_postfix(
                        train=f"{train_rmse:.3g}",
                        val=f"{vrmse:.3g}",
                        rules=f"{d['expected_rules']:.2f}",
                        factors=f"{d['expected_factors']:.2f}",
                        refresh=False,
                    )
                    tqdm.write(detail_line)
                else:
                    print(detail_line)

        if restore_best_each_stage and best_state is not None:
            model.load_state_dict(best_state)
            if best_meta is not None:
                (
                    model.structure_hardening,
                    model.factor_hardening,
                    model.rule_hardening,
                    model.symbolic_hardening,
                    model.variable_temperature,
                    model.operator_temperature,
                    model.rule_gate.temperature,
                ) = best_meta
        history.append({
            "stage": stage.name,
            "best_val_rmse": best_val,
            "diagnostics": model.diagnostics(),
            "symbolic_manifold_dual": float(manifold_dual),
        })
    # Leave the model fully trainable for optional user-defined continuation.
    model.variable_logits.requires_grad_(True)
    model.rule_gate.log_alpha.requires_grad_(True)
    model.factor_gate.log_alpha.requires_grad_(True)
    model.operator_logits.requires_grad_(True)
    model.symbolic_affine.requires_grad_(True)
    model.spline_logits.requires_grad_(True)
    model.symbolic_enabled = bool(stages[-1].symbolic_enabled) if len(stages) else model.symbolic_enabled
    return history


@torch.no_grad()
def conformal_prune_sum_product_rules(
    model: SumProductKAN,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    *,
    alpha: float = 0.10,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.0,
    min_rules: int = 1,
) -> Dict[str, object]:
    """Sequentially remove hard rules only when fresh-fold conformal checks pass.

    This is deliberately a *commit certificate*, not the source of sparsity.
    The differentiable q/pi/rho/gamma gates discover a compact structure first;
    conformal calibration only decides whether making a learned low-contribution
    rule exactly zero is safe.
    """
    if not bool(model.discretized.item()):
        raise ValueError("conformal rule pruning requires a discretized model")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")

    active = torch.nonzero(model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
    if len(active) <= int(min_rules):
        return {"model": model, "accepted": [], "rejected": None}

    # Rank by actual calibration contribution RMS; gates have already been
    # discretized, so this measures what survives in the hard predictive model.
    _, details = model(x_cal, return_details=True)
    contrib = details["contributions"]
    rms = torch.sqrt(contrib.square().mean(dim=0) + 1e-12)
    candidates = sorted(active, key=lambda r: float(rms[r]))

    n_checks = max(1, len(active) - int(min_rules))
    ids = torch.arange(x_cal.shape[0], device=x_cal.device)
    folds = [z for z in torch.tensor_split(ids, n_checks) if z.numel() > 0]
    accepted: List[int] = []
    current = model
    rejected = None

    for ci, r in enumerate(candidates):
        if int(current.hard_rule_choice.sum()) <= int(min_rules):
            break
        fold = folds[min(ci, len(folds) - 1)]
        xb, yb = x_cal[fold], y_cal[fold]
        pb = current(xb)
        trial = copy.deepcopy(current)
        trial.hard_rule_choice[r] = False
        pc = trial(xb)

        base_abs = (pb - yb).abs()
        cand_abs = (pc - yb).abs()
        if base_abs.ndim > 1:
            base_abs = base_abs.mean(dim=tuple(range(1, base_abs.ndim)))
            cand_abs = cand_abs.mean(dim=tuple(range(1, cand_abs.ndim)))
        score = cand_abs - base_abs
        score = score[torch.isfinite(score)]
        if score.numel() == 0:
            rejected = r
            break
        n = int(score.numel())
        k = min(n, max(1, int(math.ceil((n + 1) * (1.0 - float(alpha))))))
        upper = float(torch.sort(score).values[k - 1])
        base_rmse = float(torch.sqrt(torch.mean((pb - yb) ** 2)))
        tol = max(float(absolute_tolerance), float(relative_tolerance) * base_rmse)
        if upper <= tol:
            current = trial
            accepted.append(r)
        else:
            rejected = r
            break

    return {
        "model": current,
        "accepted": accepted,
        "rejected": rejected,
        "active_rules": int(current.hard_rule_choice.sum()),
    }


def conformal_refit_prune_sum_product_rules(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    *,
    alpha: float = 0.10,
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.0,
    validation_rel_budget: float = 0.15,
    refit_steps: int = 160,
    refit_lr: float = 2e-4,
    min_rules: int = 1,
    show_progress: bool = False,
) -> Dict[str, object]:
    """In-context hard-rule pruning with validation + conformal certification.

    The differentiable q/m/pi gates discover the low-rank architecture.  This
    routine is only the irreversible commit step: remove one hard rule, refit
    all remaining continuous KAN/symbolic parameters, require independent
    validation to stay inside a global budget, then certify per-example excess
    error on a fresh calibration fold.
    """
    if not bool(model.discretized.item()):
        raise ValueError("conformal refit pruning requires a discretized model")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")

    current = copy.deepcopy(model)
    current.eval()
    with torch.no_grad():
        base_val_rmse = float(torch.sqrt(torch.mean((current(val_x) - val_y) ** 2)))
    global_val_cap = base_val_rmse * (1.0 + max(0.0, float(validation_rel_budget)))

    n_active0 = int(current.hard_rule_choice.sum())
    n_checks = max(1, n_active0 - int(min_rules))
    ids = torch.arange(x_cal.shape[0], device=x_cal.device)
    folds = [z for z in torch.tensor_split(ids, n_checks) if z.numel() > 0]
    accepted: List[int] = []
    rejected: List[int] = []
    certificates: List[Dict[str, object]] = []
    fold_idx = 0

    while int(current.hard_rule_choice.sum()) > int(min_rules):
        active = torch.nonzero(current.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        current.eval()
        with torch.no_grad():
            _, det = current(val_x, return_details=True)
            rms = torch.sqrt(det["contributions"].square().mean(dim=0) + 1e-12)
        # Re-rank after every accepted refit; compensation can change which rule
        # is now redundant.
        candidates = sorted(active, key=lambda r: float(rms[r]))
        committed = False

        for r in candidates:
            trial = copy.deepcopy(current)
            trial.hard_rule_choice[r] = False

            params = [p for p in trial.parameters() if p.requires_grad]
            if params and int(refit_steps) > 0:
                opt = torch.optim.Adam(params, lr=float(refit_lr))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, int(refit_steps)), eta_min=max(float(refit_lr) * 0.05, 1e-6)
                )
                best_state = copy.deepcopy(trial.state_dict())
                trial.eval()
                with torch.no_grad():
                    best_val = float(torch.sqrt(torch.mean((trial(val_x) - val_y) ** 2)))
                for it in _progress_range(
                    refit_steps, desc=f"Conformal refit · drop rule {r}",
                    enabled=show_progress, leave=False,
                ):
                    trial.train()
                    opt.zero_grad(set_to_none=True)
                    pred = trial(train_x)
                    loss = torch.mean((pred - train_y) ** 2)
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 0.25)
                    opt.step()
                    sched.step()
                    if it % 20 == 0 or it == int(refit_steps) - 1:
                        trial.eval()
                        with torch.no_grad():
                            vr = float(torch.sqrt(torch.mean((trial(val_x) - val_y) ** 2)))
                        if math.isfinite(vr) and vr < best_val:
                            best_val = vr
                            best_state = copy.deepcopy(trial.state_dict())
                trial.load_state_dict(best_state)

            trial.eval()
            with torch.no_grad():
                val_rmse = float(torch.sqrt(torch.mean((trial(val_x) - val_y) ** 2)))
            if not math.isfinite(val_rmse) or val_rmse > global_val_cap:
                rejected.append(r)
                continue

            if not folds:
                current = trial
                accepted.append(r)
                committed = True
                break

            fold = folds[min(fold_idx, len(folds) - 1)]
            fold_idx += 1
            xb, yb = x_cal[fold], y_cal[fold]
            with torch.no_grad():
                pb = current(xb)
                pc = trial(xb)
                base_abs = (pb - yb).abs()
                cand_abs = (pc - yb).abs()
                if base_abs.ndim > 1:
                    dims = tuple(range(1, base_abs.ndim))
                    base_abs = base_abs.mean(dim=dims)
                    cand_abs = cand_abs.mean(dim=dims)
                scores = cand_abs - base_abs
                scores = scores[torch.isfinite(scores)]
            if scores.numel() == 0:
                rejected.append(r)
                continue
            n = int(scores.numel())
            k = min(n, max(1, int(math.ceil((n + 1) * (1.0 - float(alpha))))))
            upper = float(torch.sort(scores).values[k - 1])
            with torch.no_grad():
                cal_base_rmse = float(torch.sqrt(torch.mean((pb - yb) ** 2)))
                cal_trial_rmse = float(torch.sqrt(torch.mean((pc - yb) ** 2)))
            tol = max(float(absolute_tolerance), float(relative_tolerance) * cal_base_rmse)
            cert = {
                "rule": r,
                "upper_excess": upper,
                "tolerance": tol,
                "calibration_base_rmse": cal_base_rmse,
                "calibration_candidate_rmse": cal_trial_rmse,
                "validation_rmse": val_rmse,
                "accepted": upper <= tol,
                "n": n,
            }
            certificates.append(cert)
            if upper <= tol:
                current = trial
                accepted.append(r)
                committed = True
                break
            rejected.append(r)

        if not committed:
            break

    return {
        "model": current,
        "accepted": accepted,
        "rejected": rejected,
        "certificates": certificates,
        "active_rules": int(current.hard_rule_choice.sum()),
        "validation_cap": global_val_cap,
    }

# ----------------------------------------------------------------------
# Numerical-to-symbolic in-context extraction helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def _sumproduct_edge_curve(
    model: SumProductKAN,
    x: torch.Tensor,
    rule: int,
    slot: int,
    variable: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one selected numerical KAN factor edge on a dataset."""
    model.eval()
    numeric = model._numeric_edge_values(x)
    return x[:, int(variable)].detach().reshape(-1), numeric[:, int(rule), int(slot), int(variable)].detach().reshape(-1)


def shortlist_symbolic_edge(
    model: SumProductKAN,
    x: torch.Tensor,
    rule: int,
    slot: int,
    variable: int,
    *,
    library: Optional[Sequence[str]] = None,
    topk: int = 3,
    max_samples: int = 512,
    max_abs_value: float = 1e5,
) -> List[Dict[str, object]]:
    """Cheap local screen used only to propose GSR candidates.

    The final decision is never based on local curve fit.  A coarse affine-input
    search and analytic affine-output fit reduce the operator library to a small
    shortlist; every shortlisted candidate is then scored after an in-context
    full-model refit by :func:`in_context_symbolic_rule_gsr`.
    """
    lib = tuple(model.symbolic_library if library is None else library)
    xx, yy = _sumproduct_edge_curve(model, x, rule, slot, variable)
    if xx.numel() > int(max_samples):
        ids = torch.linspace(0, xx.numel() - 1, int(max_samples), device=xx.device).long()
        xx, yy = xx[ids], yy[ids]
    ymean = yy.mean()
    yc = yy - ymean
    sst = yc.square().sum().clamp_min(1e-12)

    bvals = torch.tensor(
        [-4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
          0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        device=xx.device, dtype=xx.dtype,
    )
    c_general = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=xx.device, dtype=xx.dtype)
    c_periodic = torch.tensor(
        [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, math.pi],
        device=xx.device, dtype=xx.dtype,
    )
    out: List[Dict[str, object]] = []
    for name in lib:
        if name not in SYMBOLIC_LIB or name not in model.symbolic_library:
            continue
        fun = SYMBOLIC_LIB[name][0]
        cvals = c_periodic if name in {"sin", "cos"} else c_general
        z = bvals[:, None, None] * xx[None, None, :] + cvals[None, :, None]
        try:
            gg = fun(z)
        except Exception:
            continue
        finite = torch.isfinite(gg).all(dim=-1)
        gg = torch.nan_to_num(gg, nan=0.0, posinf=max_abs_value, neginf=-max_abs_value)
        gg = gg.clamp(-float(max_abs_value), float(max_abs_value))
        flat = gg.reshape(-1, xx.numel())
        valid = finite.reshape(-1)
        gm = flat.mean(dim=1, keepdim=True)
        gc = flat - gm
        var = gc.square().sum(dim=1).clamp_min(1e-12)
        aa = (gc * yc.unsqueeze(0)).sum(dim=1) / var
        dd = ymean - aa * gm.squeeze(1)
        pred = aa[:, None] * flat + dd[:, None]
        sse = (pred - yy.unsqueeze(0)).square().sum(dim=1)
        sse = torch.where(valid, sse, torch.full_like(sse, float("inf")))
        idx = int(torch.argmin(sse).item())
        if not torch.isfinite(sse[idx]):
            continue
        nb = int(cvals.numel())
        ib = idx // nb
        ic = idx % nb
        r2 = float((1.0 - sse[idx] / sst).detach().cpu())
        out.append({
            "name": name,
            "r2": r2,
            "affine": (
                float(aa[idx].detach().cpu()),
                float(bvals[ib].detach().cpu()),
                float(cvals[ic].detach().cpu()),
                float(dd[idx].detach().cpu()),
            ),
        })
    out.sort(key=lambda z: float(z["r2"]), reverse=True)
    return out[: max(1, int(topk))]


def _refit_sumproduct_candidate(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    steps: int,
    lr: float,
    grad_clip: float = 0.20,
) -> Tuple[SumProductKAN, float]:
    """Brief full-network in-context refit with validation checkpointing."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or int(steps) <= 0:
        model.eval()
        with torch.no_grad():
            score = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
        return model, score
    opt = torch.optim.Adam(params, lr=float(lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, int(steps)), eta_min=max(float(lr) * 0.05, 5e-6)
    )
    model.eval()
    with torch.no_grad():
        best_mse = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
    best = copy.deepcopy(model.state_dict())
    for it in range(int(steps)):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(train_x)
        loss = torch.mean((pred - train_y) ** 2)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
        opt.step()
        sched.step()
        if it % 10 == 0 or it == int(steps) - 1:
            model.eval()
            with torch.no_grad():
                vmse = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
            if math.isfinite(vmse) and vmse < best_mse:
                best_mse = vmse
                best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model, best_mse




def _stabilize_pruned_candidate(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    initial_steps: int,
    lr: float,
    extra_rounds: int = 3,
    extra_steps: int = 80,
    min_relative_improvement: float = 1e-3,
) -> Tuple[SumProductKAN, float, int]:
    """Refit a structural-deletion trial until its validation fit stabilizes.

    A deletion can require the remaining rules to redistribute amplitude and
    spline shape.  The first refit decides whether the proposal is plausible;
    additional short rounds continue only while validation improves materially.
    The best validation checkpoint is restored after every round.
    """
    model, mse = _refit_sumproduct_candidate(
        model, train_x, train_y, val_x, val_y,
        steps=max(0, int(initial_steps)), lr=float(lr), grad_clip=0.25,
    )
    rounds_used = 0
    prev = float(mse)
    for _ in range(max(0, int(extra_rounds))):
        trial, new_mse = _refit_sumproduct_candidate(
            model, train_x, train_y, val_x, val_y,
            steps=max(0, int(extra_steps)), lr=max(float(lr) * 0.65, 5e-5),
            grad_clip=0.25,
        )
        rel = (prev - float(new_mse)) / max(prev, 1e-18)
        if math.isfinite(new_mse) and new_mse <= prev:
            model, prev = trial, float(new_mse)
        rounds_used += 1
        if rel < float(min_relative_improvement):
            break
    return model, prev, rounds_used


def prune_numeric_structure_to_stability(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    refit_steps: int = 180,
    refit_lr: float = 2e-4,
    stabilize_rounds: int = 3,
    stabilize_steps: int = 80,
    stabilize_min_relative_improvement: float = 1e-3,
    recovery_probe_count: int = 2,
    max_candidates_per_pass: int = 6,
    local_rel_mse_budget: float = 0.01,
    global_rel_mse_budget: float = 0.03,
    min_rules: int = 1,
    prune_optional_factors: bool = True,
    max_accepted_deletions: int = 0,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Physically prune numerical rules/factors until no safe deletion remains.

    Every candidate is tried on a deep copy.  The surviving model is refit and
    then allowed several short stabilization rounds.  A deletion is committed
    only when the stabilized validation MSE respects both a per-deletion budget
    and a cumulative budget relative to the beginning of the pruning pass.
    Destructive trials are discarded exactly.

    Accepted deletions update persistent alive masks, so subsequent hardening or
    fitting cannot reopen a dead rule/factor.
    """
    if float(local_rel_mse_budget) < 0 or float(global_rel_mse_budget) < 0:
        raise ValueError("pruning MSE budgets must be non-negative")
    if int(min_rules) < 0:
        raise ValueError("min_rules must be >= 0")

    current = copy.deepcopy(model)
    current.symbolic_enabled = False
    current.eval()
    with torch.no_grad():
        start_mse = float(torch.mean((current(val_x) - val_y) ** 2).cpu())
    if not math.isfinite(start_mse):
        raise FloatingPointError("non-finite validation MSE before pruning")
    global_cap = start_mse * (1.0 + float(global_rel_mse_budget))

    accepted: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    iteration = 0
    if verbose:
        print("\n=== NUMERICAL PRUNE-TO-STABILITY ===")
        print(
            f"  start val RMSE={math.sqrt(start_mse):.6g}; "
            f"local MSE budget={float(local_rel_mse_budget):.3g}; "
            f"global MSE budget={float(global_rel_mse_budget):.3g}"
        )

    while True:
        if int(max_accepted_deletions) > 0 and len(accepted) >= int(max_accepted_deletions):
            break
        current.eval()
        with torch.no_grad():
            pred, det = current(val_x, return_details=True)
            base_mse = float(torch.mean((pred - val_y) ** 2).cpu())
            contrib_rms = torch.sqrt(det["contributions"].square().mean(dim=0) + 1e-18)
            factor_gate = det["factor_gate_values"].detach()
        local_cap = base_mse * (1.0 + float(local_rel_mse_budget))
        active_rules = torch.nonzero(current.rule_alive_mask, as_tuple=False).squeeze(-1).tolist()
        candidates: List[Tuple[float, str, int, Optional[int], float]] = []

        if len(active_rules) > int(min_rules):
            for r in active_rules:
                trial = copy.deepcopy(current)
                trial.rule_alive_mask[r] = False
                trial.eval()
                with torch.no_grad():
                    mse0 = float(torch.mean((trial(val_x) - val_y) ** 2).cpu())
                candidates.append((mse0, "rule", int(r), None, float(contrib_rms[r].cpu())))

        if bool(prune_optional_factors) and current.max_factors > current.min_order:
            for r in active_rules:
                for slot in range(current.min_order, current.max_factors):
                    if not bool(current.factor_alive_mask[r, slot]):
                        continue
                    trial = copy.deepcopy(current)
                    trial.factor_alive_mask[r, slot] = False
                    trial.eval()
                    with torch.no_grad():
                        mse0 = float(torch.mean((trial(val_x) - val_y) ** 2).cpu())
                    candidates.append((
                        mse0, "factor", int(r), int(slot), float(factor_gate[r, slot].cpu())
                    ))

        if not candidates:
            break
        candidates.sort(key=lambda z: (z[0], 0 if z[1] == "factor" else 1, z[2], -1 if z[3] is None else z[3]))
        iteration += 1
        committed = False

        # Dead branches should have little immediate ablation damage.  Refit all
        # candidates that already lie inside the safety envelope, plus a small
        # number of best destructive candidates to allow redistribution to rescue
        # a deletion.  This avoids spending hundreds of optimizer steps on every
        # obviously destructive branch.
        immediate_cap = min(local_cap, global_cap)
        promising = [z for z in candidates if math.isfinite(z[0]) and z[0] <= immediate_cap]
        probe_ids = {id(z) for z in promising}
        for z in candidates[:max(0, int(recovery_probe_count))]:
            if id(z) not in probe_ids:
                promising.append(z); probe_ids.add(id(z))
        # Preserve the original ranking after unioning the sets and bound the
        # refit frontier. If the least-destructive candidates are all unsafe,
        # worse immediate ablations are not useful pruning proposals.
        promising.sort(key=lambda z: candidates.index(z))
        if int(max_candidates_per_pass) > 0:
            promising = promising[:int(max_candidates_per_pass)]

        bar = tqdm(
            promising, total=len(promising),
            desc=f"Prune pass {iteration}: rollback-safe trials",
            disable=not bool(show_progress), leave=False, dynamic_ncols=True,
            mininterval=0.25,
        )
        for immediate_mse, kind, r, slot, importance in bar:
            trial = copy.deepcopy(current)
            if kind == "rule":
                trial.rule_alive_mask[r] = False
                label = f"rule {r}"
            else:
                assert slot is not None
                trial.factor_alive_mask[r, slot] = False
                label = f"rule {r} factor {slot}"

            # Cheap trial refit first. Only candidates already inside the error
            # budget receive additional stabilization fitting.
            trial, trial_mse = _refit_sumproduct_candidate(
                trial, train_x, train_y, val_x, val_y,
                steps=max(0, int(refit_steps)), lr=float(refit_lr), grad_clip=0.25,
            )
            prelim_safe = (
                math.isfinite(trial_mse)
                and trial_mse <= local_cap
                and trial_mse <= global_cap
            )
            stable_rounds = 0
            if prelim_safe and int(stabilize_rounds) > 0:
                trial, trial_mse, stable_rounds = _stabilize_pruned_candidate(
                    trial, train_x, train_y, val_x, val_y,
                    initial_steps=0, lr=max(float(refit_lr) * 0.65, 5e-5),
                    extra_rounds=int(stabilize_rounds), extra_steps=int(stabilize_steps),
                    min_relative_improvement=float(stabilize_min_relative_improvement),
                )
            safe = (
                math.isfinite(trial_mse)
                and trial_mse <= local_cap
                and trial_mse <= global_cap
            )
            rec = {
                "iteration": iteration, "kind": kind, "rule": int(r),
                "slot": None if slot is None else int(slot),
                "importance": float(importance),
                "base_rmse": math.sqrt(max(base_mse, 0.0)),
                "immediate_rmse": math.sqrt(max(immediate_mse, 0.0)) if math.isfinite(immediate_mse) else float("inf"),
                "refit_rmse": math.sqrt(max(trial_mse, 0.0)) if math.isfinite(trial_mse) else float("inf"),
                "stabilization_rounds": int(stable_rounds), "accepted": bool(safe),
            }
            if safe:
                current = trial
                accepted.append(rec)
                committed = True
                if verbose:
                    print(
                        f"  prune {label}: {rec['base_rmse']:.6g} -> {rec['refit_rmse']:.6g} RMSE; "
                        f"commit after {stable_rounds} stabilization round(s)"
                    )
                break
            rejected.append(rec)
            if verbose and (immediate_mse == candidates[0][0] or len(promising) <= 4):
                print(
                    f"  try {label}: refit RMSE={rec['refit_rmse']:.6g}; destructive -> rollback"
                )
        if not committed:
            if verbose:
                print(
                    f"  no safe deletion among {len(promising)} validation-promising/recovery candidates; "
                    "pruning fixed point reached"
                )
            break

    current.eval()
    with torch.no_grad():
        end_mse = float(torch.mean((current(val_x) - val_y) ** 2).cpu())
    active_optional = 0
    if current.max_factors > current.min_order:
        active_optional = int(
            (current.factor_alive_mask[:, current.min_order:] & current.rule_alive_mask[:, None]).sum()
        )
    summary = {
        "accepted": accepted, "rejected": rejected,
        "start_rmse": math.sqrt(max(start_mse, 0.0)),
        "end_rmse": math.sqrt(max(end_mse, 0.0)),
        "active_rules": int(current.rule_alive_mask.sum()),
        "active_optional_factors": active_optional,
        "global_validation_cap_rmse": math.sqrt(max(global_cap, 0.0)),
        "iterations": iteration,
    }
    if verbose:
        print(
            f"  stable structure: rules={summary['active_rules']}, "
            f"optional factors={summary['active_optional_factors']}, "
            f"val RMSE={summary['end_rmse']:.6g}"
        )
    return current, summary


def in_context_symbolic_rule_gsr(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    library: Optional[Sequence[str]] = None,
    local_topk: int = 3,
    max_rule_candidates: int = 12,
    trial_steps: int = 80,
    lr: float = 2e-4,
    global_rel_mse_budget: float = 0.25,
    require_complete: bool = False,
    verbose: bool = True,
) -> Tuple[SumProductKAN, List[Dict[str, object]]]:
    """Joint in-context symbolic rule conversion for a discretized model.

    A separate copy is converted rule by rule in descending contribution
    importance.  For a pairwise rule, factor operators are proposed locally but
    evaluated as a tuple after a full-network refit.  Local curve fit therefore
    acts only as a proposal mechanism; validation loss of the complete model is
    the commit criterion.
    """
    if not bool(numeric_model.discretized.item()):
        raise ValueError("GSR requires a discretized hard numerical SumProductKAN")
    model = copy.deepcopy(numeric_model)
    model.symbolic_enabled = True
    model.eval()
    with torch.no_grad():
        base_mse = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
        _, det = model(val_x, return_details=True)
        importance = torch.sqrt(det["contributions"].square().mean(dim=0) + 1e-12)
    global_cap = base_mse * (1.0 + max(0.0, float(global_rel_mse_budget)))
    active_rules = torch.nonzero(model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
    active_rules.sort(key=lambda r: float(importance[r]), reverse=True)
    history: List[Dict[str, object]] = []
    if verbose:
        print("\n=== HYBRID SYMBOLIC RULE-GSR ===")
        print(f"  numerical baseline RMSE={math.sqrt(base_mse):.6g}; symbolic global cap={math.sqrt(global_cap):.6g}")

    for pos, r in enumerate(active_rules, 1):
        factors: List[Tuple[int, int]] = []
        for s in range(model.max_factors):
            j = int(model.hard_variable_choice[r, s].item())
            if j != model.in_dim:
                factors.append((s, j))
        if not factors:
            continue

        shortlists: List[List[Dict[str, object]]] = []
        for s, j in factors:
            if not bool(model.hard_spline_choice[r, s, j].item()):
                k = int(model.hard_operator_choice[r, s, j].item())
                shortlists.append([{
                    "name": model.symbolic_library[k],
                    "r2": 1.0,
                    "affine": tuple(float(v) for v in model.symbolic_affine[r, s, j, k].detach().cpu()),
                }])
            else:
                cand = shortlist_symbolic_edge(
                    model, train_x, r, s, j, library=library, topk=local_topk
                )
                if not cand:
                    cand = [{"name": "x", "r2": -float("inf"), "affine": (1.0, 1.0, 0.0, 0.0)}]
                cand = [
                    _refine_symbolic_affine_local(model, train_x, r, s, j, z)
                    for z in cand
                ]
                shortlists.append(cand)

        combos = list(itertools.product(*shortlists))[: max(1, int(max_rule_candidates))]
        best_trial: Optional[SumProductKAN] = None
        best_combo = None
        best_mse = float("inf")
        for combo in combos:
            trial = copy.deepcopy(model)
            trial.symbolic_enabled = True
            with torch.no_grad():
                for (s, j), c in zip(factors, combo):
                    name = str(c["name"])
                    if name not in trial.symbolic_library:
                        continue
                    k = trial.symbolic_library.index(name)
                    trial.hard_spline_choice[r, s, j] = False
                    trial.hard_operator_choice[r, s, j] = int(k)
                    trial.symbolic_affine[r, s, j, k] = torch.tensor(
                        c["affine"], device=trial.symbolic_affine.device,
                        dtype=trial.symbolic_affine.dtype,
                    )
            trial, mse = _refit_sumproduct_candidate(
                trial, train_x, train_y, val_x, val_y,
                steps=trial_steps, lr=lr, grad_clip=0.20,
            )
            if mse < best_mse:
                best_mse, best_trial, best_combo = mse, trial, combo

        accepted = best_trial is not None and (best_mse <= global_cap or require_complete)
        if accepted:
            model = best_trial
        names = [] if best_combo is None else [str(c["name"]) for c in best_combo]
        record = {
            "rule": int(r), "factors": factors, "operators": names,
            "validation_rmse": math.sqrt(best_mse) if math.isfinite(best_mse) else float("inf"),
            "accepted": bool(accepted),
        }
        history.append(record)
        if verbose:
            desc = " * ".join(f"{name}(x{j})" for (_, j), name in zip(factors, names)) or "FAILED"
            print(
                f"  rule {pos}/{len(active_rules)} id={r}: {desc}; "
                f"val RMSE={record['validation_rmse']:.6g}; {'commit' if accepted else 'keep spline'}"
            )
    return model, history


def hard_numeric_plateau_polish(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    rounds: int = 4,
    steps_per_round: int = 350,
    lr: float = 3e-4,
    min_relative_improvement: float = 2e-3,
    patience: int = 2,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, List[float]]:
    """Keep fitting the exact hard numerical structure until validation plateaus.

    Structural, rank/order, and symbolic gates are frozen.  Only the numerical
    KAN functions plus continuous readout scales/bias adapt.  This directly
    implements the requested "learn a bit longer until it stably converges"
    without giving the model a chance to escape into a new series expansion.
    """
    model = copy.deepcopy(model)
    model.symbolic_enabled = False
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    gate_params = [
        model.variable_logits, model.rule_gate.log_alpha, model.factor_gate.log_alpha,
        model.operator_logits, model.spline_logits, model.symbolic_affine,
    ]
    old_flags = [(p, p.requires_grad) for p in model.parameters()]
    gate_ids = {id(p) for p in gate_params}
    for p, flag in old_flags:
        if id(p) in gate_ids:
            p.requires_grad_(False)
    vals: List[float] = []
    stale = 0
    best_global = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        best_val = float(torch.sqrt(torch.mean((model(val_x) - val_y) ** 2)).cpu())
    vals.append(best_val)
    if verbose:
        print(f"\n=== HARD NUMERICAL PLATEAU POLISH ===\n  start val RMSE={best_val:.6g}")
    for rr in _progress_range(
        rounds, desc="Hard numerical plateau", enabled=show_progress, leave=True,
    ):
        before = best_val
        model, _ = _refit_sumproduct_candidate(
            model, train_x, train_y, val_x, val_y,
            steps=steps_per_round, lr=lr, grad_clip=0.25,
        )
        model.eval()
        with torch.no_grad():
            now = float(torch.sqrt(torch.mean((model(val_x) - val_y) ** 2)).cpu())
        if now < best_val:
            best_val = now
            best_global = copy.deepcopy(model.state_dict())
        vals.append(now)
        rel = max(0.0, (before - now) / max(before, 1e-12))
        if verbose:
            print(f"  round {rr+1}/{rounds}: val RMSE={now:.6g}, relative improvement={rel:.3g}")
        if rel < float(min_relative_improvement):
            stale += 1
        else:
            stale = 0
        if stale >= int(patience):
            if verbose:
                print("  plateau detected; stop")
            break
    model.load_state_dict(best_global)
    # Keep gate flags frozen in returned plateau model; discretize/GSR expects
    # structure to remain fixed. Other continuous parameters retain their flags.
    for p, flag in old_flags:
        if id(p) not in gate_ids:
            p.requires_grad_(flag)
    return model, vals


def hard_symbolic_plateau_polish(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    rounds: int = 3,
    steps_per_round: int = 220,
    lr: float = 1.5e-4,
    min_relative_improvement: float = 1e-3,
    patience: int = 2,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, List[float]]:
    """Final fixed-structure polish for a mixed/fully symbolic hard model."""
    model = copy.deepcopy(model)
    model.symbolic_enabled = True
    if not bool(model.discretized.item()):
        raise ValueError("symbolic plateau polish requires a discretized model")
    # Keep all discrete gates frozen; allow symbolic affine parameters, rule
    # scales/bias, and any intentionally retained numerical spline edges to fit.
    gate_params = [
        model.variable_logits, model.rule_gate.log_alpha, model.factor_gate.log_alpha,
        model.operator_logits, model.spline_logits,
    ]
    gate_ids = {id(p) for p in gate_params}
    old_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in old_flags:
        if id(p) in gate_ids:
            p.requires_grad_(False)
    model.symbolic_affine.requires_grad_(True)

    vals: List[float] = []
    stale = 0
    model.eval()
    with torch.no_grad():
        best_val = float(torch.sqrt(torch.mean((model(val_x) - val_y) ** 2)).cpu())
    best = copy.deepcopy(model.state_dict())
    vals.append(best_val)
    if verbose:
        print(f"\n=== SYMBOLIC FIXED-STRUCTURE PLATEAU POLISH ===\n  start val RMSE={best_val:.6g}")
    for rr in _progress_range(
        rounds, desc="Hard symbolic plateau", enabled=show_progress, leave=True,
    ):
        before = best_val
        model, _ = _refit_sumproduct_candidate(
            model, train_x, train_y, val_x, val_y,
            steps=steps_per_round, lr=lr, grad_clip=0.15,
        )
        model.eval()
        with torch.no_grad():
            now = float(torch.sqrt(torch.mean((model(val_x) - val_y) ** 2)).cpu())
        if now < best_val:
            best_val = now
            best = copy.deepcopy(model.state_dict())
        vals.append(now)
        rel = max(0.0, (before - now) / max(before, 1e-12))
        if verbose:
            print(f"  round {rr+1}/{rounds}: val RMSE={now:.6g}, relative improvement={rel:.3g}")
        stale = stale + 1 if rel < float(min_relative_improvement) else 0
        if stale >= int(patience):
            if verbose:
                print("  symbolic plateau detected; stop")
            break
    model.load_state_dict(best)
    return model, vals


def compact_sumproduct_for_symbolic(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    min_rules: int = 3,
    relative_mse_budget: float = 0.35,
    refit_steps: int = 180,
    lr: float = 2e-4,
    candidate_pool: int = 5,
    verbose: bool = True,
) -> Tuple[SumProductKAN, List[Dict[str, object]]]:
    """Compact a symbolic-only copy before operator conversion.

    This helper can use a looser validation budget than the protected numerical
    checkpoint because the copy is optimized for interpretability.  After every
    accepted deletion, all remaining continuous factors are refit in context.
    """
    if not bool(numeric_model.discretized.item()):
        raise ValueError("symbolic compaction requires a discretized numerical model")
    model = copy.deepcopy(numeric_model)
    model.symbolic_enabled = True
    model.eval()
    with torch.no_grad():
        base_mse = float(torch.mean((model(val_x) - val_y) ** 2).cpu())
    cap = base_mse * (1.0 + max(0.0, float(relative_mse_budget)))
    history: List[Dict[str, object]] = []
    if verbose:
        print("\n=== SYMBOLIC-COPY IN-CONTEXT COMPACTION ===")
        print(
            f"  start rules={int(model.hard_rule_choice.sum())}, "
            f"RMSE={math.sqrt(base_mse):.6g}, cap={math.sqrt(cap):.6g}"
        )

    while int(model.hard_rule_choice.sum()) > int(min_rules):
        model.eval()
        with torch.no_grad():
            _, det = model(val_x, return_details=True)
            rms = torch.sqrt(det["contributions"].square().mean(dim=0) + 1e-12)
        active = torch.nonzero(model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        candidates = sorted(active, key=lambda r: float(rms[r]))[: max(1, int(candidate_pool))]
        best_trial = None
        best_rule = None
        best_mse = float("inf")
        for r in candidates:
            trial = copy.deepcopy(model)
            trial.hard_rule_choice[r] = False
            trial, mse = _refit_sumproduct_candidate(
                trial, train_x, train_y, val_x, val_y,
                steps=refit_steps, lr=lr, grad_clip=0.20,
            )
            if mse < best_mse:
                best_trial, best_rule, best_mse = trial, int(r), mse
        accepted = best_trial is not None and best_mse <= cap
        history.append({
            "removed_rule": best_rule,
            "validation_rmse": math.sqrt(best_mse) if math.isfinite(best_mse) else float("inf"),
            "accepted": bool(accepted),
        })
        if not accepted:
            if verbose:
                print(
                    f"  stop at {int(model.hard_rule_choice.sum())} rules; best deletion "
                    f"RMSE={math.sqrt(best_mse):.6g} exceeds cap"
                )
            break
        model = best_trial
        if verbose:
            print(
                f"  remove rule {best_rule}: rules={int(model.hard_rule_choice.sum())}, "
                f"val RMSE={math.sqrt(best_mse):.6g}"
            )
    return model, history


def _refine_symbolic_affine_local(
    model: SumProductKAN,
    x: torch.Tensor,
    rule: int,
    slot: int,
    variable: int,
    candidate: Dict[str, object],
    *,
    steps: int = 80,
    lr: float = 0.03,
) -> Dict[str, object]:
    """Refine a shortlist atom against the learned KAN edge before GSR trial."""
    name = str(candidate["name"])
    if name not in SYMBOLIC_LIB:
        return candidate
    xx, yy = _sumproduct_edge_curve(model, x, rule, slot, variable)
    if xx.numel() > 512:
        ids = torch.linspace(0, xx.numel() - 1, 512, device=xx.device).long()
        xx, yy = xx[ids], yy[ids]
    # Identity has an exact closed-form affine fit.  Using iterative Adam here
    # introduced avoidable error in the exact x_i*x_j regression test and made
    # GSR sensitive to the local-refine step budget.
    if name == "x":
        xm = xx.mean(); ym = yy.mean()
        xc = xx - xm; yc = yy - ym
        slope = (xc * yc).sum() / xc.square().sum().clamp_min(1e-12)
        intercept = ym - slope * xm
        out = dict(candidate)
        out["affine"] = (1.0, float(slope.detach().cpu()), 0.0, float(intercept.detach().cpu()))
        return out

    theta = torch.tensor(
        candidate["affine"], device=xx.device, dtype=xx.dtype, requires_grad=True
    )
    opt = torch.optim.Adam([theta], lr=float(lr))
    fun = SYMBOLIC_LIB[name][0]
    best = theta.detach().clone()
    best_mse = float("inf")
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        a, b, c, d = theta.unbind()
        z = b * xx + c
        try:
            g = fun(z)
        except Exception:
            break
        g = torch.nan_to_num(g, nan=0.0, posinf=1e5, neginf=-1e5).clamp(-1e5, 1e5)
        pred = a * g + d
        loss = torch.mean((pred - yy) ** 2)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 10.0)
        opt.step()
        with torch.no_grad():
            theta[0].clamp_(-1e3, 1e3)
            theta[1].clamp_(-8.0, 8.0)
            theta[2].clamp_(-4.0 * math.pi, 4.0 * math.pi)
            theta[3].clamp_(-1e3, 1e3)
            lm = float(loss.detach().cpu())
            if lm < best_mse:
                best_mse = lm
                best = theta.detach().clone()
    out = dict(candidate)
    out["affine"] = tuple(float(v) for v in best.cpu())
    return out


# ----------------------------------------------------------------------
# Mandatory fully-symbolic search and high-accuracy numerical polishing
# ----------------------------------------------------------------------

def _freeze_for_hard_numeric_polish(model: SumProductKAN) -> None:
    """Freeze every structural/symbolic parameter; keep numerical KAN/readout trainable."""
    model.symbolic_enabled = False
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    for p in (
        model.variable_logits,
        model.rule_gate.log_alpha,
        model.factor_gate.log_alpha,
        model.operator_logits,
        model.spline_logits,
        model.symbolic_affine,
    ):
        p.requires_grad_(False)
    for p in model.numeric_edges.parameters():
        p.requires_grad_(True)
    model.rule_scale.requires_grad_(True)
    model.bias.requires_grad_(True)


def refine_sumproduct_numeric_grid(
    model: SumProductKAN,
    train_x: torch.Tensor,
    new_grid: int,
) -> SumProductKAN:
    """Increase numerical basis resolution while preserving the learned function.

    B-spline RuleKAN uses the native KAN parent-grid interpolation. RuleKAN-RBF
    creates a denser Gaussian-RBF bank and transfers each learned univariate
    residual by least squares on the supplied training inputs. Structural gates
    remain frozen in both cases.
    """
    new_grid = int(new_grid)
    if new_grid <= int(model.numeric_edges.num):
        return copy.deepcopy(model)
    out = copy.deepcopy(model)
    parent = out.numeric_edges

    if isinstance(parent, FastRBFEdgeBank):
        device = train_x.device
        lo, hi = parent.grid_range
        child = FastRBFEdgeBank(
            in_dim=parent.in_dim,
            out_dim=parent.out_dim,
            num=new_grid,
            grid_range=[lo, hi],
            base_fun=parent.base_fun,
            noise_scale=0.0,
            scale_base_mu=0.0,
            scale_base_sigma=0.0,
            scale_sp=0.0,
            train_grid=parent.basis.grid.requires_grad,
            train_width=parent.basis.log_denominator.requires_grad,
            width_scale=getattr(out, "rbf_width_scale", 1.0),
            sp_trainable=parent.scale_sp.requires_grad,
            sb_trainable=parent.scale_base.requires_grad,
            device=str(device),
        ).to(device)
        with torch.no_grad():
            # Keep the parent's learned Gaussian width during basis expansion.
            # Narrowing the width merely because more centres are introduced can
            # make the old coarse RBF span unnecessarily hard to reproduce.
            child.basis.log_denominator.copy_(parent.basis.log_denominator)
            # Transfer the raw nonlinear component.  Base path, masks and edge
            # scales are copied exactly, so only the RBF expansion is refit.
            target = parent.raw_basis_component(train_x)  # [B,I,O]
            phi = child.basis(train_x)                    # [B,I,Gnew]
            for i in range(parent.in_dim):
                sol = torch.linalg.lstsq(phi[:, i, :], target[:, i, :]).solution
                # lstsq returns [Gnew,O]
                child.coef[i].copy_(sol.T)
            child.scale_base.copy_(parent.scale_base)
            child.scale_sp.copy_(parent.scale_sp)
            child.mask.copy_(parent.mask)
        out.numeric_edges = child
        _freeze_for_hard_numeric_polish(out)
        return out

    device = parent.grid.device
    lo = float(parent.grid[:, parent.k].min().detach().cpu())
    hi = float(parent.grid[:, -parent.k-1].max().detach().cpu())
    child = KANLayer(
        in_dim=parent.in_dim,
        out_dim=parent.out_dim,
        num=new_grid,
        k=parent.k,
        noise_scale=0.0,
        scale_base_mu=0.0,
        scale_base_sigma=0.0,
        scale_sp=0.0,
        base_fun=parent.base_fun,
        grid_eps=parent.grid_eps,
        grid_range=[lo, hi],
        sp_trainable=parent.scale_sp.requires_grad,
        sb_trainable=parent.scale_base.requires_grad,
        save_plot_data=getattr(parent, "save_plot_data", True),
        device=str(device),
    ).to(device)
    with torch.no_grad():
        child.initialize_grid_from_parent(parent, train_x)
        child.scale_base.copy_(parent.scale_base)
        child.scale_sp.copy_(parent.scale_sp)
        child.mask.copy_(parent.mask)
    out.numeric_edges = child
    _freeze_for_hard_numeric_polish(out)
    return out


def hard_numeric_precision_polish(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    target_rmse: float = 1e-4,
    grid_schedule: Sequence[int] = (24, 32),
    adam_steps_per_grid: int = 800,
    adam_lr: float = 3e-4,
    lbfgs_steps: int = 250,
    extra_final_rounds: int = 4,
    min_relative_improvement: float = 2e-4,
    patience: int = 2,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, List[Dict[str, float]]]:
    """Refine a fixed hard numerical model beyond a coarse-grid plateau.

    Small Adam improvements on the current grid can indicate a resolution or
    optimizer plateau rather than convergence of the target function.  The
    routine refines the univariate spline grid, polishes with Adam, and finishes
    with full-batch strong-Wolfe LBFGS.  Structural gates never reopen.
    """
    cur = copy.deepcopy(model)
    _freeze_for_hard_numeric_polish(cur)
    history: List[Dict[str, float]] = []

    def vrmse(m):
        m.eval()
        with torch.no_grad():
            return float(torch.sqrt(torch.mean((m(val_x) - val_y) ** 2)).cpu())

    best_val = vrmse(cur)
    best = copy.deepcopy(cur.state_dict())
    history.append({"grid": float(cur.numeric_edges.num), "val_rmse": best_val})
    if verbose:
        print("\n=== HARD NUMERICAL PRECISION POLISH ===")
        print(f"  start grid={cur.numeric_edges.num}, val RMSE={best_val:.8g}, target={target_rmse:.3g}")

    grid_values = tuple(int(g) for g in grid_schedule)
    for grid in tqdm(
        grid_values, total=len(grid_values), desc="Numeric grid refinement",
        disable=not show_progress, leave=True, dynamic_ncols=True, mininterval=0.25,
    ):
        if best_val <= float(target_rmse):
            break
        before_model = copy.deepcopy(cur)
        before_val = best_val
        cur = refine_sumproduct_numeric_grid(cur, train_x, grid)
        # Verify interpolation itself is not destructive; rollback if it is.
        interp_val = vrmse(cur)
        if not math.isfinite(interp_val) or interp_val > max(before_val * 1.5, before_val + 1e-4):
            cur = before_model
            if verbose:
                print(f"  grid {grid}: interpolation rejected ({interp_val:.6g} vs {before_val:.6g})")
            continue
        params = [p for p in cur.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=float(adam_lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(adam_steps_per_grid)), eta_min=max(float(adam_lr) * 0.03, 2e-6)
        )
        local_best = copy.deepcopy(cur.state_dict())
        local_val = interp_val
        for it in _progress_range(
            adam_steps_per_grid, desc=f"Grid {grid} Adam polish",
            enabled=show_progress, leave=False,
        ):
            cur.train(); opt.zero_grad(set_to_none=True)
            loss = torch.mean((cur(train_x) - train_y) ** 2)
            if not torch.isfinite(loss):
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 0.25); opt.step(); sched.step()
            if it % 40 == 0 or it == int(adam_steps_per_grid) - 1:
                vv = vrmse(cur)
                if vv < local_val:
                    local_val = vv; local_best = copy.deepcopy(cur.state_dict())
        cur.load_state_dict(local_best)
        best_val = local_val
        best = copy.deepcopy(cur.state_dict())
        history.append({"grid": float(grid), "val_rmse": best_val})
        if verbose:
            print(f"  grid {grid}: val RMSE={best_val:.8g}")

    # If resolution is no longer the bottleneck, keep optimizing the final
    # dense grid until validation improvement itself plateaus rather than
    # treating a coarse-grid plateau as convergence.
    stale = 0
    for rr in _progress_range(
        extra_final_rounds, desc="Final-grid Adam rounds",
        enabled=show_progress, leave=True,
    ):
        if best_val <= float(target_rmse):
            break
        before = best_val
        cur.load_state_dict(best)
        params = [p for p in cur.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=max(float(adam_lr) * 0.55, 5e-5))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(adam_steps_per_grid)), eta_min=2e-6
        )
        local_best = copy.deepcopy(cur.state_dict()); local_val = best_val
        for it in _progress_range(
            adam_steps_per_grid, desc=f"Final-grid round {rr+1}",
            enabled=show_progress, leave=False,
        ):
            cur.train(); opt.zero_grad(set_to_none=True)
            loss = torch.mean((cur(train_x) - train_y) ** 2)
            if not torch.isfinite(loss): break
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 0.20); opt.step(); sched.step()
            if it % 40 == 0 or it == int(adam_steps_per_grid) - 1:
                vv = vrmse(cur)
                if vv < local_val: local_val = vv; local_best = copy.deepcopy(cur.state_dict())
        cur.load_state_dict(local_best)
        if local_val < best_val: best_val = local_val; best = copy.deepcopy(cur.state_dict())
        rel = max(0.0, (before - best_val) / max(before, 1e-12))
        history.append({"grid": float(cur.numeric_edges.num), "val_rmse": best_val})
        if verbose:
            print(f"  final-grid round {rr+1}/{extra_final_rounds}: val RMSE={best_val:.8g}, rel={rel:.3g}")
        stale = stale + 1 if rel < float(min_relative_improvement) else 0
        if stale >= int(patience):
            if verbose: print("  final-grid numerical plateau detected")
            break

    if int(lbfgs_steps) > 0 and best_val > float(target_rmse):
        cur.load_state_dict(best)
        params = [p for p in cur.parameters() if p.requires_grad]
        opt = torch.optim.LBFGS(
            params, lr=0.5, max_iter=int(lbfgs_steps), max_eval=int(lbfgs_steps * 1.5),
            history_size=50, line_search_fn="strong_wolfe",
            tolerance_grad=1e-10, tolerance_change=1e-12,
        )
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((cur(train_x) - train_y) ** 2)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        try:
            opt.step(closure)
            vv = vrmse(cur)
            if math.isfinite(vv) and vv < best_val:
                best_val = vv; best = copy.deepcopy(cur.state_dict())
        except Exception as exc:
            if verbose:
                print(f"  LBFGS polish skipped after optimizer failure: {exc}")
        cur.load_state_dict(best)
        history.append({"grid": float(cur.numeric_edges.num), "val_rmse": best_val})
        if verbose:
            print(f"  LBFGS: val RMSE={best_val:.8g}")

    return cur, history


def _make_fully_symbolic_shell(numeric_model: SumProductKAN) -> SumProductKAN:
    """Copy hard structure but forbid all spline use in the predictive model."""
    if not bool(numeric_model.discretized.item()):
        raise ValueError("fully symbolic takeover requires a discretized numerical model")
    m = copy.deepcopy(numeric_model)
    m.symbolic_enabled = True
    m.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=1.0)
    # Numerical physical pruning must not constrain the independent symbolic
    # structure bank. Re-open all symbolic rule/factor slots before pursuit.
    m.rule_alive_mask.fill_(True)
    m.factor_alive_mask.fill_(True)
    # Start with no active rules. Matching pursuit must earn every symbolic term.
    m.hard_rule_choice.zero_()
    # Every possible selected edge is symbolic. There is no fallback path.
    m.hard_spline_choice.zero_()
    # Numeric KANs are frozen permanently for this copy.
    for p in m.numeric_edges.parameters():
        p.requires_grad_(False)
    for p in (m.variable_logits, m.rule_gate.log_alpha, m.factor_gate.log_alpha,
              m.operator_logits, m.spline_logits):
        p.requires_grad_(False)
    m.symbolic_affine.requires_grad_(True)
    m.rule_scale.requires_grad_(True)
    m.bias.requires_grad_(True)
    with torch.no_grad():
        m.bias.copy_(torch.zeros_like(m.bias))
    return m


def _candidate_rule_value(
    model: SumProductKAN,
    x: torch.Tensor,
    rule: int,
    factors: Sequence[Tuple[int, int]],
    combo: Sequence[Dict[str, object]],
) -> torch.Tensor:
    """Evaluate one proposed symbolic product rule without mutating the model."""
    out = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    for (slot, var), cand in zip(factors, combo):
        name = str(cand["name"])
        fun = SYMBOLIC_LIB[name][0]
        a, b, c, d = [torch.as_tensor(v, device=x.device, dtype=x.dtype) for v in cand["affine"]]
        z = b * x[:, int(var)] + c
        g = fun(z)
        if not torch.isfinite(g).all():
            return torch.full_like(out, float("nan"))
        out = out * (a * g + d)
    return out


def _install_symbolic_template(
    model: SumProductKAN,
    rule: int,
    factors: Sequence[Tuple[int, int]],
    combo: Sequence[Dict[str, object]],
    initial_scale: Optional[float] = None,
) -> None:
    with torch.no_grad():
        rr = int(rule)
        model.hard_rule_choice[rr] = True
        # A symbolic template owns the complete factor structure of its rule.
        # Reset every slot to identity first, then install the requested factors.
        # This also makes alternate/self-product structures safe to propose.
        model.hard_variable_choice[rr, :].fill_(model.in_dim)
        for (slot, var), cand in zip(factors, combo):
            ss, jj = int(slot), int(var)
            model.hard_variable_choice[rr, ss] = jj
            name = str(cand["name"])
            k = model.symbolic_library.index(name)
            model.hard_spline_choice[rr, ss, jj] = False
            model.hard_operator_choice[rr, ss, jj] = int(k)
            model.symbolic_affine[rr, ss, jj, int(k)] = torch.tensor(
                _canonical_symbolic_affine(cand["affine"], name),
                device=model.symbolic_affine.device, dtype=model.symbolic_affine.dtype
            )
        if initial_scale is not None and math.isfinite(float(initial_scale)):
            model.rule_scale[rr] = float(initial_scale)


def _fully_symbolic_continuous_refit(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    steps: int = 500,
    lr: float = 3e-4,
    lbfgs_steps: int = 0,
    frozen_symbolic_factors: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    complementary_gate_pairs: Optional[Sequence[
        Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]
    ]] = None,
    show_progress: bool = False,
    progress_desc: str = "Fully symbolic refit",
) -> Tuple[SumProductKAN, float]:
    """Global refit with *only* symbolic affine constants/readout trainable."""
    m = copy.deepcopy(model)
    m.symbolic_enabled = True
    m.hard_spline_choice.zero_()
    for p in m.numeric_edges.parameters(): p.requires_grad_(False)
    for p in (m.variable_logits, m.rule_gate.log_alpha, m.factor_gate.log_alpha,
              m.operator_logits, m.spline_logits): p.requires_grad_(False)
    m.symbolic_affine.requires_grad_(True); m.rule_scale.requires_grad_(True); m.bias.requires_grad_(True)
    with torch.no_grad():
        m.symbolic_affine[..., 0].fill_(1.0)
        m.symbolic_affine[..., 3].zero_()
    frozen_symbolic_factors = tuple(frozen_symbolic_factors or ())
    complementary_gate_pairs = tuple(complementary_gate_pairs or ())
    frozen_values = {}
    with torch.no_grad():
        for rr, ss, jj, kk in frozen_symbolic_factors:
            key = (int(rr), int(ss), int(jj), int(kk))
            frozen_values[key] = m.symbolic_affine[key].detach().clone()

    def _freeze_symbolic_affine_grad() -> None:
        if m.symbolic_affine.grad is None:
            return
        m.symbolic_affine.grad[..., 0].zero_()
        m.symbolic_affine.grad[..., 3].zero_()
        for comp_key0, base_key0 in complementary_gate_pairs:
            comp_key = tuple(int(v) for v in comp_key0)
            base_key = tuple(int(v) for v in base_key0)
            g_comp = m.symbolic_affine.grad[comp_key]
            g_base = m.symbolic_affine.grad[base_key]
            # If complement(z)=1-base(z), the shared chart derivatives are
            # dL/dbeta = g_base_beta - g_comp_beta and
            # dL/dgamma = g_base_gamma - g_comp_gamma.  Split that tied
            # gradient symmetrically across the two tensor entries.
            beta_grad = 0.5 * (g_base[1] - g_comp[1])
            gamma_grad = 0.5 * (g_base[2] - g_comp[2])
            g_base[1] = beta_grad
            g_comp[1] = -beta_grad
            g_base[2] = gamma_grad
            g_comp[2] = -gamma_grad
        for key in frozen_values:
            m.symbolic_affine.grad[key].zero_()

    def _restore_frozen_symbolic_affines() -> None:
        with torch.no_grad():
            m.symbolic_affine[..., 0].fill_(1.0)
            m.symbolic_affine[..., 1].clamp_(-10.0, 10.0)
            m.symbolic_affine[..., 2].clamp_(-6.0 * math.pi, 6.0 * math.pi)
            m.symbolic_affine[..., 3].zero_()
            for comp_key0, base_key0 in complementary_gate_pairs:
                comp_key = tuple(int(v) for v in comp_key0)
                base_key = tuple(int(v) for v in base_key0)
                beta = 0.5 * (
                    m.symbolic_affine[base_key][1] - m.symbolic_affine[comp_key][1]
                )
                gamma = 0.5 * (
                    m.symbolic_affine[base_key][2] + 1.0 - m.symbolic_affine[comp_key][2]
                )
                m.symbolic_affine[base_key][1] = beta
                m.symbolic_affine[base_key][2] = gamma
                m.symbolic_affine[comp_key][1] = -beta
                m.symbolic_affine[comp_key][2] = 1.0 - gamma
            for key, value in frozen_values.items():
                m.symbolic_affine[key].copy_(value)
    _restore_frozen_symbolic_affines()
    params = [p for p in m.parameters() if p.requires_grad]
    m.eval()
    with torch.no_grad(): best_val = float(torch.mean((m(val_x)-val_y)**2).cpu())
    best = copy.deepcopy(m.state_dict())
    if int(steps) > 0:
        opt = torch.optim.Adam(params, lr=float(lr))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,int(steps)), eta_min=max(float(lr)*0.03,2e-6))
        for it in _progress_range(
            steps, desc=progress_desc, enabled=show_progress, leave=False,
        ):
            m.train(); opt.zero_grad(set_to_none=True)
            loss = torch.mean((m(train_x)-train_y)**2)
            if not torch.isfinite(loss): break
            loss.backward()
            _freeze_symbolic_affine_grad()
            torch.nn.utils.clip_grad_norm_(params,0.15); opt.step(); sched.step()
            # Canonical symbolic factors are pure functions g(beta*x+gamma).
            # Rule scale absorbs all output amplitude and the model bias absorbs
            # constants; factor offsets would create lower-order leakage inside
            # products.
            _restore_frozen_symbolic_affines()
            if it % 20 == 0 or it == int(steps)-1:
                m.eval()
                with torch.no_grad(): vv=float(torch.mean((m(val_x)-val_y)**2).cpu())
                if math.isfinite(vv) and vv < best_val: best_val=vv; best=copy.deepcopy(m.state_dict())
    m.load_state_dict(best)
    if int(lbfgs_steps)>0:
        opt2=torch.optim.LBFGS(params,lr=0.35,max_iter=int(lbfgs_steps),max_eval=int(lbfgs_steps*1.5),history_size=40,line_search_fn="strong_wolfe",tolerance_grad=1e-10,tolerance_change=1e-12)
        def closure():
            _restore_frozen_symbolic_affines()
            opt2.zero_grad(set_to_none=True); loss=torch.mean((m(train_x)-train_y)**2)
            if torch.isfinite(loss):
                loss.backward()
                _freeze_symbolic_affine_grad()
            return loss
        try:
            opt2.step(closure)
            _restore_frozen_symbolic_affines()
            m.eval()
            with torch.no_grad(): vv=float(torch.mean((m(val_x)-val_y)**2).cpu())
            if math.isfinite(vv) and vv < best_val: best_val=vv; best=copy.deepcopy(m.state_dict())
        except Exception:
            pass
        m.load_state_dict(best)
    return m, best_val




def symbolic_structure_bank(
    in_dim: int,
    max_factors: int = 2,
    *,
    allow_self_products: bool = True,
) -> List[Tuple[int, ...]]:
    """Enumerate symbolic dependency structures independently of spline rules.

    Numerical KAN training need not contain ``(x_j, x_j)`` product nodes: for
    expressive splines that decomposition is unidentifiable.  The symbolic
    phase gets its own structure bank, where repeated variables are meaningful
    because each slot must choose a restricted symbolic operator.
    """
    d = int(in_dim)
    S = max(1, int(max_factors))
    structures: List[Tuple[int, ...]] = [(j,) for j in range(d)]
    if S >= 2:
        if allow_self_products:
            structures.extend(tuple(z) for z in itertools.combinations_with_replacement(range(d), 2))
        else:
            structures.extend(tuple(z) for z in itertools.combinations(range(d), 2))
    # General higher-order fallback.  Orders >2 are opt-in through max_factors;
    # lower orders remain in the bank so matching pursuit can pay only for the
    # complexity it actually needs.
    for order in range(3, S + 1):
        iterator = (
            itertools.combinations_with_replacement(range(d), order)
            if allow_self_products else itertools.combinations(range(d), order)
        )
        structures.extend(tuple(z) for z in iterator)
    # Preserve order while removing accidental duplicates (unary entries are
    # intentionally distinct from (j,j)).
    seen = set(); out = []
    for z in structures:
        if z not in seen:
            seen.add(z); out.append(z)
    return out


def _gmp_safe_operator_value(
    name: str,
    x: torch.Tensor,
    affine: torch.Tensor,
    *,
    gradient_scale: float = 6.0,
) -> torch.Tensor:
    """Exact-forward symbolic atom with asinh-compressed backward gradients."""
    fun = SYMBOLIC_LIB[name][0]
    a, b, c, d = affine.unbind()
    z = b * x + c
    raw = fun(z)
    if not torch.isfinite(raw).all():
        return torch.full_like(raw, float("nan"))
    raw = a * raw + d
    if gradient_scale > 0:
        gs = torch.as_tensor(float(gradient_scale), device=x.device, dtype=x.dtype)
        stable = gs * torch.asinh(raw / gs)
        raw = raw.detach() + stable - stable.detach()
    return raw


def scaled_gmp_screening_sizes(
    library_size: int,
    *,
    topk: Optional[int] = None,
    unary_topk: Optional[int] = None,
    self_product_topk: Optional[int] = None,
    reference_library_size: int = 11,
    reference_topk: int = 3,
    reference_unary_topk: int = 5,
    reference_self_product_topk: int = 4,
) -> Tuple[int, int, int]:
    """Scale GMP screening width with symbolic-library size.

    The reference screening budget is 3/5/4 candidates for a
    compact 11-atom library (generic/unary/self-product).  When the library is
    enlarged, keeping those absolute top-k values silently reduces the fraction
    of operators that survive GMP.  Unspecified values are therefore scaled
    proportionally and rounded *up* so that the retained fraction never gets
    smaller solely because the vocabulary grew.  Explicit user overrides are
    preserved.
    """
    K = max(1, int(library_size))
    refK = max(1, int(reference_library_size))

    def resolve(value: Optional[int], base: int) -> int:
        if value is not None:
            return min(K, max(1, int(value)))
        return min(K, max(1, int(math.ceil(float(base) * K / refK))))

    k = resolve(topk, reference_topk)
    uk = max(k, resolve(unary_topk, reference_unary_topk))
    sk = max(k, resolve(self_product_topk, reference_self_product_topk))
    return int(k), int(uk), int(sk)


def _local_torch_generator(device: torch.device, seed: int) -> torch.Generator:
    """Return a CPU-local RNG independent of PyTorch's global RNG state.

    Symbolic proposal random tensors are tiny. Keeping their RNG on CPU gives
    deterministic seeds across CPU/CUDA/MPS and avoids backend-specific
    generator limitations; helpers transfer the sampled tensors to the target
    device before use.
    """
    _ = device
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _rng_randn(shape, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, device="cpu", dtype=dtype, generator=generator).to(device=device)


def _rng_rand(shape, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.rand(shape, device="cpu", dtype=dtype, generator=generator).to(device=device)


def _structure_rng_seed(base_seed: int, structure: Sequence[int]) -> int:
    """Stable order-independent RNG seed for one symbolic variable structure.

    GMP used to consume a single generator sequentially across the structure
    bank.  That meant merely reordering otherwise-identical candidate
    structures changed every later structure's random logits.  Give each
    structure its own deterministic stream instead, keyed only by the symbolic
    seed and the variable tuple.
    """
    # 64-bit FNV-1a style mixing, restricted to the non-negative signed range
    # accepted uniformly by torch.Generator.manual_seed.
    mask = (1 << 63) - 1
    h = (int(base_seed) ^ 0xCBF29CE484222325) & mask
    for value in (len(tuple(structure)), *tuple(int(v) for v in structure)):
        h ^= (int(value) + 0x9E3779B9) & mask
        h = (h * 0x100000001B3) & mask
    return int(h)



def _gmp_normalized_curvature(atoms: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Dimensionless finite-difference curvature for GMP operator selection.

    The returned value is zero for an affine atom up to numerical precision.
    It is detached by callers when used as a categorical selection prior, so
    the prior changes *which family* GMP prefers without flattening the
    continuous parameters of nonlinear atoms.
    """
    if atoms.ndim != 2 or atoms.shape[0] < 5:
        return torch.zeros(atoms.shape[-1], device=atoms.device, dtype=atoms.dtype)
    order = torch.argsort(x)
    xs = x[order]
    ys = atoms[order]
    dx = (xs[1:] - xs[:-1]).clamp_min(1e-5)
    d1 = (ys[1:] - ys[:-1]) / dx[:, None]
    span = 0.5 * (dx[1:] + dx[:-1]).clamp_min(1e-5)
    d2 = (d1[1:] - d1[:-1]) / span[:, None]
    xvar = torch.var(xs, unbiased=False).clamp_min(1e-6)
    num = xvar * torch.mean(d2.square(), dim=0)
    den = torch.mean(d1.square(), dim=0).clamp_min(1e-8)
    # log1p keeps extreme exp/polynomial curvature from dominating the loss.
    return torch.log1p((num / den).clamp_min(0.0).clamp_max(1e6))


def _gmp_hierarchical_probs(
    logits: torch.Tensor,
    nonlinearity_logits: torch.Tensor,
    identity_index: int,
    temperature: float,
) -> torch.Tensor:
    """Factor probabilities with an explicit affine-vs-nonlinear decision."""
    K = int(logits.shape[-1])
    if K <= 1:
        return torch.ones_like(logits)
    temp = max(float(temperature), 1e-4)
    q = torch.sigmoid(nonlinearity_logits)
    mask = torch.ones(K, device=logits.device, dtype=torch.bool)
    mask[int(identity_index)] = False
    nonlinear = torch.softmax(logits[:, mask] / temp, dim=-1)
    out = torch.zeros_like(logits)
    out[:, int(identity_index)] = 1.0 - q
    out[:, mask] = q[:, None] * nonlinear
    return out



def _resolve_gmp_local_policy(
    structure: Sequence[int],
    *,
    steps: int,
    identity_chart: str,
    atom_backward_normalization: str,
) -> Tuple[str, str]:
    """Resolve a requested GMP proposal policy for one structure.

    ``raw``/``none`` is the default policy.  The optional ``auto`` mode enables
    dual affine charts on cross-variable products and RMS gradient balancing on
    longer GMP runs.
    """
    vars_ = tuple(int(v) for v in structure)
    is_cross_product = len(vars_) > 1 and len(set(vars_)) == len(vars_)
    local_chart = ("data_dual" if is_cross_product else "raw") if identity_chart == "auto" else identity_chart
    if atom_backward_normalization == "auto":
        local_norm = "rms" if is_cross_product and int(steps) >= 50 else "none"
    else:
        local_norm = atom_backward_normalization
    return local_chart, local_norm

def resolve_gmp_local_policy(
    structure: Sequence[int],
    *,
    steps: int,
    identity_chart: str = "raw",
    atom_backward_normalization: str = "none",
) -> Tuple[str, str]:
    """Public, side-effect-free resolver for the effective GMP proposal policy."""
    identity_chart = str(identity_chart).strip().lower()
    atom_backward_normalization = str(atom_backward_normalization).strip().lower()
    return _resolve_gmp_local_policy(
        structure, steps=int(steps), identity_chart=identity_chart,
        atom_backward_normalization=atom_backward_normalization,
    )

def gmp_symbolic_operator_preselection(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    structures: Sequence[Tuple[int, ...]],
    library: Sequence[str],
    *,
    topk: Optional[int] = None,
    unary_topk: Optional[int] = None,
    self_product_topk: Optional[int] = None,
    steps: int = 120,
    lr: float = 2e-2,
    temperature_start: float = 1.5,
    temperature_end: float = 0.35,
    entropy_weight: float = 2e-4,
    same_input_diversity_weight: float = 1e-2,
    relaxation_mode: str = "soft",
    atom_backward_normalization: str = "none",
    gumbel_noise_scale: float = 1.0,
    complexity_weight: float = 0.0,
    complexity_logit_prior: float = 0.0,
    nonlinearity_logit_prior: float = 0.0,
    curvature_weight: float = 0.0,
    nonlinearity_weight: float = 0.0,
    identity_chart: str = "raw",
    max_samples: int = 768,
    debug_topk: int = 0,
    debug_log_style: str = "compact",
    debug_label: str = "GMP",
    show_progress: bool = False,
    seed: int = 0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """GMP-style symbolic operator pruning before expensive in-context GSR.

    Each *symbolic structure* (including repeated-variable structures such as
    ``(x_j,x_j)``) gets independent operator gates per factor.  A temporary
    symbolic product is fitted against the regression target while the gates and
    affine operator parameters are optimized jointly.  Only the top-k operators
    per factor survive.  This is an amortized proposal stage: final commits are
    still decided by full-model matching-pursuit refits.

    This stage is intentionally spline-free.  Same-input factorization therefore
    has a reason to exist: ``exp(x)*sin(x)`` competes against a unary symbolic
    atom, rather than against an unrestricted spline h(x).
    """
    canonical_lib = tuple(name for name in library if name in SYMBOLIC_LIB)
    lib = canonical_lib
    relaxation_mode = str(relaxation_mode).strip().lower()
    if relaxation_mode not in {"soft", "st", "st_surrogate", "gumbel_st", "annealed_st", "hierarchical_soft", "hierarchical_st"}:
        raise ValueError("relaxation_mode must be soft, st, st_surrogate, gumbel_st, annealed_st, hierarchical_soft, or hierarchical_st")
    atom_backward_normalization = str(atom_backward_normalization).strip().lower()
    if atom_backward_normalization not in {"auto", "none", "rms", "maxabs", "rms_affine_orth"}:
        raise ValueError("atom_backward_normalization must be auto, none, rms, maxabs, or rms_affine_orth")
    identity_chart = str(identity_chart).strip().lower()
    if identity_chart not in {"auto", "raw", "unit_offset", "dual", "data_unit", "data_dual", "data_mid"}:
        raise ValueError("identity_chart must be auto, raw, unit_offset, dual, data_unit, data_dual, or data_mid")
    if not canonical_lib:
        raise ValueError("GMP symbolic preselection requires a non-empty symbolic library")
    canonical_K = len(canonical_lib)
    # ``data_dual`` uses two *latent charts* of the one affine family during
    # optimization.  They are collapsed back to one canonical ``x`` family
    # before top-k pruning, so the final grammar and shortlist width never grow.
    if identity_chart in {"data_dual", "auto"} and canonical_lib.count("x") == 1:
        xid = canonical_lib.index("x")
        lib = canonical_lib[:xid + 1] + ("x",) + canonical_lib[xid + 1:]
    K = len(lib)
    latent_lib = lib
    max_latent_chart_count = canonical_K
    complexity = torch.tensor([float(SYMBOLIC_LIB[name][2]) for name in lib], device=train_x.device, dtype=train_x.dtype)
    identity_indices = tuple(ii for ii, name in enumerate(lib) if name == "x")
    identity_index = identity_indices[0] if identity_indices else -1
    if relaxation_mode.startswith("hierarchical") and identity_index < 0:
        raise ValueError("hierarchical GMP requires the identity operator 'x' in the symbolic library")
    topk, unary_topk, self_product_topk = scaled_gmp_screening_sizes(
        canonical_K, topk=topk, unary_topk=unary_topk,
        self_product_topk=self_product_topk,
    )
    base_kkeep = min(canonical_K, int(topk))
    # Use the supplied generator only as a source of the *base seed*.  Each
    # structure receives its own RNG stream below, so results are invariant to
    # structure ordering and to adding/removing unrelated structures.
    base_seed = int(generator.initial_seed()) if generator is not None else int(seed)
    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0] - 1, int(max_samples), device=train_x.device).long()
        xfit, yfit = train_x[ids], train_y[ids]
    else:
        xfit, yfit = train_x, train_y
    yvec = yfit.reshape(yfit.shape[0], -1).mean(dim=1)

    total_inner = max(1, len(structures) * max(1, int(steps)))
    bar = tqdm(
        total=total_inner, desc="Symbolic GMP operator pruning",
        disable=not bool(show_progress), leave=True, dynamic_ncols=True,
        mininterval=0.25, smoothing=0.08,
    )
    results: List[Dict[str, object]] = []
    naive_combos = 0
    pruned_combos = 0

    for si, structure in enumerate(structures):
        vars_ = tuple(int(v) for v in structure)
        order = len(vars_)
        local_identity_chart, local_atom_backward_normalization = _resolve_gmp_local_policy(
            vars_, steps=int(steps), identity_chart=identity_chart,
            atom_backward_normalization=atom_backward_normalization,
        )
        # ``auto`` is resolved per structure.  Do not merely mask an unused
        # latent affine chart for unary/repeated-variable structures: even an
        # inactive extra column changes RNG layout and floating-point reduction
        # order.  Using the canonical library here makes auto/raw *identical*
        # to an explicit raw-chart run, while cross-variable structures
        # still receive the latent complementary affine chart.
        if identity_chart == "auto":
            lib = latent_lib if local_identity_chart == "data_dual" else canonical_lib
        else:
            lib = latent_lib
        K = len(lib)
        max_latent_chart_count = max(max_latent_chart_count, K)
        complexity = torch.tensor(
            [float(SYMBOLIC_LIB[name][2]) for name in lib],
            device=train_x.device, dtype=train_x.dtype,
        )
        identity_indices = tuple(ii for ii, name in enumerate(lib) if name == "x")
        identity_index = identity_indices[0] if identity_indices else -1
        rng = _local_torch_generator(
            train_x.device, _structure_rng_seed(base_seed, vars_)
        )
        unary_k = max(int(topk), int(unary_topk))
        self_k = max(int(topk), int(self_product_topk))
        if order == 1:
            kkeep = min(canonical_K, unary_k)
        elif len(set(vars_)) < order:
            kkeep = min(canonical_K, self_k)
        else:
            kkeep = base_kkeep
        naive_combos += canonical_K ** order
        pruned_combos += kkeep ** order

        logits = torch.nn.Parameter(0.02 * _rng_randn((order, K), device=xfit.device, dtype=xfit.dtype, generator=rng))
        affine = torch.nn.Parameter(torch.zeros(order, K, 4, device=xfit.device, dtype=xfit.dtype))
        scale = torch.nn.Parameter(torch.tensor(0.20, device=xfit.device, dtype=xfit.dtype))
        offset = torch.nn.Parameter(yvec.mean().detach().clone())
        nonlinearity_logits = torch.nn.Parameter(torch.zeros((order,), device=xfit.device, dtype=xfit.dtype))
        with torch.no_grad():
            affine[..., 0] = 1.0
            affine[..., 1] = 1.0
            affine[..., 3] = 0.0
            # Diverse frequency/scale seeds make periodic/exponential gates move
            # immediately instead of all atoms beginning at exactly the same local
            # linearization.
            for kk, name in enumerate(lib):
                if name == "x" and local_identity_chart == "data_mid":
                    affine[:, kk, 1] = 0.0
                    affine[:, kk, 2] = 0.5
                elif name == "x" and local_identity_chart in {"data_unit", "data_dual"}:
                    occurrence = sum(1 for z in lib[:kk] if z == "x")
                    for slot, var in enumerate(vars_):
                        xmin = float(torch.min(xfit[:, var]).detach().cpu())
                        xmax = float(torch.max(xfit[:, var]).detach().cpu())
                        xr = max(xmax - xmin, 1e-6)
                        if local_identity_chart == "data_dual" and occurrence % 2 == 1:
                            affine[slot, kk, 1] = -1.0 / xr
                            affine[slot, kk, 2] = xmax / xr
                        else:
                            affine[slot, kk, 1] = 1.0 / xr
                            affine[slot, kk, 2] = -xmin / xr
                elif name == "x" and local_identity_chart == "unit_offset":
                    affine[:, kk, 1] = 0.0
                    affine[:, kk, 2] = 1.0
                elif name == "x" and local_identity_chart == "dual" and sum(1 for z in lib[:kk] if z == "x") % 2 == 1:
                    # Second latent chart of the same final affine family.
                    affine[:, kk, 1] = 0.0
                    affine[:, kk, 2] = 1.0
                elif name == "x" and local_identity_chart == "raw" and sum(1 for z in lib[:kk] if z == "x") % 2 == 1:
                    affine[:, kk, 1] = -1.0
                    affine[:, kk, 2] = 1.0
                elif name in {"sin", "cos"}:
                    affine[:, kk, 1] = 2.5
                elif name == "exp":
                    affine[:, kk, 1] = -0.5
                elif name in {"log1p_sq", "sqrt1p_sq", "inv1p_sq", "x^2"}:
                    affine[:, kk, 1] = 1.0
        opt_params = [logits, affine, scale, offset]
        if relaxation_mode.startswith("hierarchical"):
            opt_params.append(nonlinearity_logits)
        opt = torch.optim.Adam(opt_params, lr=float(lr))
        best_loss = float("inf")
        best = None
        nsteps = max(1, int(steps))
        for it in range(nsteps):
            u = 1.0 if nsteps <= 1 else it / float(nsteps - 1)
            temp = float(temperature_start + (temperature_end - temperature_start) * u)
            prior = float(complexity_logit_prior) * complexity[None, :]
            if identity_index >= 0 and float(nonlinearity_logit_prior) != 0.0:
                nonlin_mask = torch.ones(K, device=logits.device, dtype=logits.dtype)
                nonlin_mask[int(identity_index)] = 0.0
                prior = prior + float(nonlinearity_logit_prior) * nonlin_mask[None, :]
            effective_logits = logits - prior
            if identity_chart == "auto" and local_identity_chart == "raw" and lib.count("x") > 1:
                effective_logits = effective_logits.clone()
                xids_local = [ii for ii, nm in enumerate(lib) if nm == "x"]
                effective_logits[:, xids_local[1:]] = -1e6
            if relaxation_mode.startswith("hierarchical"):
                base_probs = _gmp_hierarchical_probs(effective_logits, nonlinearity_logits, identity_index, temp)
            else:
                base_probs = torch.softmax(effective_logits / max(temp, 1e-4), dim=-1)
            if relaxation_mode == "gumbel_st":
                u_noise = _rng_rand(logits.shape, device=logits.device, dtype=logits.dtype, generator=rng).clamp_(1e-6, 1.0 - 1e-6)
                g_noise = -torch.log(-torch.log(u_noise)) * float(gumbel_noise_scale)
                soft_probs = torch.softmax((effective_logits + g_noise) / max(temp, 1e-4), dim=-1)
            else:
                soft_probs = base_probs
            hard_ids = torch.argmax(soft_probs, dim=-1)
            hard_probs = F.one_hot(hard_ids, num_classes=K).to(dtype=soft_probs.dtype)
            if relaxation_mode in {"st", "gumbel_st", "hierarchical_st"}:
                probs = hard_probs + soft_probs - soft_probs.detach()
            elif relaxation_mode == "annealed_st":
                # Smooth continuation: soft mixture early, exactly categorical
                # late, while retaining the soft categorical gradient throughout.
                h = max(0.0, min(1.0, (u - 0.30) / 0.55))
                h = h * h * (3.0 - 2.0 * h)
                forward_probs = (1.0 - h) * soft_probs + h * hard_probs
                probs = forward_probs.detach() + soft_probs - soft_probs.detach()
            else:
                probs = soft_probs
            factors = []
            curvature_terms = []
            valid = True
            for slot, var in enumerate(vars_):
                atoms = []
                xx = xfit[:, var]
                for kk, name in enumerate(lib):
                    try:
                        atoms.append(_gmp_safe_operator_value(name, xx, affine[slot, kk]))
                    except Exception:
                        atoms.append(torch.zeros_like(xx))
                A = torch.stack(atoms, dim=-1)
                if float(curvature_weight) > 0:
                    curv = _gmp_normalized_curvature(A.detach(), xx.detach())
                    curvature_terms.append(torch.sum(base_probs[slot] * curv))
                if local_atom_backward_normalization == "rms_affine_orth":
                    xc = xx - torch.mean(xx)
                    xvar = torch.mean(xc.square()).clamp_min(1e-8)
                    amean = torch.mean(A, dim=0, keepdim=True)
                    slope = torch.mean((A - amean) * xc[:, None], dim=0, keepdim=True) / xvar
                    A_orth = A - amean - xc[:, None] * slope
                    # Preserve the explicit affine family itself; only nonlinear
                    # atoms are quotiented by span{1,x} in the backward surrogate.
                    if identity_indices:
                        A_orth = A_orth.clone()
                        for xid in identity_indices:
                            A_orth[:, xid] = A[:, xid]
                    denom = torch.sqrt(torch.mean(A_orth.detach().square(), dim=0, keepdim=True) + 1e-8)
                    A_sur = A_orth / denom
                elif local_atom_backward_normalization == "rms":
                    denom = torch.sqrt(torch.mean(A.detach().square(), dim=0, keepdim=True) + 1e-8)
                    A_sur = A / denom
                elif local_atom_backward_normalization == "maxabs":
                    denom = torch.amax(torch.abs(A.detach()), dim=0, keepdim=True).clamp_min(1e-4)
                    A_sur = A / denom
                else:
                    A_sur = A
                if relaxation_mode == "st_surrogate":
                    hard_mix = (A * hard_probs[slot][None, :]).sum(dim=-1)
                    soft_mix = (A_sur * soft_probs[slot][None, :]).sum(dim=-1)
                    mix = hard_mix.detach() + soft_mix - soft_mix.detach()
                else:
                    mix_raw = (A * probs[slot][None, :]).sum(dim=-1)
                    if local_atom_backward_normalization != "none":
                        mix_sur = (A_sur * probs[slot][None, :]).sum(dim=-1)
                        mix = mix_raw.detach() + mix_sur - mix_sur.detach()
                    else:
                        mix = mix_raw
                if not torch.isfinite(mix).all():
                    valid = False; break
                factors.append(mix)
            if not valid:
                loss = torch.as_tensor(float("inf"), device=xfit.device, dtype=xfit.dtype)
            else:
                rule = torch.ones_like(yvec)
                for f in factors:
                    rule = rule * f
                pred = offset + scale * rule
                mse = torch.mean((pred - yvec) ** 2)
                ent = _entropy(probs, dim=-1).mean()
                diversity = torch.zeros((), device=xfit.device, dtype=xfit.dtype)
                if len(vars_) > 1:
                    for aa in range(len(vars_)):
                        for bb in range(aa + 1, len(vars_)):
                            if vars_[aa] == vars_[bb]:
                                diversity = diversity + torch.sum(probs[aa] * probs[bb])
                expected_complexity = torch.mean(torch.sum(base_probs * complexity[None, :], dim=-1))
                curvature_penalty = (torch.stack(curvature_terms).mean() if curvature_terms
                                     else torch.zeros((), device=xfit.device, dtype=xfit.dtype))
                if relaxation_mode.startswith("hierarchical"):
                    nonlinear_mass = torch.sigmoid(nonlinearity_logits).mean()
                else:
                    nonlinear_mass = (
                        1.0 - base_probs[:, list(identity_indices)].sum(dim=-1).mean()
                        if identity_indices else torch.ones((), device=xfit.device, dtype=xfit.dtype)
                    )
                loss = (mse + float(entropy_weight) * ent
                        + float(same_input_diversity_weight) * diversity
                        + float(complexity_weight) * expected_complexity
                        + float(curvature_weight) * curvature_penalty
                        + float(nonlinearity_weight) * nonlinear_mass)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                if affine.grad is not None:
                    affine.grad[..., 0].zero_()
                    affine.grad[..., 3].zero_()
                torch.nn.utils.clip_grad_norm_([logits, affine, scale, offset], 5.0)
                opt.step()
                with torch.no_grad():
                    affine[..., 0].fill_(1.0)
                    affine[..., 1].clamp_(-8.0, 8.0)
                    affine[..., 2].clamp_(-4.0 * math.pi, 4.0 * math.pi)
                    affine[..., 3].zero_()
                    scale.clamp_(-1e3, 1e3); offset.clamp_(-1e3, 1e3)
                    lv = float(loss.detach().cpu())
                    if lv < best_loss:
                        best_loss = lv
                        best = (
                            logits.detach().clone(), affine.detach().clone(),
                            scale.detach().clone(), offset.detach().clone(), temp,
                            nonlinearity_logits.detach().clone(),
                        )
            if show_progress:
                bar.update(1)
                if it % 8 == 0 or it == nsteps - 1:
                    bar.set_postfix(
                        structure="*".join(str(v) for v in vars_),
                        loss=f"{best_loss:.3g}" if math.isfinite(best_loss) else "inf",
                        refresh=False,
                    )
        if not show_progress:
            # keep accounting consistent when tqdm is disabled
            pass
        if best is None:
            continue
        blogits, baffine, bscale, boffset, btemp, bnonlin = best
        final_prior = float(complexity_logit_prior) * complexity[None, :]
        if identity_indices and float(nonlinearity_logit_prior) != 0.0:
            final_nonlin_mask = torch.ones(K, device=blogits.device, dtype=blogits.dtype)
            for xid in identity_indices:
                final_nonlin_mask[int(xid)] = 0.0
            final_prior = final_prior + float(nonlinearity_logit_prior) * final_nonlin_mask[None, :]
        final_logits = blogits - final_prior
        if identity_chart == "auto" and local_identity_chart == "raw" and lib.count("x") > 1:
            final_logits = final_logits.clone()
            xids_local = [ii for ii, nm in enumerate(lib) if nm == "x"]
            final_logits[:, xids_local[1:]] = -1e6
        if relaxation_mode.startswith("hierarchical"):
            final_probs = _gmp_hierarchical_probs(final_logits, bnonlin, identity_index, float(temperature_end))
        else:
            final_probs = torch.softmax(final_logits / max(float(temperature_end), 1e-4), dim=-1)
        factor_shortlists: List[List[Dict[str, object]]] = []
        for slot in range(order):
            # Rank *canonical families*, not latent charts.  For a duplicated
            # affine family, sum the chart probabilities but keep the affine
            # parameters of the better-oriented chart.
            grouped = []
            for name in canonical_lib:
                ids = [ii for ii, nm in enumerate(lib) if nm == name]
                mass = final_probs[slot, ids].sum()
                best_id = max(ids, key=lambda ii: float(final_probs[slot, ii].detach().cpu()))
                grouped.append((float(mass.detach().cpu()), int(best_id), name))
            grouped.sort(key=lambda z: z[0], reverse=True)
            cands=[]
            for prob, kk, name in grouped[:kkeep]:
                cands.append({
                    "name": name,
                    "gmp_prob": float(prob),
                    "affine": _canonical_symbolic_affine(baffine[slot, kk].cpu().tolist()),
                })
            factor_shortlists.append(cands)
        result = {
            "structure": vars_,
            "factor_shortlists": factor_shortlists,
            "gmp_loss": float(best_loss),
            "gmp_scale": float(bscale.cpu()),
            "gmp_offset": float(boffset.cpu()),
            "topk_used": int(kkeep),
        }
        results.append(result)
    if show_progress:
        bar.close()
    if int(debug_topk) > 0 and results:
        _print_gmp_debug(
            results, debug_topk=int(debug_topk), log_style=str(debug_log_style),
            label=str(debug_label), show_progress=show_progress,
        )
    stats = {
        "structures": float(len(results)),
        "library_size": float(canonical_K),
        "latent_chart_count": float(max_latent_chart_count),
        "topk": float(base_kkeep),
        "unary_topk": float(min(canonical_K, max(int(topk), int(unary_topk)))),
        "self_product_topk": float(min(canonical_K, max(int(topk), int(self_product_topk)))),
        "naive_operator_tuples": float(naive_combos),
        "gmp_operator_tuples": float(pruned_combos),
        "tuple_reduction_fraction": float(1.0 - pruned_combos / max(1, naive_combos)),
    }
    return results, stats




def _hard_affine_seed_grid(name: str) -> Tuple[Tuple[float, float], ...]:
    """Deterministic multistart seeds for hard symbolic operator screening.

    Screening is deliberately cheap and deterministic: it does not optimize a
    soft mixture over operators.  Instead, each analytic atom is evaluated at a
    small set of plausible input scales/shifts and the best hard product tuple
    is ranked by its optimal linear projection onto the target.  The expensive
    continuous refit still happens later in matching pursuit.
    """
    name = str(name)
    if name in {"sin", "cos"}:
        bs = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -1.0, -2.0)
        cs = (0.0, -0.5 * math.pi, 0.5 * math.pi)
    elif name == "exp":
        bs = (-2.0, -1.5, -1.0, -0.75, -0.5, 0.5, 1.0, 1.5, 2.0)
        cs = (0.0, -0.5, 0.5)
    elif name in {"tanh", "arctan"}:
        bs = (0.5, 1.0, 1.5, 2.0, 2.5, -0.5, -1.0, -2.0)
        cs = (0.0, -0.5, 0.5)
    elif name == "x":
        # Input shift matters inside a product: x-1 is the canonical fuzzy
        # complement factor even though an output offset is forbidden.
        bs = (1.0, -1.0)
        cs = (0.0, -1.0, 1.0)
    elif name in {"x^2", "x^3", "x^4", "x^5", "sqrt", "log", "gaussian",
                  "log1p_sq", "sqrt1p_sq", "inv1p_sq", "1/x", "1/x^2",
                  "1/x^3", "1/x^4", "1/x^5"}:
        bs = (0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5)
        cs = (0.0, -0.5, 0.5)
    else:
        bs = (1.0, -1.0, 0.5, 2.0)
        cs = (0.0, -0.5, 0.5)
    # Preserve order while removing duplicates.
    out = []
    seen = set()
    for b in bs:
        for c in cs:
            key = (float(b), float(c))
            if key not in seen:
                seen.add(key); out.append(key)
    return tuple(out)


def _data_unit_affine_seed_grid(
    x: torch.Tensor,
    variable: int,
    name: str,
) -> Tuple[Tuple[float, float], ...]:
    """Return raw and data-unit reparameterized affine starts for one atom.

    The hard-start grid is written in a convenient unit-coordinate
    chart.  RuleKAN normally trains on standardized coordinates, so a useful
    basin such as ``sin(2.4 * x_raw)`` can be far from those starts after input
    normalization.  If ``u=(z-z_min)/(z_max-z_min)``, the chart

        g(beta*u + gamma)

    is exactly

        g((beta/range)*z + gamma - beta*z_min/range).

    Keeping both charts is cheap, generic across the operator library, and does
    not inject any task formula or unsupported variable dependency.
    """
    base = _hard_affine_seed_grid(str(name))
    if x.numel() == 0:
        return base
    col = x[:, int(variable)].reshape(-1)
    finite = col[torch.isfinite(col)]
    if finite.numel() < 2:
        return base
    lo = float(torch.min(finite).detach().cpu())
    hi = float(torch.max(finite).detach().cpu())
    width = hi - lo
    if not math.isfinite(width) or width <= 1e-8:
        return base
    out: List[Tuple[float, float]] = []
    seen = set()
    for b, c in base:
        for bb, cc in (
            (float(b), float(c)),
            (float(b) / width, float(c) - float(b) * lo / width),
        ):
            key = (round(bb, 12), round(cc, 12))
            if key in seen:
                continue
            seen.add(key); out.append((bb, cc))
    return tuple(out)


def _affine_partition_gate_chart(
    x: torch.Tensor,
    variable: int,
    complement: bool,
) -> Tuple[float, float, float, float]:
    """Canonical identity chart for ``u`` or ``1-u`` on the observed range."""
    col = x[:, int(variable)].reshape(-1)
    finite = col[torch.isfinite(col)]
    if finite.numel() < 2:
        lo, hi = 0.0, 1.0
    else:
        lo = float(torch.min(finite).detach().cpu())
        hi = float(torch.max(finite).detach().cpu())
    width = max(hi - lo, 1e-8)
    if complement:
        beta = -1.0 / width
        gamma = 1.0 + lo / width
    else:
        beta = 1.0 / width
        gamma = -lo / width
    return (1.0, float(beta), float(gamma), 0.0)


def _affine_partition_structure_pairs(
    structure_candidates: Sequence[Tuple[int, ...]],
    *,
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], int, int, int]]:
    """Enumerate two-rule partition mechanisms without leaving learned supports.

    A mechanism has two order-2 structures that share one gate variable.  The
    repeated-variable case ``(j,j)+(j,j)`` is included because it is the
    identifiable symbolic decomposition needed for same-variable fuzzy rules.
    Identical cross-variable supports are intentionally not duplicated.
    """
    structures = []
    seen = set()
    for z in structure_candidates:
        zz = tuple(sorted(int(v) for v in z))
        if len(zz) != 2 or zz in seen:
            continue
        seen.add(zz); structures.append(zz)
    allowed = None
    if allowed_supports is not None:
        allowed = {frozenset(int(v) for v in s) for s in allowed_supports}
        for z in structures:
            if frozenset(z) not in allowed:
                raise ValueError(f"affine partition rescue received unsupported structure {z}")
    out = []
    out_seen = set()
    for ia, za in enumerate(structures):
        for ib in range(ia, len(structures)):
            zb = structures[ib]
            if za == zb and za[0] != za[1]:
                continue
            shared = sorted(set(za).intersection(zb))
            for gate in shared:
                aa = list(za); bb = list(zb)
                aa.remove(gate); bb.remove(gate)
                if len(aa) != 1 or len(bb) != 1:
                    continue
                branch_a, branch_b = int(aa[0]), int(bb[0])
                key = (za, zb, int(gate), branch_a, branch_b)
                if key in out_seen:
                    continue
                out_seen.add(key); out.append(key)
    return out



def _active_symbolic_support_multiset(model: SumProductKAN) -> List[Tuple[int, ...]]:
    """Return active rule supports as sorted variable-index tuples.

    Multiplicity is preserved so repeated-variable symbolic products such as
    ``(j,j)`` remain distinguishable from unary ``(j,)`` rules.  This helper
    examines only the selected symbolic model; it does not consult task labels
    or target formulas.
    """
    out: List[Tuple[int, ...]] = []
    active = torch.nonzero(
        model.hard_rule_choice & model.rule_alive_mask,
        as_tuple=False,
    ).squeeze(-1).tolist()
    for rr in active:
        support: List[int] = []
        for ss in range(model.max_factors):
            jj = int(model.hard_variable_choice[int(rr), ss].item())
            if jj != model.in_dim:
                support.append(jj)
        if support:
            out.append(tuple(sorted(support)))
    return out


def _has_affine_partition_refactor_opportunity(
    model: SumProductKAN,
    structure_candidates: Sequence[Tuple[int, ...]],
    *,
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
) -> bool:
    """Detect a generic distributive/partition refactor opportunity.

    A very accurate model can represent

        (1-u) f(a) + u g(b)

    distributively as ``f(a) - u*f(a) + u*g(b)``.  Prediction error alone then
    cannot tell us to try the canonical partition form.  Trigger the partition
    rescue when the *selected symbolic model* already contains both admissible
    gate/branch product supports for a shared gate.  This uses only learned
    RuleKAN structures and the incumbent model -- never fuzzy benchmark labels.

    Repeated-variable ``(j,j)`` pairs are not used as an accuracy-independent
    trigger because many ordinary one-variable products legitimately use that
    support; those cases still trigger through the usual validation-error gate.
    """
    active = _active_symbolic_support_multiset(model)
    if not active:
        return False
    aset = set(active)
    for za, zb, gate, branch_a, branch_b in _affine_partition_structure_pairs(
        structure_candidates, allowed_supports=allowed_supports,
    ):
        if za == zb:
            continue
        if tuple(za) in aset and tuple(zb) in aset:
            return True
    return False


def _two_term_linear_fit(
    h0: torch.Tensor,
    h1: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[float, float, float, float]:
    """Least-squares ``bias + a*h0 + b*h1`` fit and MSE."""
    yy = y.reshape(y.shape[0], -1).mean(dim=1)
    X = torch.stack((torch.ones_like(h0), h0, h1), dim=1)
    try:
        coef = torch.linalg.lstsq(X, yy[:, None]).solution[:3, 0]
    except Exception:
        gram = X.T @ X + torch.eye(3, device=X.device, dtype=X.dtype) * 1e-8
        coef = torch.linalg.solve(gram, X.T @ yy)
    pred = X @ coef
    mse = float(torch.mean((pred - yy) ** 2).detach().cpu())
    return mse, float(coef[0].detach().cpu()), float(coef[1].detach().cpu()), float(coef[2].detach().cpu())


def _best_two_term_seed_pair(
    terms0: torch.Tensor,
    terms1: torch.Tensor,
    y: torch.Tensor,
    *,
    ridge: float = 1e-9,
) -> Tuple[float, int, int, float, float, float]:
    """Vectorized best two-regressor fit across all seed-pair combinations.

    ``terms0`` and ``terms1`` are ``[n_samples, n_seeds]`` matrices.  The
    intercept is handled by centering, so every seed pair is scored with its
    exact two-variable least-squares coefficients without a Python loop over
    combinations.  This is what lets the affine-partition rescue delay pruning
    until *after* a complete operator family has had all of its data-aware
    initial charts considered jointly.
    """
    # Coarse family ranking is sensitive to near-collinear branch atoms.  Do
    # the tiny normal-equation algebra in float64 even when the model trains in
    # float32; otherwise cancellation can make a spurious family appear to have
    # exactly zero residual and crowd out the correct family before refinement.
    yy = y.reshape(y.shape[0], -1).mean(dim=1).to(torch.float64)
    t0 = terms0.to(torch.float64); t1 = terms1.to(torch.float64)
    m0 = t0.mean(dim=0); m1 = t1.mean(dim=0); ym = yy.mean()
    a = t0 - m0; b = t1 - m1; yc = yy - ym
    aa = (a * a).sum(dim=0) + float(ridge)
    bb = (b * b).sum(dim=0) + float(ridge)
    ay = a.T @ yc
    by = b.T @ yc
    ab = a.T @ b
    det = aa[:, None] * bb[None, :] - ab.square()
    valid = det.abs() > 1e-14
    det_safe = torch.where(valid, det, torch.ones_like(det))
    ca = (ay[:, None] * bb[None, :] - by[None, :] * ab) / det_safe
    cb = (by[None, :] * aa[:, None] - ay[:, None] * ab) / det_safe
    explained = ca * ay[:, None] + cb * by[None, :]
    sst = (yc * yc).sum()
    sse = (sst - explained).clamp_min(0.0)
    sse = torch.where(valid & torch.isfinite(sse), sse, torch.full_like(sse, float("inf")))
    flat = int(torch.argmin(sse).item())
    n1 = int(terms1.shape[1])
    i, j = flat // n1, flat % n1
    mse = float((sse[i, j] / max(1, int(yy.numel()))).detach().cpu())
    scale0 = float(ca[i, j].detach().cpu())
    scale1 = float(cb[i, j].detach().cpu())
    bias = float((ym - ca[i, j] * m0[i] - cb[i, j] * m1[j]).detach().cpu())
    return mse, int(i), int(j), bias, scale0, scale1


def affine_partition_symbolic_rescue(
    numeric_model: SumProductKAN,
    incumbent: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    structure_candidates: Sequence[Tuple[int, ...]],
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    family_beam: int = 18,
    seed_topk: int = 0,
    max_samples: int = 384,
    max_support_pairs: int = 12,
    refine_steps: int = 120,
    refine_lr: float = 1.0e-3,
    lbfgs_steps: int = 20,
    final_polish_topk: int = 8,
    final_polish_steps: int = 500,
    final_polish_lbfgs_steps: int = 80,
    equivalence_rel_mse: float = 3.0,
    equivalence_nrmse: float = 5e-4,
    cancellation_weight: float = 1.0,
    complexity_weight: float = 0.02,
    min_improvement_rel: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Validation-gated, structure-conditioned affine partition rescue.

    This is a generic symbolic-search rescue, not a fuzzy benchmark shortcut.
    It considers only pairs of already-admissible order-2 RuleKAN structures
    that share a variable.  That variable is represented by the canonical
    partition-of-unity charts ``u`` and ``1-u``; branch atoms are selected from
    the ordinary symbolic library with raw + data-unit affine multistarts.

    Operator *families* survive coarse screening first. Only then are the best
    families continuously refined. One affine gate chart is optimized while its
    partner is constrained to remain the exact complement ``1-u``. Selection is
    validation-first: a partition can replace the incumbent by material held-out
    improvement or, inside the configured validation-equivalence band, by lower
    cancellation/description preference.
    """
    meta: Dict[str, object] = {
        "attempted": False, "selected": False, "reason": "not_applicable",
    }
    if numeric_model.max_factors < 2 or numeric_model.n_rules < 2:
        return incumbent, meta
    lib = tuple(
        str(n) for n in (numeric_model.symbolic_library if library is None else library)
        if n in SYMBOLIC_LIB
    )
    if "x" not in lib or not lib:
        meta["reason"] = "identity_not_in_library"
        return incumbent, meta
    support_pairs = _affine_partition_structure_pairs(
        structure_candidates, allowed_supports=allowed_supports,
    )
    if int(max_support_pairs) > 0:
        support_pairs = support_pairs[:int(max_support_pairs)]
    if not support_pairs:
        meta["reason"] = "no_partition_support_pair"
        return incumbent, meta
    meta["attempted"] = True

    incumbent.eval()
    with torch.no_grad():
        incumbent_mse = float(torch.mean((incumbent(val_x) - val_y) ** 2).detach().cpu())
    meta["incumbent_validation_mse"] = incumbent_mse

    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0] - 1, int(max_samples), device=train_x.device).long()
        xx, yy = train_x[ids], train_y[ids]
    else:
        xx, yy = train_x, train_y
    yvec = yy.reshape(yy.shape[0], -1).mean(dim=1)

    # Precompute affine starts for each gate/branch/operator.  Crucially, starts
    # are NOT ranked by a one-rule objective: a fuzzy/partition branch can be
    # individually weak and jointly exact.  Family scoring below evaluates all
    # retained starts jointly with the complementary branch.
    atom_cache: Dict[Tuple[int, str, float, float], torch.Tensor] = {}
    def atom(var: int, name: str, b: float, c: float) -> Optional[torch.Tensor]:
        key = (int(var), str(name), round(float(b), 12), round(float(c), 12))
        if key in atom_cache:
            return atom_cache[key]
        try:
            g = SYMBOLIC_LIB[str(name)][0](float(b) * xx[:, int(var)] + float(c))
        except Exception:
            return None
        if not torch.isfinite(g).all():
            return None
        atom_cache[key] = g
        return g

    family_candidates: List[Tuple[float, Dict[str, object]]] = []
    for za, zb, gate, branch_a, branch_b in support_pairs:
        for swapped in (False, True):
            # Define the partition gauge from the complete training range, not
            # the coarse-screen subsample; subsampling should never change the
            # semantic chart used to initialize tied complementary refinement.
            gate_a_aff = _affine_partition_gate_chart(train_x, gate, complement=bool(swapped))
            gate_b_aff = _affine_partition_gate_chart(train_x, gate, complement=not bool(swapped))
            ga = gate_a_aff[1] * xx[:, gate] + gate_a_aff[2]
            gb = gate_b_aff[1] * xx[:, gate] + gate_b_aff[2]
            seeded: Dict[Tuple[str, int], Tuple[torch.Tensor, List[Tuple[float, float]]]] = {}
            for side, (gcurve, branch_var) in enumerate(((ga, branch_a), (gb, branch_b))):
                for name in lib:
                    curves = []
                    seeds = []
                    # Seed charts must be defined from the complete training
                    # coordinate range.  The coarse-screen subsample is only an
                    # evaluation shortcut; letting it define the coordinate map
                    # shifts the useful basin from run to run.
                    for b, c in _data_unit_affine_seed_grid(train_x, branch_var, name):
                        aval = atom(branch_var, name, b, c)
                        if aval is None:
                            continue
                        term = gcurve * aval
                        if not torch.isfinite(term).all():
                            continue
                        curves.append(term); seeds.append((float(b), float(c)))
                    if int(seed_topk) > 0 and len(curves) > int(seed_topk):
                        # Optional explicit runtime cap.  Preserve the complete
                        # data-aware family by default (seed_topk=0); callers
                        # that choose a cap accept the corresponding recall loss.
                        curves = curves[:int(seed_topk)]; seeds = seeds[:int(seed_topk)]
                    if curves:
                        seeded[(name, side)] = (torch.stack(curves, dim=1), seeds)

            for name_a in lib:
                left = seeded.get((name_a, 0))
                if not left:
                    continue
                for name_b in lib:
                    right = seeded.get((name_b, 1))
                    if not right:
                        continue
                    left_terms, left_seeds = left
                    right_terms, right_seeds = right
                    mse, ia, ib, bias, scale_a, scale_b = _best_two_term_seed_pair(
                        left_terms, right_terms, yy,
                    )
                    if not math.isfinite(float(mse)):
                        continue
                    ba, ca = left_seeds[int(ia)]
                    bb, cb = right_seeds[int(ib)]
                    yscale_coarse = max(float(torch.std(yvec).detach().cpu()), 1e-8)
                    h0_best = left_terms[:, int(ia)]
                    h1_best = right_terms[:, int(ib)]
                    coarse_cancel = (
                        float(torch.mean(
                            torch.abs(float(scale_a) * h0_best)
                            + torch.abs(float(scale_b) * h1_best)
                        ).detach().cpu())
                        + abs(float(bias))
                    ) / yscale_coarse
                    coarse_complexity = (
                        float(SYMBOLIC_LIB[str(name_a)][2])
                        + float(SYMBOLIC_LIB[str(name_b)][2])
                    )
                    coarse_preference = (
                        float(cancellation_weight) * float(coarse_cancel)
                        + float(complexity_weight) * float(coarse_complexity)
                    )
                    family_candidates.append((float(mse), {
                        "support_a": za, "support_b": zb, "gate": int(gate),
                        "branch_a": int(branch_a), "branch_b": int(branch_b),
                        "gate_a_affine": gate_a_aff, "gate_b_affine": gate_b_aff,
                        "operator_a": str(name_a), "operator_b": str(name_b),
                        "branch_a_affine": (1.0, float(ba), float(ca), 0.0),
                        "branch_b_affine": (1.0, float(bb), float(cb), 0.0),
                        "bias": float(bias), "scale_a": float(scale_a), "scale_b": float(scale_b),
                        "coarse_cancellation_score": float(coarse_cancel),
                        "coarse_description_complexity": float(coarse_complexity),
                        "coarse_preference_score": float(coarse_preference),
                    }))

    if not family_candidates:
        meta["reason"] = "no_finite_partition_family"
        return incumbent, meta

    # Keep one best start per support/gate/operator family before expensive
    # continuous refinement.  This is the delayed-pruning rule: affine starts
    # compete *within* a family first; different operator families retain their
    # own lane until after a joint two-rule screen.
    best_by_family: Dict[Tuple[object, ...], Tuple[float, Dict[str, object]]] = {}
    for score, rec in family_candidates:
        key = (
            tuple(rec["support_a"]), tuple(rec["support_b"]), int(rec["gate"]),
            str(rec["operator_a"]), str(rec["operator_b"]),
        )
        if key not in best_by_family or score < best_by_family[key][0]:
            best_by_family[key] = (score, rec)
    all_families = list(best_by_family.values())
    family_k = max(1, int(family_beam))
    loss_k = max(1, int(math.ceil(family_k / 2.0)))
    selected_families: List[Tuple[float, Dict[str, object]]] = []
    selected_family_keys = set()
    for item in sorted(all_families, key=lambda z: z[0])[:loss_k]:
        rec = item[1]
        key = (
            tuple(rec["support_a"]), tuple(rec["support_b"]), int(rec["gate"]),
            str(rec["operator_a"]), str(rec["operator_b"]),
        )
        if key not in selected_family_keys:
            selected_family_keys.add(key); selected_families.append(item)
    by_coarse_pref = sorted(
        all_families,
        key=lambda z: (float(z[1].get("coarse_preference_score", float("inf"))), z[0]),
    )
    for item in by_coarse_pref:
        if len(selected_families) >= family_k:
            break
        rec = item[1]
        key = (
            tuple(rec["support_a"]), tuple(rec["support_b"]), int(rec["gate"]),
            str(rec["operator_a"]), str(rec["operator_b"]),
        )
        if key in selected_family_keys:
            continue
        selected_family_keys.add(key); selected_families.append(item)
    ranked = selected_families
    meta["families_screened"] = int(len(best_by_family))
    meta["families_refined"] = int(len(ranked))

    best_model = incumbent
    best_mse = incumbent_mse
    best_rec = None
    selection_pool: List[Tuple[float, SumProductKAN, Dict[str, object]]] = []
    refined_families: List[Tuple[
        float, SumProductKAN, Dict[str, object], Tuple[Tuple[int, int, int, int], ...]
    ]] = []
    x_index = numeric_model.symbolic_library.index("x")
    for coarse_mse, rec in ranked:
        trial = _make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            trial.hard_rule_choice.zero_()
            trial.rule_scale.zero_()
            trial.bias.fill_(float(rec["bias"]))
        combo_a = (
            {"name": "x", "affine": rec["gate_a_affine"], "partition_gate": True},
            {"name": rec["operator_a"], "affine": rec["branch_a_affine"]},
        )
        combo_b = (
            {"name": "x", "affine": rec["gate_b_affine"], "partition_gate": True},
            {"name": rec["operator_b"], "affine": rec["branch_b_affine"]},
        )
        factors_a = ((0, int(rec["gate"])), (1, int(rec["branch_a"])))
        factors_b = ((0, int(rec["gate"])), (1, int(rec["branch_b"])))
        _install_symbolic_template(trial, 0, factors_a, combo_a, initial_scale=float(rec["scale_a"]))
        _install_symbolic_template(trial, 1, factors_b, combo_b, initial_scale=float(rec["scale_b"]))
        frozen = (
            (0, 0, int(rec["gate"]), int(x_index)),
            (1, 0, int(rec["gate"]), int(x_index)),
        )
        # Refine one shared affine gate chart while preserving exact complementarity.
        if float(rec["gate_a_affine"][1]) >= 0.0:
            gate_pair = (((1, 0, int(rec["gate"]), int(x_index)),
                          (0, 0, int(rec["gate"]), int(x_index))),)
        else:
            gate_pair = (((0, 0, int(rec["gate"]), int(x_index)),
                          (1, 0, int(rec["gate"]), int(x_index))),)
        trial, mse = _fully_symbolic_continuous_refit(
            trial, train_x, train_y, val_x, val_y,
            steps=max(0, int(refine_steps)), lr=float(refine_lr),
            lbfgs_steps=max(0, int(lbfgs_steps)),
            complementary_gate_pairs=gate_pair, show_progress=False,
        )
        if math.isfinite(float(mse)):
            rec2 = dict(rec); rec2["coarse_train_mse"] = float(coarse_mse)
            refined_families.append((float(mse), trial, rec2, tuple(frozen)))
            selection_pool.append((float(mse), trial, rec2))
        if math.isfinite(float(mse)) and float(mse) < float(best_mse):
            best_model, best_mse, best_rec = trial, float(mse), dict(rec)
            best_rec["coarse_train_mse"] = float(coarse_mse)

    # A short family-level refinement is enough to eliminate bad basins but can
    # leave genuinely different symbolic families numerically tied.  Deeply
    # polish only the best few survivors before validation selection.  This is
    # much cheaper than giving every family a large optimization budget and is
    # the step that lets a correctly initialized tanh/sin family separate from
    # a flexible phase-shifted trigonometric surrogate.
    refined_families.sort(key=lambda z: z[0])
    def _shallow_preference(candidate: SumProductKAN) -> Tuple[float, float, float]:
        candidate.eval()
        with torch.no_grad():
            _, det = candidate(val_x, return_details=True)
            active = torch.nonzero(
                candidate.hard_rule_choice & candidate.rule_alive_mask,
                as_tuple=False,
            ).squeeze(-1).tolist()
            if active:
                contrib = det["contributions"][:, active]
                contribution_mass = float(
                    torch.mean(torch.sum(torch.abs(contrib), dim=1)).detach().cpu()
                )
            else:
                contribution_mass = 0.0
            yscale0 = max(float(torch.std(val_y.reshape(-1)).detach().cpu()), 1e-8)
            bias_mass = abs(float(candidate.bias.detach().reshape(-1)[0].cpu()))
            cancellation = (contribution_mass + bias_mass) / yscale0
            complexity = 0.0
            for rr in active:
                complexity += 0.5
                for ss in range(candidate.max_factors):
                    jj = int(candidate.hard_variable_choice[rr, ss].item())
                    if jj == candidate.in_dim:
                        continue
                    kk = int(candidate.hard_operator_choice[rr, ss, jj].item())
                    name = str(candidate.symbolic_library[kk])
                    complexity += float(SYMBOLIC_LIB.get(name, (None, None, 1.0))[2])
            score = float(cancellation_weight) * cancellation + float(complexity_weight) * complexity
        return float(cancellation), float(complexity), float(score)

    shallow_diag = []
    for z in refined_families:
        cancel0, complexity0, pref0 = _shallow_preference(z[1])
        shallow_diag.append((z, cancel0, complexity0, pref0))
    meta["shallow_family_validation"] = [
        {
            "operator_a": str(z[0][2]["operator_a"]),
            "operator_b": str(z[0][2]["operator_b"]),
            "validation_mse": float(z[0][0]),
            "cancellation_score": float(z[1]),
            "preference_score": float(z[3]),
        }
        for z in shallow_diag[:min(16, len(shallow_diag))]
    ]

    # Deep-polish a union of predictive leaders and low-cancellation leaders.
    # This prevents the extra optimization budget from becoming another
    # prediction-only pruning stage that can starve a mechanistic family before
    # the validation-equivalence tie-break is applied.
    deep_k = max(0, int(final_polish_topk))
    deep_selected = []
    deep_seen = set()
    if deep_k > 0:
        loss_k = max(1, int(math.ceil(deep_k / 2.0)))
        pref_k = max(0, deep_k - loss_k)
        for item in refined_families[:loss_k]:
            key = (str(item[2]["operator_a"]), str(item[2]["operator_b"]),
                   tuple(item[2]["support_a"]), tuple(item[2]["support_b"]))
            if key not in deep_seen:
                deep_seen.add(key); deep_selected.append(item)
        by_pref = sorted(shallow_diag, key=lambda z: (z[3], z[0][0]))
        for item, _, _, _ in by_pref:
            if len(deep_selected) >= deep_k:
                break
            key = (str(item[2]["operator_a"]), str(item[2]["operator_b"]),
                   tuple(item[2]["support_a"]), tuple(item[2]["support_b"]))
            if key in deep_seen:
                continue
            deep_seen.add(key); deep_selected.append(item)
    for _, shallow_model, rec, frozen in deep_selected:
        if float(rec["gate_a_affine"][1]) >= 0.0:
            gate_pair = (((1, 0, int(rec["gate"]), int(x_index)),
                          (0, 0, int(rec["gate"]), int(x_index))),)
        else:
            gate_pair = (((0, 0, int(rec["gate"]), int(x_index)),
                          (1, 0, int(rec["gate"]), int(x_index))),)
        polished, mse = _fully_symbolic_continuous_refit(
            shallow_model, train_x, train_y, val_x, val_y,
            steps=max(0, int(final_polish_steps)), lr=max(float(refine_lr) * 0.5, 2e-4),
            lbfgs_steps=max(0, int(final_polish_lbfgs_steps)),
            complementary_gate_pairs=gate_pair, show_progress=False,
        )
        if math.isfinite(float(mse)) and float(mse) < float(best_mse):
            best_model, best_mse, best_rec = polished, float(mse), dict(rec)
        if math.isfinite(float(mse)):
            selection_pool.append((float(mse), polished, dict(rec)))

    def _preference_components(candidate: SumProductKAN) -> Tuple[float, float, float]:
        """Return cancellation, description complexity, and combined score."""
        candidate.eval()
        with torch.no_grad():
            _, det = candidate(val_x, return_details=True)
            active = torch.nonzero(
                candidate.hard_rule_choice & candidate.rule_alive_mask,
                as_tuple=False,
            ).squeeze(-1).tolist()
            if active:
                contrib = det["contributions"][:, active]
                contribution_mass = float(
                    torch.mean(torch.sum(torch.abs(contrib), dim=1)).detach().cpu()
                )
            else:
                contribution_mass = 0.0
            yscale = float(torch.std(val_y.reshape(-1)).detach().cpu())
            yscale = max(yscale, 1e-8)
            bias_mass = abs(float(candidate.bias.detach().reshape(-1)[0].cpu()))
            cancellation = (contribution_mass + bias_mass) / yscale
            complexity = 0.0
            for rr in active:
                complexity += 0.5  # one symbolic rule/readout coefficient
                for ss in range(candidate.max_factors):
                    jj = int(candidate.hard_variable_choice[rr, ss].item())
                    if jj == candidate.in_dim:
                        continue
                    kk = int(candidate.hard_operator_choice[rr, ss, jj].item())
                    name = str(candidate.symbolic_library[kk])
                    complexity += float(SYMBOLIC_LIB.get(name, (None, None, 1.0))[2])
            score = float(cancellation_weight) * cancellation + float(complexity_weight) * complexity
        return float(cancellation), float(complexity), float(score)

    # Predictive equivalence comes first.  Within that validation-defined band,
    # prefer the representation requiring less internal cancellation and, only
    # weakly, lower symbolic description complexity.  This resolves cases where
    # a phase-shifted surrogate matches finite validation points microscopically
    # better by cancelling large constants, while the mechanistic partition is
    # numerically indistinguishable at the task scale.
    predictive_best = min([incumbent_mse] + [z[0] for z in selection_pool])
    yscale = max(float(torch.std(val_y.reshape(-1)).detach().cpu()), 1e-8)
    equivalence_cap = (
        predictive_best * (1.0 + max(0.0, float(equivalence_rel_mse)))
        + (max(0.0, float(equivalence_nrmse)) * yscale) ** 2
    )
    incumbent_cancel, incumbent_complexity, incumbent_pref = _preference_components(incumbent)
    eligible: List[Tuple[float, float, float, SumProductKAN, Optional[Dict[str, object]]]] = []
    if incumbent_mse <= equivalence_cap:
        eligible.append((incumbent_pref, incumbent_mse, incumbent_cancel, incumbent, None))
    for mse, cand_model, rec in selection_pool:
        if float(mse) > float(equivalence_cap):
            continue
        cancel, complexity, pref = _preference_components(cand_model)
        rec = dict(rec)
        rec["cancellation_score"] = float(cancel)
        rec["description_complexity"] = float(complexity)
        rec["preference_score"] = float(pref)
        eligible.append((float(pref), float(mse), float(cancel), cand_model, rec))
    partition_eligible = [z for z in eligible if z[4] is not None]
    if partition_eligible:
        _best_pref_part = min(partition_eligible, key=lambda z: (z[0], z[1]))
        _best_loss_part = min(partition_eligible, key=lambda z: (z[1], z[0]))
        for _prefix, _item in (("best_partition_preference", _best_pref_part), ("best_partition_predictive", _best_loss_part)):
            _pref, _mse, _cancel, _model, _rec = _item
            meta[f"{_prefix}_validation_mse"] = float(_mse)
            meta[f"{_prefix}_cancellation_score"] = float(_cancel)
            meta[f"{_prefix}_description_complexity"] = float(_rec.get("description_complexity", float("nan")))
            meta[f"{_prefix}_preference_score"] = float(_pref)
            meta[f"{_prefix}_operator_a"] = str(_rec.get("operator_a"))
            meta[f"{_prefix}_operator_b"] = str(_rec.get("operator_b"))
            meta[f"{_prefix}_support_a"] = list(_rec.get("support_a", ()))
            meta[f"{_prefix}_support_b"] = list(_rec.get("support_b", ()))
    meta["incumbent_preference_score"] = float(incumbent_pref)
    eligible.sort(key=lambda z: (z[0], z[1]))
    if eligible:
        pref, chosen_mse, chosen_cancel, chosen_model, chosen_rec = eligible[0]
        if chosen_rec is not None:
            best_model, best_mse, best_rec = chosen_model, float(chosen_mse), dict(chosen_rec)
        else:
            best_model, best_mse, best_rec = incumbent, incumbent_mse, None
    meta["predictive_best_validation_mse"] = float(predictive_best)
    meta["validation_equivalence_cap"] = float(equivalence_cap)
    meta["incumbent_cancellation_score"] = float(incumbent_cancel)
    meta["incumbent_description_complexity"] = float(incumbent_complexity)

    rel = (incumbent_mse - best_mse) / max(incumbent_mse, 1e-18)
    meta["best_validation_mse"] = float(best_mse)
    meta["relative_validation_improvement"] = float(rel)
    if best_rec is None:
        meta["reason"] = "incumbent_preferred_within_validation_band"
        return incumbent, meta

    # A candidate may replace the incumbent either by a genuine validation
    # improvement or as the lower-cancellation representation inside the same
    # validation-equivalence band.  No test labels or benchmark rule metadata
    # participate in this decision.
    validation_improved = rel >= float(min_improvement_rel)
    validation_equivalent = float(best_mse) <= float(equivalence_cap)
    if not validation_improved and not validation_equivalent:
        meta["reason"] = "validation_improvement_too_small"
        return incumbent, meta

    # Hard safety check: the selected model may use only supports from the
    # canonical effective RuleKAN support contract supplied to this rescue.
    if allowed_supports is not None:
        allowed = {frozenset(int(v) for v in s) for s in allowed_supports}
        for support in (best_rec["support_a"], best_rec["support_b"]):
            if frozenset(int(v) for v in support) not in allowed:
                raise ValueError(f"affine partition rescue selected unsupported structure {support}")
    meta.update({
        "selected": True,
        "reason": "validation_improved" if validation_improved else "validation_equivalent_lower_cancellation",
        "gate_variable": int(best_rec["gate"]),
        "support_a": list(best_rec["support_a"]),
        "support_b": list(best_rec["support_b"]),
        "operator_a": str(best_rec["operator_a"]),
        "operator_b": str(best_rec["operator_b"]),
        "gate_a_affine": list(best_rec["gate_a_affine"]),
        "gate_b_affine": list(best_rec["gate_b_affine"]),
        "coarse_train_mse": float(best_rec["coarse_train_mse"]),
        "cancellation_score": float(best_rec.get("cancellation_score", float("nan"))),
        "description_complexity": float(best_rec.get("description_complexity", float("nan"))),
        "preference_score": float(best_rec.get("preference_score", float("nan"))),
    })
    if verbose:
        print(
            "  affine partition rescue: "
            f"x{best_rec['gate']} -> {best_rec['operator_a']} / {best_rec['operator_b']}; "
            f"val RMSE {math.sqrt(incumbent_mse):.8g} -> {math.sqrt(best_mse):.8g}"
        )
    return best_model, meta




def complementary_two_rule_symbolic_rescue(
    numeric_model: SumProductKAN,
    incumbent: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    input_mean: Optional[torch.Tensor],
    input_std: Optional[torch.Tensor],
    structure_candidates: Sequence[Tuple[int, ...]],
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    max_input_dim: int = 2,
    max_samples: int = 384,
    max_support_pairs: int = 12,
    coarse_global_topk: int = 16,
    coarse_final_topk: int = 32,
    seed_restarts: int = 3,
    local_polish_steps: int = 28,
    local_polish_lr: float = 1.0e-2,
    local_lbfgs_topk: int = 32,
    local_lbfgs_steps: int = 35,
    family_final_topk: int = 10,
    final_steps: int = 240,
    final_lbfgs_steps: int = 50,
    min_improvement_rel: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Validation-gated exhaustive complementary two-rule rescue.

    Search ``(1-g) f + g h`` without changing either model's hypothesis class.
    RuleKAN is restricted to the learned/controlled support bank supplied by the
    caller; SISP may pass no support restriction.  The gate is frozen to the
    raw-coordinate membership chart, so the rescue cannot gain accuracy by
    changing the semantics of ``x``/``1-x``.

    The expensive part is deliberately staged.  Every operator pair is scored
    jointly by exact two-term least squares over a data-aware affine seed grid.
    Then we keep global leaders plus one best-counterpart lane for every left
    and right operator.  Each retained family gets several seed restarts through
    a very small native Torch optimizer before only a handful of winners are
    installed into a full SumProductKAN shell and deeply polished.  This closes
    the old ``sin/tanh`` starvation failure without turning quick benchmarks
    into an O(|library|^2) collection of full-model refits.
    """
    meta: Dict[str, object] = {
        "attempted": False, "selected": False, "reason": "not_applicable",
    }
    if int(numeric_model.in_dim) > int(max_input_dim):
        meta["reason"] = "input_dimension_above_limit"; return incumbent, meta
    if int(numeric_model.max_factors) < 2 or int(numeric_model.n_rules) < 2:
        meta["reason"] = "insufficient_rule_capacity"; return incumbent, meta
    lib = tuple(
        str(n) for n in (numeric_model.symbolic_library if library is None else library)
        if n in SYMBOLIC_LIB
    )
    if "x" not in lib or not lib:
        meta["reason"] = "identity_not_in_library"; return incumbent, meta

    gate_candidates = _raw_membership_gate_candidates(
        train_x, input_mean=input_mean, input_std=input_std,
    )
    if not gate_candidates:
        meta["reason"] = "no_unit_interval_gate_candidate"; return incumbent, meta
    gate_set = {int(v) for v in gate_candidates}
    pairs = [
        z for z in _affine_partition_structure_pairs(
            structure_candidates, allowed_supports=allowed_supports,
        ) if int(z[2]) in gate_set
    ]
    if int(max_support_pairs) > 0:
        pairs = pairs[:int(max_support_pairs)]
    if not pairs:
        meta["reason"] = "no_admissible_complementary_support_pair"; return incumbent, meta

    meta["attempted"] = True
    meta["gate_candidates"] = sorted(gate_set)
    incumbent.eval()
    with torch.no_grad():
        incumbent_mse = float(torch.mean((incumbent(val_x) - val_y) ** 2).detach().cpu())
    meta["incumbent_validation_mse"] = incumbent_mse

    if int(max_samples) > 0 and train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0]-1, int(max_samples), device=train_x.device).long()
        xx, yy = train_x[ids], train_y[ids]
    else:
        xx, yy = train_x, train_y

    def gate_affines(j: int, swapped: bool):
        if input_mean is not None and input_std is not None:
            mu = float(input_mean[int(j)].detach().cpu())
            sd = float(input_std[int(j)].detach().cpu())
            base = (1.0, sd, mu, 0.0)
            comp = (1.0, -sd, 1.0-mu, 0.0)
        else:
            base = _affine_partition_gate_chart(train_x, int(j), complement=False)
            comp = _affine_partition_gate_chart(train_x, int(j), complement=True)
        return (base, comp) if bool(swapped) else (comp, base)

    atom_cache: Dict[Tuple[int, str, float, float], torch.Tensor] = {}
    def atom(var: int, name: str, b: float, c: float) -> Optional[torch.Tensor]:
        key=(int(var),str(name),round(float(b),12),round(float(c),12))
        if key in atom_cache: return atom_cache[key]
        try: g=SYMBOLIC_LIB[str(name)][0](float(b)*xx[:,int(var)]+float(c))
        except Exception: return None
        if not torch.isfinite(g).all(): return None
        atom_cache[key]=g; return g

    # Store several affine starts per operator family.  Family selection uses
    # only the best start, but retained families later receive all top restarts.
    starts_by_family: Dict[Tuple[object, ...], List[Tuple[float, Dict[str, object]]]] = {}
    for za, zb, gate, branch_a, branch_b in pairs:
        # When both rules use the same repeated-variable support, swapping the
        # gate orientation is algebraically redundant because the exhaustive
        # operator cross-product already contains the swapped branch pair.
        orientations = (False,) if (tuple(za) == tuple(zb) and int(branch_a) == int(branch_b)) else (False, True)
        for swapped in orientations:
            aff_a, aff_b = gate_affines(int(gate), bool(swapped))
            ga = float(aff_a[1])*xx[:,int(gate)] + float(aff_a[2])
            gb = float(aff_b[1])*xx[:,int(gate)] + float(aff_b[2])
            seeded: Dict[Tuple[str,int], Tuple[torch.Tensor,List[Tuple[float,float]]]] = {}
            for side,(gcurve,bvar) in enumerate(((ga,branch_a),(gb,branch_b))):
                for name in lib:
                    curves=[]; seeds=[]
                    for b,c in _data_unit_affine_seed_grid(train_x,int(bvar),str(name)):
                        av=atom(int(bvar),str(name),float(b),float(c))
                        if av is None: continue
                        term=gcurve*av
                        if torch.isfinite(term).all():
                            curves.append(term); seeds.append((float(b),float(c)))
                    if curves: seeded[(str(name),int(side))]=(torch.stack(curves,dim=1),seeds)
            for name_a in lib:
                left=seeded.get((str(name_a),0))
                if left is None: continue
                for name_b in lib:
                    right=seeded.get((str(name_b),1))
                    if right is None: continue
                    rows=_vectorized_two_family_screen(
                        left[0].to(torch.float64), right[0].to(torch.float64),
                        yy.to(torch.float64), topk=max(1,int(seed_restarts)),
                    )
                    if not rows: continue
                    fkey=(tuple(int(v) for v in za),tuple(int(v) for v in zb),int(gate),
                          tuple(round(float(v),10) for v in aff_a[1:3]),str(name_a),str(name_b))
                    bucket=[]
                    for rank,row in enumerate(rows):
                        mse,ia,ib,bias,sa,sb=row
                        ba,ca=left[1][int(ia)]; bb,cb=right[1][int(ib)]
                        bucket.append((float(mse),{
                            "support_a":tuple(int(v) for v in za),"support_b":tuple(int(v) for v in zb),
                            "gate":int(gate),"branch_a":int(branch_a),"branch_b":int(branch_b),
                            "operator_a":str(name_a),"operator_b":str(name_b),
                            "gate_a_affine":tuple(float(v) for v in aff_a),
                            "gate_b_affine":tuple(float(v) for v in aff_b),
                            "branch_a_affine":(1.0,float(ba),float(ca),0.0),
                            "branch_b_affine":(1.0,float(bb),float(cb),0.0),
                            "bias":float(bias),"scale_a":float(sa),"scale_b":float(sb),
                            "seed_rank":int(rank),"family_key":fkey,
                        }))
                    starts_by_family[fkey]=bucket
    if not starts_by_family:
        meta["reason"]="no_finite_family_proposal"; return incumbent,meta

    leaders=[min(v,key=lambda z:z[0]) for v in starts_by_family.values()]
    meta["families_screened"]=int(len(leaders))
    groups: Dict[Tuple[object,...],List[Tuple[float,Dict[str,object]]]]={}
    for item in leaders:
        rec=item[1]
        gkey=(tuple(rec["support_a"]),tuple(rec["support_b"]),int(rec["gate"]),
              tuple(round(float(v),10) for v in rec["gate_a_affine"][1:3]))
        groups.setdefault(gkey,[]).append(item)
    selected=[]; selected_keys=set()
    def add(item):
        key=item[1]["family_key"]
        if key not in selected_keys:
            selected_keys.add(key); selected.append(item)
    for item in sorted(leaders,key=lambda z:z[0])[:max(1,int(coarse_global_topk),int(coarse_final_topk))]: add(item)
    for grows in groups.values():
        best_left={}; best_right={}
        for item in grows:
            a=str(item[1]["operator_a"]); b=str(item[1]["operator_b"])
            if a not in best_left or item[0]<best_left[a][0]: best_left[a]=item
            if b not in best_right or item[0]<best_right[b][0]: best_right[b]=item
        for item in best_left.values(): add(item)
        for item in best_right.values(): add(item)
    selected.sort(key=lambda z:z[0])
    meta["family_lane_candidates"]=int(len(selected))
    # Families in this coarse set get a small quasi-Newton polish *after* the
    # cheap multistart Adam pass.  Selection is based on the original joint
    # family screen so a slowly converging correct pair cannot lose its slot to
    # a surrogate merely because Adam has not yet reached its basin.
    _lbfgs_keys = {
        item[1]["family_key"]
        for item in selected[:max(0, int(local_lbfgs_topk))]
    }

    # Cheap native-Torch multistart polishing; gate is frozen exactly.
    polished=[]
    for _,leader in selected:
        best=None
        for coarse,rec in starts_by_family[leader["family_key"]][:max(1,int(seed_restarts))]:
            out=_polish_product_partition_pair(
                xx,yy,gate=int(rec["gate"]),gate_a_affine=rec["gate_a_affine"],
                branch_a=(int(rec["branch_a"]),),branch_b=(int(rec["branch_b"]),),
                combo_a=({"name":rec["operator_a"],"affine":rec["branch_a_affine"]},),
                combo_b=({"name":rec["operator_b"],"affine":rec["branch_b_affine"]},),
                steps=max(0,int(local_polish_steps)),lr=float(local_polish_lr),freeze_gate=True,
            )
            rr=dict(rec); rr["coarse_train_mse"]=float(coarse)
            if out is not None:
                rr["gate_a_affine"]=tuple(rec["gate_a_affine"])
                rr["gate_b_affine"]=tuple(rec["gate_b_affine"])
                rr["branch_a_affine"]=tuple(out["combo_a"][0]["affine"])
                rr["branch_b_affine"]=tuple(out["combo_b"][0]["affine"])
                rr["scale_a"]=float(out["scale_a"]); rr["scale_b"]=float(out["scale_b"]); rr["bias"]=float(out["bias"])
                score=float(out["train_mse"])
            else: score=float(coarse)
            if best is None or score<best[0]: best=(score,rr)
        if best is not None: polished.append(best)
    if not polished:
        meta["reason"]="all_local_family_polishes_nonfinite"; return incumbent,meta

    # Finish the strongest *coarse* families with a tiny LBFGS problem containing
    # only branch beta/gamma, two rule scales and a bias.  This is much cheaper
    # than a full SumProductKAN refit and remains entirely within the chosen
    # operator/support family.
    if int(local_lbfgs_steps) > 0 and _lbfgs_keys:
        qn_polished=[]
        qn_count=0
        qn_restarts=0
        for score,rec in polished:
            if rec["family_key"] not in _lbfgs_keys:
                qn_polished.append((score,rec)); continue
            best_qn=(float(score),dict(rec))
            # Revisit each of the retained *coarse* affine basins.  A family can
            # have several symmetry-equivalent starts (for example tanh(-z)
            # with a negative rule scale); choosing the best short-Adam basin
            # before quasi-Newton refinement can discard the one that converges
            # to the exact constants.
            for _coarse,_seed_rec in starts_by_family.get(rec["family_key"],())[:max(1,int(seed_restarts))]:
                out=_polish_product_partition_pair(
                    xx,yy,gate=int(_seed_rec["gate"]),gate_a_affine=_seed_rec["gate_a_affine"],
                    branch_a=(int(_seed_rec["branch_a"]),),branch_b=(int(_seed_rec["branch_b"]),),
                    combo_a=({"name":_seed_rec["operator_a"],"affine":_seed_rec["branch_a_affine"]},),
                    combo_b=({"name":_seed_rec["operator_b"],"affine":_seed_rec["branch_b_affine"]},),
                    steps=max(1,int(local_polish_steps)),lr=float(local_polish_lr),
                    lbfgs_steps=max(0,int(local_lbfgs_steps)),freeze_gate=True,
                )
                qn_restarts += 1
                if out is None or not math.isfinite(float(out["train_mse"])):
                    continue
                rr=dict(_seed_rec)
                rr["coarse_train_mse"]=float(_coarse)
                rr["gate_a_affine"]=tuple(_seed_rec["gate_a_affine"])
                rr["gate_b_affine"]=tuple(_seed_rec["gate_b_affine"])
                rr["branch_a_affine"]=tuple(out["combo_a"][0]["affine"])
                rr["branch_b_affine"]=tuple(out["combo_b"][0]["affine"])
                rr["scale_a"]=float(out["scale_a"]); rr["scale_b"]=float(out["scale_b"]); rr["bias"]=float(out["bias"])
                candidate=(float(out["train_mse"]),rr)
                if candidate[0] < best_qn[0]:
                    best_qn=candidate
            if best_qn[0] < float(score):
                qn_count += 1
            qn_polished.append(best_qn)
        polished=qn_polished
        meta["families_lbfgs_refined"]=int(qn_count)
        meta["family_lbfgs_restarts_run"]=int(qn_restarts)
    else:
        meta["families_lbfgs_refined"]=0
        meta["family_lbfgs_restarts_run"]=0

    polished.sort(key=lambda z:z[0])
    meta["families_joint_refined"]=int(len(polished))
    meta["local_best_train_mse"]=float(polished[0][0])
    meta["local_top_families"]=[
        {"operator_a":str(z[1]["operator_a"]),"operator_b":str(z[1]["operator_b"]),"train_mse":float(z[0])}
        for z in polished[:min(20,len(polished))]
    ]
    _right_diag={}
    for z in polished:
        b=str(z[1]["operator_b"])
        if b not in _right_diag: _right_diag[b]={"operator_a":str(z[1]["operator_a"]),"train_mse":float(z[0])}
    meta["local_best_by_right"]=_right_diag

    # Full-model refinement uses predictive leaders *plus* one preserved lane
    # for every operator on each side.  The latter is selected from the coarse
    # joint screen, not from the short local optimizer, so a family such as
    # sin/tanh cannot be starved merely because its best basin needs a stronger
    # refit.  Across the whole rescue this adds at most ~2*|library| lanes.
    polished_by_key={z[1]["family_key"]:z for z in polished}
    final_candidates=[]; final_keys=set()
    def _add_final(item):
        key=item[1]["family_key"]
        if key not in final_keys:
            final_keys.add(key); final_candidates.append(item)
    for item in polished[:max(1,int(family_final_topk))]: _add_final(item)
    # Coarse-family preservation: every one of the best coarse joint families
    # receives the deep full-model refit even when the cheap Adam scorer ranks
    # it poorly.  This is crucial for slowly converging families such as a
    # sin/tanh complementary pair, while the bounded top-k keeps cost finite.
    _coarse_promoted = 0
    for leader in sorted(leaders,key=lambda z:z[0])[:max(0,int(coarse_final_topk))]:
        item=polished_by_key.get(leader[1]["family_key"])
        if item is not None:
            before=len(final_keys); _add_final(item)
            if len(final_keys)>before: _coarse_promoted += 1
    meta["coarse_families_promoted_to_full_refit"] = int(_coarse_promoted)
    meta["coarse_family_promotion_topk"] = int(max(0,int(coarse_final_topk)))
    for side in ("operator_a","operator_b"):
        best_side={}
        for leader in selected:
            op=str(leader[1][side]); key=leader[1]["family_key"]
            item=polished_by_key.get(key)
            if item is None: continue
            # Preserve the lane whose *coarse joint* family score was best for
            # this operator, independent of the local-polish ranking.
            if op not in best_side or float(leader[0]) < float(best_side[op][0]):
                best_side[op]=(float(leader[0]),item)
        for _,item in best_side.values(): _add_final(item)
    final_candidates.sort(key=lambda z:z[0])
    meta["full_refit_candidates"]=int(len(final_candidates))

    x_index=numeric_model.symbolic_library.index("x")
    finalists=[]
    for local_score,rec in final_candidates:
        trial=_make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            trial.hard_rule_choice.zero_(); trial.rule_scale.zero_(); trial.bias.fill_(float(rec["bias"]))
        combo_a=({"name":"x","affine":rec["gate_a_affine"],"partition_gate":True},
                 {"name":rec["operator_a"],"affine":rec["branch_a_affine"]})
        combo_b=({"name":"x","affine":rec["gate_b_affine"],"partition_gate":True},
                 {"name":rec["operator_b"],"affine":rec["branch_b_affine"]})
        factors_a=((0,int(rec["gate"])),(1,int(rec["branch_a"])))
        factors_b=((0,int(rec["gate"])),(1,int(rec["branch_b"])))
        _install_symbolic_template(trial,0,factors_a,combo_a,initial_scale=float(rec["scale_a"]))
        _install_symbolic_template(trial,1,factors_b,combo_b,initial_scale=float(rec["scale_b"]))
        frozen=((0,0,int(rec["gate"]),int(x_index)),(1,0,int(rec["gate"]),int(x_index)))
        rr=dict(rec); rr["local_joint_train_mse"]=float(local_score)
        # The cheap native joint optimizer can occasionally land in a much
        # better basin than the generic full-model refitter.  Evaluate that
        # hard symbolic candidate on validation *before* refitting, and never
        # allow the generic optimizer to destroy it.
        trial.eval()
        with torch.no_grad():
            pre_vmse=float(torch.mean((trial(val_x)-val_y)**2).detach().cpu())
        if math.isfinite(pre_vmse):
            rr0=dict(rr); rr0["full_refit_stage"]="pre_refit"
            finalists.append((pre_vmse,copy.deepcopy(trial),rr0))
        fitted,vmse=_fully_symbolic_continuous_refit(
            trial,train_x,train_y,val_x,val_y,
            steps=max(0,int(final_steps)),lr=5e-4,lbfgs_steps=max(0,int(final_lbfgs_steps)),
            frozen_symbolic_factors=frozen,show_progress=False,
        )
        if math.isfinite(float(vmse)):
            rr1=dict(rr); rr1["full_refit_stage"]="post_refit"
            finalists.append((float(vmse),fitted,rr1))
    if not finalists:
        meta["reason"]="all_full_family_refits_nonfinite"; return incumbent,meta
    finalists.sort(key=lambda z:z[0])
    best_mse,best_model,best_rec=finalists[0]
    rel=(incumbent_mse-float(best_mse))/max(incumbent_mse,1e-18)
    meta.update({
        "best_validation_mse":float(best_mse),"relative_validation_improvement":float(rel),
        "gate_variable":int(best_rec["gate"]),"support_a":list(best_rec["support_a"]),
        "support_b":list(best_rec["support_b"]),"operator_a":str(best_rec["operator_a"]),
        "operator_b":str(best_rec["operator_b"]),"winning_seed_rank":int(best_rec.get("seed_rank",0)),
        "winning_refit_stage":str(best_rec.get("full_refit_stage","unknown")),
    })
    if rel<float(min_improvement_rel):
        meta["reason"]="validation_improvement_too_small"; return incumbent,meta
    if allowed_supports is not None:
        allowed={frozenset(int(v) for v in s) for s in allowed_supports}
        for support in (best_rec["support_a"],best_rec["support_b"]):
            if frozenset(int(v) for v in support) not in allowed:
                raise ValueError(f"complementary two-rule rescue selected unsupported structure {support}")
    meta["selected"]=True; meta["reason"]="validation_improved"
    if verbose:
        print("  complementary two-rule rescue: "
              f"x{best_rec['gate']} -> {best_rec['operator_a']} / {best_rec['operator_b']}; "
              f"val RMSE {math.sqrt(max(incumbent_mse,0.0)):.8g} -> {math.sqrt(max(float(best_mse),0.0)):.8g}")
    return best_model,meta

def low_dimensional_symbolic_family_rescue(
    numeric_model: SumProductKAN,
    incumbent: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    structure_candidates: Sequence[Tuple[int, ...]],
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    max_input_dim: int = 2,
    max_order: int = 2,
    family_refine_topk: int = 18,
    max_samples: int = 512,
    refine_steps: int = 180,
    refine_lr: float = 8e-4,
    lbfgs_steps: int = 40,
    final_topk: int = 5,
    final_steps: int = 500,
    final_lbfgs_steps: int = 100,
    min_improvement_rel: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Complete low-dimensional operator-family rescue.

    For one- and two-factor structures, the number of operator families is
    small enough to preserve one hard multistart proposal for *every* family
    through coarse screening.  Only the best validation candidates receive a
    full-model continuous refit.  The incumbent remains the default unless a
    candidate improves held-out validation error.
    """
    meta={"attempted":False,"selected":False,"reason":"not_applicable"}
    if int(numeric_model.in_dim) > int(max_input_dim):
        meta["reason"]="input_dimension_above_limit"; return incumbent,meta
    lib=tuple(str(n) for n in (numeric_model.symbolic_library if library is None else library) if n in SYMBOLIC_LIB)
    if not lib:
        meta["reason"]="empty_library"; return incumbent,meta
    allowed=None
    if allowed_supports is not None:
        allowed={frozenset(int(v) for v in ss) for ss in allowed_supports}
    structures=[]; seen=set()
    for z in structure_candidates:
        zz=tuple(int(v) for v in z)
        if not zz or len(zz)>int(max_order):
            continue
        zz=tuple(sorted(zz))
        if zz in seen:
            continue
        if allowed is not None and frozenset(zz) not in allowed:
            continue
        seen.add(zz); structures.append(zz)
    if not structures:
        meta["reason"]="no_low_dimensional_structures"; return incumbent,meta
    meta["attempted"]=True
    incumbent.eval()
    with torch.no_grad():
        incumbent_mse=float(torch.mean((incumbent(val_x)-val_y)**2).detach().cpu())
    meta["incumbent_validation_mse"]=incumbent_mse

    coarse=[]
    for struct in structures:
        n_families=max(1,len(lib)**len(struct))
        combos=hard_symbolic_tuple_screening(
            train_x,train_y,struct,lib,
            beam_width=max(n_families,len(lib)),top_tuples=n_families,
            max_samples=max(32,int(max_samples)),seeds_per_operator_prefix=1,
            use_data_unit_seeds=True,
        )
        # One proposal per operator-name tuple is guaranteed by the hard screen.
        xx=train_x
        if train_x.shape[0] > int(max_samples):
            ids=torch.linspace(0,train_x.shape[0]-1,int(max_samples),device=train_x.device).long()
            xx=train_x[ids]; yy=train_y[ids]
        else:
            yy=train_y
        yvec=yy.reshape(yy.shape[0],-1).mean(dim=1)
        for combo in combos:
            curve=_symbolic_combo_curve(xx,struct,combo)
            if curve is None: continue
            rmse,scale,offset=_best_affine_residual_rmse(curve,yvec)
            if math.isfinite(float(rmse)):
                coarse.append((float(rmse),struct,tuple(combo),float(scale),float(offset)))
    if not coarse:
        meta["reason"]="no_finite_low_dimensional_family"; return incumbent,meta
    coarse.sort(key=lambda z:z[0])
    meta["families_screened"]=int(len(coarse))
    selected=coarse[:max(1,int(family_refine_topk))]
    meta["families_refined"]=int(len(selected))
    refined=[]
    for score,struct,combo,scale,offset in selected:
        trial=_make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            trial.hard_rule_choice.zero_(); trial.rule_scale.zero_(); trial.bias.fill_(float(offset))
        factors=tuple((i,int(v)) for i,v in enumerate(struct))
        _install_symbolic_template(trial,0,factors,combo,initial_scale=float(scale))
        trial,mse=_fully_symbolic_continuous_refit(
            trial,train_x,train_y,val_x,val_y,
            steps=max(0,int(refine_steps)),lr=float(refine_lr),lbfgs_steps=max(0,int(lbfgs_steps)),show_progress=False,
        )
        if math.isfinite(float(mse)):
            refined.append((float(mse),trial,{"structure":struct,"combo":combo,"coarse_rmse":float(score)}))
    if not refined:
        meta["reason"]="all_low_dimensional_refits_nonfinite"; return incumbent,meta
    refined.sort(key=lambda z:z[0])
    pool=list(refined)
    for _,shallow,rec in refined[:max(0,int(final_topk))]:
        polished,mse=_fully_symbolic_continuous_refit(
            shallow,train_x,train_y,val_x,val_y,
            steps=max(0,int(final_steps)),lr=max(float(refine_lr)*0.5,2e-4),lbfgs_steps=max(0,int(final_lbfgs_steps)),show_progress=False,
        )
        if math.isfinite(float(mse)):
            pool.append((float(mse),polished,rec))
    best_mse,best_model,best_rec=min(pool,key=lambda z:z[0])
    rel=(incumbent_mse-float(best_mse))/max(incumbent_mse,1e-18)
    meta.update({"best_validation_mse":float(best_mse),"relative_validation_improvement":float(rel)})
    if rel < float(min_improvement_rel):
        meta["reason"]="validation_improvement_too_small"; return incumbent,meta
    names=[str(c["name"]) for c in best_rec["combo"]]
    meta.update({"selected":True,"reason":"validation_improved","structure":list(best_rec["structure"]),"operators":names,
                 "coarse_rmse":float(best_rec["coarse_rmse"])})
    if verbose:
        print("  low-dimensional family rescue: "
              f"{best_rec['structure']} {names}; val RMSE {math.sqrt(incumbent_mse):.8g} -> {math.sqrt(float(best_mse)):.8g}")
    return best_model,meta

def _product_partition_structure_pairs(
    structure_candidates: Sequence[Tuple[int, ...]],
    *,
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    min_order: int = 3,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], int, Tuple[int, ...], Tuple[int, ...]]]:
    """Enumerate complementary-gate mechanisms with product-valued branches.

    Each selected rule contains one gate factor plus one or more branch factors.
    Multiplicity is preserved, so structures such as ``(0,0,1)`` correctly
    represent a gate on ``x0`` together with a branch atom on ``x0`` and another
    atom on ``x1``.  The routine is support-contract preserving: when
    ``allowed_supports`` is supplied it never introduces a distinct-variable
    dependency absent from the numerical RuleKAN evidence.
    """
    structures=[]; seen=set()
    for z in structure_candidates:
        zz=tuple(sorted(int(v) for v in z))
        if len(zz) < int(min_order) or zz in seen:
            continue
        seen.add(zz); structures.append(zz)
    if allowed_supports is not None:
        allowed={frozenset(int(v) for v in ss) for ss in allowed_supports}
        structures=[z for z in structures if frozenset(z) in allowed]
    out=[]; out_seen=set()
    for ia,za in enumerate(structures):
        for ib in range(ia,len(structures)):
            zb=structures[ib]
            for gate in sorted(set(za).intersection(zb)):
                aa=list(za); bb=list(zb)
                aa.remove(int(gate)); bb.remove(int(gate))
                if not aa or not bb:
                    continue
                key=(za,zb,int(gate),tuple(aa),tuple(bb))
                if key in out_seen:
                    continue
                out_seen.add(key); out.append(key)
    return out


def _symbolic_combo_curve(
    x: torch.Tensor,
    structure: Sequence[int],
    combo: Sequence[Dict[str, object]],
) -> Optional[torch.Tensor]:
    """Evaluate one exact hard symbolic product proposal."""
    if len(structure) != len(combo):
        return None
    prod=torch.ones(x.shape[0],device=x.device,dtype=x.dtype)
    try:
        for var,cand in zip(structure,combo):
            name=str(cand["name"]); fun=SYMBOLIC_LIB[name][0]
            aff=_canonical_symbolic_affine(cand["affine"],name)
            _,b,c,_=aff
            g=fun(float(b)*x[:,int(var)] + float(c))
            if not torch.isfinite(g).all():
                return None
            prod=prod*g
    except Exception:
        return None
    return prod if torch.isfinite(prod).all() else None


def _vectorized_two_family_screen(
    left: torch.Tensor,
    right: torch.Tensor,
    y: torch.Tensor,
    *,
    topk: int,
) -> List[Tuple[float,int,int,float,float,float]]:
    """Rank ``bias + a*left_i + b*right_j`` fits for all family pairs."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        return []
    yy=y.reshape(y.shape[0],-1).mean(dim=1)
    ym=yy.mean(); yc=yy-ym
    L=left-left.mean(dim=0,keepdim=True); R=right-right.mean(dim=0,keepdim=True)
    g00=torch.sum(L*L,dim=0).clamp_min(1e-12)
    g11=torch.sum(R*R,dim=0).clamp_min(1e-12)
    g01=L.T@R
    r0=L.T@yc; r1=R.T@yc
    det=g00[:,None]*g11[None,:]-g01.square()
    good=torch.isfinite(det) & (det.abs()>1e-12)
    a=torch.zeros_like(g01); b=torch.zeros_like(g01)
    a[good]=(r0[:,None]*g11[None,:]-r1[None,:]*g01)[good]/det[good]
    b[good]=(r1[None,:]*g00[:,None]-r0[:,None]*g01)[good]/det[good]
    yss=torch.sum(yc*yc)
    sse=yss-(a*r0[:,None]+b*r1[None,:])
    mse=(sse/max(1,int(yy.numel()))).clamp_min(0.0)
    mse[~good]=float("inf")
    flat=mse.reshape(-1)
    k=min(max(1,int(topk)),int(torch.isfinite(flat).sum().item()))
    if k<=0:
        return []
    vals,ids=torch.topk(flat,k=k,largest=False)
    nr=right.shape[1]
    out=[]
    lmean=left.mean(dim=0); rmean=right.mean(dim=0)
    for vv,idx in zip(vals.tolist(),ids.tolist()):
        i=int(idx//nr); j=int(idx%nr)
        aa=float(a[i,j].detach().cpu()); bb=float(b[i,j].detach().cpu())
        bias=float((ym-aa*lmean[i]-bb*rmean[j]).detach().cpu())
        out.append((float(vv),i,j,bias,aa,bb))
    return out



def _polish_product_partition_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gate: int,
    gate_a_affine: Sequence[float],
    branch_a: Sequence[int],
    branch_b: Sequence[int],
    combo_a: Sequence[Dict[str, object]],
    combo_b: Sequence[Dict[str, object]],
    steps: int = 40,
    lr: float = 1.2e-2,
    lbfgs_steps: int = 0,
    freeze_gate: bool = False,
) -> Optional[Dict[str, object]]:
    """Cheap joint polish for one complementary product-family pair.

    The expensive full :class:`SumProductKAN` shell is unnecessary while we
    are still deciding which *operator families* deserve a GSR refit.  This
    helper optimizes only the tied affine gate chart, the branch beta/gamma
    constants, and the two linear rule scales/bias.  It is intentionally a
    proposal scorer: the winning families are subsequently reinstalled in a
    normal symbolic model and validation-gated there.

    Keeping the two branches in one objective is important for fuzzy routing.
    A correct branch can look mediocre when fitted in isolation because the
    complementary branch explains a correlated part of the same target.
    """
    if len(branch_a) != len(combo_a) or len(branch_b) != len(combo_b):
        return None
    yy = y.reshape(y.shape[0], -1).mean(dim=1)
    dtype, device = x.dtype, x.device

    # Represent gate B exactly as 1 - gate A throughout optimization.
    ga0 = _canonical_symbolic_affine(gate_a_affine, "x")
    gate_bc = torch.tensor(
        [float(ga0[1]), float(ga0[2])], device=device, dtype=dtype,
        requires_grad=not bool(freeze_gate),
    )
    aff_a = torch.tensor(
        [_canonical_symbolic_affine(c["affine"], str(c["name"])) for c in combo_a],
        device=device, dtype=dtype, requires_grad=True,
    )
    aff_b = torch.tensor(
        [_canonical_symbolic_affine(c["affine"], str(c["name"])) for c in combo_b],
        device=device, dtype=dtype, requires_grad=True,
    )
    scale_a = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
    scale_b = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
    bias = yy.mean().detach().clone().requires_grad_(True)

    # Initialize the readout by least squares for the supplied hard curves.
    with torch.no_grad():
        def branch_curve(branch, combo, aff):
            out = torch.ones_like(yy)
            for slot, (var, cand) in enumerate(zip(branch, combo)):
                fun = SYMBOLIC_LIB[str(cand["name"])][0]
                out = out * fun(aff[slot, 1] * x[:, int(var)] + aff[slot, 2])
            return out
        try:
            ua = gate_bc[0] * x[:, int(gate)] + gate_bc[1]
            ha = ua * branch_curve(branch_a, combo_a, aff_a)
            hb = (1.0 - ua) * branch_curve(branch_b, combo_b, aff_b)
            fit = _vectorized_two_family_screen(ha[:, None], hb[:, None], yy[:, None], topk=1)
            if fit:
                _, _, _, b0, sa0, sb0 = fit[0]
                bias.copy_(torch.as_tensor(b0, device=device, dtype=dtype))
                scale_a.copy_(torch.as_tensor(sa0, device=device, dtype=dtype))
                scale_b.copy_(torch.as_tensor(sb0, device=device, dtype=dtype))
        except Exception:
            pass

    params = [aff_a, aff_b, scale_a, scale_b, bias]
    if not bool(freeze_gate):
        params.insert(0, gate_bc)
    opt = torch.optim.Adam(params, lr=float(lr))
    best_mse = float("inf")
    best = None
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        try:
            ua = gate_bc[0] * x[:, int(gate)] + gate_bc[1]
            pa = torch.ones_like(yy)
            for slot, (var, cand) in enumerate(zip(branch_a, combo_a)):
                fun = SYMBOLIC_LIB[str(cand["name"])][0]
                g = fun(aff_a[slot, 1] * x[:, int(var)] + aff_a[slot, 2])
                if not torch.isfinite(g).all():
                    raise FloatingPointError
                pa = pa * g
            pb = torch.ones_like(yy)
            for slot, (var, cand) in enumerate(zip(branch_b, combo_b)):
                fun = SYMBOLIC_LIB[str(cand["name"])][0]
                g = fun(aff_b[slot, 1] * x[:, int(var)] + aff_b[slot, 2])
                if not torch.isfinite(g).all():
                    raise FloatingPointError
                pb = pb * g
            pred = bias + scale_a * ua * pa + scale_b * (1.0 - ua) * pb
            loss = torch.mean((pred - yy) ** 2)
        except Exception:
            break
        if not torch.isfinite(loss):
            break
        loss.backward()
        # Canonical factors use only beta/gamma.  Amplitude belongs to the
        # rule scales and constants to the global bias.
        if aff_a.grad is not None:
            aff_a.grad[:, 0].zero_(); aff_a.grad[:, 3].zero_()
        if aff_b.grad is not None:
            aff_b.grad[:, 0].zero_(); aff_b.grad[:, 3].zero_()
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        with torch.no_grad():
            if not bool(freeze_gate):
                gate_bc[0].clamp_(-10.0, 10.0); gate_bc[1].clamp_(-4.0, 4.0)
            for aff in (aff_a, aff_b):
                aff[:, 0].fill_(1.0); aff[:, 1].clamp_(-10.0, 10.0)
                aff[:, 2].clamp_(-6.0 * math.pi, 6.0 * math.pi); aff[:, 3].zero_()
            scale_a.clamp_(-1e3, 1e3); scale_b.clamp_(-1e3, 1e3); bias.clamp_(-1e3, 1e3)
            mse = float(loss.detach().cpu())
            if math.isfinite(mse) and mse < best_mse:
                best_mse = mse
                best = (
                    gate_bc.detach().clone(), aff_a.detach().clone(), aff_b.detach().clone(),
                    scale_a.detach().clone(), scale_b.detach().clone(), bias.detach().clone(),
                )

    # A short quasi-Newton finish is disproportionately useful for the tiny
    # complementary-family problem: coarse screening often finds the correct
    # operator pair but Adam needs hundreds of steps to settle its beta/gamma
    # constants.  This optimizer changes no supports, operators, or gate
    # semantics; it only refines continuous constants inside the already chosen
    # family.  Restore the best Adam iterate first so a bad late iterate cannot
    # poison the polish.
    if best is not None and int(lbfgs_steps) > 0:
        with torch.no_grad():
            gbc0, aa0, ab0, sa0, sb0, bb0 = best
            gate_bc.copy_(gbc0); aff_a.copy_(aa0); aff_b.copy_(ab0)
            scale_a.copy_(sa0); scale_b.copy_(sb0); bias.copy_(bb0)

        def _pair_loss(backward: bool = False):
            ua = gate_bc[0] * x[:, int(gate)] + gate_bc[1]
            pa = torch.ones_like(yy)
            for slot, (var, cand) in enumerate(zip(branch_a, combo_a)):
                fun = SYMBOLIC_LIB[str(cand["name"])][0]
                pa = pa * fun(aff_a[slot, 1] * x[:, int(var)] + aff_a[slot, 2])
            pb = torch.ones_like(yy)
            for slot, (var, cand) in enumerate(zip(branch_b, combo_b)):
                fun = SYMBOLIC_LIB[str(cand["name"])][0]
                pb = pb * fun(aff_b[slot, 1] * x[:, int(var)] + aff_b[slot, 2])
            pred = bias + scale_a * ua * pa + scale_b * (1.0 - ua) * pb
            loss = torch.mean((pred - yy) ** 2)
            if backward:
                loss.backward()
                if aff_a.grad is not None:
                    aff_a.grad[:, 0].zero_(); aff_a.grad[:, 3].zero_()
                if aff_b.grad is not None:
                    aff_b.grad[:, 0].zero_(); aff_b.grad[:, 3].zero_()
            return loss

        try:
            qn = torch.optim.LBFGS(
                params, max_iter=max(1, int(lbfgs_steps)), history_size=12,
                tolerance_grad=1e-10, tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )
            def closure():
                qn.zero_grad(set_to_none=True)
                loss = _pair_loss(backward=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite complementary-family LBFGS loss")
                return loss
            qn.step(closure)
            with torch.no_grad():
                if not bool(freeze_gate):
                    gate_bc[0].clamp_(-10.0, 10.0); gate_bc[1].clamp_(-4.0, 4.0)
                for aff in (aff_a, aff_b):
                    aff[:, 0].fill_(1.0); aff[:, 1].clamp_(-10.0, 10.0)
                    aff[:, 2].clamp_(-6.0 * math.pi, 6.0 * math.pi); aff[:, 3].zero_()
                scale_a.clamp_(-1e3, 1e3); scale_b.clamp_(-1e3, 1e3); bias.clamp_(-1e3, 1e3)
                loss = _pair_loss(backward=False)
                if torch.isfinite(loss):
                    mse = float(loss.detach().cpu())
                    if mse < best_mse:
                        best_mse = mse
                        best = (
                            gate_bc.detach().clone(), aff_a.detach().clone(), aff_b.detach().clone(),
                            scale_a.detach().clone(), scale_b.detach().clone(), bias.detach().clone(),
                        )
        except Exception:
            # The family search is a rescue.  A pathological operator domain or
            # line-search step must never invalidate the incumbent model.
            pass

    if best is None:
        return None
    gbc, aa, ab, sa, sb, bb = best
    ca=[]; cb=[]
    for slot,cand in enumerate(combo_a):
        cc=dict(cand); cc["affine"]=_canonical_symbolic_affine(aa[slot].cpu().tolist(), str(cand["name"])); ca.append(cc)
    for slot,cand in enumerate(combo_b):
        cc=dict(cand); cc["affine"]=_canonical_symbolic_affine(ab[slot].cpu().tolist(), str(cand["name"])); cb.append(cc)
    beta=float(gbc[0].cpu()); gamma=float(gbc[1].cpu())
    return {
        "train_mse": float(best_mse),
        "gate_a_affine": (1.0, beta, gamma, 0.0),
        "gate_b_affine": (1.0, -beta, 1.0 - gamma, 0.0),
        "combo_a": tuple(ca), "combo_b": tuple(cb),
        "scale_a": float(sa.cpu()), "scale_b": float(sb.cpu()), "bias": float(bb.cpu()),
    }

def product_partition_symbolic_rescue(
    numeric_model: SumProductKAN,
    incumbent: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    structure_candidates: Sequence[Tuple[int, ...]],
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    branch_family_pool: int = 72,
    pair_beam: int = 28,
    family_lane_left: int = 48,
    family_lane_right: int = 16,
    family_lane_support_pairs: int = 2,
    family_lane_polish_steps: int = 40,
    family_lane_polish_lr: float = 1.2e-2,
    max_samples: int = 384,
    max_support_pairs: int = 32,
    shallow_steps: int = 100,
    shallow_lr: float = 1.0e-3,
    shallow_lbfgs_steps: int = 16,
    deep_topk: int = 8,
    deep_steps: int = 420,
    deep_lbfgs_steps: int = 70,
    min_improvement_rel: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Validation-gated complementary partition rescue with product branches.

    This extends :func:`affine_partition_symbolic_rescue` from
    ``gate*unary + complement*unary`` to arbitrary branch products that fit in
    the model's symbolic factor budget.

    The search intentionally has two lanes.  A global joint least-squares lane
    keeps the best complete branch pairs, while *family lanes* preserve the
    best counterpart for many individual left/right operator families on the
    most promising support mechanisms.  Those lane candidates receive a cheap
    joint affine polish before the expensive full-model refit.  This prevents a
    mechanistically correct family such as ``exp * sin`` from being discarded
    merely because its initial constants rank poorly when the complementary
    branch is fitted at the same coarse seed.
    """
    meta={"attempted":False,"selected":False,"reason":"not_applicable"}
    if numeric_model.max_factors < 3 or numeric_model.n_rules < 2:
        return incumbent,meta
    lib=tuple(str(n) for n in (numeric_model.symbolic_library if library is None else library) if n in SYMBOLIC_LIB)
    if "x" not in lib or not lib:
        meta["reason"]="identity_not_in_library"; return incumbent,meta
    mechanisms=_product_partition_structure_pairs(
        structure_candidates,allowed_supports=allowed_supports,min_order=3,
    )
    if not mechanisms:
        meta["reason"]="no_product_partition_support_pair"; return incumbent,meta
    # Prefer smaller supports deterministically; this is task-label independent.
    mechanisms.sort(key=lambda z:(len(z[0])+len(z[1]),z[0],z[1],z[2]))
    if int(max_support_pairs)>0:
        mechanisms=mechanisms[:int(max_support_pairs)]
    meta["attempted"]=True
    incumbent.eval()
    with torch.no_grad():
        incumbent_mse=float(torch.mean((incumbent(val_x)-val_y)**2).detach().cpu())
    meta["incumbent_validation_mse"]=incumbent_mse

    if train_x.shape[0] > int(max_samples):
        ids=torch.linspace(0,train_x.shape[0]-1,int(max_samples),device=train_x.device).long()
        xx,yy=train_x[ids],train_y[ids]
    else:
        xx,yy=train_x,train_y
    yvec=yy.reshape(yy.shape[0],-1).mean(dim=1)
    x_index=numeric_model.symbolic_library.index("x")

    # Each pack corresponds to one (support pair, gate orientation) mechanism.
    # Store all pair rankings cheaply, but only expand family lanes for the
    # strongest few packs below.
    packs=[]
    global_records=[]

    def make_record(pack,mse,ia,ib,bias,sa,sb,source):
        return (float(mse),{
            "support_a":pack["support_a"],"support_b":pack["support_b"],"gate":int(pack["gate"]),
            "branch_a":pack["branch_a"],"branch_b":pack["branch_b"],
            "gate_a_affine":pack["gate_a_affine"],"gate_b_affine":pack["gate_b_affine"],
            "combo_a":pack["ca"][int(ia)],"combo_b":pack["cb"][int(ib)],
            "bias":float(bias),"scale_a":float(sa),"scale_b":float(sb),
            "proposal_source":str(source),
        })

    for za,zb,gate,branch_a,branch_b in mechanisms:
        for swapped in (False,True):
            aff_a=_affine_partition_gate_chart(train_x,gate,complement=bool(swapped))
            aff_b=_affine_partition_gate_chart(train_x,gate,complement=not bool(swapped))
            ga=aff_a[1]*xx[:,gate]+aff_a[2]
            gb=aff_b[1]*xx[:,gate]+aff_b[2]

            def branch_candidates(branch,gate_curve):
                # Conditional branch screening where this gate dominates.  The
                # fallback keeps the full coarse set if a narrow gate leaves too
                # few points.
                mask=torch.abs(gate_curve)>=0.55
                if int(mask.sum().item())>=max(24,4*len(branch)):
                    bx=xx[mask]
                    denom=gate_curve[mask]
                    safe=torch.where(denom.abs()<1e-3, torch.sign(denom).where(denom!=0, torch.ones_like(denom))*1e-3, denom)
                    by=(yvec[mask]/safe)[:,None]
                else:
                    bx=xx; by=yy
                combos=hard_symbolic_tuple_screening(
                    bx,by,branch,lib,
                    beam_width=max(int(branch_family_pool),len(lib)**min(2,len(branch))),
                    top_tuples=max(1,int(branch_family_pool)),
                    max_samples=min(int(max_samples),int(bx.shape[0])),
                    seeds_per_operator_prefix=1,
                    use_data_unit_seeds=True,
                )
                curves=[]; kept=[]; seen=set()
                for combo in combos:
                    names=tuple(str(c["name"]) for c in combo)
                    if names in seen: continue
                    seen.add(names)
                    curve=_symbolic_combo_curve(xx,branch,combo)
                    if curve is None: continue
                    curves.append(gate_curve*curve); kept.append(tuple(combo))
                return kept,curves

            ca,ha=branch_candidates(branch_a,ga); cb,hb=branch_candidates(branch_b,gb)
            if not ha or not hb:
                continue
            H0=torch.stack(ha,dim=1); H1=torch.stack(hb,dim=1)
            # Ranking the full A*B matrix is cheap and lets us later preserve a
            # best-counterpart lane for individual operator families.
            ranked=_vectorized_two_family_screen(H0,H1,yy,topk=max(1,len(ca)*len(cb)))
            if not ranked:
                continue
            pack={
                "support_a":za,"support_b":zb,"gate":int(gate),
                "branch_a":tuple(branch_a),"branch_b":tuple(branch_b),
                "gate_a_affine":aff_a,"gate_b_affine":aff_b,
                "ca":ca,"cb":cb,"ranked":ranked,"best_mse":float(ranked[0][0]),
            }
            packs.append(pack)
            for mse,ia,ib,bias,sa,sb in ranked[:max(1,int(pair_beam))]:
                global_records.append(make_record(pack,mse,ia,ib,bias,sa,sb,"global"))

    if not packs or not global_records:
        meta["reason"]="no_finite_product_partition_family"; return incumbent,meta

    # Add family-diverse lanes only for the most promising support mechanisms.
    # Within each lane the best counterpart is selected from the *joint* A*B
    # objective, so a branch is not judged in isolation.
    lane_records=[]
    top_packs=sorted(packs,key=lambda p:p["best_mse"])[:max(0,int(family_lane_support_pairs))]
    for pack in top_packs:
        best_left={}; best_right={}
        for row in pack["ranked"]:
            mse,ia,ib,bias,sa,sb=row
            if int(ia) not in best_left:
                best_left[int(ia)]=row
            if int(ib) not in best_right:
                best_right[int(ib)]=row
        for ia in range(min(len(pack["ca"]),max(0,int(family_lane_left)))):
            row=best_left.get(int(ia))
            if row is not None:
                lane_records.append(make_record(pack,*row,"left_family_lane"))
        for ib in range(min(len(pack["cb"]),max(0,int(family_lane_right)))):
            row=best_right.get(int(ib))
            if row is not None:
                lane_records.append(make_record(pack,*row,"right_family_lane"))

    candidates=global_records+lane_records
    # De-duplicate exact family/support/orientation proposals before the cheap
    # joint affine polish.
    dedup=[]; seen=set()
    for item in sorted(candidates,key=lambda z:z[0]):
        rec=item[1]
        key=(tuple(rec["support_a"]),tuple(rec["support_b"]),int(rec["gate"]),
             tuple(round(float(v),8) for v in rec["gate_a_affine"][1:3]),
             tuple(str(c["name"]) for c in rec["combo_a"]),tuple(str(c["name"]) for c in rec["combo_b"]))
        if key in seen: continue
        seen.add(key); dedup.append(item)
    meta["families_screened"]=int(sum(len(p["ranked"]) for p in packs))
    meta["family_lane_candidates"]=int(len(lane_records))
    meta["prepolish_candidates"]=int(len(dedup))

    # Cheap joint family polish.  This is the key anti-pruning step: only after
    # both branches and their tied gate have had a chance to adjust constants do
    # we choose the small expensive full-model beam.
    polished_records=[]
    for coarse,rec in dedup:
        polished=_polish_product_partition_pair(
            xx,yy,gate=int(rec["gate"]),gate_a_affine=rec["gate_a_affine"],
            branch_a=rec["branch_a"],branch_b=rec["branch_b"],
            combo_a=rec["combo_a"],combo_b=rec["combo_b"],
            steps=max(0,int(family_lane_polish_steps)),lr=float(family_lane_polish_lr),
        )
        rr=dict(rec); rr["coarse_train_mse"]=float(coarse)
        if polished is not None:
            rr.update({k:v for k,v in polished.items() if k!="train_mse"})
            score=float(polished["train_mse"])
        else:
            score=float(coarse)
        polished_records.append((score,rr))
    polished_records.sort(key=lambda z:z[0])

    selected=[]; seen=set()
    for item in polished_records:
        rec=item[1]
        key=(tuple(str(c["name"]) for c in rec["combo_a"]),tuple(str(c["name"]) for c in rec["combo_b"]),
             tuple(rec["support_a"]),tuple(rec["support_b"]),int(rec["gate"]),
             tuple(round(float(v),6) for v in rec["gate_a_affine"][1:3]))
        if key in seen: continue
        seen.add(key); selected.append(item)
        if len(selected)>=max(1,int(pair_beam)): break
    meta["families_refined"]=int(len(selected))

    refined=[]
    for local_score,rec in selected:
        trial=_make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            trial.hard_rule_choice.zero_(); trial.rule_scale.zero_(); trial.bias.fill_(float(rec["bias"]))
        combo_a=({"name":"x","affine":rec["gate_a_affine"],"partition_gate":True},)+tuple(rec["combo_a"])
        combo_b=({"name":"x","affine":rec["gate_b_affine"],"partition_gate":True},)+tuple(rec["combo_b"])
        factors_a=((0,int(rec["gate"])),)+tuple((i+1,int(v)) for i,v in enumerate(rec["branch_a"]))
        factors_b=((0,int(rec["gate"])),)+tuple((i+1,int(v)) for i,v in enumerate(rec["branch_b"]))
        if len(factors_a)>trial.max_factors or len(factors_b)>trial.max_factors:
            continue
        _install_symbolic_template(trial,0,factors_a,combo_a,initial_scale=float(rec["scale_a"]))
        _install_symbolic_template(trial,1,factors_b,combo_b,initial_scale=float(rec["scale_b"]))
        if float(rec["gate_a_affine"][1])>=0:
            gate_pair=(((1,0,int(rec["gate"]),int(x_index)),(0,0,int(rec["gate"]),int(x_index))),)
        else:
            gate_pair=(((0,0,int(rec["gate"]),int(x_index)),(1,0,int(rec["gate"]),int(x_index))),)
        trial,mse=_fully_symbolic_continuous_refit(
            trial,train_x,train_y,val_x,val_y,
            steps=max(0,int(shallow_steps)),lr=float(shallow_lr),lbfgs_steps=max(0,int(shallow_lbfgs_steps)),
            complementary_gate_pairs=gate_pair,show_progress=False,
        )
        if math.isfinite(float(mse)):
            rr=dict(rec); rr["local_joint_train_mse"]=float(local_score); rr["gate_pair"]=gate_pair
            refined.append((float(mse),trial,rr))
    if not refined:
        meta["reason"]="all_product_partition_refits_nonfinite"; return incumbent,meta
    refined.sort(key=lambda z:z[0])
    pool=list(refined)
    for _,shallow,rec in refined[:max(0,int(deep_topk))]:
        polished,mse=_fully_symbolic_continuous_refit(
            shallow,train_x,train_y,val_x,val_y,
            steps=max(0,int(deep_steps)),lr=max(float(shallow_lr)*0.5,2e-4),
            lbfgs_steps=max(0,int(deep_lbfgs_steps)),complementary_gate_pairs=rec["gate_pair"],show_progress=False,
        )
        if math.isfinite(float(mse)):
            pool.append((float(mse),polished,rec))
    best_mse,best_model,best_rec=min(pool,key=lambda z:z[0])
    rel=(incumbent_mse-float(best_mse))/max(incumbent_mse,1e-18)
    meta.update({"best_validation_mse":float(best_mse),"relative_validation_improvement":float(rel)})
    if rel < float(min_improvement_rel):
        meta["reason"]="validation_improvement_too_small"; return incumbent,meta
    if allowed_supports is not None:
        allowed={frozenset(int(v) for v in ss) for ss in allowed_supports}
        for support in (best_rec["support_a"],best_rec["support_b"]):
            if frozenset(int(v) for v in support) not in allowed:
                raise ValueError(f"product partition rescue selected unsupported structure {support}")
    meta.update({
        "selected":True,"reason":"validation_improved","gate_variable":int(best_rec["gate"]),
        "support_a":list(best_rec["support_a"]),"support_b":list(best_rec["support_b"]),
        "operators_a":[str(c["name"]) for c in best_rec["combo_a"]],
        "operators_b":[str(c["name"]) for c in best_rec["combo_b"]],
        "coarse_train_mse":float(best_rec.get("coarse_train_mse",float("nan"))),
        "local_joint_train_mse":float(best_rec.get("local_joint_train_mse",float("nan"))),
        "proposal_source":str(best_rec.get("proposal_source","unknown")),
    })
    if verbose:
        print("  product partition rescue: "
              f"x{best_rec['gate']} -> {meta['operators_a']} / {meta['operators_b']}; "
              f"val RMSE {math.sqrt(incumbent_mse):.8g} -> {math.sqrt(float(best_mse)):.8g}")
    return best_model,meta


def _raw_membership_gate_candidates(
    train_x: torch.Tensor,
    *,
    input_mean: Optional[torch.Tensor] = None,
    input_std: Optional[torch.Tensor] = None,
    lower: float = -0.08,
    upper: float = 1.08,
    min_span: float = 0.70,
) -> List[int]:
    """Infer plausible fuzzy-membership variables from observed raw ranges.

    RuleKAN is normally trained on standardized coordinates.  This helper
    reconstructs raw coordinates using the benchmark's training-only affine
    normalization and marks variables whose observed support is approximately
    the unit interval.  No task labels or expected symbolic structures are
    consulted.
    """
    if train_x.ndim != 2:
        return []
    d = int(train_x.shape[1])
    if input_mean is None:
        mean = torch.zeros(d, device=train_x.device, dtype=train_x.dtype)
    else:
        mean = input_mean.to(device=train_x.device, dtype=train_x.dtype).reshape(-1)
    if input_std is None:
        std = torch.ones(d, device=train_x.device, dtype=train_x.dtype)
    else:
        std = input_std.to(device=train_x.device, dtype=train_x.dtype).reshape(-1)
    if mean.numel() != d or std.numel() != d:
        return []
    raw = train_x * std[None, :] + mean[None, :]
    out: List[int] = []
    for j in range(d):
        col = raw[:, j]
        finite = col[torch.isfinite(col)]
        if finite.numel() < 8:
            continue
        lo = float(finite.min().detach().cpu())
        hi = float(finite.max().detach().cpu())
        span = hi - lo
        if lo >= float(lower) and hi <= float(upper) and span >= float(min_span):
            out.append(int(j))
    return out


def _exact_raw_gate_affine(
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    variable: int,
    *,
    complement: bool,
) -> Tuple[float, float, float, float]:
    """Identity chart equal to raw ``x`` or ``1-x`` on standardized inputs."""
    j = int(variable)
    mean = float(input_mean.reshape(-1)[j].detach().cpu())
    std = float(input_std.reshape(-1)[j].detach().cpu())
    if complement:
        return (1.0, -std, 1.0 - mean, 0.0)
    return (1.0, std, mean, 0.0)


def _weighted_unary_leaf_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    weight_train: torch.Tensor,
    weight_val: torch.Tensor,
    *,
    variables: Sequence[int],
    library: Sequence[str],
    input_mean: Optional[torch.Tensor] = None,
    input_std: Optional[torch.Tensor] = None,
    topk: int = 10,
    max_samples: int = 384,
) -> List[Dict[str, object]]:
    """Rank unary branch atoms for one leaf of a recursive fuzzy partition.

    Unlike the generic hard tuple screen, this routine does *not* collapse an
    operator family to the seed that best explains a conditional pseudo-target
    before routing is accounted for.  Every deterministic affine seed is first
    scored in the actual routed one-term model ``b + a*w*g(beta*x+gamma)``;
    only then is the best seed retained per (variable, operator) family.  This
    family-preserving order is important for nested trees, where the other
    branches contaminate a conditional target and can otherwise make the right
    frequency/shift look poor too early.
    """
    lib = tuple(str(n) for n in library if n in SYMBOLIC_LIB and str(n) != "x")
    if not lib:
        return []
    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0]-1, int(max_samples), device=train_x.device).long()
        tx = train_x[ids]; ty = train_y[ids]
        wt = weight_train.reshape(-1)[ids]
    else:
        tx = train_x; ty = train_y; wt = weight_train.reshape(-1)
    ytr = ty.reshape(ty.shape[0], -1).mean(dim=1)
    yv = val_y.reshape(val_y.shape[0], -1).mean(dim=1)
    wv = weight_val.reshape(-1)
    best_by_family: Dict[Tuple[int,str], Tuple[float, Dict[str, object]]] = {}
    for var in variables:
        j=int(var)
        for name in lib:
            seed_grid = list(_data_unit_affine_seed_grid(train_x, j, name))
            # Add a raw-coordinate chart.  The benchmark feeds standardized z
            # to RuleKAN, but meaningful symbolic frequencies live naturally in
            # the original x units.  These starts are generic dense scientific
            # scales, transformed exactly through x = mean + std*z.
            if input_mean is not None and input_std is not None:
                mu=float(input_mean.reshape(-1)[j].detach().cpu())
                sd=float(input_std.reshape(-1)[j].detach().cpu())
                if name in {"sin","cos"}:
                    rbs=(0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,-1.0,-2.0)
                    rcs=(0.0,-0.5*math.pi,0.5*math.pi)
                elif name=="exp":
                    rbs=(-2.0,-1.5,-1.2,-1.0,-0.8,-0.6,-0.4,0.4,0.6,0.8,1.0,1.2,1.5,2.0)
                    rcs=(0.0,-0.5,0.5)
                elif name in {"tanh","arctan"}:
                    rbs=(0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,-0.5,-1.0,-2.0)
                    rcs=(0.0,-0.5,0.5)
                else:
                    rbs=(0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,-0.5,-1.0,-1.5)
                    rcs=(0.0,-0.5,0.5)
                seed_grid.extend((float(rb)*sd, float(rb)*mu+float(rc)) for rb in rbs for rc in rcs)
            # preserve order while de-duplicating
            uniq=[];seen_seed=set()
            for b,c in seed_grid:
                key=(round(float(b),12),round(float(c),12))
                if key in seen_seed: continue
                seen_seed.add(key);uniq.append((float(b),float(c)))
            for b,c in uniq:
                try:
                    gtr = SYMBOLIC_LIB[name][0](float(b)*tx[:,j] + float(c))
                    gv = SYMBOLIC_LIB[name][0](float(b)*val_x[:,j] + float(c))
                except Exception:
                    continue
                if not torch.isfinite(gtr).all() or not torch.isfinite(gv).all():
                    continue
                # Prefer conditional branch identification where this route is
                # genuinely active.  At a fuzzy-tree corner, y/w is dominated
                # by the corresponding leaf, whereas a global one-term fit is
                # contaminated by the other branches and systematically favors
                # polynomial surrogates.  Fall back to the routed full-target
                # score when too few high-membership samples are available.
                mtr = wt.abs() >= 0.55
                mv = wv.abs() >= 0.55
                if int(mtr.sum().item()) >= 24 and int(mv.sum().item()) >= 10:
                    swt=wt[mtr]; swv=wv[mv]
                    swt=torch.where(swt.abs()<1e-3,torch.where(swt>=0,torch.full_like(swt,1e-3),torch.full_like(swt,-1e-3)),swt)
                    swv=torch.where(swv.abs()<1e-3,torch.where(swv>=0,torch.full_like(swv,1e-3),torch.full_like(swv,-1e-3)),swv)
                    target_tr=ytr[mtr]/swt; target_v=yv[mv]/swv
                    Atr=torch.stack([torch.ones_like(gtr[mtr]),gtr[mtr]],dim=1)
                    try:
                        coef=torch.linalg.lstsq(Atr,target_tr[:,None]).solution[:2,0]
                    except Exception:
                        continue
                    pred=coef[0]+coef[1]*gv[mv]
                    mse=float(torch.mean((pred-target_v)**2).detach().cpu())
                else:
                    phi = wt*gtr; phiv=wv*gv
                    A=torch.stack([torch.ones_like(phi),phi],dim=1)
                    try:
                        coef=torch.linalg.lstsq(A,ytr[:,None]).solution[:2,0]
                    except Exception:
                        continue
                    pred=coef[0]+coef[1]*phiv
                    mse=float(torch.mean((pred-yv)**2).detach().cpu())
                if not math.isfinite(mse):
                    continue
                key=(j,str(name))
                prev=best_by_family.get(key)
                if prev is None or mse<prev[0]:
                    best_by_family[key]=(mse,{
                        "variable":j,"name":str(name),
                        "affine":(1.0,float(b),float(c),0.0),
                        "leaf_screen_validation_mse":mse,
                    })
    ranked=sorted(best_by_family.values(),key=lambda z:z[0])
    return [rec for _,rec in ranked[:max(1,int(topk))]]


def recursive_partition_symbolic_rescue(
    numeric_model: SumProductKAN,
    incumbent: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    leaf_topk: int = 10,
    coarse_topk: int = 16,
    max_samples: int = 384,
    shallow_topk: int = 8,
    shallow_steps: int = 90,
    shallow_lr: float = 8e-4,
    shallow_lbfgs_steps: int = 14,
    deep_topk: int = 3,
    deep_steps: int = 360,
    deep_lbfgs_steps: int = 70,
    min_improvement_rel: float = 1e-4,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Validation-gated depth-two complementary fuzzy-tree rescue.

    The search is deliberately narrow and interpretable.  Gate variables are
    inferred only from raw observed ranges resembling membership values; each
    gate is fixed exactly to raw ``x``/``1-x`` after accounting for input
    standardization.  Three unary branch atoms are then searched and the tree
    is expanded into the ordinary three-rule SumProductKAN DNF:

        (1-g0)(1-g1)F0 + (1-g0)g1 F1 + g0 F2

    (plus the symmetric orientation where the nested branch is on ``g0``).
    The resulting model is a normal RuleKAN model, not a separate predictor.
    """
    meta: Dict[str, object] = {"attempted": False, "selected": False, "reason": "not_applicable"}
    if numeric_model.max_factors < 3 or numeric_model.n_rules < 3:
        return incumbent, meta
    gates = _raw_membership_gate_candidates(
        train_x, input_mean=input_mean, input_std=input_std,
    )
    meta["gate_candidates"] = list(gates)
    if len(gates) < 2:
        meta["reason"] = "fewer_than_two_membership_gate_candidates"
        return incumbent, meta
    mean = input_mean.to(device=train_x.device, dtype=train_x.dtype).reshape(-1)
    std = input_std.to(device=train_x.device, dtype=train_x.dtype).reshape(-1)
    if mean.numel() != numeric_model.in_dim or std.numel() != numeric_model.in_dim:
        meta["reason"] = "normalization_shape_mismatch"
        return incumbent, meta
    lib = tuple(str(n) for n in (numeric_model.symbolic_library if library is None else library) if n in SYMBOLIC_LIB)
    leaf_lib = tuple(n for n in lib if n != "x")
    if not leaf_lib:
        meta["reason"] = "no_nonidentity_leaf_library"
        return incumbent, meta
    allowed = None if allowed_supports is None else {frozenset(int(v) for v in s) for s in allowed_supports}
    leaf_vars = [j for j in range(numeric_model.in_dim) if j not in set(gates)]
    if not leaf_vars:
        # Same-variable nested mechanisms remain possible in principle; this
        # fallback keeps the rescue generic without preferring task labels.
        leaf_vars = list(range(numeric_model.in_dim))
    meta["leaf_variables"] = list(leaf_vars)
    meta["attempted"] = True
    incumbent.eval()
    with torch.no_grad():
        incumbent_mse = float(torch.mean((incumbent(val_x) - val_y) ** 2).detach().cpu())
    meta["incumbent_validation_mse"] = incumbent_mse

    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0] - 1, int(max_samples), device=train_x.device).long()
        sx, sy = train_x[ids], train_y[ids]
    else:
        sx, sy = train_x, train_y
    # Use exact raw membership values; these are train-only affine transforms.
    raw_tr = sx * std[None, :] + mean[None, :]
    raw_v = val_x * std[None, :] + mean[None, :]
    ytr = sy.reshape(sy.shape[0], -1).mean(dim=1)
    yv = val_y.reshape(val_y.shape[0], -1).mean(dim=1)

    coarse: List[Tuple[float, Dict[str, object]]] = []
    support_rejected = 0
    mechanisms_tested = 0
    for outer, inner in itertools.permutations(gates, 2):
        go = raw_tr[:, int(outer)]; gi = raw_tr[:, int(inner)]
        gov = raw_v[:, int(outer)]; giv = raw_v[:, int(inner)]
        for nested_on_complement in (True, False):
            if nested_on_complement:
                W = ((1.0-go)*(1.0-gi), (1.0-go)*gi, go)
                WV = ((1.0-gov)*(1.0-giv), (1.0-gov)*giv, gov)
            else:
                W = (go*(1.0-gi), go*gi, 1.0-go)
                WV = (gov*(1.0-giv), gov*giv, 1.0-gov)
            slot_candidates = [
                _weighted_unary_leaf_candidates(
                    sx, sy, val_x, val_y, W[k], WV[k],
                    variables=leaf_vars, library=leaf_lib,
                    input_mean=mean, input_std=std,
                    topk=int(leaf_topk), max_samples=int(max_samples),
                ) for k in range(3)
            ]
            if any(not z for z in slot_candidates):
                continue
            mechanisms_tested += 1
            local: List[Tuple[float, Dict[str, object]]] = []
            for leaves in itertools.product(*slot_candidates):
                v0, v1, v2 = [int(c["variable"]) for c in leaves]
                if nested_on_complement:
                    supports = (
                        frozenset((int(outer), int(inner), v0)),
                        frozenset((int(outer), int(inner), v1)),
                        frozenset((int(outer), v2)),
                    )
                else:
                    supports = (
                        frozenset((int(outer), int(inner), v0)),
                        frozenset((int(outer), int(inner), v1)),
                        frozenset((int(outer), v2)),
                    )
                if allowed is not None and any(s not in allowed for s in supports):
                    support_rejected += 1
                    continue
                cols_tr = [torch.ones_like(ytr)]
                cols_v = [torch.ones_like(yv)]
                finite = True
                for k, leaf in enumerate(leaves):
                    cand = {"name": leaf["name"], "affine": leaf["affine"]}
                    ctr = _symbolic_combo_curve(sx, (int(leaf["variable"]),), (cand,))
                    cv = _symbolic_combo_curve(val_x, (int(leaf["variable"]),), (cand,))
                    if ctr is None or cv is None:
                        finite = False; break
                    cols_tr.append(W[k] * ctr); cols_v.append(WV[k] * cv)
                if not finite:
                    continue
                A = torch.stack(cols_tr, dim=1)
                AV = torch.stack(cols_v, dim=1)
                try:
                    coef = torch.linalg.lstsq(A, ytr[:, None]).solution[:4, 0]
                except Exception:
                    continue
                pred = AV @ coef
                mse = float(torch.mean((pred - yv) ** 2).detach().cpu())
                if not math.isfinite(mse):
                    continue
                local.append((mse, {
                    "outer": int(outer), "inner": int(inner),
                    "nested_on_complement": bool(nested_on_complement),
                    "leaves": tuple(dict(z) for z in leaves),
                    "bias": float(coef[0].detach().cpu()),
                    "scales": tuple(float(z.detach().cpu()) for z in coef[1:4]),
                    "supports": tuple(tuple(sorted(int(v) for v in s)) for s in supports),
                }))
            local.sort(key=lambda z: z[0])
            coarse.extend(local[:max(1, int(coarse_topk))])
    meta["mechanisms_tested"] = int(mechanisms_tested)
    meta["support_rejected_candidates"] = int(support_rejected)
    if not coarse:
        meta["reason"] = "no_admissible_recursive_partition_candidate"
        return incumbent, meta
    coarse.sort(key=lambda z: z[0])
    meta["coarse_candidates"] = int(len(coarse))

    x_idx = numeric_model.symbolic_library.index("x") if "x" in numeric_model.symbolic_library else -1
    if x_idx < 0:
        meta["reason"] = "identity_not_in_library"
        return incumbent, meta

    refined: List[Tuple[float, SumProductKAN, Dict[str, object]]] = []
    seen = set()
    for coarse_mse, rec in coarse:
        sig = (rec["outer"], rec["inner"], rec["nested_on_complement"],
               tuple((z["variable"], z["name"]) for z in rec["leaves"]))
        if sig in seen:
            continue
        seen.add(sig)
        trial = _make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            trial.hard_rule_choice.zero_(); trial.rule_scale.zero_(); trial.bias.fill_(float(rec["bias"]))
        outer = int(rec["outer"]); inner = int(rec["inner"])
        nested_comp = bool(rec["nested_on_complement"])
        outer_nested_aff = _exact_raw_gate_affine(mean, std, outer, complement=nested_comp)
        outer_leaf_aff = _exact_raw_gate_affine(mean, std, outer, complement=not nested_comp)
        inner_c_aff = _exact_raw_gate_affine(mean, std, inner, complement=True)
        inner_aff = _exact_raw_gate_affine(mean, std, inner, complement=False)
        frozen: List[Tuple[int,int,int,int]] = []
        leaf_specs = rec["leaves"]
        for rr in range(3):
            leaf = leaf_specs[rr]
            leaf_cand = {"name": str(leaf["name"]), "affine": tuple(leaf["affine"])}
            if rr < 2:
                combo = (
                    {"name":"x", "affine":outer_nested_aff, "recursive_gate":True},
                    {"name":"x", "affine":inner_c_aff if rr == 0 else inner_aff, "recursive_gate":True},
                    leaf_cand,
                )
                factors = ((0,outer),(1,inner),(2,int(leaf["variable"])))
                _install_symbolic_template(trial, rr, factors, combo, initial_scale=float(rec["scales"][rr]))
                frozen.extend([(rr,0,outer,x_idx),(rr,1,inner,x_idx)])
            else:
                combo = (
                    {"name":"x", "affine":outer_leaf_aff, "recursive_gate":True},
                    leaf_cand,
                )
                factors = ((0,outer),(1,int(leaf["variable"])))
                _install_symbolic_template(trial, rr, factors, combo, initial_scale=float(rec["scales"][rr]))
                frozen.append((rr,0,outer,x_idx))
        trial, vmse = _fully_symbolic_continuous_refit(
            trial, train_x, train_y, val_x, val_y,
            steps=max(0,int(shallow_steps)), lr=float(shallow_lr),
            lbfgs_steps=max(0,int(shallow_lbfgs_steps)),
            frozen_symbolic_factors=tuple(frozen), show_progress=False,
        )
        if math.isfinite(float(vmse)):
            rr = dict(rec); rr["frozen_gate_factors"] = tuple(tuple(int(v) for v in z) for z in frozen)
            rr["coarse_validation_mse"] = float(coarse_mse)
            refined.append((float(vmse), trial, rr))
        if len(refined) >= max(1, int(shallow_topk)):
            # Coarse candidates are already sorted.  Stop after enough finite
            # shallow refits rather than spending a full benchmark budget here.
            break
    if not refined:
        meta["reason"] = "all_recursive_partition_refits_nonfinite"
        return incumbent, meta
    refined.sort(key=lambda z:z[0])
    pool = list(refined)
    for _, shallow, rec in refined[:max(0,int(deep_topk))]:
        polished, vmse = _fully_symbolic_continuous_refit(
            shallow, train_x, train_y, val_x, val_y,
            steps=max(0,int(deep_steps)), lr=max(float(shallow_lr)*0.5,2e-4),
            lbfgs_steps=max(0,int(deep_lbfgs_steps)),
            frozen_symbolic_factors=rec["frozen_gate_factors"], show_progress=False,
        )
        if math.isfinite(float(vmse)):
            pool.append((float(vmse), polished, rec))
    best_mse, best_model, best_rec = min(pool, key=lambda z:z[0])
    rel = (incumbent_mse - float(best_mse)) / max(incumbent_mse, 1e-18)
    meta.update({"best_validation_mse":float(best_mse), "relative_validation_improvement":float(rel)})
    if rel < float(min_improvement_rel):
        meta["reason"] = "validation_improvement_too_small"
        return incumbent, meta
    meta.update({
        "selected": True, "reason":"validation_improved",
        "outer_gate": int(best_rec["outer"]), "inner_gate": int(best_rec["inner"]),
        "nested_on_complement": bool(best_rec["nested_on_complement"]),
        "leaf_variables": [int(z["variable"]) for z in best_rec["leaves"]],
        "leaf_operators": [str(z["name"]) for z in best_rec["leaves"]],
        "supports": [list(z) for z in best_rec["supports"]],
        "frozen_gate_factors": [list(z) for z in best_rec["frozen_gate_factors"]],
    })
    if verbose:
        print(
            "  recursive partition rescue: "
            f"outer=x{best_rec['outer']} inner=x{best_rec['inner']} "
            f"ops={meta['leaf_operators']}; val RMSE "
            f"{math.sqrt(max(incumbent_mse,0.0)):.8g} -> {math.sqrt(max(float(best_mse),0.0)):.8g}"
        )
    return best_model, meta

def hard_symbolic_tuple_screening(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    structure: Sequence[int],
    library: Sequence[str],
    *,
    beam_width: int = 64,
    top_tuples: int = 12,
    max_samples: int = 512,
    seeds_per_operator_prefix: int = 3,
    use_data_unit_seeds: bool = False,
) -> List[Tuple[Dict[str, object], ...]]:
    """Deterministically propose hard operator tuples with multistart affines.

    This is complementary to GMP, not a replacement.  GMP is good at amortized
    soft screening but can suppress the exact atom before hard refitting.  This
    routine never mixes operators: it performs a small beam search over exact
    analytic atoms and deterministic affine seeds, ranking every partial product
    by the best scalar affine projection onto the target.

    At depth one, at least one seed for every operator is retained whenever the
    beam can hold the library.  At higher product order the beam limits the
    combinatorics.  Final GSR validation/refitting remains the sole commit rule.
    """
    lib = tuple(str(n) for n in library if n in SYMBOLIC_LIB)
    vars_ = tuple(int(v) for v in structure)
    if not lib or not vars_:
        return []
    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0] - 1, int(max_samples), device=train_x.device).long()
        xx, yy = train_x[ids], train_y[ids]
    else:
        xx, yy = train_x, train_y
    yvec = yy.reshape(yy.shape[0], -1).mean(dim=1)

    # Cache hard atom curves by (variable, operator, beta, gamma).
    cache: Dict[Tuple[int, str, float, float], torch.Tensor] = {}
    def atom(var: int, name: str, b: float, c: float) -> Optional[torch.Tensor]:
        key = (int(var), str(name), float(b), float(c))
        if key in cache:
            return cache[key]
        try:
            g = SYMBOLIC_LIB[name][0](float(b) * xx[:, int(var)] + float(c))
        except Exception:
            return None
        if not torch.isfinite(g).all():
            return None
        cache[key] = g
        return g

    # beam entries: (rmse, product_curve, tuple(candidate_dicts))
    beam: List[Tuple[float, torch.Tensor, Tuple[Dict[str, object], ...]]] = [
        (float("inf"), torch.ones_like(yvec), tuple())
    ]
    keep = max(int(beam_width), len(lib))
    for depth, var in enumerate(vars_):
        # For each operator-name prefix retain its best affine seed.  This keeps
        # hard operator diversity while avoiding a beam dominated by many nearly
        # identical frequency seeds of one atom.
        best_by_names: Dict[Tuple[str, ...], List[Tuple[float, torch.Tensor, Tuple[Dict[str, object], ...]]]] = {}
        per_prefix = max(1, int(seeds_per_operator_prefix))
        for _, prefix_curve, prefix_combo in beam:
            prefix_names = tuple(str(c["name"]) for c in prefix_combo)
            for name in lib:
                seed_grid = (
                    _data_unit_affine_seed_grid(train_x, var, name)
                    if bool(use_data_unit_seeds) else _hard_affine_seed_grid(name)
                )
                for b, c in seed_grid:
                    g = atom(var, name, b, c)
                    if g is None:
                        continue
                    prod = prefix_curve * g
                    if not torch.isfinite(prod).all():
                        continue
                    score, _, _ = _best_affine_residual_rmse(prod, yvec)
                    if not math.isfinite(score):
                        continue
                    cand = {"name": name, "affine": (1.0, float(b), float(c), 0.0),
                            "gmp_prob": 0.0, "proposal_source": "hard_multistart"}
                    combo = prefix_combo + (cand,)
                    names = prefix_names + (name,)
                    bucket = best_by_names.setdefault(names, [])
                    bucket.append((score, prod, combo))
                    bucket.sort(key=lambda z: z[0])
                    del bucket[per_prefix:]
        ranked = sorted((z for bucket in best_by_names.values() for z in bucket), key=lambda z: z[0])
        if not ranked:
            return []
        # At unary depth keep all operator identities when affordable.  At later
        # depths cap combinatorics with the configured beam.
        depth_keep = max(len(lib) * per_prefix, keep) if depth == 0 else keep
        beam = ranked[:max(1, int(depth_keep))]

    beam.sort(key=lambda z: z[0])
    out: List[Tuple[Dict[str, object], ...]] = []
    seen_names = set()
    for _, _, combo in beam:
        names = tuple(str(c["name"]) for c in combo)
        if names in seen_names:
            continue
        seen_names.add(names); out.append(combo)
        if len(out) >= max(1, int(top_tuples)):
            break
    return out


def _polish_symbolic_operator_tuple(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    structure: Sequence[int],
    combo: Sequence[Dict[str, object]],
    *,
    steps: int = 60,
    lr: float = 1.5e-2,
    max_samples: int = 512,
) -> Tuple[Tuple[Dict[str, object], ...], float]:
    """Briefly refit one hard GMP top-k tuple before costly full-model GSR.

    Soft GMP mixtures can fit well while their top-1 hard operators still have
    mismatched affine constants.  This inexpensive hard-tuple polish closes that
    soft/hard gap using only the proposed symbolic rule; it is orders of
    magnitude cheaper than a full in-context model refit.
    """
    vars_ = tuple(int(v) for v in structure)
    if train_x.shape[0] > int(max_samples):
        ids = torch.linspace(0, train_x.shape[0] - 1, int(max_samples), device=train_x.device).long()
        xx, yy = train_x[ids], train_y[ids]
    else:
        xx, yy = train_x, train_y
    yvec = yy.reshape(yy.shape[0], -1).mean(dim=1)
    aff = torch.tensor(
        [_canonical_symbolic_affine(c["affine"], str(c["name"])) for c in combo],
        device=xx.device, dtype=xx.dtype, requires_grad=True,
    )
    scale = torch.tensor(0.2, device=xx.device, dtype=xx.dtype, requires_grad=True)
    offset = yvec.mean().detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([aff, scale, offset], lr=float(lr))
    best_aff = aff.detach().clone(); best_scale = scale.detach().clone(); best_offset = offset.detach().clone()
    best_mse = float("inf")
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        prod = torch.ones_like(yvec)
        ok=True
        for slot,(var,cand) in enumerate(zip(vars_, combo)):
            name=str(cand["name"]); fun=SYMBOLIC_LIB[name][0]
            a,b,c,d=aff[slot]
            try:
                g=fun(b*xx[:,var]+c)
            except Exception:
                ok=False; break
            if not torch.isfinite(g).all():
                ok=False; break
            prod=prod*(a*g+d)
        if not ok: break
        pred=offset+scale*prod
        loss=torch.mean((pred-yvec)**2)
        if not torch.isfinite(loss): break
        loss.backward()
        if aff.grad is not None:
            aff.grad[:,0].zero_(); aff.grad[:,3].zero_()
        torch.nn.utils.clip_grad_norm_([aff,scale,offset],5.0); opt.step()
        with torch.no_grad():
            aff[:,0].fill_(1.0); aff[:,1].clamp_(-8.0,8.0)
            aff[:,2].clamp_(-4.0*math.pi,4.0*math.pi); aff[:,3].zero_()
            scale.clamp_(-1e3,1e3); offset.clamp_(-1e3,1e3)
            mse=float(loss.detach().cpu())
            if mse<best_mse:
                best_mse=mse; best_aff=aff.detach().clone(); best_scale=scale.detach().clone(); best_offset=offset.detach().clone()
    out=[]
    for slot,cand in enumerate(combo):
        cc=dict(cand); cc["affine"]=_canonical_symbolic_affine(best_aff[slot].cpu().tolist(), str(cand["name"])); out.append(cc)
    # The tuple's scale/offset are only proposal diagnostics; matching pursuit
    # recomputes an optimal residual scale and full-model GSR refits everything.
    for cc in out:
        cc["tuple_scale_hint"]=float(best_scale.cpu()); cc["tuple_offset_hint"]=float(best_offset.cpu())
    return tuple(out), float(best_mse)



def _best_affine_residual_rmse(
    h: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[float, float, float]:
    """Return RMSE of the best ``offset + scale*h`` residual projection."""
    hv = h.reshape(-1)
    rv = residual.reshape(-1)
    hm, rm = hv.mean(), rv.mean()
    hc, rc = hv - hm, rv - rm
    denom = hc.square().sum().clamp_min(1e-12)
    scale_t = (hc * rc).sum() / denom
    offset_t = rm - scale_t * hm
    err = rv - (offset_t + scale_t * hv)
    rmse_t = torch.sqrt(torch.mean(err.square()).clamp_min(0.0))
    return (
        float(rmse_t.detach().cpu()),
        float(scale_t.detach().cpu()),
        float(offset_t.detach().cpu()),
    )


def _format_symbolic_combo(combo: Sequence[Dict[str, object]]) -> str:
    return " * ".join(str(c["name"]) for c in combo)


def _fmt_structure(structure: Sequence[int]) -> str:
    vals = tuple(int(v) for v in structure)
    return "*".join(f"x{v}" for v in vals) if vals else "bias"


def _emit_debug(text: str, *, show_progress: bool) -> None:
    tqdm.write(text) if show_progress else print(text)


def _fmt_gmp_slot(cands: Sequence[Dict[str, object]], topk: int, *, verbose: bool) -> str:
    shown = list(cands)[:max(1, int(topk))]
    if verbose:
        return " | ".join(
            f"{c['name']} p={float(c.get('gmp_prob',0.0)):.3f} "
            f"b={float(c['affine'][1]):+.3g} c={float(c['affine'][2]):+.3g}"
            for c in shown
        )
    return " / ".join(f"{c['name']} {float(c.get('gmp_prob',0.0)):.3f}" for c in shown)


def _print_gmp_debug(
    results: Sequence[Dict[str, object]],
    *,
    debug_topk: int,
    log_style: str = "compact",
    label: str = "GMP",
    show_progress: bool = False,
) -> None:
    """Print GMP operator shortlists in a compact, scan-friendly block."""
    style = str(log_style).lower()
    detailed = style == "verbose"
    k = max(1, int(debug_topk))
    lines = [f"  [{label}] top-{k} operator shortlists"]
    for entry in results:
        struct = _fmt_structure(entry["structure"])
        slots = []
        for slot, cands in enumerate(entry["factor_shortlists"]):
            desc = _fmt_gmp_slot(cands, k, verbose=detailed)
            slots.append((f"s{slot}: " if len(entry["factor_shortlists"]) > 1 else "") + desc)
        lines.append(f"    {struct:<7} " + " ; ".join(slots))
    _emit_debug("\n".join(lines), show_progress=show_progress)


def _format_ranked_inline(
    entries: Sequence[Tuple[float, int]],
    templates: Sequence[Dict[str, object]],
    *,
    topk: int,
    selected_ti: Optional[int] = None,
) -> str:
    parts=[]
    for rank,(score,ti) in enumerate(list(entries)[:max(1,int(topk))],1):
        tpl=templates[int(ti)]
        mark="*" if selected_ti is not None and int(ti)==int(selected_ti) else ""
        parts.append(
            f"{rank}:{_fmt_structure(tpl['structure'])} "
            f"{_format_symbolic_combo(tpl['combo'])} {score:.6g}{mark}"
        )
    return " | ".join(parts)




def drop_negligible_symbolic_rules(
    model: SumProductKAN,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    current_mse: Optional[float] = None,
    scale_threshold: float = 1e-4,
    contribution_rmse_threshold: float = 1e-5,
    relative_mse_budget: float = 2e-3,
    absolute_rmse_budget: float = 0.0,
    min_rules: int = 1,
    verbose: bool = False,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, float, List[int]]:
    """Drop rules that are tiny in both coefficient and actual contribution.

    The deletion is evaluated on validation data before it is committed.  This
    prevents a deceptively small coefficient from being removed when it
    multiplies a large analytic factor.  No optimizer is used here, so the pass
    is cheap and always terminates after at most the number of active rules.
    """
    current = copy.deepcopy(model)
    current.eval()
    if current_mse is None:
        with torch.no_grad():
            mse = float(torch.mean((current(val_x) - val_y) ** 2).cpu())
    else:
        mse = float(current_mse)
    removed: List[int] = []
    while True:
        active = torch.nonzero(current.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        if len(active) <= int(min_rules):
            break
        with torch.no_grad():
            _, det = current(val_x, return_details=True)
            crms = torch.sqrt(torch.mean(det["contributions"] ** 2, dim=0) + 1e-18)
        candidates = []
        for rr in active:
            scale_abs = abs(float(current.rule_scale[rr].detach().cpu()))
            contribution = float(crms[rr].detach().cpu())
            if scale_abs <= float(scale_threshold) and contribution <= float(contribution_rmse_threshold):
                candidates.append((contribution, scale_abs, int(rr)))
        if not candidates:
            break
        candidates.sort()
        committed = False
        for contribution, scale_abs, rr in candidates:
            trial = copy.deepcopy(current)
            with torch.no_grad():
                trial.hard_rule_choice[rr] = False
                trial.rule_scale[rr] = 0.0
                trial_mse = float(torch.mean((trial(val_x) - val_y) ** 2).cpu())
            rel = (trial_mse - mse) / max(mse, 1e-18)
            below_absolute_target = (
                float(absolute_rmse_budget) > 0.0
                and math.sqrt(max(0.0, float(trial_mse))) <= float(absolute_rmse_budget)
            )
            if math.isfinite(trial_mse) and (rel <= float(relative_mse_budget) or below_absolute_target):
                current, mse = trial, trial_mse
                removed.append(rr)
                committed = True
                if verbose:
                    msg=(
                        f"  final tiny-term drop rule {rr}: |scale|={scale_abs:.3g}, "
                        f"contribution RMSE={contribution:.3g}, relative MSE change={rel:.3g}"
                    )
                    tqdm.write(msg) if show_progress else print(msg)
                break
        if not committed:
            break
    return current, mse, removed



def rule_contribution_redundancy(
    model: SumProductKAN,
    x: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> Dict[str, object]:
    """Diagnose redundancy among actual end-to-end rule contributions.

    Returns contribution RMS, pairwise Pearson correlation, and for every rule
    the R² obtained by regressing that rule's centered contribution against the
    span of all other active rule contributions.  Span-R² detects redundancy
    such as ``c0 ~= c1 + c2`` that pairwise correlation can miss.
    """
    model.eval()
    with torch.no_grad():
        _, det = model(x, return_details=True)
        c = det["contributions"].detach()
    active = torch.nonzero(
        model.hard_rule_choice & model.rule_alive_mask,
        as_tuple=False,
    ).squeeze(-1).tolist() if bool(model.discretized.item()) else list(range(model.n_rules))
    if not active:
        return {"active_rules": [], "rms": {}, "corr": {}, "span_r2": {}}
    C = c[:, active]
    rms = torch.sqrt(C.square().mean(dim=0) + 1e-18)
    Cc = C - C.mean(dim=0, keepdim=True)
    std = torch.sqrt(Cc.square().mean(dim=0) + 1e-12)
    Z = Cc / std
    corr = (Z.T @ Z) / float(max(1, Z.shape[0]))
    span = {}
    eye_eps = float(ridge)
    for ii, rr in enumerate(active):
        y = Cc[:, ii]
        others = [jj for jj in range(len(active)) if jj != ii]
        if not others or float(y.square().sum()) <= 1e-18:
            span[int(rr)] = 0.0
            continue
        X = Cc[:, others]
        gram = X.T @ X + eye_eps * torch.eye(len(others), device=X.device, dtype=X.dtype)
        try:
            beta = torch.linalg.solve(gram, X.T @ y)
        except RuntimeError:
            beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
        resid = y - X @ beta
        r2 = 1.0 - float(resid.square().sum().cpu()) / max(float(y.square().sum().cpu()), 1e-18)
        span[int(rr)] = float(max(0.0, min(1.0, r2)))
    corr_map = {}
    for i, ri in enumerate(active):
        for j in range(i + 1, len(active)):
            rj = active[j]
            corr_map[(int(ri), int(rj))] = float(corr[i, j].cpu())
    return {
        "active_rules": [int(r) for r in active],
        "rms": {int(r): float(rms[i].cpu()) for i, r in enumerate(active)},
        "corr": corr_map,
        "span_r2": span,
        "max_abs_corr": max((abs(v) for v in corr_map.values()), default=0.0),
        "max_span_r2": max(span.values(), default=0.0),
    }



def numeric_logic_diagnostics(
    model: SumProductKAN,
    x: torch.Tensor,
) -> Dict[str, object]:
    """Diagnostics for whether a hard numerical RuleKAN is logic-compressed.

    ``cancellation_index`` is the triangle-inequality ratio

        sum_r ||c_r||_2 / ||sum_r c_r||_2,

    over active rule contributions.  It is >= 1 (up to numerical noise); large
    values indicate compensating/cancelling rules.  ``active_structures`` is the
    unique set of selected variable tuples among active numerical rules.
    """
    diag = rule_contribution_redundancy(model, x)
    model.eval()
    with torch.no_grad():
        _, details = model(x, return_details=True)
        active = list(diag.get("active_rules", []))
        if active:
            C = details["contributions"][:, active]
            numerator = C.norm(dim=0).sum()
            denominator = C.sum(dim=1).norm().clamp_min(1e-12)
            cancellation = float((numerator / denominator).cpu())
        else:
            cancellation = 0.0
    structures = []
    if bool(model.discretized.item()):
        seen = set()
        for r in active:
            vals = [
                int(v) for v in model.hard_variable_choice[int(r)].detach().cpu().tolist()
                if int(v) != int(model.in_dim)
            ]
            if not vals:
                continue
            key = tuple(sorted(vals))
            if key not in seen:
                seen.add(key); structures.append(key)
    return {
        **diag,
        "cancellation_index": cancellation,
        "active_structures": structures,
        "n_active_structures": len(structures),
    }


def compress_numeric_rule_bank(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    min_rules: int = 1,
    max_deletions: int = 0,
    candidate_trials: int = 4,
    lbfgs_steps: int = 24,
    relative_rmse_tolerance: float = 0.25,
    nrmse_tolerance: float = 3e-3,
    verbose: bool = False,
) -> Tuple[SumProductKAN, Dict[str, object]]:
    """Rollback-safe logic compression of a polished hard numerical RuleKAN.

    The model stays at its original *maximum* width, but redundant hard rules are
    deleted one at a time.  For each deletion we re-optimize only the continuous
    numerical edge/readout parameters with fixed structure.  A deletion is kept
    only when validation RMSE stays inside a pre-declared tolerance envelope.

    The absolute part of the envelope is scale-free: ``nrmse_tolerance`` is
    multiplied by the validation-target standard deviation.  This avoids the
    pathological situation where an already near-perfect precursor can never
    simplify because any tiny absolute increase is a huge relative MSE change.
    """
    if not bool(model.discretized.item()):
        raise ValueError("compress_numeric_rule_bank requires a discretized model")
    cur = copy.deepcopy(model)
    cur.eval()

    def rmse(m: SumProductKAN) -> float:
        m.eval()
        with torch.no_grad():
            return float(torch.sqrt(torch.mean((m(val_x) - val_y) ** 2)).cpu())

    start_rmse = rmse(cur)
    y_scale = float(val_y.detach().std().cpu())
    if not math.isfinite(y_scale) or y_scale <= 1e-12:
        y_scale = 1.0
    cap = start_rmse + max(
        float(relative_rmse_tolerance) * start_rmse,
        float(nrmse_tolerance) * y_scale,
    )
    before = numeric_logic_diagnostics(cur, val_x)
    history: List[Dict[str, object]] = []

    while True:
        active = torch.nonzero(
            cur.hard_rule_choice & cur.rule_alive_mask, as_tuple=False
        ).squeeze(-1).tolist()
        if len(active) <= int(min_rules):
            break
        if int(max_deletions) > 0 and len(history) >= int(max_deletions):
            break

        immediate: List[Tuple[float, int, SumProductKAN]] = []
        for r in active:
            trial = copy.deepcopy(cur)
            with torch.no_grad():
                trial.hard_rule_choice[int(r)] = False
                trial.rule_alive_mask[int(r)] = False
                trial.rule_scale[int(r)] = 0.0
            v0 = rmse(trial)
            if math.isfinite(v0):
                immediate.append((v0, int(r), trial))
        if not immediate:
            break
        immediate.sort(key=lambda z: (z[0], z[1]))

        best = None
        for v0, r, trial in immediate[:max(1, int(candidate_trials))]:
            polished, _ = hard_numeric_precision_polish(
                trial, train_x, train_y, val_x, val_y,
                target_rmse=0.0, grid_schedule=(), adam_steps_per_grid=0,
                lbfgs_steps=max(0, int(lbfgs_steps)), extra_final_rounds=0,
                verbose=False, show_progress=False,
            )
            vv = rmse(polished)
            rec = {
                "rule": int(r), "immediate_val_rmse": float(v0),
                "refit_val_rmse": float(vv),
            }
            if best is None or vv < best[0]:
                best = (vv, r, polished, rec)

        if best is None or not math.isfinite(best[0]) or best[0] > cap:
            break
        vv, r, cur, rec = best
        rec["accepted"] = True
        rec["active_rules_after"] = int((cur.hard_rule_choice & cur.rule_alive_mask).sum().item())
        history.append(rec)
        if verbose:
            print(
                f"  numeric logic-compress: drop r{r}, val RMSE={vv:.6g}, "
                f"active={rec['active_rules_after']} (cap={cap:.6g})"
            )

    after_rmse = rmse(cur)
    after = numeric_logic_diagnostics(cur, val_x)
    return cur, {
        "start_val_rmse": float(start_rmse),
        "final_val_rmse": float(after_rmse),
        "validation_rmse_cap": float(cap),
        "accepted_deletions": history,
        "before": before,
        "after": after,
    }

def orthogonal_rule_scale_refit(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    ridge: float = 1e-8,
) -> Tuple[SumProductKAN, float]:
    """Exact OMP-style refit of all active rule amplitudes and global bias.

    Operator identities and affine input parameters stay fixed.  The active rule
    products are treated as a design matrix and all coefficients are solved
    jointly, making the residual orthogonal to the selected rule span up to the
    ridge term.
    """
    m = copy.deepcopy(model)
    m.eval()
    active = torch.nonzero(m.hard_rule_choice & m.rule_alive_mask, as_tuple=False).squeeze(-1).tolist()
    if not active:
        with torch.no_grad():
            m.bias.fill_(float(train_y.mean()))
            mse = float(torch.mean((m(val_x) - val_y) ** 2).cpu())
        return m, mse
    with torch.no_grad():
        # Divide current contributions by their rule scales to recover the fixed
        # product atom values.  For near-zero scales, temporarily set scale=1 and
        # evaluate once so no information is lost.
        saved = m.rule_scale.detach().clone()
        m.rule_scale.zero_()
        for r in active:
            m.rule_scale[r] = 1.0
        m.bias.zero_()
        _, det = m(train_x, return_details=True)
        H = det["contributions"][:, active]
        ones = torch.ones(H.shape[0], 1, device=H.device, dtype=H.dtype)
        A = torch.cat([H, ones], dim=1)
        y = train_y.reshape(train_y.shape[0], -1).mean(dim=1, keepdim=True)
        gram = A.T @ A
        reg = float(ridge) * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        reg[-1, -1] = 0.0
        rhs = A.T @ y
        try:
            coef = torch.linalg.solve(gram + reg, rhs)[:, 0]
        except RuntimeError:
            coef = torch.linalg.lstsq(A, y).solution[:, 0]
        m.rule_scale.copy_(saved)
        for i, r in enumerate(active):
            m.rule_scale[r] = coef[i]
        m.bias.fill_(coef[-1])
        mse = float(torch.mean((m(val_x) - val_y) ** 2).cpu())
    return m, mse


def redundancy_aware_symbolic_prune(
    model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    span_r2_threshold: float = 0.995,
    corr_threshold: float = 0.995,
    relative_mse_budget: float = 2e-3,
    refit_steps: int = 120,
    refit_lr: float = 3e-4,
    min_rules: int = 1,
    verbose: bool = False,
) -> Tuple[SumProductKAN, List[Dict[str, object]]]:
    """Rollback-safe deletion of contribution-redundant symbolic rules."""
    cur = copy.deepcopy(model)
    hist = []
    while int((cur.hard_rule_choice & cur.rule_alive_mask).sum()) > int(min_rules):
        diag = rule_contribution_redundancy(cur, val_x)
        active = diag["active_rules"]
        candidates = []
        for r in active:
            maxcorr = max(
                [abs(v) for (a, b), v in diag["corr"].items() if a == r or b == r] or [0.0]
            )
            span = float(diag["span_r2"].get(r, 0.0))
            if span >= float(span_r2_threshold) or maxcorr >= float(corr_threshold):
                candidates.append((float(diag["rms"].get(r, 0.0)), -span, -maxcorr, int(r)))
        if not candidates:
            break
        candidates.sort()
        with torch.no_grad():
            base_mse = float(torch.mean((cur(val_x) - val_y) ** 2).cpu())
        committed = False
        for _, nspan, ncorr, r in candidates:
            trial = copy.deepcopy(cur)
            with torch.no_grad():
                trial.hard_rule_choice[r] = False
                trial.rule_scale[r] = 0.0
            trial, mse = _fully_symbolic_continuous_refit(
                trial, train_x, train_y, val_x, val_y,
                steps=int(refit_steps), lr=float(refit_lr), lbfgs_steps=0,
            )
            rel = (float(mse) - base_mse) / max(base_mse, 1e-18)
            rec = {"rule": r, "span_r2": -nspan, "max_abs_corr": -ncorr,
                   "relative_mse_change": rel, "accepted": bool(rel <= float(relative_mse_budget))}
            hist.append(rec)
            if rel <= float(relative_mse_budget):
                cur = trial; committed = True
                if verbose:
                    print(f"  redundancy prune r{r}: spanR2={-nspan:.4f} corr={-ncorr:.4f} dMSE={rel:.3g}")
                break
        if not committed:
            break
    return cur, hist


def distill_numeric_structure_to_symbolic(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    library: Optional[Sequence[str]] = None,
    gmp_topk: Optional[int] = None,
    gmp_unary_topk: Optional[int] = None,
    gmp_self_product_topk: Optional[int] = None,
    gmp_steps: int = 70,
    gmp_lr: float = 2e-2,
    gmp_temperature_start: float = 1.5,
    gmp_temperature_end: float = 0.35,
    gmp_entropy_weight: float = 2e-4,
    gmp_relaxation_mode: str = "soft",
    gmp_atom_backward_normalization: str = "none",
    gmp_identity_chart: str = "raw",
    tuple_refine_steps: int = 30,
    tuple_refine_lr: float = 1.5e-2,
    max_rule_candidates: int = 16,
    final_steps: int = 200,
    final_lr: float = 2e-4,
    final_lbfgs_steps: int = 30,
    symbolic_seed: int = 0,
    verbose: bool = False,
) -> Tuple[SumProductKAN, List[Dict[str, object]]]:
    """Distill the learned hard numerical RuleKAN decomposition into symbols.

    Unlike :func:`mandatory_symbolic_matching_pursuit`, this routine does *not*
    discard the numerical dependency decomposition.  It preserves the active
    numerical rules and each rule's selected variables/product order.  For each
    learned rule it independently fits symbolic operator tuples to that rule's
    numerical end-to-end contribution, installs the best tuple in the same rule
    slot, and finally refits all symbolic affine input parameters, rule scales,
    and the global bias jointly against the original target.

    Per-rule affine projection offsets are accumulated into the global bias.
    Symbolic factors themselves remain canonical ``g(beta*x + gamma)`` atoms,
    so this does not enlarge the symbolic grammar with hidden factor offsets.
    """
    if not bool(numeric_model.discretized.item()):
        raise ValueError("numeric-structure distillation requires a discretized numerical model")
    lib = tuple(
        name for name in (numeric_model.symbolic_library if library is None else library)
        if name in SYMBOLIC_LIB
    )
    if not lib:
        raise ValueError("symbolic library is empty")
    gmp_topk, gmp_unary_topk, gmp_self_product_topk = scaled_gmp_screening_sizes(
        len(lib), topk=gmp_topk, unary_topk=gmp_unary_topk,
        self_product_topk=gmp_self_product_topk,
    )
    symbolic_rng = _local_torch_generator(train_x.device, int(symbolic_seed))

    numeric_model.eval()
    with torch.no_grad():
        _, num_train_det = numeric_model(train_x, return_details=True)
        _, num_val_det = numeric_model(val_x, return_details=True)

    active = torch.nonzero(
        numeric_model.hard_rule_choice & numeric_model.rule_alive_mask,
        as_tuple=False,
    ).squeeze(-1).tolist()
    if not active:
        raise RuntimeError("numerical model has no active rules to distill")

    # Preserve the numerical hard dependency structure, but forbid every spline
    # in the symbolic copy.  Unlike _make_fully_symbolic_shell, rule/factor masks
    # and hard rule choices are deliberately *not* reopened/reset here.
    model = copy.deepcopy(numeric_model)
    model.symbolic_enabled = True
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=1.0)
    model.hard_spline_choice.zero_()
    for p in model.numeric_edges.parameters():
        p.requires_grad_(False)
    for p in (
        model.variable_logits, model.rule_gate.log_alpha, model.factor_gate.log_alpha,
        model.operator_logits, model.spline_logits,
    ):
        p.requires_grad_(False)
    model.symbolic_affine.requires_grad_(True)
    model.rule_scale.requires_grad_(True)
    model.bias.requires_grad_(True)

    history: List[Dict[str, object]] = []
    bias_offset = float(numeric_model.bias.detach().cpu())

    for rr in active:
        vars_: List[int] = []
        factors: List[Tuple[int, int]] = []
        for ss in range(numeric_model.max_factors):
            if not bool(numeric_model.factor_alive_mask[rr, ss]):
                continue
            jj = int(numeric_model.hard_variable_choice[rr, ss].item())
            if jj == numeric_model.in_dim:
                continue
            vars_.append(jj)
            factors.append((ss, jj))
        if not vars_:
            # A degenerate active constant rule cannot be represented by the
            # canonical factor grammar; fold its mean contribution into bias.
            c = num_train_det["contributions"][:, rr]
            bias_offset += float(c.mean().cpu())
            with torch.no_grad():
                model.hard_rule_choice[rr] = False
                model.rule_scale[rr] = 0.0
            history.append({"rule": int(rr), "status": "folded_constant"})
            continue

        target_train = num_train_det["contributions"][:, rr:rr+1].detach()
        target_val = num_val_det["contributions"][:, rr:rr+1].detach()
        structure = tuple(vars_)
        gmp_results, _ = gmp_symbolic_operator_preselection(
            train_x, target_train, [structure], lib,
            topk=int(gmp_topk), unary_topk=int(gmp_unary_topk),
            self_product_topk=int(gmp_self_product_topk),
            generator=symbolic_rng,
            steps=max(1, int(gmp_steps)), lr=float(gmp_lr),
            temperature_start=float(gmp_temperature_start),
            temperature_end=float(gmp_temperature_end),
            entropy_weight=float(gmp_entropy_weight),
            relaxation_mode=str(gmp_relaxation_mode),
            atom_backward_normalization=str(gmp_atom_backward_normalization),
            identity_chart=str(gmp_identity_chart),
            debug_topk=0, debug_log_style="compact",
            debug_label=f"distill-r{rr}", show_progress=False,
        )
        if not gmp_results:
            raise RuntimeError(f"no symbolic candidates generated for numerical rule {rr}")
        shortlists = gmp_results[0]["factor_shortlists"]
        combos = list(itertools.product(*shortlists))
        combos.sort(
            key=lambda cc: sum(
                math.log(max(float(z.get("gmp_prob", 1e-12)), 1e-12)) for z in cc
            ),
            reverse=True,
        )
        combos = combos[:max(1, int(max_rule_candidates))]

        scored: List[Tuple[float, Tuple[Dict[str, object], ...], float, float, float]] = []
        for combo in combos:
            try:
                polished, _ = _polish_symbolic_operator_tuple(
                    train_x, target_train, structure, combo,
                    steps=max(0, int(tuple_refine_steps)), lr=float(tuple_refine_lr),
                )
                htr = _candidate_rule_value(model, train_x, rr, factors, polished)
                hva = _candidate_rule_value(model, val_x, rr, factors, polished)
                if not torch.isfinite(htr).all() or not torch.isfinite(hva).all():
                    continue
                _, scale, offset = _best_affine_residual_rmse(htr, target_train)
                pred_val = offset + scale * hva
                val_rmse = float(torch.sqrt(torch.mean((pred_val.reshape_as(target_val) - target_val) ** 2)).cpu())
                if math.isfinite(val_rmse):
                    scored.append((val_rmse, tuple(polished), float(scale), float(offset), float(torch.sqrt(torch.mean((target_val) ** 2)).cpu())))
            except Exception:
                continue
        if not scored:
            raise RuntimeError(f"all symbolic distillation candidates failed for numerical rule {rr}")
        scored.sort(key=lambda z: z[0])
        val_rmse, best_combo, scale, offset, target_rms = scored[0]
        _install_symbolic_template(model, rr, factors, best_combo, initial_scale=scale)
        bias_offset += offset
        history.append({
            "rule": int(rr),
            "structure": structure,
            "operators": [str(c["name"]) for c in best_combo],
            "rule_distill_val_rmse": float(val_rmse),
            "numeric_rule_val_rms": float(target_rms),
            "scale": float(scale),
            "offset_to_global_bias": float(offset),
        })
        if verbose:
            print(
                f"  distill r{rr} {'*'.join('x'+str(v) for v in structure)} -> "
                f"{' * '.join(str(c['name']) for c in best_combo)}; "
                f"rule val RMSE={val_rmse:.6g}"
            )

    with torch.no_grad():
        model.bias.fill_(bias_offset)
    # Jointly optimize the exact symbolic model against the original target.
    model, final_mse = _fully_symbolic_continuous_refit(
        model, train_x, train_y, val_x, val_y,
        steps=max(0, int(final_steps)), lr=float(final_lr),
        lbfgs_steps=max(0, int(final_lbfgs_steps)),
        show_progress=False, progress_desc="Numerical-structure symbolic distillation",
    )
    history.append({
        "status": "global_refit",
        "validation_rmse": math.sqrt(max(float(final_mse), 0.0)),
        "active_rules": int((model.hard_rule_choice & model.rule_alive_mask).sum().item()),
    })
    return model, history


def project_constrained_numeric_to_symbolic(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    final_steps: int = 120,
    final_lr: float = 2e-4,
    final_lbfgs_steps: int = 20,
) -> SumProductKAN:
    """Project a manifold-constrained hard numerical model directly to symbols.

    The hard dependency structure and the operator choices learned by the
    augmented-Lagrangian numerical stages are retained.  Nonlinear per-factor
    amplitudes are absorbed into the rule coefficient, leaving the canonical
    ``g(beta*x+gamma)`` SumProduct grammar.  A rollback-safe joint continuous
    symbolic refit then adjusts beta/gamma, rule scales, and the global bias.
    """
    if not bool(numeric_model.discretized.item()):
        raise ValueError("manifold projection requires a discretized numerical model")
    model = copy.deepcopy(numeric_model)
    model.symbolic_enabled = True
    model.hard_spline_choice.zero_()
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=1.0)
    with torch.no_grad():
        active = torch.nonzero(model.hard_rule_choice & model.rule_alive_mask, as_tuple=False).squeeze(-1).tolist()
        for r in active:
            amplitude = 1.0
            for s in range(model.max_factors):
                if not bool(model.factor_alive_mask[r, s]):
                    continue
                j = int(model.hard_variable_choice[r, s].item())
                if j == model.in_dim:
                    continue
                k = int(model.hard_operator_choice[r, s, j].item())
                name = model.symbolic_library[k]
                if name != "x":
                    amplitude *= float(model.symbolic_affine[r, s, j, k, 0].item())
                model.symbolic_affine[r, s, j, k, 0] = 1.0
                model.symbolic_affine[r, s, j, k, 3] = 0.0
            model.rule_scale[r] *= float(amplitude)
    for p in model.numeric_edges.parameters():
        p.requires_grad_(False)
    for p in (model.variable_logits, model.rule_gate.log_alpha, model.factor_gate.log_alpha, model.operator_logits, model.spline_logits):
        p.requires_grad_(False)
    model.symbolic_affine.requires_grad_(True)
    model.rule_scale.requires_grad_(True)
    model.bias.requires_grad_(True)
    model, _ = _fully_symbolic_continuous_refit(
        model, train_x, train_y, val_x, val_y,
        steps=max(0, int(final_steps)), lr=float(final_lr),
        lbfgs_steps=max(0, int(final_lbfgs_steps)),
        show_progress=False, progress_desc="Constrained symbolic projection",
    )
    return model




def _fit_symbolic_shape_curve(
    xx: torch.Tensor, yy: torch.Tensor, library: Sequence[str], *, topk: int = 4, max_abs_value: float = 1e5,
) -> List[Dict[str, object]]:
    """Fit symbolic *shape* families to an arbitrary 1-D sampled curve.

    Output amplitude/offset are used only for ranking; SumProduct canonicalization
    later keeps only the input chart ``g(beta*x+gamma)``.
    """
    xx = xx.reshape(-1); yy = yy.reshape(-1)
    good = torch.isfinite(xx) & torch.isfinite(yy)
    xx, yy = xx[good], yy[good]
    if xx.numel() < 8 or float(torch.std(yy).detach().cpu()) < 1e-8:
        return []
    ymean = yy.mean(); yc = yy - ymean; sst = yc.square().sum().clamp_min(1e-12)
    bvals = torch.tensor([-4.0,-3.0,-2.0,-1.5,-1.0,-0.75,-0.5,-0.25,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0], device=xx.device, dtype=xx.dtype)
    c_general = torch.tensor([-1.0,-0.5,0.0,0.5,1.0], device=xx.device, dtype=xx.dtype)
    c_periodic = torch.tensor([-math.pi,-math.pi/2.0,0.0,math.pi/2.0,math.pi], device=xx.device, dtype=xx.dtype)
    out=[]
    for name in library:
        if name not in SYMBOLIC_LIB: continue
        fun=SYMBOLIC_LIB[name][0]; cvals=c_periodic if name in {"sin","cos"} else c_general
        z=bvals[:,None,None]*xx[None,None,:]+cvals[None,:,None]
        try: gg=fun(z)
        except Exception: continue
        finite=torch.isfinite(gg).all(dim=-1)
        gg=torch.nan_to_num(gg,nan=0.0,posinf=max_abs_value,neginf=-max_abs_value).clamp(-max_abs_value,max_abs_value)
        flat=gg.reshape(-1,xx.numel()); valid=finite.reshape(-1)
        gm=flat.mean(dim=1,keepdim=True); gc=flat-gm; var=gc.square().sum(dim=1).clamp_min(1e-12)
        aa=(gc*yc.unsqueeze(0)).sum(dim=1)/var; dd=ymean-aa*gm.squeeze(1)
        pred=aa[:,None]*flat+dd[:,None]; sse=(pred-yy.unsqueeze(0)).square().sum(dim=1)
        sse=torch.where(valid,sse,torch.full_like(sse,float("inf")))
        idx=int(torch.argmin(sse).item())
        if not torch.isfinite(sse[idx]): continue
        nb=int(cvals.numel()); ib=idx//nb; ic=idx%nb
        out.append({"name":name,"r2":float((1.0-sse[idx]/sst).detach().cpu()),
                    "affine":(float(aa[idx].detach().cpu()),float(bvals[ib].detach().cpu()),float(cvals[ic].detach().cpu()),float(dd[idx].detach().cpu()))})
    out.sort(key=lambda z:float(z["r2"]),reverse=True)
    return out[:max(1,int(topk))]


def interaction_shape_symbolic_shortlists(
    x: torch.Tensor, y: torch.Tensor, structure: Sequence[int], library: Sequence[str], *, bins: int = 24, topk: int = 4, min_bin_count: int = 2,
) -> Optional[List[List[Dict[str, object]]]]:
    """Shape-only operator proposals from the two-way interaction surface.

    For a cross-variable product f(x_i)g(x_j), removing additive row/column
    effects leaves an approximately rank-one interaction.  The leading SVD
    factors therefore reveal the two multiplicative shapes without asking one
    candidate product to explain the entire additive target.  This routine only
    prunes/proposes operator families; GSR still decides the final rules.
    """
    vars_=tuple(int(v) for v in structure)
    if len(vars_) != 2 or vars_[0] == vars_[1]: return None
    i,j=vars_; xi=x[:,i].reshape(-1); xj=x[:,j].reshape(-1); yy=y.reshape(y.shape[0],-1).mean(dim=1)
    B=max(8,int(bins))
    qi=torch.linspace(0,1,B+1,device=x.device,dtype=x.dtype)
    ei=torch.quantile(xi,qi); ej=torch.quantile(xj,qi)
    # Guard duplicated quantiles.
    ei=ei+torch.arange(B+1,device=x.device,dtype=x.dtype)*torch.finfo(x.dtype).eps*8
    ej=ej+torch.arange(B+1,device=x.device,dtype=x.dtype)*torch.finfo(x.dtype).eps*8
    bi=torch.bucketize(xi.contiguous(),ei[1:-1].contiguous()).clamp(0,B-1); bj=torch.bucketize(xj.contiguous(),ej[1:-1].contiguous()).clamp(0,B-1)
    sums=torch.zeros((B,B),device=x.device,dtype=x.dtype); counts=torch.zeros((B,B),device=x.device,dtype=x.dtype)
    sums.index_put_((bi,bj),yy,accumulate=True); counts.index_put_((bi,bj),torch.ones_like(yy),accumulate=True)
    mask=counts>=float(min_bin_count); global_mean=yy.mean()
    M=torch.where(mask,sums/counts.clamp_min(1.0),global_mean)
    # Fill sparse cells with additive row/column estimates before removing them.
    row_num=(M*mask).sum(1); row_den=mask.sum(1).clamp_min(1); row=row_num/row_den
    col_num=(M*mask).sum(0); col_den=mask.sum(0).clamp_min(1); col=col_num/col_den
    fill=row[:,None]+col[None,:]-global_mean
    M=torch.where(mask,M,fill)
    I=M-M.mean(1,keepdim=True)-M.mean(0,keepdim=True)+M.mean()
    try: U,S,Vh=torch.linalg.svd(I,full_matrices=False)
    except Exception: return None
    if S.numel()==0 or float(S[0].detach().cpu())<1e-8: return None
    rank1_fraction=float((S[0].square()/S.square().sum().clamp_min(1e-12)).detach().cpu())
    ci=0.5*(ei[:-1]+ei[1:]); cj=0.5*(ej[:-1]+ej[1:])
    si=U[:,0]*torch.sqrt(S[0]); sj=Vh[0,:]*torch.sqrt(S[0])
    li=_fit_symbolic_shape_curve(ci,si,library,topk=topk); lj=_fit_symbolic_shape_curve(cj,sj,library,topk=topk)
    if li and lj:
        for _slot in (li,lj):
            for _cand in _slot: _cand["interaction_rank1_fraction"]=rank1_fraction
        return [li,lj]
    return None



def capture_numeric_support_evidence(
    model: SumProductKAN,
    x: torch.Tensor,
    *,
    factor_open_threshold: float = 0.20,
) -> List[Dict[str, object]]:
    """Capture a high-recall numerical support snapshot before destructive pruning.

    The snapshot is deliberately support-level rather than a literal symbolic
    decomposition.  For every numerical rule we record the most likely input
    variable of each sufficiently open factor slot together with a soft evidence
    score.  Inactive rules are retained: a rule may be numerically redundant yet
    carry the only evidence for a symbolically necessary interaction support.
    """
    model.eval()
    with torch.no_grad():
        _, details = model(x, return_details=True)
        vprob = details["variable_probabilities"]
        rprob = model.rule_gate.open_probability().reshape(model.n_rules)
        fprob = model.factor_gate.open_probability().reshape(model.n_rules, model.max_factors)
        raw = details["raw_rules"]
        scales = model.rule_scale.detach()
        raw_strength = (raw * scales[None, :]).std(dim=0)
    out: List[Dict[str, object]] = []
    for r in range(model.n_rules):
        vars_: List[int] = []
        slot_conf: List[float] = []
        for s in range(model.max_factors):
            fp = 1.0 if s < int(model.min_order) else float(fprob[r, s].detach().cpu())
            if s >= int(model.min_order) and fp < float(factor_open_threshold):
                continue
            probs = vprob[r, s, : model.in_dim]
            j = int(torch.argmax(probs).item())
            vars_.append(j)
            slot_conf.append(float(probs[j].detach().cpu()) * fp)
        if not vars_:
            continue
        support = tuple(sorted(set(vars_)))
        multiplicity = tuple(sorted(vars_))
        structure_prob = float(rprob[r].detach().cpu())
        for vv in slot_conf:
            structure_prob *= max(vv, 1e-8)
        strength = float(raw_strength[r].detach().cpu())
        out.append({
            "rule": int(r),
            "support": support,
            "multiplicity": multiplicity,
            "rule_probability": float(rprob[r].detach().cpu()),
            "structure_probability": float(structure_prob),
            "numeric_strength": float(strength),
            "active": bool(
                bool(model.discretized.item())
                and bool(model.hard_rule_choice[r].item())
                and bool(model.rule_alive_mask[r].item())
            ),
        })
    return out


def learned_numeric_support_classes(
    model: SumProductKAN,
    *,
    evidence: Optional[Sequence[Dict[str, object]]] = None,
    max_supports: int = 0,
    include_all_active: bool = True,
    score_importance_mix: float = 0.50,
) -> List[Dict[str, object]]:
    """Group numerical rows into distinct variable-support classes.

    ``evidence`` should normally be captured before numerical pruning.  Final
    active supports are merged in so pruning cannot delete a support from the
    symbolic admissible set.  The ranking score mixes soft structural confidence
    with observed numerical contribution strength, but the returned object is a
    support prior only; it does not constrain symbolic factor multiplicity or
    operator identity.
    """
    rows = list(evidence or [])
    if bool(model.discretized.item()):
        fprob = model.factor_gate.open_probability().reshape(model.n_rules, model.max_factors).detach()
        for r in range(model.n_rules):
            if not bool(model.hard_rule_choice[r].item() and model.rule_alive_mask[r].item()):
                continue
            vals = []
            for s in range(model.max_factors):
                if s >= int(model.min_order) and not bool(model.factor_alive_mask[r, s].item()):
                    continue
                j = int(model.hard_variable_choice[r, s].item())
                if j < model.in_dim:
                    vals.append(j)
            if vals:
                rows.append({
                    "rule": int(r), "support": tuple(sorted(set(vals))),
                    "multiplicity": tuple(sorted(vals)), "rule_probability": 1.0,
                    "structure_probability": 1.0,
                    "numeric_strength": float(abs(model.rule_scale[r].detach().cpu())),
                    "active": True,
                })
    by: Dict[Tuple[int, ...], Dict[str, object]] = {}
    mix = min(1.0, max(0.0, float(score_importance_mix)))
    for rec in rows:
        support = tuple(sorted(int(v) for v in rec.get("support", ())))
        if not support:
            continue
        cls = by.setdefault(support, {
            "support": support, "members": [], "active_members": [],
            "structure_probability": 0.0, "numeric_strength": 0.0,
            "observed_multiplicities": [],
        })
        cls["members"].append(int(rec.get("rule", -1)))
        if bool(rec.get("active", False)):
            cls["active_members"].append(int(rec.get("rule", -1)))
        sp = float(rec.get("structure_probability", 0.0))
        st = float(rec.get("numeric_strength", 0.0))
        cls["structure_probability"] = max(float(cls["structure_probability"]), sp)
        cls["numeric_strength"] = max(float(cls["numeric_strength"]), st)
        mult = tuple(int(v) for v in rec.get("multiplicity", ()))
        if mult and mult not in cls["observed_multiplicities"]:
            cls["observed_multiplicities"].append(mult)
    if not by:
        return []
    max_strength = max(float(c["numeric_strength"]) for c in by.values()) or 1.0
    out = []
    for cls in by.values():
        sp = float(cls["structure_probability"])
        st = float(cls["numeric_strength"]) / max_strength
        cls["score"] = (1.0 - mix) * sp + mix * st
        cls["representative"] = int(cls["active_members"][0] if cls["active_members"] else cls["members"][0])
        out.append(cls)
    out.sort(key=lambda c: (
        bool(c["active_members"]), float(c["score"]), float(c["structure_probability"]),
        float(c["numeric_strength"])
    ), reverse=True)
    if int(max_supports) > 0 and len(out) > int(max_supports):
        if include_all_active:
            active = [c for c in out if c["active_members"]]
            rest = [c for c in out if not c["active_members"]]
            keep = active[:int(max_supports)]
            if len(keep) < int(max_supports):
                keep += rest[:int(max_supports) - len(keep)]
            out = keep
        else:
            out = out[:int(max_supports)]
    return out


def learned_structure_symbolic_bank(
    support_classes: Sequence[Dict[str, object]],
    *,
    max_factors: int,
) -> List[Tuple[int, ...]]:
    """Return the complete repeated-factor grammar inside learned supports only."""
    bank: List[Tuple[int, ...]] = []
    seen = set()
    q = max(1, int(max_factors))
    for cls in support_classes:
        support = tuple(sorted(int(v) for v in cls["support"]))
        s = len(support)
        if s == 0 or s > q:
            continue
        # Positive compositions of total order m across all variables in support.
        def compositions(total: int, parts: int):
            if parts == 1:
                yield (total,); return
            for first in range(1, total - parts + 2):
                for rest in compositions(total-first, parts-1):
                    yield (first,) + rest
        for m in range(s, q+1):
            for counts in compositions(m, s):
                z = tuple(v for v, c in zip(support, counts) for _ in range(c))
                if z not in seen:
                    seen.add(z); bank.append(z)
    bank.sort(key=lambda z: (len(z), z))
    return bank


def learned_support_symbolic_gsr(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    structure_candidates: Sequence[Tuple[int, ...]],
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    **kwargs,
):
    """Hard in-context GSR that cannot leave the numerical RuleKAN supports."""
    if structure_candidates is None or len(structure_candidates) == 0:
        raise ValueError("learned_support_symbolic_gsr requires an explicit learned-support bank")
    allowed = (
        {frozenset(int(v) for v in s) for s in allowed_supports}
        if allowed_supports is not None
        else {frozenset(c["support"]) for c in learned_numeric_support_classes(numeric_model)}
    )
    checked = []
    for z in structure_candidates:
        zz = tuple(sorted(int(v) for v in z))
        if frozenset(zz) not in allowed:
            raise ValueError(f"unlearned support in RuleKAN GSR: {zz}")
        checked.append(zz)
    kwargs.setdefault(
        "affine_partition_allowed_supports",
        [tuple(sorted(int(v) for v in s)) for s in allowed],
    )
    return mandatory_symbolic_matching_pursuit(
        numeric_model, train_x, train_y, val_x, val_y,
        structure_candidates=checked, **kwargs,
    )

def mandatory_symbolic_matching_pursuit(
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    library: Optional[Sequence[str]] = None,
    local_topk: int = 4,
    max_rule_candidates: int = 16,
    max_symbolic_rules: int = 6,
    min_symbolic_rules: int = 1,
    shortlist_refine_steps: int = 120,
    beam_width: int = 4,
    structure_diverse_beam: bool = False,
    beam_max_per_structure: int = 1,
    residual_structure_topk: int = 0,
    residual_structure_mass: float = 0.0,
    residual_structure_min: int = 1,
    residual_structure_max: int = 0,
    residual_structure_gap_rel: float = 0.0,
    joint_scale_refit_each_commit: bool = False,
    trial_steps: int = 180,
    trial_lr: float = 3e-4,
    final_steps: int = 1400,
    final_lr: float = 2e-4,
    final_lbfgs_steps: int = 250,
    target_val_rmse: float = 1e-4,
    include_inactive_structures: bool = True,
    allow_symbolic_self_products: bool = True,
    structure_candidates: Optional[Sequence[Tuple[int, ...]]] = None,
    use_gmp_preselection: bool = True,
    gmp_topk: Optional[int] = None,
    gmp_unary_topk: Optional[int] = None,
    gmp_self_product_topk: Optional[int] = None,
    gmp_steps: int = 120,
    gmp_lr: float = 2e-2,
    gmp_temperature_start: float = 1.5,
    gmp_temperature_end: float = 0.35,
    gmp_entropy_weight: float = 2e-4,
    gmp_relaxation_mode: str = "soft",
    gmp_atom_backward_normalization: str = "none",
    gmp_gumbel_noise_scale: float = 1.0,
    gmp_complexity_weight: float = 0.0,
    gmp_complexity_logit_prior: float = 0.0,
    gmp_nonlinearity_logit_prior: float = 0.0,
    gmp_curvature_weight: float = 0.0,
    gmp_nonlinearity_weight: float = 0.0,
    gmp_identity_chart: str = "raw",
    gmp_tuple_refine_steps: int = 50,
    gmp_tuple_refine_lr: float = 1.5e-2,
    interaction_shape_screening: bool = True,
    interaction_shape_topk: int = 0,
    interaction_shape_bins: int = 8,
    interaction_shape_min_rank1: float = 0.65,
    interaction_shape_require_multiple: bool = True,
    hard_proposal_union: bool = False,
    hard_proposal_top_tuples: int = 8,
    hard_proposal_beam_width: int = 48,
    hard_proposal_max_samples: int = 384,
    hard_proposal_max_per_factor: int = 6,
    initial_block_pursuit: bool = True,
    initial_block_pool: int = 28,
    initial_block_pair_beam: int = 10,
    initial_block_refit_steps: int = 100,
    initial_block_lbfgs_steps: int = 20,
    initial_block_consolidate_steps: int = 200,
    initial_block_consolidate_lbfgs_steps: int = 60,
    hybrid_hard_screening: bool = False,
    hard_screen_beam_width: int = 64,
    hard_screen_top_tuples: int = 12,
    hard_screen_max_samples: int = 512,
    hard_screen_start_step: int = 1,
    hard_screen_matching_steps: int = 4,
    hard_screen_global_candidates: int = 2,
    hard_screen_structure_topk: int = 2,
    hard_screen_residual_rescue: bool = False,
    commit_refit_steps: int = 300,
    commit_lbfgs_steps: int = 80,
    backfit: bool = True,
    backfit_beam_width: int = 3,
    backfit_steps: int = 120,
    backfit_lbfgs_steps: int = 40,
    pursuit_mode: str = "gsr",
    omp_ridge: float = 1e-8,
    omp_extra_steps: int = 120,
    omp_redundancy_each_commit: bool = True,
    min_rule_improvement_rel: float = 5e-4,
    max_rules_per_structure: int = 0,
    residual_operator_rescue: bool = True,
    residual_rescue_gmp_topk: Optional[int] = None,
    residual_rescue_gmp_steps: int = 80,
    residual_rescue_beam_width: int = 3,
    residual_rescue_steps: int = 80,
    residual_rescue_lbfgs_steps: int = 30,
    elimination_rel_mse_budget: float = 1e-2,
    debug_topk: int = 5,
    debug_log_style: str = "compact",
    tiny_rule_scale_threshold: float = 1e-4,
    tiny_rule_contribution_rmse: float = 1e-5,
    tiny_rule_rel_mse_budget: float = 2e-3,
    cleanup_max_seconds: float = 120.0,
    cleanup_max_trials: int = 32,
    cleanup_min_rules: int = 1,
    redundancy_cleanup: bool = True,
    redundancy_span_r2_threshold: float = 0.995,
    redundancy_corr_threshold: float = 0.995,
    redundancy_rel_mse_budget: float = 2e-3,
    affine_partition_rescue: bool = True,
    affine_partition_allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    affine_partition_family_beam: int = 18,
    affine_partition_seed_topk: int = 0,
    affine_partition_max_samples: int = 384,
    affine_partition_max_support_pairs: int = 12,
    affine_partition_refine_steps: int = 120,
    affine_partition_refine_lr: float = 1.0e-3,
    affine_partition_lbfgs_steps: int = 20,
    affine_partition_final_polish_topk: int = 8,
    affine_partition_final_polish_steps: int = 500,
    affine_partition_final_polish_lbfgs_steps: int = 80,
    affine_partition_equivalence_rel_mse: float = 3.0,
    affine_partition_equivalence_nrmse: float = 5e-4,
    affine_partition_cancellation_weight: float = 1.0,
    affine_partition_complexity_weight: float = 0.02,
    affine_partition_min_improvement_rel: float = 1e-4,
    symbolic_seed: int = 0,
    verbose: bool = True,
    show_progress: bool = False,
) -> Tuple[SumProductKAN, List[Dict[str, object]]]:
    """Build a fully symbolic model by forward matching pursuit.

    Numerical dependency discovery and symbolic factorization are separate.
    Same-input products are proposed through an independent symbolic structure
    bank, where decompositions such as ``exp(x)*sin(x)`` are meaningful because
    both factors must come from the restricted operator library.

    A GMP-style gate optimization prunes the operator library to top-k
    candidates per factor before expensive in-context trials.  Matching pursuit
    remains the final arbiter: every committed product rule is evaluated after a
    full symbolic-model refit on the training set and validation scoring.

    Fully symbolic factors use the canonical form ``g(beta*x + gamma)``.
    Per-factor output scales are absorbed into the rule coefficient, and
    per-factor output offsets are forbidden because multiplying shifted factors
    would generate unintended lower-order additive terms.

    GMP pruning is not treated as permanent.  After the global symbolic fit, a
    leave-one-rule-out residual rescue can reintroduce operators that were weak
    under the initial residual but become appropriate after other mechanisms are
    explained.  This is especially important for same-variable decompositions.
    """
    if not bool(numeric_model.discretized.item()):
        raise ValueError("mandatory symbolic matching pursuit requires a discretized numerical model")
    lib = tuple(name for name in (numeric_model.symbolic_library if library is None else library) if name in SYMBOLIC_LIB)
    gmp_topk, gmp_unary_topk, gmp_self_product_topk = scaled_gmp_screening_sizes(
        len(lib), topk=gmp_topk, unary_topk=gmp_unary_topk,
        self_product_topk=gmp_self_product_topk,
    )
    # Symbolic search has its own local RNG stream.  It is deliberately
    # independent of how many random numbers numerical model construction or
    # training consumed before symbolic takeover.
    symbolic_rng = _local_torch_generator(train_x.device, int(symbolic_seed))
    if residual_rescue_gmp_topk is None:
        residual_rescue_gmp_topk = scaled_gmp_screening_sizes(
            len(lib), topk=None, unary_topk=None, self_product_topk=None,
            reference_topk=4, reference_unary_topk=4, reference_self_product_topk=4,
        )[0]
    if str(debug_log_style).lower() not in {"compact", "verbose"}:
        raise ValueError("debug_log_style must be 'compact' or 'verbose'")
    pursuit_mode = str(pursuit_mode).strip().lower()
    if pursuit_mode not in {"gsr", "omp_linear", "omp_nonlinear", "omp_full"}:
        raise ValueError("pursuit_mode must be gsr, omp_linear, omp_nonlinear, or omp_full")
    if not lib:
        raise ValueError("symbolic library is empty")
    pursuit_t0 = time.perf_counter()
    gmp_seconds = 0.0; template_seconds = 0.0; gsr_seconds = 0.0; final_seconds = 0.0
    gsr_trial_count = 0

    if structure_candidates is None:
        structures = symbolic_structure_bank(
            numeric_model.in_dim, numeric_model.max_factors,
            allow_self_products=bool(allow_symbolic_self_products),
        )
    else:
        seen_structures = set()
        structures = []
        for raw in structure_candidates:
            z = tuple(int(v) for v in raw)
            if not z or len(z) > int(numeric_model.max_factors):
                continue
            if any(v < 0 or v >= int(numeric_model.in_dim) for v in z):
                continue
            z = tuple(sorted(z))
            if z not in seen_structures:
                seen_structures.add(z); structures.append(z)
        if not structures:
            raise ValueError("structure_candidates produced an empty symbolic structure bank")
    # ``include_inactive_structures`` is retained for API compatibility.  The
    # symbolic bank is independent of numerical active/inactive rows by design.
    _ = include_inactive_structures

    if use_gmp_preselection:
        _gmp_t0 = time.perf_counter()
        gmp_results, gmp_stats = gmp_symbolic_operator_preselection(
            train_x, train_y, structures, lib,
            topk=gmp_topk, unary_topk=gmp_unary_topk,
            self_product_topk=gmp_self_product_topk, generator=symbolic_rng,
            steps=gmp_steps, lr=gmp_lr,
            temperature_start=gmp_temperature_start,
            temperature_end=gmp_temperature_end,
            entropy_weight=gmp_entropy_weight,
            relaxation_mode=gmp_relaxation_mode,
            atom_backward_normalization=gmp_atom_backward_normalization,
            gumbel_noise_scale=gmp_gumbel_noise_scale,
            complexity_weight=gmp_complexity_weight,
            complexity_logit_prior=gmp_complexity_logit_prior,
            nonlinearity_logit_prior=gmp_nonlinearity_logit_prior,
            curvature_weight=gmp_curvature_weight,
            nonlinearity_weight=gmp_nonlinearity_weight,
            identity_chart=gmp_identity_chart,
            debug_topk=debug_topk, debug_log_style=debug_log_style, debug_label="GMP",
            show_progress=show_progress,
        )
        by_structure = {tuple(z["structure"]): z for z in gmp_results}
        _interaction_shapes = {}
        if bool(interaction_shape_screening):
            _shape_topk = len(lib) if int(interaction_shape_topk) <= 0 else min(len(lib), int(interaction_shape_topk))
            for _structure in structures:
                _vars = tuple(int(v) for v in _structure)
                _shape = interaction_shape_symbolic_shortlists(
                    train_x, train_y, _vars, lib, bins=int(interaction_shape_bins), topk=int(_shape_topk)
                )
                if _shape is None: continue
                _strength=float(_shape[0][0].get("interaction_rank1_fraction",0.0)) if _shape[0] else 0.0
                if _strength >= float(interaction_shape_min_rank1): _interaction_shapes[_vars]=_shape
            _enable_shape = bool(_interaction_shapes)
            if bool(interaction_shape_require_multiple): _enable_shape = len(_interaction_shapes) >= 2
            if _enable_shape:
                for _vars,_shape in _interaction_shapes.items():
                    if _vars not in by_structure: continue
                    _merged=[]
                    for _base,_extra in zip(by_structure[_vars]["factor_shortlists"],_shape):
                        _seen=set(); _slot=[]; _annot=[]
                        for _rank,_cand0 in enumerate(_extra):
                            _cand=dict(_cand0); _cand["interaction_shape_r2"]=float(_cand0.get("r2",-1.0)); _cand["interaction_shape_rank"]=int(_rank)
                            _annot.append(_cand)
                        for _cand in list(_annot)+list(_base):
                            _name=str(_cand["name"])
                            if _name in _seen: continue
                            _seen.add(_name); _slot.append(_cand)
                            if len(_slot) >= max(int(_shape_topk), len(_base)): break
                        _merged.append(_slot)
                    by_structure[_vars]["factor_shortlists"]=_merged
        # High-recall static hard proposals. GMP is an efficient differentiable
        # ranking mechanism, but a strict top-k can permanently remove the
        # correct operator when several affine-reparameterised families are near
        # ties.  For RuleKAN's small learned-support bank, run one cheap hard
        # multistart screen per structure and union the operators appearing in
        # its best complete tuples with the GMP shortlist.  Final selection is
        # still in-context GSR; these proposals only prevent false-negative
        # pruning by the soft gate stage.
        _hard_tuple_proposals: Dict[Tuple[int, ...], List[Tuple[Dict[str, object], ...]]] = {}
        if bool(hard_proposal_union):
            for _structure in structures:
                _vars = tuple(int(v) for v in _structure)
                if _vars not in by_structure:
                    continue
                _hard = hard_symbolic_tuple_screening(
                    train_x, train_y, _vars, lib,
                    beam_width=max(1, int(hard_proposal_beam_width)),
                    top_tuples=max(1, int(hard_proposal_top_tuples)),
                    max_samples=max(32, int(hard_proposal_max_samples)),
                    seeds_per_operator_prefix=3,
                )
                if not _hard:
                    continue
                _hard_tuple_proposals[_vars] = [tuple(dict(c) for c in combo) for combo in _hard]
                _per_slot = [[] for _ in _vars]
                for _combo in _hard:
                    for _ss, _cand0 in enumerate(_combo):
                        _cand = dict(_cand0)
                        _cand["hard_proposal"] = True
                        _per_slot[_ss].append(_cand)
                _merged=[]
                for _base,_extra in zip(by_structure[_vars]["factor_shortlists"],_per_slot):
                    _seen=set(); _slot=[]
                    # Keep the differentiable GMP proposals first, then add hard
                    # proposals until the explicit per-factor recall budget.
                    for _cand in list(_base)+list(_extra):
                        _name=str(_cand["name"])
                        if _name in _seen: continue
                        _seen.add(_name); _slot.append(_cand)
                        if len(_slot) >= max(len(_base), int(hard_proposal_max_per_factor)):
                            break
                    _merged.append(_slot)
                by_structure[_vars]["factor_shortlists"]=_merged
        gmp_seconds = time.perf_counter() - _gmp_t0
    else:
        by_structure = {}
        _hard_tuple_proposals = {}
        K=len(lib); naive=sum(K**len(z) for z in structures)
        gmp_stats={
            "structures":float(len(structures)),"library_size":float(K),"topk":float(K),
            "unary_topk":float(K),"self_product_topk":float(K),
            "naive_operator_tuples":float(naive),"gmp_operator_tuples":float(naive),
            "tuple_reduction_fraction":0.0,
        }

    _template_t0 = time.perf_counter()
    templates: List[Dict[str, object]] = []
    template_bar = tqdm(
        structures, total=len(structures), desc="Symbolic template generation",
        disable=not bool(show_progress), leave=True, dynamic_ncols=True,
        mininterval=0.25, smoothing=0.08,
    )
    for structure in template_bar:
        vars_ = tuple(int(v) for v in structure)
        factors = tuple((slot, var) for slot, var in enumerate(vars_))
        shortlists: List[List[Dict[str, object]]] = []
        if use_gmp_preselection and vars_ in by_structure:
            shortlists = by_structure[vars_]["factor_shortlists"]
        else:
            # Exhaustive fallback: generic affine seeds. This path is slower but
            # preserves the mandatory-symbolic contract when GMP is disabled.
            for slot, var in factors:
                cands=[]
                for name in lib:
                    b0 = 2.5 if name in {"sin","cos"} else (-0.5 if name=="exp" else 1.0)
                    cands.append({"name":name,"affine":(1.0,b0,0.0,0.0),"gmp_prob":1.0/len(lib)})
                shortlists.append(cands)
        if len(shortlists) != len(factors) or any(not z for z in shortlists):
            continue
        _all_combos = list(itertools.product(*shortlists))
        _gmp_combos = sorted(
            _all_combos,
            key=lambda cc: sum(math.log(max(float(z.get("gmp_prob",1e-12)),1e-12)) for z in cc),
            reverse=True,
        )[:max(1,int(max_rule_candidates))]
        combos = list(_gmp_combos)
        if bool(hard_proposal_union) and vars_ in _hard_tuple_proposals:
            _seen_hard = {
                (tuple(str(z["name"]) for z in cc),
                 tuple(tuple(float(v) for v in z["affine"]) for z in cc))
                for cc in combos
            }
            for _cc in _hard_tuple_proposals[vars_][:max(1, int(hard_proposal_top_tuples))]:
                _key=(tuple(str(z["name"]) for z in _cc),
                      tuple(tuple(float(v) for v in z["affine"]) for z in _cc))
                if _key in _seen_hard:
                    continue
                _seen_hard.add(_key); combos.append(_cc)
        if bool(interaction_shape_screening):
            _shape_combos=[]
            for _cc in _all_combos:
                if not all("interaction_shape_r2" in _z for _z in _cc):
                    continue
                _r2=sum(float(_z["interaction_shape_r2"]) for _z in _cc)/len(_cc)
                _cx=sum(float(SYMBOLIC_LIB[str(_z["name"])][2]) for _z in _cc)/len(_cc)
                _shape_combos.append((_r2-0.02*_cx,_cc))
            _shape_combos.sort(key=lambda z:z[0],reverse=True)
            _seen={(tuple(str(z["name"]) for z in cc), tuple(tuple(float(v) for v in z["affine"]) for z in cc)) for cc in combos}
            for _score,_cc in _shape_combos[:max(1,int(max_rule_candidates))]:
                _key=(tuple(str(z["name"]) for z in _cc), tuple(tuple(float(v) for v in z["affine"]) for z in _cc))
                if _key in _seen: continue
                _seen.add(_key); combos.append(_cc)
        polished=[]
        for combo in combos:
            if use_gmp_preselection and int(gmp_tuple_refine_steps)>0:
                pc, pmse = _polish_symbolic_operator_tuple(
                    train_x, train_y, vars_, combo, steps=gmp_tuple_refine_steps,
                    lr=gmp_tuple_refine_lr,
                )
            else:
                pc, pmse = tuple(combo), float("inf")
            polished.append((pmse,pc))
        polished.sort(key=lambda z:z[0])
        for pmse,combo in polished:
            templates.append({"structure":vars_,"factors":factors,"combo":tuple(combo),"tuple_mse":pmse})
        if show_progress:
            template_bar.set_postfix(
                structure="*".join(str(v) for v in vars_), templates=len(templates), refresh=False
            )
    template_seconds = time.perf_counter() - _template_t0
    if not templates:
        raise RuntimeError("no all-symbolic product templates could be generated")

    model=_make_fully_symbolic_shell(numeric_model)
    with torch.no_grad(): model.bias.fill_(float(train_y.mean().cpu()))
    history=[]; used_templates=set(); rule_template_idx = {}
    model.eval()
    with torch.no_grad(): current_mse=float(torch.mean((model(val_x)-val_y)**2).cpu())
    if verbose:
        print("\n=== FULLY SYMBOLIC GMP + GSR ===")
        print(
            f"  [setup] bias={math.sqrt(current_mse):.6g} | structures={len(structures)} | "
            f"templates={len(templates)} | self-products={bool(allow_symbolic_self_products)}"
        )
        if use_gmp_preselection:
            print(
                f"  [GMP] tuples {int(gmp_stats['naive_operator_tuples'])} -> "
                f"{int(gmp_stats['gmp_operator_tuples'])} "
                f"(-{100.0*gmp_stats['tuple_reduction_fraction']:.1f}%)"
            )
        print(f"  [GSR] beam={max(1,int(beam_width))} full-model refits/add | pursuit={pursuit_mode}")

    _gsr_t0 = time.perf_counter()
    initial_committed = 0
    # Optional two-rule block initialization.  This changes the greedy objective
    # at the level where SumProduct is additive: candidate rules are still
    # proposed factor-wise, but the first two mechanisms are scored jointly with
    # a cheap linear coefficient/bias refit before any nonlinear polishing.
    if bool(initial_block_pursuit) and int(max_symbolic_rules) >= 2 and model.n_rules >= 2:
        with torch.no_grad():
            _res0=(train_y-model(train_x)).reshape(train_y.shape[0],-1).mean(dim=1)
        _rank0=[]
        for _ti,_tpl in enumerate(templates):
            _h=_candidate_rule_value(model,train_x,0,_tpl["factors"],_tpl["combo"])
            _score,_scale,_off=_best_affine_residual_rmse(_h,_res0)
            if math.isfinite(_score): _rank0.append((_score,_ti,_scale))
        _rank0.sort(key=lambda z:z[0])
        # Build a shape-first candidate pool when interaction evidence exists.
        # This is the crucial difference from ordinary greedy pursuit: candidates
        # that are individually mediocre against the *whole* target can still
        # enter the initial joint block if their factor shapes match a rank-one
        # interaction mechanism.
        _shape_rank=[]
        for _item in _rank0:
            _tpl=templates[int(_item[1])]; _vals=[]; _complex=0.0
            for _c in _tpl["combo"]:
                if "interaction_shape_r2" in _c: _vals.append(float(_c["interaction_shape_r2"]))
                _complex += float(SYMBOLIC_LIB[str(_c["name"])][2])
            if len(_vals)==len(_tpl["combo"]) and _vals:
                _shape_mean=sum(_vals)/len(_vals)
                _shape_obj=-_shape_mean + 0.02*(_complex/max(1,len(_vals)))
                _shape_rank.append((_shape_obj, _complex, float(_item[0]), _item))
        _shape_rank.sort(key=lambda z:(z[0],z[1],z[2]))
        _pool=[]; _seen_ti=set(); _pc={}
        _cap=max(2,int(math.ceil(max(2,int(initial_block_pool))/max(1,len(structures)))))
        # First reserve shape-aligned candidates, capped per dependency structure.
        for _negshape,_complex,_coarse,_item in _shape_rank:
            _st=tuple(int(v) for v in templates[int(_item[1])]["structure"])
            if _pc.get(_st,0)>=_cap: continue
            _pool.append(_item); _seen_ti.add(int(_item[1])); _pc[_st]=_pc.get(_st,0)+1
            if len(_pool)>=max(2,int(initial_block_pool)): break
        # Fill remaining capacity with ordinary whole-target screening as safety net.
        for _item in _rank0:
            if int(_item[1]) in _seen_ti: continue
            _st=tuple(int(v) for v in templates[int(_item[1])]["structure"])
            if _pc.get(_st,0)>=_cap: continue
            _pool.append(_item); _seen_ti.add(int(_item[1])); _pc[_st]=_pc.get(_st,0)+1
            if len(_pool)>=max(2,int(initial_block_pool)): break
        if len(_pool)<2: _pool=_rank0[:max(2,int(initial_block_pool))]
        _pairs=[]
        for _a in range(len(_pool)):
            for _b in range(_a+1,len(_pool)):
                _sa,_tia,_sca=_pool[_a]; _sb,_tib,_scb=_pool[_b]
                if int(_tia)==int(_tib): continue
                if bool(interaction_shape_screening):
                    if tuple(int(v) for v in templates[int(_tia)]["structure"]) == tuple(int(v) for v in templates[int(_tib)]["structure"]):
                        continue
                    _ca=templates[int(_tia)]["combo"]; _cb=templates[int(_tib)]["combo"]
                    if not all("interaction_shape_r2" in _c for _c in _ca): continue
                    if not all("interaction_shape_r2" in _c for _c in _cb): continue
                _trial=copy.deepcopy(model)
                _install_symbolic_template(_trial,0,templates[int(_tia)]["factors"],templates[int(_tia)]["combo"],initial_scale=float(_sca))
                _install_symbolic_template(_trial,1,templates[int(_tib)]["factors"],templates[int(_tib)]["combo"],initial_scale=float(_scb))
                _trial,_mse=orthogonal_rule_scale_refit(_trial,train_x,train_y,val_x,val_y,ridge=float(omp_ridge))
                if math.isfinite(float(_mse)): _pairs.append((float(_mse),int(_tia),int(_tib),_trial))
        _pairs.sort(key=lambda z:z[0])
        _refine_pairs=list(_pairs[:max(1,int(initial_block_pair_beam))])
        if bool(interaction_shape_screening):
            _shape_pairs=[]
            for _item in _pairs:
                _mse,_tia,_tib,_trial=_item; _vals=[]; _cx=0.0
                for _ti in (_tia,_tib):
                    for _c in templates[int(_ti)]["combo"]:
                        if "interaction_shape_r2" in _c: _vals.append(float(_c["interaction_shape_r2"]))
                        _cx += float(SYMBOLIC_LIB[str(_c["name"])][2])
                if len(_vals)==sum(len(templates[int(_ti)]["combo"]) for _ti in (_tia,_tib)):
                    _score=sum(_vals)/len(_vals)-0.02*(_cx/max(1,len(_vals)))
                    _shape_pairs.append((_score,_item))
            _shape_pairs.sort(key=lambda z:z[0],reverse=True)
            _seen_pair={tuple(sorted((int(z[1]),int(z[2])))) for z in _refine_pairs}
            for _score,_item in _shape_pairs[:max(1,int(initial_block_pair_beam))]:
                _key=tuple(sorted((int(_item[1]),int(_item[2]))))
                if _key in _seen_pair: continue
                _seen_pair.add(_key); _refine_pairs.append(_item)
        _best_block=None
        for _mse,_tia,_tib,_trial in _refine_pairs:
            if int(initial_block_refit_steps)>0:
                _trial,_mse=_fully_symbolic_continuous_refit(
                    _trial,train_x,train_y,val_x,val_y,steps=int(initial_block_refit_steps),
                    lr=max(float(trial_lr)*2.0,8e-4),lbfgs_steps=int(initial_block_lbfgs_steps),show_progress=False,
                )
                _trial,_mse=orthogonal_rule_scale_refit(_trial,train_x,train_y,val_x,val_y,ridge=float(omp_ridge))
            if _best_block is None or float(_mse)<float(_best_block[0]):
                _best_block=(float(_mse),int(_tia),int(_tib),_trial)
        # Once the best two-rule mechanism block is selected, spend a stronger
        # continuous polish on that *one* block before allowing greedy pursuit to
        # add compensating rules. This is materially cheaper and more stable than
        # giving every candidate pair a large nonlinear optimization budget.
        if _best_block is not None and math.sqrt(max(0.0, float(_best_block[0]))) > float(target_val_rmse):
            _bmse,_btia,_btib,_btrial=_best_block
            if int(initial_block_consolidate_steps) > 0 or int(initial_block_consolidate_lbfgs_steps) > 0:
                _btrial,_bmse=_fully_symbolic_continuous_refit(
                    _btrial,train_x,train_y,val_x,val_y,
                    steps=max(0,int(initial_block_consolidate_steps)),
                    lr=max(float(trial_lr)*2.0,8e-4),
                    lbfgs_steps=max(0,int(initial_block_consolidate_lbfgs_steps)),
                    show_progress=False,
                )
                _btrial,_bmse=orthogonal_rule_scale_refit(
                    _btrial,train_x,train_y,val_x,val_y,ridge=float(omp_ridge)
                )
                _best_block=(float(_bmse),int(_btia),int(_btib),_btrial)
        if _best_block is not None and float(_best_block[0]) < float(current_mse)*(1.0-1e-7):
            current_mse,_tia,_tib,model=_best_block
            used_templates.update({_tia,_tib}); rule_template_idx[0]=_tia; rule_template_idx[1]=_tib
            initial_committed=2
            for _rr,_ti in ((0,_tia),(1,_tib)):
                _tpl=templates[_ti]; history.append({
                    "rule":_rr,"structure":tuple(int(v) for v in _tpl["structure"]),
                    "operators":[str(c["name"]) for c in _tpl["combo"]],
                    "validation_rmse":math.sqrt(float(current_mse)),"block_initial":True,
                })
            if verbose:
                _ta,_tb=templates[_tia],templates[_tib]
                print(f"  [block commit] {_fmt_structure(_ta['structure'])} {_format_symbolic_combo(_ta['combo'])} + "
                      f"{_fmt_structure(_tb['structure'])} {_format_symbolic_combo(_tb['combo'])} | val={math.sqrt(float(current_mse)):.8g}")

    mp_bar = tqdm(
        range(initial_committed, max(1, int(max_symbolic_rules))),
        total=max(1, int(max_symbolic_rules)), desc="Symbolic matching pursuit",
        disable=not bool(show_progress), leave=True, dynamic_ncols=True,
        mininterval=0.25, smoothing=0.08,
    )
    for step in mp_bar:
        free_rules = torch.nonzero(~model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        if not free_rules:
            break
        target_rule = int(free_rules[0])
        model.eval()
        with torch.no_grad():
            residual=(train_y-model(train_x)).reshape(train_y.shape[0],-1).mean(dim=1)

        ranked=[]
        structure_counts: Dict[Tuple[int, ...], int] = {}
        if int(max_rules_per_structure) > 0:
            for _rr, _ti in rule_template_idx.items():
                _s = tuple(int(v) for v in templates[int(_ti)]["structure"])
                structure_counts[_s] = structure_counts.get(_s, 0) + 1
        for ti,tpl in enumerate(templates):
            if ti in used_templates: continue
            if int(max_rules_per_structure) > 0:
                _s = tuple(int(v) for v in tpl["structure"])
                if structure_counts.get(_s, 0) >= int(max_rules_per_structure):
                    continue
            h=_candidate_rule_value(model,train_x,target_rule,tpl["factors"],tpl["combo"])
            coarse_rmse, scale, offset = _best_affine_residual_rmse(h, residual)
            if math.isfinite(coarse_rmse): ranked.append((coarse_rmse,ti,scale,offset))

        # Compact hybrid shortlist union.  Soft GMP remains the primary proposal
        # bank.  During only the first few pursuit steps, deterministic hard
        # multistart screening contributes at most a tiny global number of
        # residual-conditioned candidates.  They are *not* trusted directly:
        # each still has to win the same in-context full-model refit/validation
        # used by GMP candidates before it can be committed.
        if (bool(hybrid_hard_screening) and step >= max(0, int(hard_screen_start_step))
                and step < max(0, int(hard_screen_matching_steps))):
            hard_ranked = []
            residual_target = residual.reshape(-1, 1)
            # First use the cheap already-materialised symbolic templates to
            # identify which learned supports currently explain the residual.
            # Expensive multistart hard screening is then restricted to that
            # tiny support set rather than repeated over the whole grammar.
            _best_struct: Dict[Tuple[int, ...], float] = {}
            for _score, _ti, _scale, _offset in ranked:
                _s = tuple(int(v) for v in templates[int(_ti)]["structure"])
                if _s not in _best_struct or float(_score) < _best_struct[_s]:
                    _best_struct[_s] = float(_score)
            _ordered_hard_structs = [z[0] for z in sorted(_best_struct.items(), key=lambda kv: kv[1])]
            if int(hard_screen_structure_topk) > 0:
                _ordered_hard_structs = _ordered_hard_structs[:int(hard_screen_structure_topk)]
            for hard_struct in _ordered_hard_structs:
                hard_vars = tuple(int(v) for v in hard_struct)
                hard_combos = hard_symbolic_tuple_screening(
                    train_x, residual_target, hard_vars, lib,
                    beam_width=int(hard_screen_beam_width),
                    top_tuples=max(1, int(hard_screen_top_tuples)), max_samples=int(hard_screen_max_samples),
                    seeds_per_operator_prefix=3,
                )
                if not hard_combos:
                    continue
                combo = tuple(hard_combos[0])
                factors_h = tuple((slot, var) for slot, var in enumerate(hard_vars))
                h = _candidate_rule_value(model, train_x, target_rule, factors_h, combo)
                coarse_rmse, scale, offset = _best_affine_residual_rmse(h, residual)
                if math.isfinite(coarse_rmse):
                    hard_ranked.append((coarse_rmse, hard_vars, factors_h, combo, scale, offset))
            hard_ranked.sort(key=lambda z: z[0])
            for coarse_rmse, hard_vars, factors_h, combo, scale, offset in hard_ranked[:max(0, int(hard_screen_global_candidates))]:
                if int(max_rules_per_structure) > 0 and structure_counts.get(tuple(hard_vars), 0) >= int(max_rules_per_structure):
                    continue
                # Do not add an exact duplicate hard proposal already present.
                names = tuple(str(c["name"]) for c in combo)
                duplicate = any(
                    tuple(int(v) for v in t["structure"]) == hard_vars and
                    tuple(str(c["name"]) for c in t["combo"]) == names and
                    all(abs(float(a["affine"][1])-float(b["affine"][1])) < 1e-9 and
                        abs(float(a["affine"][2])-float(b["affine"][2])) < 1e-9
                        for a,b in zip(t["combo"], combo))
                    for t in templates
                )
                if duplicate:
                    continue
                templates.append({
                    "structure": hard_vars, "factors": factors_h, "combo": combo,
                    "tuple_mse": float("nan"), "hybrid_hard_residual": True,
                })
                ti = len(templates) - 1
                ranked.append((coarse_rmse, ti, scale, offset))
        ranked.sort(key=lambda z:z[0])
        if not ranked: break

        # Residual-adaptive structure sparsification.  The symbolic structure
        # bank is intentionally overcomplete, but GSR's expensive in-context
        # refit beam should not be diluted across every possible variable edge.
        # Rank structures by their best current residual candidate and retain
        # only the top-k structures for this pursuit step.  This is recomputed
        # after every commit, so different mechanisms can enter as the residual
        # changes. A value <=0 keeps all structures in the search.
        if int(residual_structure_topk) > 0 or float(residual_structure_mass) > 0.0:
            best_by_structure: Dict[Tuple[int, ...], float] = {}
            for _score, _ti, _scale, _offset in ranked:
                _s = tuple(int(v) for v in templates[int(_ti)]["structure"])
                if _s not in best_by_structure or float(_score) < best_by_structure[_s]:
                    best_by_structure[_s] = float(_score)
            ordered_structures = sorted(best_by_structure.items(), key=lambda kv: kv[1])
            if float(residual_structure_mass) > 0.0 and ordered_structures:
                # Turn residual-RMSE screening into nonnegative explanatory gains.
                # A structure gets credit only for improving over predicting the
                # current residual by zero.  Keep the smallest prefix explaining
                # the requested fraction of total positive screening gain.
                residual_rmse = float(torch.sqrt(torch.mean(residual.square())).detach().cpu())
                gains = [max(0.0, residual_rmse - float(score)) for _, score in ordered_structures]
                total_gain = sum(gains)
                k = max(1, int(residual_structure_min))
                if total_gain > 1e-15:
                    target = max(0.0, min(1.0, float(residual_structure_mass))) * total_gain
                    accum = 0.0
                    k = 0
                    for gain in gains:
                        accum += gain; k += 1
                        if accum >= target: break
                    k = max(k, max(1, int(residual_structure_min)))
                if int(residual_structure_max) > 0:
                    k = min(k, int(residual_structure_max))
            else:
                k = max(1, int(residual_structure_topk))
                # Confidence-adaptive contraction: when the best structure is
                # clearly separated from the runner-up on the current residual,
                # spend the expensive in-context refit budget on that one
                # mechanism only.  When scores are close, retain the configured
                # top-k safety net.
                if float(residual_structure_gap_rel) > 0.0 and k > 1 and len(ordered_structures) > 1:
                    best_score = max(float(ordered_structures[0][1]), 1e-12)
                    second_score = float(ordered_structures[1][1])
                    if (second_score - best_score) / best_score >= float(residual_structure_gap_rel):
                        k = 1
            selected_structures = {s for s, _ in ordered_structures[:k]}
            ranked = [z for z in ranked if tuple(int(v) for v in templates[int(z[1])]["structure"]) in selected_structures]
            if not ranked:
                break

        if verbose and int(debug_topk) > 0:
            if str(debug_log_style).lower() == "verbose":
                msg=[f"  [GSR add {step+1}] proposal top-{min(len(ranked),int(debug_topk))}"]
                for rank,(score,ti,scale,offset) in enumerate(ranked[:max(1,int(debug_topk))],1):
                    tpl=templates[ti]
                    msg.append(
                        f"    {rank:>2}. {_fmt_structure(tpl['structure'])} "
                        f"{_format_symbolic_combo(tpl['combo'])} "
                        f"screen={score:.6g} scale={scale:+.4g} bias={offset:+.4g}"
                    )
                _emit_debug("\n".join(msg), show_progress=show_progress)
            else:
                inline=[]
                for rank,(score,ti,_,_) in enumerate(ranked[:max(1,int(debug_topk))],1):
                    tpl=templates[ti]
                    inline.append(f"{rank}:{_fmt_structure(tpl['structure'])} {_format_symbolic_combo(tpl['combo'])} {score:.6g}")
                _emit_debug(f"  [GSR add {step+1}] screen | " + " | ".join(inline), show_progress=show_progress)
        best_trial=None; best_tpl=None; best_ti=None; best_val=float("inf")
        def _structure_diverse_take(items, width):
            width = max(1, int(width))
            if not bool(structure_diverse_beam):
                return list(items[:width])
            cap = max(1, int(beam_max_per_structure))
            out = []
            counts: Dict[Tuple[int, ...], int] = {}
            for item in items:
                _s = tuple(int(v) for v in templates[int(item[1])]["structure"])
                if counts.get(_s, 0) >= cap:
                    continue
                out.append(item); counts[_s] = counts.get(_s, 0) + 1
                if len(out) >= width:
                    break
            # If strict diversity produced fewer than width candidates, fill the
            # remaining slots in original rank order.  Thus this can never make
            # the beam smaller merely because there are few structures.
            if len(out) < width:
                used_ids = {int(z[1]) for z in out}
                for item in items:
                    if int(item[1]) in used_ids:
                        continue
                    out.append(item); used_ids.add(int(item[1]))
                    if len(out) >= width:
                        break
            return out

        if bool(hybrid_hard_screening):
            static_ranked = [z for z in ranked if not bool(templates[int(z[1])].get("hybrid_hard_residual", False))]
            hard_ranked_now = [z for z in ranked if bool(templates[int(z[1])].get("hybrid_hard_residual", False))]
            # Hard screening is additive: it never steals a slot from the
            # GMP/GSR beam.
            beam = _structure_diverse_take(static_ranked, beam_width) + hard_ranked_now[:max(0,int(hard_screen_global_candidates))]
        else:
            beam = _structure_diverse_take(ranked, beam_width)
        beam_bar = tqdm(
            beam, total=len(beam),
            desc=f"MP step {step+1}: in-context candidate refits",
            disable=not bool(show_progress), leave=False, dynamic_ncols=True,
            mininterval=0.25, smoothing=0.08,
        )
        gsr_debug=[]
        for _,ti,scale,_ in beam_bar:
            gsr_trial_count += 1
            tpl=templates[ti]; trial=copy.deepcopy(model)
            _install_symbolic_template(trial,target_rule,tpl["factors"],tpl["combo"],initial_scale=scale)
            trial,mse=_fully_symbolic_continuous_refit(
                trial,train_x,train_y,val_x,val_y,steps=trial_steps,lr=trial_lr,lbfgs_steps=0
            )
            gsr_debug.append((math.sqrt(max(float(mse),0.0)), int(ti)))
            if mse<best_val:
                best_val=mse; best_trial=trial; best_tpl=tpl; best_ti=ti
            if show_progress:
                beam_bar.set_postfix(
                    trial_rmse=f"{math.sqrt(max(mse,0.0)):.3g}",
                    best_rmse=f"{math.sqrt(max(best_val,0.0)):.3g}",
                    refresh=False,
                )
        if verbose and int(debug_topk) > 0 and gsr_debug:
            gsr_debug.sort(key=lambda z:z[0])
            if str(debug_log_style).lower() == "verbose":
                lines=[f"  [GSR add {step+1}] refit top-{min(len(gsr_debug),int(debug_topk))}"]
                for rank,(score,ti) in enumerate(gsr_debug[:max(1,int(debug_topk))],1):
                    tpl=templates[ti]
                    mark=" *" if best_ti is not None and int(ti)==int(best_ti) else ""
                    lines.append(
                        f"    {rank:>2}. {_fmt_structure(tpl['structure'])} "
                        f"{_format_symbolic_combo(tpl['combo'])} val={score:.6g}{mark}"
                    )
                _emit_debug("\n".join(lines), show_progress=show_progress)
            else:
                _emit_debug(
                    f"  [GSR add {step+1}] refit  | " + _format_ranked_inline(
                        gsr_debug, templates, topk=debug_topk, selected_ti=best_ti
                    ),
                    show_progress=show_progress,
                )
        if best_trial is None: break
        improves=best_val < current_mse*(1.0-1e-7)
        if not improves and step+1 > int(min_symbolic_rules):
            break
        # Keep a complete rollback point. Candidate acceptance is checked again
        # *after* the stronger commit refit and any overlap backfitting.
        previous_model = copy.deepcopy(model)
        previous_mse = float(current_mse)
        previous_used_templates = set(used_templates)
        previous_rule_template_idx = dict(rule_template_idx)
        model=best_trial; current_mse=best_val; used_templates.add(int(best_ti)); rule_template_idx[target_rule]=int(best_ti)
        # One stronger optimization is cheap after a commit and prevents matching
        # pursuit from adding garbage rules merely to compensate for poorly fitted
        # frequencies/scales in otherwise-correct symbolic atoms.
        if int(commit_refit_steps) > 0 or int(commit_lbfgs_steps) > 0:
            model, current_mse = _fully_symbolic_continuous_refit(
                model, train_x, train_y, val_x, val_y,
                steps=commit_refit_steps, lr=max(float(trial_lr)*2.0, 8e-4),
                lbfgs_steps=commit_lbfgs_steps, show_progress=False,
            )

        ops=[str(c["name"]) for c in best_tpl["combo"]]
        vars_=tuple(int(v) for v in best_tpl["structure"])

        # Restricted orthogonal-matching-pursuit/backfitting: when a newly added
        # rule overlaps variables with an earlier rule, revisit that earlier
        # operator choice.  Example: sin(control) selected before the self-product
        # arrives can be corrected to tanh(control) afterwards.
        if bool(backfit) and len(rule_template_idx) > 1:
            new_vars=set(vars_)
            for rr, old_ti in list(rule_template_idx.items()):
                if rr == target_rule: continue
                old_tpl=templates[int(old_ti)]
                old_vars=tuple(int(v) for v in old_tpl["structure"])
                if not (new_vars & set(old_vars)): continue
                same=[(ti,tpl) for ti,tpl in enumerate(templates)
                      if tuple(int(v) for v in tpl["structure"]) == old_vars and ti != old_ti]
                same.sort(key=lambda z: float(z[1].get("tuple_mse", float("inf"))))
                bf_best=model; bf_mse=current_mse; bf_ti=old_ti
                bf_debug=[(math.sqrt(max(float(current_mse),0.0)), int(old_ti))]
                for ti,tpl in same[:max(1,int(backfit_beam_width))]:
                    trial=copy.deepcopy(model)
                    _install_symbolic_template(trial,rr,tpl["factors"],tpl["combo"],initial_scale=float(trial.rule_scale[rr].detach().cpu()))
                    trial,mse=_fully_symbolic_continuous_refit(
                        trial,train_x,train_y,val_x,val_y,steps=backfit_steps,
                        lr=max(float(trial_lr)*2.0,8e-4),lbfgs_steps=backfit_lbfgs_steps,
                    )
                    bf_debug.append((math.sqrt(max(float(mse),0.0)), int(ti)))
                    if mse < bf_mse*(1.0-1e-6):
                        bf_best,bf_mse,bf_ti=trial,mse,ti
                bf_debug.sort(key=lambda z:z[0])
                if verbose and int(debug_topk) > 0:
                    oldops=_format_symbolic_combo(old_tpl["combo"])
                    if str(debug_log_style).lower() == "verbose":
                        lines=[f"  [BF r{rr} {_fmt_structure(old_vars)}] from {oldops}; top-{min(len(bf_debug),int(debug_topk))}"]
                        for rank,(score,ti) in enumerate(bf_debug[:max(1,int(debug_topk))],1):
                            tpl=templates[ti]
                            mark=" *" if int(ti)==int(bf_ti) else ""
                            prior=tpl.get("tuple_mse", float("nan"))
                            prior_txt=f" tupleMSE={float(prior):.3g}" if math.isfinite(float(prior)) else ""
                            lines.append(f"    {rank:>2}. {_format_symbolic_combo(tpl['combo'])} val={score:.6g}{prior_txt}{mark}")
                        _emit_debug("\n".join(lines), show_progress=show_progress)
                    else:
                        items=[]
                        for rank,(score,ti) in enumerate(bf_debug[:max(1,int(debug_topk))],1):
                            tpl=templates[ti]
                            mark="*" if int(ti)==int(bf_ti) else ""
                            items.append(f"{rank}:{_format_symbolic_combo(tpl['combo'])} {score:.6g}{mark}")
                        _emit_debug(
                            f"  [BF r{rr} {_fmt_structure(old_vars)}] {oldops} -> " + " | ".join(items),
                            show_progress=show_progress,
                        )
                if bf_ti != old_ti:
                    used_templates.discard(int(old_ti)); used_templates.add(int(bf_ti)); rule_template_idx[rr]=int(bf_ti)
                    model,current_mse=bf_best,bf_mse
                    if verbose and int(debug_topk) <= 0:
                        newops=[str(c["name"]) for c in templates[bf_ti]["combo"]]
                        msg=f"    backfit rule {rr} {'*'.join(f'x{v}' for v in old_vars)} -> {' * '.join(newops)}; val RMSE={math.sqrt(current_mse):.8g}"
                        _emit_debug(msg, show_progress=show_progress)

        # Optional lightweight orthogonalization for plain GSR: jointly refit
        # only the selected rule amplitudes and bias after every commit.  This
        # cannot change operator identities or affine factor shapes; it simply
        # prevents coefficient error from leaking back into the residual and
        # causing duplicate/compensating structures to be selected later.
        if bool(joint_scale_refit_each_commit) and pursuit_mode == "gsr":
            _joint_model, _joint_mse = orthogonal_rule_scale_refit(
                model, train_x, train_y, val_x, val_y, ridge=float(omp_ridge)
            )
            if float(_joint_mse) <= float(current_mse):
                model, current_mse = _joint_model, float(_joint_mse)

        # Optional orthogonal-pursuit variants.  GSR remains the reference
        # implementation.  OMP variants jointly re-estimate all selected rule
        # amplitudes/bias so the residual is orthogonal to the selected rule span;
        # nonlinear/full variants spend progressively more compute afterwards.
        if pursuit_mode in {"omp_linear", "omp_nonlinear", "omp_full"}:
            model, current_mse = orthogonal_rule_scale_refit(
                model, train_x, train_y, val_x, val_y, ridge=float(omp_ridge)
            )
            if pursuit_mode in {"omp_nonlinear", "omp_full"} and int(omp_extra_steps) > 0:
                model, current_mse = _fully_symbolic_continuous_refit(
                    model, train_x, train_y, val_x, val_y,
                    steps=int(omp_extra_steps), lr=max(float(trial_lr), 3e-4), lbfgs_steps=0,
                )
                model, current_mse = orthogonal_rule_scale_refit(
                    model, train_x, train_y, val_x, val_y, ridge=float(omp_ridge)
                )
            if pursuit_mode == "omp_full" and bool(omp_redundancy_each_commit):
                model, _omp_red = redundancy_aware_symbolic_prune(
                    model, train_x, train_y, val_x, val_y,
                    span_r2_threshold=float(redundancy_span_r2_threshold),
                    corr_threshold=float(redundancy_corr_threshold),
                    relative_mse_budget=float(redundancy_rel_mse_budget),
                    refit_steps=max(20, int(omp_extra_steps)//2),
                    refit_lr=max(float(trial_lr), 3e-4),
                    min_rules=max(1, int(min_symbolic_rules)), verbose=False,
                )
                with torch.no_grad():
                    current_mse = float(torch.mean((model(val_x)-val_y)**2).cpu())

        # Do not keep residual-cleanup rules whose gain disappears once the
        # committed model has been properly re-optimized/backfit.  This catches
        # cases where a trial looks microscopically better but finishes with the
        # same validation loss and a near-zero rule amplitude.
        rel_gain = (previous_mse - float(current_mse)) / max(previous_mse, 1e-18)
        if step + 1 > int(min_symbolic_rules) and rel_gain < float(min_rule_improvement_rel):
            if verbose:
                msg = (
                    f"  reject add {step+1}: post-refit relative MSE gain={rel_gain:.3g} "
                    f"< {float(min_rule_improvement_rel):.3g}; stop pursuit"
                )
                tqdm.write(msg) if show_progress else print(msg)
            model = previous_model
            current_mse = previous_mse
            used_templates = previous_used_templates
            rule_template_idx = previous_rule_template_idx
            break

        # Refresh this rule's displayed operators after any backfit activity.
        best_ti = rule_template_idx[target_rule]
        best_tpl = templates[best_ti]
        ops=[str(c["name"]) for c in best_tpl["combo"]]
        vars_=tuple(int(v) for v in best_tpl["structure"])
        history.append({
            "rule":target_rule,"structure":vars_,"operators":ops,
            "validation_rmse":math.sqrt(current_mse),
        })
        struct_text="*".join(f"x{v}" for v in vars_)
        detail = (
            f"  [commit {step+1}] r{target_rule} {struct_text} "
            f"{' * '.join(ops)} | val={math.sqrt(current_mse):.8g}"
        )
        if verbose:
            tqdm.write(detail) if show_progress else print(detail)
        if show_progress:
            mp_bar.set_postfix(
                active=int(model.hard_rule_choice.sum()), val=f"{math.sqrt(current_mse):.3g}",
                last=f"{struct_text}:{'*'.join(ops)}", refresh=True,
            )
        if math.sqrt(current_mse) <= float(target_val_rmse) and step+1 >= int(min_symbolic_rules):
            break

    gsr_seconds = time.perf_counter() - _gsr_t0
    _final_t0 = time.perf_counter()
    model,final_mse=_fully_symbolic_continuous_refit(
        model,train_x,train_y,val_x,val_y,steps=final_steps,lr=final_lr,
        lbfgs_steps=final_lbfgs_steps, show_progress=show_progress,
        progress_desc="Final fully-symbolic polish",
    )

    # Residual-conditioned operator rescue.  The initial GMP shortlist is only a
    # proposal distribution for the *initial* residual.  Once other mechanisms
    # have been committed, the best operator for an existing structure may
    # change.  Recompute a leave-one-rule-out residual, screen the full operator
    # library cheaply against that residual, and spend full-model refits only on
    # a small rescue beam.  Unary rules screen the complete library directly;
    # product rules use a fresh residual-conditioned GMP shortlist first.
    cleanup_start = time.perf_counter()
    sweep_start = cleanup_start
    cleanup_trials = 0
    def cleanup_exhausted() -> bool:
        by_time = float(cleanup_max_seconds) > 0 and (time.perf_counter() - cleanup_start) >= float(cleanup_max_seconds)
        by_trials = int(cleanup_max_trials) > 0 and cleanup_trials >= int(cleanup_max_trials)
        return bool(by_time or by_trials)

    if bool(residual_operator_rescue):
        for sweep in range(2):
            if cleanup_exhausted():
                break
            changed = False
            for rr, old_ti in list(rule_template_idx.items()):
                if cleanup_exhausted():
                    break
                old_tpl = templates[int(old_ti)]
                struct = tuple(int(v) for v in old_tpl["structure"])
                factors = tuple((slot, var) for slot, var in enumerate(struct))

                model.eval()
                with torch.no_grad():
                    pred_tr, det_tr = model(train_x, return_details=True)
                    pred_va, det_va = model(val_x, return_details=True)
                    residual_train = train_y - (pred_tr - det_tr["contributions"][:, rr:rr+1])
                    residual_val = val_y - (pred_va - det_va["contributions"][:, rr:rr+1])

                candidate_combos = []
                if len(struct) == 1:
                    # A unary rescue is cheap enough to screen the complete
                    # analytic library.  This is what allows tanh to re-enter
                    # after an early GMP pass preferred sin/arctan while the
                    # same-variable product was still unexplained.
                    for name in lib:
                        b0 = 2.5 if name in {"sin", "cos"} else (-0.5 if name == "exp" else 1.0)
                        candidate_combos.append(({
                            "name": name, "affine": (1.0, b0, 0.0, 0.0),
                            "gmp_prob": 1.0 / max(1, len(lib)),
                        },))
                else:
                    rescue, _ = gmp_symbolic_operator_preselection(
                        train_x, residual_train, [struct], lib,
                        topk=max(1, int(residual_rescue_gmp_topk)),
                        unary_topk=max(1, int(residual_rescue_gmp_topk)),
                        self_product_topk=max(1, int(residual_rescue_gmp_topk)),
                        steps=max(1, int(residual_rescue_gmp_steps)),
                        lr=gmp_lr, temperature_start=gmp_temperature_start,
                        temperature_end=gmp_temperature_end,
                        entropy_weight=gmp_entropy_weight,
                        relaxation_mode=gmp_relaxation_mode,
                        atom_backward_normalization=gmp_atom_backward_normalization,
                        gumbel_noise_scale=gmp_gumbel_noise_scale, complexity_weight=gmp_complexity_weight,
                        complexity_logit_prior=gmp_complexity_logit_prior,
                        nonlinearity_logit_prior=gmp_nonlinearity_logit_prior,
                        curvature_weight=gmp_curvature_weight, nonlinearity_weight=gmp_nonlinearity_weight,
                        identity_chart=gmp_identity_chart, debug_topk=0,
                        debug_log_style=debug_log_style, debug_label=f"rescue r{rr}",
                        generator=symbolic_rng,
                        show_progress=False,
                    )
                    if rescue:
                        candidate_combos = list(itertools.product(*rescue[0]["factor_shortlists"]))
                    if bool(hard_screen_residual_rescue):
                        candidate_combos.extend(hard_symbolic_tuple_screening(
                            train_x, residual_train, struct, lib,
                            beam_width=int(hard_screen_beam_width),
                            top_tuples=max(int(hard_screen_top_tuples), int(residual_rescue_beam_width) * 2),
                            max_samples=int(hard_screen_max_samples),
                            seeds_per_operator_prefix=3,
                        ))

                scored = []
                for combo in candidate_combos:
                    if cleanup_exhausted():
                        break
                    try:
                        pc, pmse = _polish_symbolic_operator_tuple(
                            train_x, residual_train, struct, combo,
                            steps=max(20, int(gmp_tuple_refine_steps)),
                            lr=gmp_tuple_refine_lr,
                        )
                    except Exception:
                        continue
                    if math.isfinite(float(pmse)):
                        scored.append((float(pmse), tuple(pc)))
                scored.sort(key=lambda z: z[0])

                old_names = tuple(str(c["name"]) for c in old_tpl["combo"])
                # Always keep the current operator tuple in the comparison set,
                # but do not waste a full-model refit on an identical proposal.
                rescue_beam = []
                for pmse, combo in scored:
                    names = tuple(str(c["name"]) for c in combo)
                    if names == old_names:
                        continue
                    rescue_beam.append((pmse, combo))
                    if len(rescue_beam) >= max(1, int(residual_rescue_beam_width)):
                        break

                best_model, best_mse, best_combo = model, float(final_mse), None
                for _, combo in rescue_beam:
                    if cleanup_exhausted():
                        break
                    cleanup_trials += 1
                    trial = copy.deepcopy(model)
                    _install_symbolic_template(
                        trial, rr, factors, combo,
                        initial_scale=float(trial.rule_scale[rr].detach().cpu()),
                    )
                    trial, mse = _fully_symbolic_continuous_refit(
                        trial, train_x, train_y, val_x, val_y,
                        steps=max(40, int(residual_rescue_steps)),
                        lr=max(float(trial_lr) * 2.0, 8e-4),
                        lbfgs_steps=max(10, int(residual_rescue_lbfgs_steps)),
                        show_progress=False,
                    )
                    if mse < best_mse * (1.0 - 1e-6):
                        best_model, best_mse, best_combo = trial, mse, combo

                if best_combo is not None:
                    new_tpl = {
                        "structure": struct, "factors": factors,
                        "combo": tuple(best_combo), "tuple_mse": float("nan"),
                        "residual_rescue": True,
                    }
                    templates.append(new_tpl)
                    new_ti = len(templates) - 1
                    if verbose:
                        newops = [str(c["name"]) for c in best_combo]
                        msg = (f"  residual operator rescue rule {rr}: {' * '.join(old_names)} -> "
                               f"{' * '.join(newops)}; val RMSE={math.sqrt(best_mse):.8g}")
                        tqdm.write(msg) if show_progress else print(msg)
                    used_templates.discard(int(old_ti)); used_templates.add(int(new_ti))
                    rule_template_idx[rr] = int(new_ti)
                    model, final_mse = best_model, best_mse
                    changed = True
            if not changed:
                break
    operator_sweep_seconds = time.perf_counter() - sweep_start


    if bool(redundancy_cleanup):
        model, redundancy_history = redundancy_aware_symbolic_prune(
            model, train_x, train_y, val_x, val_y,
            span_r2_threshold=float(redundancy_span_r2_threshold),
            corr_threshold=float(redundancy_corr_threshold),
            relative_mse_budget=float(redundancy_rel_mse_budget),
            refit_steps=max(40, int(residual_rescue_steps)),
            refit_lr=max(float(final_lr), 2e-4),
            min_rules=max(1, int(cleanup_min_rules)), verbose=bool(verbose),
        )
        with torch.no_grad():
            final_mse=float(torch.mean((model(val_x)-val_y)**2).cpu())
    else:
        redundancy_history = []

    # First remove numerically negligible symbolic rules.  A small rule-scale by
    # itself is not sufficient because the associated analytic factor could be
    # large, so contribution RMS is also measured and every deletion is verified
    # after a survivor refit.  Destructive deletions are rolled back.
    tiny_eliminated_rules = []
    while not cleanup_exhausted():
        active = torch.nonzero(model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        if len(active) <= int(cleanup_min_rules):
            break
        model.eval()
        with torch.no_grad():
            _, det = model(val_x, return_details=True)
            contrib = det["contributions"]
            crms = torch.sqrt(torch.mean(contrib * contrib, dim=0) + 1e-18)
        candidates = []
        for rr in active:
            scale_abs = abs(float(model.rule_scale[rr].detach().cpu()))
            contrib_val = float(crms[rr].detach().cpu())
            if scale_abs <= float(tiny_rule_scale_threshold) or contrib_val <= float(tiny_rule_contribution_rmse):
                candidates.append((contrib_val, scale_abs, int(rr)))
        candidates.sort()
        if not candidates:
            break
        removed = False
        for contrib_val, scale_abs, rr in candidates:
            if cleanup_exhausted():
                break
            cleanup_trials += 1
            trial = copy.deepcopy(model)
            with torch.no_grad():
                trial.hard_rule_choice[rr] = False
                trial.rule_scale[rr] = 0.0
            trial, mse = _fully_symbolic_continuous_refit(
                trial, train_x, train_y, val_x, val_y,
                steps=max(80, int(commit_refit_steps)//3),
                lr=max(float(trial_lr)*2.0, 8e-4),
                lbfgs_steps=max(20, int(commit_lbfgs_steps)//3),
                show_progress=False,
            )
            rel_increase = (float(mse) - float(final_mse)) / max(float(final_mse), 1e-18)
            below_absolute_target = (
                float(target_val_rmse) > 0.0
                and math.sqrt(max(0.0, float(mse))) <= float(target_val_rmse)
            )
            if rel_increase <= float(tiny_rule_rel_mse_budget) or below_absolute_target:
                if verbose:
                    msg=(
                        f"  remove tiny symbolic rule {rr}: |scale|={scale_abs:.3g}, "
                        f"contribution RMSE={contrib_val:.3g}, relative MSE change={rel_increase:.3g}; "
                        f"val RMSE={math.sqrt(mse):.8g}"
                    )
                    tqdm.write(msg) if show_progress else print(msg)
                model, final_mse = trial, mse
                tiny_eliminated_rules.append(int(rr))
                old_ti = rule_template_idx.pop(rr, None)
                if old_ti is not None:
                    used_templates.discard(int(old_ti))
                removed = True
                break
            elif verbose and scale_abs <= float(tiny_rule_scale_threshold):
                msg=(
                    f"  keep tiny-scale rule {rr}: |scale|={scale_abs:.3g} but deletion is destructive "
                    f"(relative MSE change={rel_increase:.3g})"
                )
                tqdm.write(msg) if show_progress else print(msg)
        if not removed:
            break

    # Post-convergence backward elimination.  A rule may look useful when added
    # but have its amplitude driven nearly to zero by the final joint fit.  Test
    # actual leave-one-rule-out deletion after final convergence, refit survivors,
    # and remove rules whose deletion is loss-neutral.
    eliminate_start = time.perf_counter()
    eliminated_rules = []
    while not cleanup_exhausted():
        active = torch.nonzero(model.hard_rule_choice, as_tuple=False).squeeze(-1).tolist()
        if len(active) <= int(cleanup_min_rules):
            break
        with torch.no_grad():
            _, det = model(val_x, return_details=True)
            contrib = det["contributions"]
            rms = torch.sqrt(torch.mean(contrib * contrib, dim=0) + 1e-18)
        order = sorted(active, key=lambda r: float(rms[r].detach().cpu()))
        removed = False
        for rr in order:
            if cleanup_exhausted():
                break
            cleanup_trials += 1
            trial = copy.deepcopy(model)
            with torch.no_grad():
                trial.hard_rule_choice[rr] = False
                trial.rule_scale[rr] = 0.0
            trial, mse = _fully_symbolic_continuous_refit(
                trial, train_x, train_y, val_x, val_y,
                steps=max(120, int(commit_refit_steps)//2),
                lr=max(float(trial_lr)*2.0, 8e-4),
                lbfgs_steps=max(40, int(commit_lbfgs_steps)//2),
                show_progress=False,
            )
            rel_increase = (float(mse) - float(final_mse)) / max(float(final_mse), 1e-18)
            # Backward elimination is a complexity decision made after final
            # convergence, so it has its own budget.  A tiny residual rule can
            # survive the forward acceptance threshold yet become negligible
            # after joint refitting; allow a small validation-MSE increase to
            # remove such cleanup terms.
            below_absolute_target = (
                float(target_val_rmse) > 0.0
                and math.sqrt(max(0.0, float(mse))) <= float(target_val_rmse)
            )
            if rel_increase <= float(elimination_rel_mse_budget) or below_absolute_target:
                if verbose:
                    msg = (f"  eliminate symbolic rule {rr}: contribution RMS="
                           f"{float(rms[rr]):.3g}, relative MSE change={rel_increase:.3g}; "
                           f"val RMSE={math.sqrt(mse):.8g}")
                    tqdm.write(msg) if show_progress else print(msg)
                model, final_mse = trial, mse
                eliminated_rules.append(int(rr))
                old_ti = rule_template_idx.pop(rr, None)
                if old_ti is not None:
                    used_templates.discard(int(old_ti))
                removed = True
                break
        if not removed:
            break
    elimination_seconds = time.perf_counter() - eliminate_start

    # Always perform one cheap last-chance cleanup independent of the expensive
    # rescue budget. This is what removes visually spurious terms such as
    # ``3e-5 * (...)`` when their actual validation contribution is also tiny.
    model, final_mse, last_chance_removed = drop_negligible_symbolic_rules(
        model, val_x, val_y, current_mse=final_mse,
        scale_threshold=tiny_rule_scale_threshold,
        contribution_rmse_threshold=tiny_rule_contribution_rmse,
        relative_mse_budget=tiny_rule_rel_mse_budget,
        absolute_rmse_budget=target_val_rmse,
        min_rules=cleanup_min_rules, verbose=verbose, show_progress=show_progress,
    )
    for rr in last_chance_removed:
        old_ti = rule_template_idx.pop(rr, None)
        if old_ti is not None:
            used_templates.discard(int(old_ti))

    # One short final refit after structural cleanup.
    model, final_mse = _fully_symbolic_continuous_refit(
        model, train_x, train_y, val_x, val_y,
        steps=max(120, int(commit_refit_steps)//2), lr=final_lr,
        lbfgs_steps=max(40, int(commit_lbfgs_steps)//2), show_progress=False,
    )

    # Affine-partition rescue is the last symbolic alternative considered.
    # Ordinary RuleKAN/GSR gets the full opportunity to solve the task first.
    # The rescue cannot leave the supplied learned-support contract; selection
    # is based on held-out validation fit and the validation-equivalence
    # cancellation/description preference defined in the rescue.
    partition_meta: Dict[str, object] = {"attempted": False, "selected": False}
    partition_seconds = 0.0
    _partition_refactor_opportunity = False
    if bool(affine_partition_rescue):
        _partition_refactor_opportunity = _has_affine_partition_refactor_opportunity(
            model, structures, allowed_supports=affine_partition_allowed_supports,
        )
    _partition_triggered = (
        bool(affine_partition_rescue)
        and (
            float(target_val_rmse) <= 0.0
            or math.sqrt(max(0.0, float(final_mse))) > float(target_val_rmse)
            or bool(_partition_refactor_opportunity)
        )
    )
    if _partition_triggered:
        _partition_t0 = time.perf_counter()
        model2, partition_meta = affine_partition_symbolic_rescue(
            numeric_model, model, train_x, train_y, val_x, val_y,
            structure_candidates=structures,
            allowed_supports=affine_partition_allowed_supports,
            library=lib,
            family_beam=max(1, int(affine_partition_family_beam)),
            seed_topk=max(0, int(affine_partition_seed_topk)),
            max_samples=max(32, int(affine_partition_max_samples)),
            max_support_pairs=max(1, int(affine_partition_max_support_pairs)),
            refine_steps=max(0, int(affine_partition_refine_steps)),
            refine_lr=float(affine_partition_refine_lr),
            lbfgs_steps=max(0, int(affine_partition_lbfgs_steps)),
            final_polish_topk=max(0, int(affine_partition_final_polish_topk)),
            final_polish_steps=max(0, int(affine_partition_final_polish_steps)),
            final_polish_lbfgs_steps=max(0, int(affine_partition_final_polish_lbfgs_steps)),
            equivalence_rel_mse=max(0.0, float(affine_partition_equivalence_rel_mse)),
            equivalence_nrmse=max(0.0, float(affine_partition_equivalence_nrmse)),
            cancellation_weight=max(0.0, float(affine_partition_cancellation_weight)),
            complexity_weight=max(0.0, float(affine_partition_complexity_weight)),
            min_improvement_rel=max(0.0, float(affine_partition_min_improvement_rel)),
            verbose=bool(verbose),
        )
        partition_seconds = time.perf_counter() - _partition_t0
        if bool(partition_meta.get("selected", False)):
            model = model2
            final_mse = float(partition_meta["best_validation_mse"])
        # Keep one compact diagnostic record whether or not the rescue wins.
        # This is useful for benchmark audits and does not affect selection.
        history.append({
            "affine_partition_rescue": bool(partition_meta.get("selected", False)),
            "affine_partition_rescue_attempted": bool(partition_meta.get("attempted", False)),
            "affine_partition_rescue_reason": str(partition_meta.get("reason", "unknown")),
            "gate_variable": partition_meta.get("gate_variable"),
            "support_a": partition_meta.get("support_a"),
            "support_b": partition_meta.get("support_b"),
            "operators": ([str(partition_meta.get("operator_a")), str(partition_meta.get("operator_b"))]
                          if partition_meta.get("operator_a") is not None else None),
            "incumbent_validation_mse": partition_meta.get("incumbent_validation_mse"),
            "predictive_best_validation_mse": partition_meta.get("predictive_best_validation_mse"),
            "best_validation_mse": partition_meta.get("best_validation_mse"),
            "validation_equivalence_cap": partition_meta.get("validation_equivalence_cap"),
            "incumbent_cancellation_score": partition_meta.get("incumbent_cancellation_score"),
            "cancellation_score": partition_meta.get("cancellation_score"),
            "preference_score": partition_meta.get("preference_score"),
            "incumbent_preference_score": partition_meta.get("incumbent_preference_score"),
            "best_partition_preference_validation_mse": partition_meta.get("best_partition_preference_validation_mse"),
            "best_partition_preference_cancellation_score": partition_meta.get("best_partition_preference_cancellation_score"),
            "best_partition_preference_description_complexity": partition_meta.get("best_partition_preference_description_complexity"),
            "best_partition_preference_preference_score": partition_meta.get("best_partition_preference_preference_score"),
            "best_partition_preference_operator_a": partition_meta.get("best_partition_preference_operator_a"),
            "best_partition_preference_operator_b": partition_meta.get("best_partition_preference_operator_b"),
            "best_partition_predictive_validation_mse": partition_meta.get("best_partition_predictive_validation_mse"),
            "best_partition_predictive_preference_score": partition_meta.get("best_partition_predictive_preference_score"),
            "best_partition_predictive_operator_a": partition_meta.get("best_partition_predictive_operator_a"),
            "best_partition_predictive_operator_b": partition_meta.get("best_partition_predictive_operator_b"),
            "partition_seconds": float(partition_seconds),
        })
    final_seconds = time.perf_counter() - _final_t0
    active_rules=torch.nonzero(model.hard_rule_choice,as_tuple=False).squeeze(-1).tolist()
    spline_violations=[]
    for r in active_rules:
        for ss in range(model.max_factors):
            j=int(model.hard_variable_choice[r,ss].item())
            if j!=model.in_dim and bool(model.hard_spline_choice[r,ss,j].item()):
                spline_violations.append((r,ss,j))
    if spline_violations:
        raise RuntimeError(f"fully symbolic invariant violated: spline factors remain {spline_violations}")
    if verbose:
        print(f"  [result] val={math.sqrt(final_mse):.8g} | active rules={len(active_rules)}")
        total_seconds = time.perf_counter() - pursuit_t0
        print(
            "  [timing] "
            f"GMP={gmp_seconds:.2f}s, tuple-build/polish={template_seconds:.2f}s, "
            f"GSR={gsr_seconds:.2f}s ({gsr_trial_count} full refits), "
            f"residual-rescue={operator_sweep_seconds:.2f}s, "
            f"backward-elimination={elimination_seconds:.2f}s, "
            f"affine-partition={partition_seconds:.2f}s"
            f"{' selected' if bool(partition_meta.get('selected', False)) else ''}, "
            f"cleanup-trials={cleanup_trials}, cleanup-budget-exhausted={cleanup_exhausted()}, "
            f"final-polish/cleanup={final_seconds:.2f}s, total={total_seconds:.2f}s"
        )
    return model,history
