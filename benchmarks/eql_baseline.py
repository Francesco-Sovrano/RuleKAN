from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch import nn

from symbolic_kan.utils import SYMBOLIC_LIB


DEFAULT_EQL_UNARY_LIBRARY = (
    "x", "x^2", "1/x", "sqrt", "log", "exp", "sin", "cos", "tanh",
)


class EQLLayer(nn.Module):
    """One EQL hidden layer: affine map -> unary units + pairwise products.

    ``units_per_type`` is the paper's width parameter: the network contains this
    many copies of each unary type and this many multiplication units.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        unary_library: Sequence[str] = DEFAULT_EQL_UNARY_LIBRARY,
        units_per_type: int = 10,
        n_binary: int | None = None,
    ) -> None:
        super().__init__()
        lib = tuple(str(x) for x in unary_library)
        if not lib:
            raise ValueError("EQL unary_library must be non-empty")
        missing = [x for x in lib if x not in SYMBOLIC_LIB]
        if missing:
            raise ValueError(f"unsupported EQL unary primitives: {missing}")
        units_per_type = max(1, int(units_per_type))
        n_binary = units_per_type if n_binary is None else max(1, int(n_binary))
        self.unary_names = tuple(name for name in lib for _ in range(units_per_type))
        self.n_unary = len(self.unary_names)
        self.n_binary = int(n_binary)
        self.out_dim = self.n_unary + self.n_binary
        # Each multiplication unit consumes two affine preactivations.
        self.linear = nn.Linear(int(in_dim), self.n_unary + 2 * self.n_binary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        vals = []
        for i, name in enumerate(self.unary_names):
            vals.append(SYMBOLIC_LIB[name][0](z[:, i : i + 1]))
        off = self.n_unary
        for j in range(self.n_binary):
            a = z[:, off + 2 * j : off + 2 * j + 1]
            b = z[:, off + 2 * j + 1 : off + 2 * j + 2]
            vals.append(a * b)
        return torch.cat(vals, dim=1)


class EQLDivRegressor(nn.Module):
    """PyTorch reproduction of the ICML-2018 EQL-Div architecture.

    Hidden layers use identity/sine/cosine and multiplication units.  The final
    scalar output is a regularized division h_theta(a,b): a/b for b>theta and
    zero otherwise, matching the EQL-Div construction for scalar regression.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        n_hidden_layers: int = 2,
        unary_library: Sequence[str] = DEFAULT_EQL_UNARY_LIBRARY,
        units_per_type: int = 10,
        n_binary: int | None = None,
        eval_division_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        if int(n_hidden_layers) < 1:
            raise ValueError("EQL-Div requires at least one hidden layer")
        layers = []
        dim = int(input_dim)
        for _ in range(int(n_hidden_layers)):
            layer = EQLLayer(
                dim,
                unary_library=unary_library,
                units_per_type=int(units_per_type),
                n_binary=n_binary,
            )
            layers.append(layer)
            dim = layer.out_dim
        self.layers = nn.ModuleList(layers)
        self.output_linear = nn.Linear(dim, 2)  # numerator, denominator
        self.eval_division_threshold = float(eval_division_threshold)

    def division_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.layers:
            h = layer(h)
        z = self.output_linear(h)
        return z[:, 0:1], z[:, 1:2]

    def forward(
        self,
        x: torch.Tensor,
        *,
        division_threshold: float | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num, den = self.division_parts(x)
        theta = self.eval_division_threshold if division_threshold is None else float(division_threshold)
        out = torch.where(den > theta, num / den.clamp_min(theta), torch.zeros_like(num))
        if return_parts:
            return out, num, den
        return out


# Backward-compatible internal name used by the benchmark adapter.
EQLRegressor = EQLDivRegressor


@dataclass
class EQLFitResult:
    model: EQLDivRegressor
    steps_run: int
    val_mse: float
    nonzero_weights: int
    active_units: int
    l1_lambda: float
    total_layers: int


def _weight_l1(model: nn.Module) -> torch.Tensor:
    dev = next(model.parameters()).device
    total = torch.zeros((), device=dev)
    for name, p in model.named_parameters():
        if name.endswith("weight"):
            total = total + p.abs().sum()
    return total


def _weight_masks(model: nn.Module, threshold: float) -> Dict[str, torch.Tensor]:
    return {
        name: (p.detach().abs() >= float(threshold)).to(dtype=p.dtype)
        for name, p in model.named_parameters()
        if name.endswith("weight")
    }


def _apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name])


def _active_unit_count(model: EQLDivRegressor, threshold: float = 0.0) -> int:
    """Approximate the paper's connected-unit sparsity count."""
    count = 0
    thr = float(threshold)
    with torch.no_grad():
        for layer in model.layers:
            w = layer.linear.weight.detach().abs()
            # Connected-unit sparsity is defined from incoming weights; biases
            # do not by themselves constitute a connection.
            for i in range(layer.n_unary):
                if bool((w[i] > thr).any()):
                    count += 1
            off = layer.n_unary
            for j in range(layer.n_binary):
                rows = slice(off + 2 * j, off + 2 * j + 2)
                if bool((w[rows] > thr).any()):
                    count += 1
        ow = model.output_linear.weight.detach().abs()
        count += int(bool((ow[0] > thr).any()))
        count += int(bool((ow[1] > thr).any()))
    return int(count)


def _nonzero_weight_count(model: nn.Module) -> int:
    with torch.no_grad():
        return int(sum(torch.count_nonzero(p).item() for n, p in model.named_parameters() if n.endswith("weight")))


def _sample_uniform_box(
    lo: torch.Tensor,
    hi: torch.Tensor,
    n: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    # CPU generator is deterministic across benchmark runs; transfer afterwards.
    r = torch.rand((int(n), int(lo.numel())), generator=generator, dtype=torch.float32)
    out = lo.detach().cpu().float() + r * (hi.detach().cpu().float() - lo.detach().cpu().float())
    return out.to(device=lo.device, dtype=lo.dtype)


def fit_eql_candidate(
    model: EQLDivRegressor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    steps: int,
    lr: float = 1e-3,
    adam_eps: float = 1e-4,
    l1_lambda: float = 1e-5,
    phase1_frac: float = 0.25,
    phase2_frac: float = 0.95,
    prune_threshold: float = 1e-3,
    batch_size: int = 20,
    penalty_every: int = 50,
    penalty_batch_size: int | None = None,
    output_bound: float | None = None,
    seed: int = 0,
) -> EQLFitResult:
    """Train one EQL-Div candidate with the paper's three regularization phases.

    The phase boundaries are t1=T/4 and t2=19T/20.  Training uses Adam with
    mini-batches.  The division threshold follows theta(t)=1/sqrt(t+1), and
    small weights are frozen to zero in the final fixed-support phase.
    """
    steps = max(1, int(steps))
    t1 = max(1, min(steps, int(round(float(phase1_frac) * steps))))
    t2 = max(t1, min(steps, int(round(float(phase2_frac) * steps))))
    batch_size = max(1, min(int(batch_size), int(train_x.shape[0])))
    penalty_batch_size = batch_size if penalty_batch_size is None else max(1, int(penalty_batch_size))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), eps=float(adam_eps))
    masks: Dict[str, torch.Tensor] | None = None

    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(int(seed))
    lo = torch.minimum(train_x.min(dim=0).values, val_x.min(dim=0).values)
    hi = torch.maximum(train_x.max(dim=0).values, val_x.max(dim=0).values)
    if output_bound is None:
        # Avoid imposing an artificial scale on benchmark targets while still
        # retaining the paper's out-of-domain blow-up penalty.
        output_bound = max(10.0, 3.0 * float(train_y.detach().abs().max().cpu()))

    n = int(train_x.shape[0])
    for step in range(steps):
        if step == t2 and masks is None:
            masks = _weight_masks(model, float(prune_threshold))
            _apply_masks(model, masks)

        theta = 1.0 / math.sqrt(float(step) + 1.0)
        opt.zero_grad(set_to_none=True)

        is_penalty_step = int(penalty_every) > 0 and (step + 1) % int(penalty_every) == 0
        if is_penalty_step:
            px = _sample_uniform_box(lo, hi, penalty_batch_size, generator=cpu_gen)
            pout, _, pden = model(px, division_threshold=theta, return_parts=True)
            denom_penalty = torch.relu(torch.as_tensor(theta, device=pden.device, dtype=pden.dtype) - pden).mean()
            bound = torch.as_tensor(float(output_bound), device=pout.device, dtype=pout.dtype)
            bound_penalty = (torch.relu(pout - bound) + torch.relu(-pout - bound)).mean()
            loss = denom_penalty + bound_penalty
        else:
            idx = torch.randint(0, n, (batch_size,), generator=cpu_gen)
            bx = train_x[idx.to(train_x.device)]
            by = train_y[idx.to(train_y.device)]
            pred, _, den = model(bx, division_threshold=theta, return_parts=True)
            loss = torch.mean((pred.reshape_as(by) - by) ** 2)
            # Eq. (8): denominator penalty is present during ordinary training.
            loss = loss + torch.relu(torch.as_tensor(theta, device=den.device, dtype=den.dtype) - den).mean()
            if t1 <= step < t2:
                loss = loss + float(l1_lambda) * _weight_l1(model)

        if not torch.isfinite(loss):
            break
        loss.backward()
        if masks is not None:
            for name, p in model.named_parameters():
                if p.grad is not None and name in masks:
                    p.grad.mul_(masks[name])
        opt.step()
        if masks is not None:
            _apply_masks(model, masks)

    with torch.no_grad():
        vpred = model(val_x, division_threshold=model.eval_division_threshold)
        val_mse = float(torch.mean((vpred.reshape_as(val_y) - val_y) ** 2).detach().cpu())
    return EQLFitResult(
        model=model,
        steps_run=steps,
        val_mse=val_mse,
        nonzero_weights=_nonzero_weight_count(model),
        active_units=_active_unit_count(model, threshold=float(prune_threshold)),
        l1_lambda=float(l1_lambda),
        total_layers=int(len(model.layers) + 1),
    )


def _normalize01(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [math.inf for _ in values]
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return [0.0 if math.isfinite(v) else math.inf for v in values]
    return [(v - lo) / (hi - lo) if math.isfinite(v) else math.inf for v in values]


def fit_eql_model_selection(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    input_dim: int,
    unary_library: Sequence[str] = DEFAULT_EQL_UNARY_LIBRARY,
    total_layers: Sequence[int] = (2, 3, 4),
    l1_lambdas: Sequence[float] = (1e-6, 1e-5, 1e-4, 3.162277660168379e-4),
    units_per_type: int = 10,
    n_binary: int | None = None,
    steps_per_hidden_layer: int = 10000,
    lr: float = 1e-3,
    adam_eps: float = 1e-4,
    phase1_frac: float = 0.25,
    phase2_frac: float = 0.95,
    prune_threshold: float = 1e-3,
    batch_size: int = 20,
    penalty_every: int = 50,
    eval_division_threshold: float = 1e-4,
    selection_error_weight: float = 0.5,
    selection_sparsity_weight: float = 0.5,
    seed: int = 0,
) -> tuple[EQLFitResult, list[dict[str, float]]]:
    """Fit depth/regularization candidates and select by validation+sparsity.

    This follows the ICML-2018 model-selection criterion using normalized
    interpolation validation error and sparsity with alpha=beta=0.5.  For the
    benchmark we use a coarse four-point subset spanning the paper's lambda
    range by default, rather than its 26-value 0.1-decade sweep, to stay inside
    the common per-job compute budget.
    """
    candidates: list[EQLFitResult] = []
    rows: list[dict[str, float]] = []
    candidate_index = 0
    for L in [int(x) for x in total_layers]:
        if L < 2:
            raise ValueError("EQL-Div total layer count L must be >=2")
        hidden = L - 1
        steps = max(1, int(steps_per_hidden_layer) * hidden)
        for lam in [float(x) for x in l1_lambdas]:
            torch.manual_seed(int(seed) + 1009 * candidate_index + 97 * L)
            model = EQLDivRegressor(
                int(input_dim),
                n_hidden_layers=hidden,
                unary_library=unary_library,
                units_per_type=int(units_per_type),
                n_binary=n_binary,
                eval_division_threshold=float(eval_division_threshold),
            ).to(device=train_x.device, dtype=train_x.dtype)
            result = fit_eql_candidate(
                model, train_x, train_y, val_x, val_y,
                steps=steps,
                lr=float(lr),
                adam_eps=float(adam_eps),
                l1_lambda=lam,
                phase1_frac=float(phase1_frac),
                phase2_frac=float(phase2_frac),
                prune_threshold=float(prune_threshold),
                batch_size=int(batch_size),
                penalty_every=int(penalty_every),
                seed=int(seed) + 7919 * candidate_index,
            )
            candidates.append(result)
            rows.append({
                "total_layers": float(L),
                "l1_lambda": float(lam),
                "val_mse": float(result.val_mse),
                "active_units": float(result.active_units),
                "nonzero_weights": float(result.nonzero_weights),
                "steps": float(result.steps_run),
            })
            candidate_index += 1

    if not candidates:
        raise RuntimeError("EQL-Div model selection produced no candidates")
    errors = _normalize01([float(r.val_mse) for r in candidates])
    sparsities = _normalize01([float(r.active_units) for r in candidates])
    a = float(selection_error_weight)
    b = float(selection_sparsity_weight)
    scores = [a * e * e + b * s * s for e, s in zip(errors, sparsities)]
    finite_idx = [i for i, score in enumerate(scores) if math.isfinite(score)]
    if not finite_idx:
        raise RuntimeError("EQL-Div model selection produced no finite validation candidate")
    best_i = min(finite_idx, key=lambda i: (scores[i], candidates[i].val_mse, candidates[i].active_units))
    for i, row in enumerate(rows):
        row["selection_score"] = float(scores[i])
        row["selected"] = float(i == best_i)
    return candidates[best_i], rows


def _linear_sympy(weight, bias, inputs, sp, threshold: float):
    expr = sp.Float(float(bias)) if abs(float(bias)) >= threshold else sp.Integer(0)
    for w, inp in zip(weight, inputs):
        wf = float(w)
        if abs(wf) >= threshold:
            expr += sp.Float(wf) * inp
    return expr


def eql_formula(
    model: EQLDivRegressor,
    feature_names: Sequence[str],
    *,
    coefficient_threshold: float = 1e-10,
    simplify: bool = False,
) -> str:
    """Serialize the selected EQL-Div candidate as a raw analytic quotient."""
    import sympy as sp

    h = [sp.Symbol(str(n), real=True) for n in feature_names]
    thr = float(coefficient_threshold)
    for layer in model.layers:
        w = layer.linear.weight.detach().cpu().double().numpy()
        b = layer.linear.bias.detach().cpu().double().numpy()
        z = [_linear_sympy(w[i], b[i], h, sp, thr) for i in range(w.shape[0])]
        out = []
        for i, name in enumerate(layer.unary_names):
            out.append(SYMBOLIC_LIB[name][1](z[i]))
        off = layer.n_unary
        for j in range(layer.n_binary):
            out.append(z[off + 2 * j] * z[off + 2 * j + 1])
        h = out

    ow = model.output_linear.weight.detach().cpu().double().numpy()
    ob = model.output_linear.bias.detach().cpu().double().numpy()
    num = _linear_sympy(ow[0], ob[0], h, sp, thr)
    den = _linear_sympy(ow[1], ob[1], h, sp, thr)
    expr = num / den
    if simplify:
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
    return str(expr)
