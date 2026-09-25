from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import torch
from torch import nn

from symbolic_kan.utils import SYMBOLIC_LIB


DEFAULT_EQL_UNARY_LIBRARY = (
    "x", "x^2", "1/x", "sqrt", "log", "exp", "sin", "cos", "tanh",
)


class EQLLayer(nn.Module):
    """Equation-Learner hidden layer: affine map -> unary units + products."""

    def __init__(
        self,
        in_dim: int,
        hidden_width: int,
        unary_library: Sequence[str] = DEFAULT_EQL_UNARY_LIBRARY,
        *,
        n_binary: int | None = None,
    ) -> None:
        super().__init__()
        lib = tuple(str(x) for x in unary_library)
        if not lib:
            raise ValueError("EQL unary_library must be non-empty")
        missing = [x for x in lib if x not in SYMBOLIC_LIB]
        if missing:
            raise ValueError(f"unsupported EQL unary primitives: {missing}")
        hidden_width = int(hidden_width)
        if hidden_width < 2:
            raise ValueError("EQL hidden_width must be >= 2")
        if n_binary is None:
            # Preserve one slot for each requested unary primitive whenever the
            # shared width permits it; remaining outputs are product units.
            n_binary = max(1, hidden_width - min(len(lib), hidden_width - 1))
        n_binary = max(1, min(int(n_binary), hidden_width - 1))
        n_unary = hidden_width - n_binary
        self.unary_names = tuple(lib[i % len(lib)] for i in range(n_unary))
        self.n_unary = int(n_unary)
        self.n_binary = int(n_binary)
        self.out_dim = int(hidden_width)
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


class EQLRegressor(nn.Module):
    """PyTorch reproduction of the EQL alternating affine/operator architecture."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_width: int = 12,
        n_hidden_layers: int = 2,
        unary_library: Sequence[str] = DEFAULT_EQL_UNARY_LIBRARY,
        n_binary: int | None = None,
    ) -> None:
        super().__init__()
        layers = []
        dim = int(input_dim)
        for _ in range(int(n_hidden_layers)):
            layer = EQLLayer(
                dim,
                int(hidden_width),
                unary_library,
                n_binary=n_binary,
            )
            layers.append(layer)
            dim = layer.out_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.readout(h)


@dataclass
class EQLFitResult:
    model: EQLRegressor
    epochs_run: int
    best_val_mse: float
    nonzero_parameters: int


def _mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(x).reshape_as(y) - y) ** 2)


def _l1_parameters(model: nn.Module) -> torch.Tensor:
    return sum((p.abs().sum() for p in model.parameters()), torch.zeros((), device=next(model.parameters()).device))


def _parameter_masks(model: nn.Module, threshold: float) -> Dict[str, torch.Tensor]:
    return {
        name: (p.detach().abs() >= float(threshold)).to(dtype=p.dtype)
        for name, p in model.named_parameters()
    }


def _apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            mask = masks.get(name)
            if mask is not None:
                p.mul_(mask)


def fit_eql(
    model: EQLRegressor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    epochs: int = 1200,
    lr: float = 1e-3,
    l1_lambda: float = 1e-5,
    phase1_frac: float = 1.0 / 3.0,
    phase2_frac: float = 2.0 / 3.0,
    prune_threshold: float = 1e-3,
) -> EQLFitResult:
    """Three-phase EQL training: dense, L1, then fixed-support refit."""
    epochs = max(1, int(epochs))
    t1 = max(1, min(epochs, int(round(float(phase1_frac) * epochs))))
    t2 = max(t1, min(epochs, int(round(float(phase2_frac) * epochs))))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    masks: Dict[str, torch.Tensor] | None = None
    best_state = None
    best_val = math.inf
    epochs_run = 0

    for epoch in range(epochs):
        if epoch == t2 and masks is None:
            masks = _parameter_masks(model, float(prune_threshold))
            _apply_masks(model, masks)
        opt.zero_grad(set_to_none=True)
        loss = _mse(model, train_x, train_y)
        if t1 <= epoch < t2:
            loss = loss + float(l1_lambda) * _l1_parameters(model)
        loss.backward()
        if masks is not None:
            for name, p in model.named_parameters():
                if p.grad is not None and name in masks:
                    p.grad.mul_(masks[name])
        opt.step()
        if masks is not None:
            _apply_masks(model, masks)
        epochs_run = epoch + 1

        # Selection is restricted to the fixed-support phase when it exists, so
        # a dense pre-pruning solution cannot silently defeat the EQL sparsity
        # schedule. For very short tests, use the final available phase.
        if epoch >= t2 or (t2 >= epochs and epoch == epochs - 1):
            with torch.no_grad():
                vmse = float(_mse(model, val_x, val_y).detach().cpu())
            if math.isfinite(vmse) and vmse < best_val:
                best_val = vmse
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        nonzero = int(sum(torch.count_nonzero(p).item() for p in model.parameters()))
        if not math.isfinite(best_val):
            best_val = float(_mse(model, val_x, val_y).detach().cpu())
    return EQLFitResult(model=model, epochs_run=epochs_run, best_val_mse=best_val, nonzero_parameters=nonzero)


def _linear_sympy(weight, bias, inputs, sp, threshold: float):
    expr = sp.Float(float(bias)) if abs(float(bias)) >= threshold else sp.Integer(0)
    for w, inp in zip(weight, inputs):
        wf = float(w)
        if abs(wf) >= threshold:
            expr += sp.Float(wf) * inp
    return expr


def eql_formula(
    model: EQLRegressor,
    feature_names: Sequence[str],
    *,
    coefficient_threshold: float = 1e-10,
    simplify: bool = False,
) -> str:
    """Serialize the trained EQL network as an explicit SymPy expression."""
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

    rw = model.readout.weight.detach().cpu().double().numpy().reshape(-1)
    rb = float(model.readout.bias.detach().cpu().double().item())
    expr = _linear_sympy(rw, rb, h, sp, thr)
    if simplify:
        try:
            expr = sp.simplify(expr)
        except Exception:
            pass
    return str(expr)
