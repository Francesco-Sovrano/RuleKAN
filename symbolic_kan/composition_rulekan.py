from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sympy
import torch
from torch import nn

from .sum_product_kan import (
    SumProductKAN,
    _candidate_rule_value,
    _data_unit_affine_seed_grid,
    _fully_symbolic_continuous_refit,
    _install_symbolic_template,
    _make_fully_symbolic_shell,
    _polish_symbolic_operator_tuple,
    hard_symbolic_tuple_screening,
)
from .utils import SYMBOLIC_LIB


_DEFAULT_OUTER = ("sin", "cos", "tanh", "exp", "sqrt")
_DEFAULT_UNARY = (
    "x", "x^2", "exp", "sin", "cos", "tanh",
    "log1p_sq", "sqrt1p_sq", "inv1p_sq",
)
_DEFAULT_PAIR = ("x", "x^2", "sin", "cos", "exp", "tanh", "log", "sqrt")
_LOW_COMPLEXITY_SUM = frozenset(("x", "x^2", "sin", "cos", "log", "sqrt"))


class Depth2CompositionAtom(nn.Module):
    """One depth-2 symbolic tree ``a*g(b*h(x)+c)+d``.

    ``h`` is a small sum of product terms. Each factor in those terms is one
    ordinary RuleKAN symbolic operator with its own affine input chart. Operator
    identities and the term topology are discrete; all scalar parameters are
    optimized continuously.
    """

    def __init__(
        self,
        outer_operator: str,
        terms: Sequence[Sequence[Tuple[int, str]]],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if outer_operator not in SYMBOLIC_LIB:
            raise KeyError(f"unknown outer symbolic operator {outer_operator!r}")
        clean_terms: List[Tuple[Tuple[int, str], ...]] = []
        for term in terms:
            tt = tuple((int(j), str(op)) for j, op in term)
            if not tt:
                raise ValueError("composition term cannot be empty")
            for _, op in tt:
                if op not in SYMBOLIC_LIB:
                    raise KeyError(f"unknown inner symbolic operator {op!r}")
            clean_terms.append(tt)
        if not clean_terms:
            raise ValueError("composition atom requires at least one inner term")
        self.outer_operator = str(outer_operator)
        self.terms = tuple(clean_terms)
        self.factor_specs = tuple(f for t in self.terms for f in t)
        n_factor = len(self.factor_specs)
        self.beta = nn.Parameter(torch.ones(n_factor, device=device, dtype=dtype))
        self.gamma = nn.Parameter(torch.zeros(n_factor, device=device, dtype=dtype))
        self.term_scale = nn.Parameter(torch.ones(len(self.terms), device=device, dtype=dtype))
        self.inner_bias = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.outer_beta = nn.Parameter(torch.ones((), device=device, dtype=dtype))
        self.outer_gamma = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.output_scale = nn.Parameter(torch.ones((), device=device, dtype=dtype))
        self.output_bias = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    @property
    def support(self) -> Tuple[int, ...]:
        return tuple(sorted({int(j) for j, _ in self.factor_specs}))

    @property
    def family_key(self) -> Tuple[str, Tuple[Tuple[Tuple[int, str], ...], ...]]:
        return self.outer_operator, self.terms

    def inner_value(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner_bias.expand(x.shape[0])
        factor_idx = 0
        for term_idx, term in enumerate(self.terms):
            h = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            for j, op in term:
                raw = SYMBOLIC_LIB[op][0](self.beta[factor_idx] * x[:, j] + self.gamma[factor_idx])
                h = h * raw
                factor_idx += 1
            z = z + self.term_scale[term_idx] * h
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.outer_beta * self.inner_value(x) + self.outer_gamma
        y = self.output_scale * SYMBOLIC_LIB[self.outer_operator][0](z) + self.output_bias
        return y[:, None]

    def symbolic_expression(
        self,
        variable_names: Optional[Sequence[str]] = None,
        *,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        digits: int = 8,
    ):
        d = max(j for j, _ in self.factor_specs) + 1
        names = list(variable_names) if variable_names is not None else [f"x{i}" for i in range(d)]
        syms = [sympy.Symbol(str(n)) for n in names]
        mean = None if input_mean is None else input_mean.detach().cpu().reshape(-1)
        std = None if input_std is None else input_std.detach().cpu().reshape(-1)

        def q(v: torch.Tensor | float):
            return sympy.Float(round(float(torch.as_tensor(v).detach().cpu()), int(digits)))

        inner = q(self.inner_bias)
        factor_idx = 0
        for term_idx, term in enumerate(self.terms):
            prod = sympy.Integer(1)
            for j, op in term:
                xj = syms[j]
                if mean is not None and std is not None:
                    xj = (xj - q(mean[j])) / q(std[j])
                arg = q(self.beta[factor_idx]) * xj + q(self.gamma[factor_idx])
                prod *= SYMBOLIC_LIB[op][1](arg)
                factor_idx += 1
            inner += q(self.term_scale[term_idx]) * prod
        outer_arg = q(self.outer_beta) * inner + q(self.outer_gamma)
        return q(self.output_scale) * SYMBOLIC_LIB[self.outer_operator][1](outer_arg) + q(self.output_bias)


class ComposedRuleKAN(nn.Module):
    """Final symbolic model containing one depth-2 atom and optional flat correction."""

    def __init__(self, composition: Depth2CompositionAtom, correction: Optional[SumProductKAN] = None):
        super().__init__()
        self.composition = composition
        self.correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.composition(x)
        if self.correction is not None:
            y = y + self.correction(x)
        return y

    def symbolic_structure_set(self) -> set[Tuple[int, ...]]:
        out = {tuple(self.composition.support)} if self.composition.support else set()
        if self.correction is not None:
            active = torch.nonzero(
                self.correction.hard_rule_choice & self.correction.rule_alive_mask,
                as_tuple=False,
            ).squeeze(-1).tolist()
            for r in active:
                z: List[int] = []
                for s in range(self.correction.max_factors):
                    if not bool(self.correction.factor_alive_mask[r, s]):
                        continue
                    j = int(self.correction.hard_variable_choice[r, s].item())
                    if j != self.correction.in_dim:
                        z.append(j)
                if z:
                    out.add(tuple(z))
        return out

    def active_symbolic_rule_count(self) -> int:
        n = 1
        if self.correction is not None:
            n += int((self.correction.hard_rule_choice & self.correction.rule_alive_mask).sum().item())
        return n

    def symbolic_formula(
        self,
        variable_names: Optional[Sequence[str]] = None,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        digits: int = 8,
        simplify: bool = False,
    ):
        expr = self.composition.symbolic_expression(
            variable_names, input_mean=input_mean, input_std=input_std, digits=digits
        )
        if self.correction is not None:
            expr = expr + self.correction.symbolic_formula(
                variable_names=variable_names,
                input_mean=input_mean,
                input_std=input_std,
                digits=digits,
                simplify=False,
            )
        return sympy.simplify(expr) if simplify else expr


@dataclass
class CompositionSearchResult:
    model: nn.Module
    selected: bool
    incumbent_validation_mse: float
    best_validation_mse: float
    metadata: Dict[str, object]


def _linear_output_initialize(model: Depth2CompositionAtom, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        z = model.outer_beta * model.inner_value(x) + model.outer_gamma
        h = SYMBOLIC_LIB[model.outer_operator][0](z)
        if not torch.isfinite(h).all():
            return float("inf")
        A = torch.stack([h, torch.ones_like(h)], dim=1)
        try:
            ab = torch.linalg.lstsq(A, y[:, :1]).solution[:, 0]
        except Exception:
            return float("inf")
        if ab.numel() < 2 or not torch.isfinite(ab).all():
            return float("inf")
        model.output_scale.copy_(ab[0])
        model.output_bias.copy_(ab[1])
        pred = A @ ab
        return float(torch.mean((pred - y[:, 0]) ** 2).detach().cpu())




_DOMAIN_SENSITIVE_INNER = frozenset(("log", "sqrt", "1/x", "1/x^2", "1/x^3", "1/sqrt(x)"))


def _seed_domain_sensitive_factors(
    model: Depth2CompositionAtom,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_seeds: int = 18,
) -> None:
    """Greedy data-coordinate multistart for domain-sensitive inner atoms."""
    for factor_idx, (j, op) in enumerate(model.factor_specs):
        if str(op) not in _DOMAIN_SENSITIVE_INNER:
            continue
        seeds = _data_unit_affine_seed_grid(x, int(j), str(op))
        if not seeds:
            continue
        base_state = copy.deepcopy(model.state_dict())
        best_state = base_state
        best_mse = float("inf")
        # Raw and data-unit grids are interleaved by the seed helper. Keep a
        # bounded prefix so this remains a rescue initialization, not a second
        # combinatorial search.
        for beta, gamma in seeds[:max(1, int(max_seeds))]:
            model.load_state_dict(base_state)
            with torch.no_grad():
                model.beta[factor_idx] = float(beta)
                model.gamma[factor_idx] = float(gamma)
            mse = _linear_output_initialize(model, x, y)
            if math.isfinite(mse) and mse < best_mse:
                best_mse = float(mse)
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)


def _coarse_composition_parameter_seed(
    model: Depth2CompositionAtom,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Choose a cheap constant seed before nonlinear refinement.

    Expression-tree SR systems carry explicit constants from the beginning.  A
    depth-2 RuleKAN tree otherwise starts every inner coefficient and the outer
    argument scale at one, which can put periodic outer functions in the wrong
    basin.  Screen a small deterministic scale/relative-weight grid using only
    training data, while keeping the discrete operator family fixed.
    """
    outer_scales = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -2.0)
    if len(model.terms) == 1:
        term_patterns = ((1.0,),)
    elif len(model.terms) == 2:
        term_patterns = (
            (1.0, 1.0), (1.0, 0.25), (0.25, 1.0),
            (1.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1.0),
            (1.0, -0.5), (-0.5, 1.0),
        )
    else:
        term_patterns = (tuple(1.0 for _ in model.terms),)
    # Periodic inner factors are highly multimodal.  Starting every affine
    # chart at beta=1 repeatedly trapped nested oscillators in the same local
    # basin.  Screen a small *global* frequency scale for trig factors before
    # gradient refinement.  This does not change the discrete family or its
    # licensed support; it only seeds continuous constants.
    trig_idx = [i for i, (_, op) in enumerate(model.factor_specs) if op in {"sin", "cos"}]
    trig_scales = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) if trig_idx else (1.0,)
    base_beta = model.beta.detach().clone()

    best = copy.deepcopy(model.state_dict())
    best_mse = float("inf")
    with torch.no_grad():
        for trig_scale in trig_scales:
            model.beta.copy_(base_beta)
            for i in trig_idx:
                model.beta[i] = base_beta[i] * float(trig_scale)
            for pattern in term_patterns:
                model.term_scale.copy_(torch.as_tensor(pattern, device=model.term_scale.device, dtype=model.term_scale.dtype))
                for scale in outer_scales:
                    model.outer_beta.fill_(float(scale))
                    mse = _linear_output_initialize(model, x, y)
                    if math.isfinite(mse) and mse < best_mse:
                        best_mse = float(mse)
                        best = copy.deepcopy(model.state_dict())
        model.load_state_dict(best)
    return float(best_mse)


def _is_simple_tree_family(atom: Depth2CompositionAtom, kind: str) -> bool:
    ops = [str(op) for term in atom.terms for _, op in term]
    if kind == "unary":
        return True
    if kind == "prod2":
        return set(ops) <= {"x", "x^2"}
    if kind == "sum2":
        return set(ops) <= _LOW_COMPLEXITY_SUM
    return False


def _fit_composition_atom(
    model: Depth2CompositionAtom,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    adam_steps: int,
    lbfgs_steps: int,
    lr: float,
) -> Tuple[Depth2CompositionAtom, float]:
    m = copy.deepcopy(model)
    _seed_domain_sensitive_factors(m, train_x, train_y)
    _coarse_composition_parameter_seed(m, train_x, train_y)
    params = [p for p in m.parameters() if p.requires_grad]
    best_state = copy.deepcopy(m.state_dict())
    with torch.no_grad():
        best_mse = float(torch.mean((m(val_x) - val_y) ** 2).detach().cpu())
    opt = torch.optim.Adam(params, lr=float(lr))
    for step in range(max(0, int(adam_steps))):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((m(train_x) - train_y) ** 2)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 4.0)
        opt.step()
        with torch.no_grad():
            m.beta.clamp_(-10.0, 10.0)
            m.gamma.clamp_(-6.0 * math.pi, 6.0 * math.pi)
            m.term_scale.clamp_(-20.0, 20.0)
            m.outer_beta.clamp_(-10.0, 10.0)
            m.outer_gamma.clamp_(-6.0 * math.pi, 6.0 * math.pi)
            m.output_scale.clamp_(-100.0, 100.0)
            m.output_bias.clamp_(-100.0, 100.0)
        if step % 8 == 0 or step + 1 == int(adam_steps):
            with torch.no_grad():
                vmse = float(torch.mean((m(val_x) - val_y) ** 2).detach().cpu())
            if math.isfinite(vmse) and vmse < best_mse:
                best_mse = vmse
                best_state = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)

    if int(lbfgs_steps) > 0:
        before_state = copy.deepcopy(m.state_dict())
        before_mse = best_mse
        opt2 = torch.optim.LBFGS(
            params, lr=0.35, max_iter=int(lbfgs_steps),
            max_eval=max(int(lbfgs_steps) + 1, int(lbfgs_steps * 1.5)),
            history_size=32, line_search_fn="strong_wolfe",
        )
        def closure():
            opt2.zero_grad(set_to_none=True)
            loss = torch.mean((m(train_x) - train_y) ** 2)
            loss.backward()
            return loss
        try:
            opt2.step(closure)
        except Exception:
            m.load_state_dict(before_state)
        with torch.no_grad():
            vmse = float(torch.mean((m(val_x) - val_y) ** 2).detach().cpu())
        if math.isfinite(vmse) and vmse < before_mse:
            best_mse = vmse
        else:
            m.load_state_dict(before_state)
            best_mse = before_mse
    return m, float(best_mse)


def _candidate_patterns(
    in_dim: int,
    allowed_supports: Optional[Sequence[Sequence[int]]],
    *,
    unary_ops: Sequence[str],
    pair_ops: Sequence[str],
    outer_ops: Sequence[str],
) -> List[Tuple[str, Tuple[Tuple[Tuple[int, str], ...], ...], str]]:
    if allowed_supports is None:
        supports = [tuple([j]) for j in range(in_dim)]
        supports += [(i, j) for i in range(in_dim) for j in range(i + 1, in_dim)]
    else:
        supports = sorted({tuple(sorted(set(int(v) for v in s))) for s in allowed_supports if s})
    out = []
    for outer in outer_ops:
        for support in supports:
            if len(support) == 1:
                j = support[0]
                for op in unary_ops:
                    out.append((outer, (((j, op),),), "unary"))
            elif len(support) == 2:
                i, j = support
                for oi in pair_ops:
                    for oj in pair_ops:
                        out.append((outer, (((i, oi), (j, oj)),), "prod2"))
                        out.append((outer, (((i, oi),), ((j, oj),)), "sum2"))
    return out


def _select_family_diverse(
    coarse: Sequence[Tuple[float, str, Tuple[Tuple[Tuple[int, str], ...], ...], str]],
    *,
    global_topk: int,
    per_family_topk: int,
    max_families: int,
) -> List[Tuple[float, str, Tuple[Tuple[Tuple[int, str], ...], ...], str]]:
    selected = []
    seen = set()
    def add(rec):
        key = (rec[1], rec[2])
        if key not in seen and len(selected) < max(1, int(max_families)):
            seen.add(key); selected.append(rec)
    ranked = sorted(coarse, key=lambda z: z[0])
    # Reserve a substantial part of the beam for low-complexity expression-tree
    # families before prediction-only candidates can fill it. This mirrors the
    # parsimony pressure that makes PySR robust on compact nested expressions.
    simple = []
    for rec in ranked:
        ops = [op for term in rec[2] for _, op in term]
        if rec[3] == "unary":
            simple.append(rec)
        elif rec[3] == "sum2" and set(ops) <= _LOW_COMPLEXITY_SUM:
            simple.append(rec)
        elif rec[3] == "prod2" and set(ops) <= {"x", "x^2"}:
            simple.append(rec)
    # Canonical low-node-count trees get a protected slice regardless of their
    # unrefined prediction rank: h(x_i)+h(x_j) and x_i*x_j are especially easy
    # for tree SR but can look poor before affine constants are optimized.
    canonical = []
    for rec in simple:
        ops = [op for term in rec[2] for _, op in term]
        if rec[3] == "sum2" and len(ops) == 2 and ops[0] == ops[1]:
            canonical.append(rec)
        elif rec[3] == "prod2" and ops == ["x", "x"]:
            canonical.append(rec)
    canonical_quota = max(1, int(math.ceil(0.20 * int(max_families))))
    for rec in canonical[:canonical_quota]:
        add(rec)
    simple_quota = max(1, int(math.ceil(0.75 * int(max_families))))
    for rec in simple:
        if len(selected) >= simple_quota:
            break
        add(rec)
    for rec in ranked[:max(1, int(global_topk))]:
        add(rec)
    outer_names = sorted({r[1] for r in coarse})
    for outer in outer_names:
        for kind in ("unary", "prod2", "sum2"):
            bucket = sorted((r for r in coarse if r[1] == outer and r[3] == kind), key=lambda z: z[0])
            for rec in bucket[:max(1, int(per_family_topk))]:
                add(rec)
    # Fill any remaining budget by prediction error.
    for rec in ranked:
        add(rec)
    return selected


def _fit_unary_correction(
    numeric_model: SumProductKAN,
    composition: Depth2CompositionAtom,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    library: Sequence[str],
    allowed_supports: Optional[Sequence[Sequence[int]]],
    screen_top_tuples: int = 8,
    polish_steps: int = 30,
    final_steps: int = 100,
    final_lbfgs_steps: int = 25,
) -> Optional[Tuple[ComposedRuleKAN, float, Dict[str, object]]]:
    with torch.no_grad():
        rtr = train_y - composition(train_x)
        rval = val_y - composition(val_x)
    if allowed_supports is None:
        variables = list(range(train_x.shape[1]))
    else:
        variables = sorted({int(s[0]) for s in allowed_supports if len(set(int(v) for v in s)) == 1})
    if not variables:
        return None

    proposals = []
    for j in variables:
        combos = hard_symbolic_tuple_screening(
            train_x, rtr, (j,), library,
            beam_width=max(24, len(library)), top_tuples=max(1, int(screen_top_tuples)),
            max_samples=384, seeds_per_operator_prefix=1,
        )
        for combo in combos:
            polished, _ = _polish_symbolic_operator_tuple(
                train_x, rtr, (j,), combo,
                steps=max(0, int(polish_steps)), lr=8e-3, max_samples=384,
            )
            factors = ((0, int(j)),)
            htr = _candidate_rule_value(numeric_model, train_x, 0, factors, polished)
            hv = _candidate_rule_value(numeric_model, val_x, 0, factors, polished)
            if not torch.isfinite(htr).all() or not torch.isfinite(hv).all():
                continue
            A = torch.stack([htr, torch.ones_like(htr)], dim=1)
            try:
                ab = torch.linalg.lstsq(A, rtr[:, :1]).solution[:, 0]
            except Exception:
                continue
            pv = ab[0] * hv + ab[1]
            vmse = float(torch.mean((pv - rval[:, 0]) ** 2).detach().cpu())
            proposals.append((vmse, int(j), polished, float(ab[0]), float(ab[1])))
    if not proposals:
        return None
    proposals.sort(key=lambda z: z[0])
    best_model = None
    best_mse = float("inf")
    best_meta: Dict[str, object] = {}
    for _, j, combo, scale, bias in proposals[:min(6, len(proposals))]:
        corr = _make_fully_symbolic_shell(numeric_model)
        with torch.no_grad():
            corr.hard_rule_choice.zero_(); corr.rule_alive_mask.zero_(); corr.rule_scale.zero_(); corr.bias.fill_(bias)
        _install_symbolic_template(corr, 0, ((0, int(j)),), combo, initial_scale=scale)
        with torch.no_grad():
            corr.rule_alive_mask[0] = True
        corr, _ = _fully_symbolic_continuous_refit(
            corr, train_x, rtr, val_x, rval,
            steps=max(0, int(final_steps)), lr=7e-4,
            lbfgs_steps=max(0, int(final_lbfgs_steps)), show_progress=False,
        )
        candidate = ComposedRuleKAN(copy.deepcopy(composition), corr)
        # Joint continuous polish: the flat correction can remove the pressure
        # that otherwise makes the composition atom absorb additive residuals.
        params = [p for p in candidate.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=7e-4)
        state = copy.deepcopy(candidate.state_dict())
        with torch.no_grad():
            vmse = float(torch.mean((candidate(val_x) - val_y) ** 2).detach().cpu())
        for step in range(100):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((candidate(train_x) - train_y) ** 2)
            if not torch.isfinite(loss):
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 4.0); opt.step()
            if step % 10 == 0:
                with torch.no_grad(): cur = float(torch.mean((candidate(val_x) - val_y) ** 2).detach().cpu())
                if math.isfinite(cur) and cur < vmse:
                    vmse = cur; state = copy.deepcopy(candidate.state_dict())
        candidate.load_state_dict(state)
        opt2 = torch.optim.LBFGS(params, lr=0.35, max_iter=60, max_eval=90, history_size=30, line_search_fn="strong_wolfe")
        before = copy.deepcopy(candidate.state_dict()); before_mse = vmse
        def closure():
            opt2.zero_grad(set_to_none=True); loss = torch.mean((candidate(train_x) - train_y) ** 2); loss.backward(); return loss
        try: opt2.step(closure)
        except Exception: candidate.load_state_dict(before)
        with torch.no_grad(): cur = float(torch.mean((candidate(val_x) - val_y) ** 2).detach().cpu())
        if not math.isfinite(cur) or cur >= before_mse:
            candidate.load_state_dict(before); cur = before_mse
        if cur < best_mse:
            best_mse = cur; best_model = candidate
            best_meta = {"correction_variable": int(j), "correction_operator": str(combo[0]["name"])}
    if best_model is None:
        return None
    return best_model, float(best_mse), best_meta


def depth2_composition_rescue(
    incumbent: nn.Module,
    numeric_model: SumProductKAN,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    allowed_supports: Optional[Sequence[Sequence[int]]] = None,
    library: Optional[Sequence[str]] = None,
    outer_operators: Sequence[str] = _DEFAULT_OUTER,
    unary_operators: Sequence[str] = _DEFAULT_UNARY,
    pair_operators: Sequence[str] = _DEFAULT_PAIR,
    trigger_nrmse: float = 3e-3,
    min_improvement_rel: float = 2e-3,
    coarse_global_topk: int = 18,
    coarse_family_topk: int = 4,
    max_families: int = 220,
    shallow_steps: int = 80,
    shallow_lbfgs_steps: int = 20,
    deep_topk: int = 6,
    deep_steps: int = 250,
    deep_lbfgs_steps: int = 80,
    correction_topk: int = 4,
    seed: int = 0,
) -> CompositionSearchResult:
    """Try one validation-gated depth-2 symbolic tree without touching test data.

    This is an additive grammar extension inspired by recursive symbolic-regression
    trees. It is deliberately a rescue: the incumbent remains the result unless a
    composed candidate improves held-out validation MSE by ``min_improvement_rel``.
    """
    torch.manual_seed(int(seed))
    incumbent.eval()
    with torch.no_grad():
        incumbent_mse = float(torch.mean((incumbent(val_x) - val_y) ** 2).detach().cpu())
        scale = max(float(torch.std(val_y).detach().cpu()), 1e-12)
    incumbent_nrmse = math.sqrt(max(incumbent_mse, 0.0)) / scale
    meta: Dict[str, object] = {
        "attempted": False, "selected": False,
        "incumbent_validation_mse": float(incumbent_mse),
        "incumbent_validation_nrmse": float(incumbent_nrmse),
    }
    if math.isfinite(incumbent_nrmse) and incumbent_nrmse <= float(trigger_nrmse):
        meta["reason"] = "incumbent_below_trigger"
        return CompositionSearchResult(incumbent, False, incumbent_mse, incumbent_mse, meta)

    lib = tuple(str(x) for x in (library or numeric_model.symbolic_library) if str(x) in SYMBOLIC_LIB)
    outer = tuple(x for x in outer_operators if x in lib and x in SYMBOLIC_LIB)
    unary = tuple(x for x in unary_operators if x in lib and x in SYMBOLIC_LIB)
    pair = tuple(x for x in pair_operators if x in lib and x in SYMBOLIC_LIB)
    if not outer or not unary:
        meta["reason"] = "composition_library_empty"
        return CompositionSearchResult(incumbent, False, incumbent_mse, incumbent_mse, meta)

    families = _candidate_patterns(
        train_x.shape[1], allowed_supports,
        unary_ops=unary, pair_ops=pair, outer_ops=outer,
    )
    if not families:
        meta["reason"] = "no_licensed_depth2_patterns"
        return CompositionSearchResult(incumbent, False, incumbent_mse, incumbent_mse, meta)
    meta["attempted"] = True
    meta["family_count"] = int(len(families))

    coarse = []
    for outer_name, terms, kind in families:
        atom = Depth2CompositionAtom(outer_name, terms, device=train_x.device, dtype=train_x.dtype)
        # Coarse family ranking stays intentionally cheap. Constant multistart
        # is applied only after the family-diverse beam has been selected.
        mse = _linear_output_initialize(atom, train_x, train_y)
        if math.isfinite(mse):
            coarse.append((mse, outer_name, terms, kind))
    selected_families = _select_family_diverse(
        coarse, global_topk=coarse_global_topk,
        per_family_topk=coarse_family_topk, max_families=max_families,
    )
    meta["refined_family_count"] = int(len(selected_families))

    trials = []
    for _, outer_name, terms, kind in selected_families:
        atom = Depth2CompositionAtom(outer_name, terms, device=train_x.device, dtype=train_x.dtype)
        atom, vmse = _fit_composition_atom(
            atom, train_x, train_y, val_x, val_y,
            adam_steps=shallow_steps, lbfgs_steps=shallow_lbfgs_steps, lr=3e-3,
        )
        if math.isfinite(vmse):
            trials.append((float(vmse), atom, kind))
    trials.sort(key=lambda z: z[0])
    if not trials:
        meta["reason"] = "all_depth2_families_failed"
        return CompositionSearchResult(incumbent, False, incumbent_mse, incumbent_mse, meta)

    deep = []
    deep_pool = []
    deep_seen = set()
    deep_k = max(0, int(deep_topk))
    # Preserve one leading candidate per outer operator before filling by loss.
    # This is the composition analogue of population/family diversity in tree SR.
    for outer_name in sorted({rec[1].outer_operator for rec in trials}):
        bucket = [rec for rec in trials if rec[1].outer_operator == outer_name]
        if bucket:
            rec = min(bucket, key=lambda z: z[0])
            key = rec[1].family_key
            if key not in deep_seen and len(deep_pool) < deep_k:
                deep_seen.add(key); deep_pool.append(rec)
    for rec in trials:
        if len(deep_pool) >= deep_k:
            break
        key = rec[1].family_key
        if key not in deep_seen:
            deep_seen.add(key); deep_pool.append(rec)
    for _, atom0, kind in deep_pool:
        atom, vmse = _fit_composition_atom(
            atom0, train_x, train_y, val_x, val_y,
            adam_steps=deep_steps, lbfgs_steps=deep_lbfgs_steps, lr=1.5e-3,
        )
        if math.isfinite(vmse):
            deep.append((float(vmse), atom, kind))
    candidates = sorted(trials + deep, key=lambda z: z[0])

    # Evaluate the strongest composed families with one flat unary correction.
    # This covers common trees such as sin(x0*x1) + x2^2 without turning the
    # entire search into a general expression-tree enumerator.
    augmented = []
    correction_pool = []
    correction_seen = set()
    def add_correction_trial(rec):
        key = rec[1].family_key
        if key not in correction_seen:
            correction_seen.add(key); correction_pool.append(rec)
    # Half of the correction budget follows predictive ranking; the other half
    # preserves simple tree families that may have an irreducible additive
    # residual until the correction term is introduced.
    ctop = max(0, int(correction_topk))
    pred_k = max(1, int(math.ceil(ctop / 2.0))) if ctop else 0
    for rec in candidates[:pred_k]:
        add_correction_trial(rec)
    simple_trials = [rec for rec in candidates if _is_simple_tree_family(rec[1], rec[2])]
    for rec in simple_trials:
        if len(correction_pool) >= ctop:
            break
        add_correction_trial(rec)
    for rec in candidates:
        if len(correction_pool) >= ctop:
            break
        add_correction_trial(rec)
    for vmse, atom, kind in correction_pool:
        corr = _fit_unary_correction(
            numeric_model, atom, train_x, train_y, val_x, val_y,
            library=lib, allowed_supports=allowed_supports,
        )
        if corr is not None:
            cm, cmse, cmeta = corr
            augmented.append((float(cmse), cm, kind, cmeta))

    best_mse, best_atom, best_kind = candidates[0]
    best_model: nn.Module = ComposedRuleKAN(best_atom)
    best_extra: Dict[str, object] = {}
    if augmented:
        augmented.sort(key=lambda z: z[0])
        if augmented[0][0] < best_mse:
            best_mse, best_model, best_kind, best_extra = augmented[0]

    improvement = (incumbent_mse - float(best_mse)) / max(incumbent_mse, 1e-18)
    meta.update({
        "best_validation_mse": float(best_mse),
        "best_validation_nrmse": math.sqrt(max(float(best_mse), 0.0)) / scale,
        "relative_improvement": float(improvement),
        "outer_operator": str(best_model.composition.outer_operator),
        "inner_terms": [[(int(j), str(op)) for j, op in term] for term in best_model.composition.terms],
        "composition_support": list(best_model.composition.support),
        "pattern_kind": str(best_kind),
        "has_flat_correction": bool(best_model.correction is not None),
        **best_extra,
    })
    if not math.isfinite(best_mse) or improvement < float(min_improvement_rel):
        meta["reason"] = "validation_improvement_too_small"
        return CompositionSearchResult(incumbent, False, incumbent_mse, incumbent_mse, meta)
    meta["selected"] = True
    meta["reason"] = "validation_improved"
    return CompositionSearchResult(best_model, True, incumbent_mse, float(best_mse), meta)
