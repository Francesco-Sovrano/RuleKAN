from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TensorFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FuzzyFactorSpec:
    """One factor in an expanded fuzzy if/then rule.

    ``role`` is either ``gate`` or ``branch``. Gate factors use the identity
    operator and ``orientation`` distinguishes membership ``x`` (+1) from its
    complement ``1-x`` (-1). Branch factors are matched by variable/operator
    family while leaving continuous affine parameters free.
    """
    variable: int
    operator: str
    role: str = "branch"
    orientation: int = 0


@dataclass(frozen=True)
class FuzzyRuleSpec:
    name: str
    factors: Tuple[FuzzyFactorSpec, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    suite: str
    description: str
    task_type: str = "regression"  # regression | binary
    n_var: Optional[int] = None
    ranges: Optional[Sequence[Sequence[float]]] = None
    fn: Optional[TensorFn] = None
    formula: Optional[str] = None
    max_factors: int = 2
    expected_structures: Tuple[Tuple[int, ...], ...] = ()
    representability: str = "unknown"
    # Optional benchmark-only expression-size metadata. ``target_tree_nodes``
    # uses the conventional syntax-tree count where variables/constants and
    # operator nodes each count once. It is used by the long-expression stress
    # suite to place targets just below common SR complexity caps.
    target_terms: Optional[int] = None
    target_tree_nodes: Optional[int] = None
    source: str = "synthetic"
    fuzzy_rules: Tuple[FuzzyRuleSpec, ...] = ()
    loader: Optional[Callable[[int, int], "BenchmarkData"]] = None

    @property
    def synthetic(self) -> bool:
        return self.fn is not None


@dataclass
class BenchmarkData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    input_mean: torch.Tensor
    input_std: torch.Tensor
    y_mean: float = 0.0
    y_std: float = 1.0
    task_type: str = "regression"
    feature_names: Optional[List[str]] = None

    def dataset_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "train_input": self.train_x,
            "train_label": self.train_y,
            "test_input": self.val_x,
            "test_label": self.val_y,
            "input_mean": self.input_mean,
            "input_std": self.input_std,
        }


# ---------------------------------------------------------------------------
# Synthetic capability suite
# ---------------------------------------------------------------------------

def _col(x: torch.Tensor, i: int) -> torch.Tensor:
    return x[:, [i]]


def f_same_var_exp_sin(x):
    z = _col(x, 0)
    return 0.8 * torch.exp(-0.7 * z) * torch.sin(2.4 * z)


def f_cross_log_cos(x):
    a, b = _col(x, 0), _col(x, 1)
    return 0.75 * torch.log1p(a.square()) * torch.cos(1.7 * b)


def f_mixed_rank4(x):
    t, phase, concentration, angle, control = [_col(x, i) for i in range(5)]
    return (
        1.3 * torch.exp(-0.9 * t) * torch.sin(2.2 * phase)
        + 0.75 * torch.log1p(concentration.square()) * torch.cos(1.6 * angle)
        + 0.25 * torch.tanh(2.0 * control)
        + 0.40 * torch.exp(-0.65 * control) * torch.sin(2.35 * control)
    )


def f_three_way_product(x):
    a, b, c = _col(x, 0), _col(x, 1), _col(x, 2)
    return 0.9 * torch.sin(2.0 * a) * torch.exp(0.55 * b) * torch.sqrt(1.0 + (1.2 * c).square())


def f_two_same_var_mechanisms(x):
    z = _col(x, 0)
    return 0.3 * torch.tanh(2.0 * z) + 0.55 * torch.exp(-0.6 * z) * torch.sin(2.5 * z)


def f_power_inv_bilinear(x):
    x0, x1 = _col(x, 0), _col(x, 1)
    return torch.reciprocal(1.0 + 0.8 * x0 * x1)


def f_power_inv_quadratic_sum(x):
    x0, x1 = _col(x, 0), _col(x, 1)
    return torch.reciprocal(1.0 + 0.6 * x0.square() + 0.4 * x1.square())


def f_power_square_affine_sum(x):
    x0, x1 = _col(x, 0), _col(x, 1)
    return (0.8 + 0.4 * x0 + 0.3 * x1).square()


def f_power_ratio_product(x):
    x0, x1 = _col(x, 0), _col(x, 1)
    return (0.7 + 0.4 * x0) / (1.0 + 0.6 * x0 * x1)


def f_rational_product_mix(x):
    x0, x1, x2, x3, x4 = [_col(x, i) for i in range(5)]
    return (
        0.9 * torch.cos(2.0 * x0) / (1.0 + 1.5 * x1.square())
        + 0.55 * torch.sqrt(1.0 + x2.square()) * torch.tanh(1.7 * x3)
        + 0.2 * torch.sin(3.0 * x4)
    )


def f_nested_same_sin_exp(x):
    z = _col(x, 0)
    return torch.sin(torch.exp(0.7 * z))


def f_nested_same_tanh_sin(x):
    z = _col(x, 0)
    return torch.tanh(1.6 * torch.sin(2.2 * z))


def f_nested_cross_sin_product(x):
    a, b, c = _col(x, 0), _col(x, 1), _col(x, 2)
    return torch.sin(1.3 * a * b) + 0.2 * c.square()


def f_pykan_exp_sin_square(x):
    a, b = _col(x, 0), _col(x, 1)
    return torch.exp(torch.sin(math.pi * a) + b.square())


def f_pykan_singularity(x):
    a, b = _col(x, 0), _col(x, 1)
    return torch.sin(2.0 * (torch.log(a) + torch.log(b)))


def f_pykan_radial(x):
    a, b = _col(x, 0), _col(x, 1)
    return torch.sqrt(a.square() + b.square())


def f_mixed_nested_products(x):
    x0, x1, x2, x3, x4 = [_col(x, i) for i in range(5)]
    return (
        0.7 * torch.exp(-0.5 * x0) * torch.sin(2.1 * x1)
        + 0.4 * torch.tanh(1.4 * torch.sin(2.0 * x2)) / (1.0 + x3.square())
        + 0.35 * torch.exp(-0.7 * x4) * torch.sin(2.4 * x4)
    )


def f_nested_cross_oscillator(x):
    a, b = _col(x, 0), _col(x, 1)
    return torch.sin(2.5 * torch.sin(1.6 * a) + 0.7 * torch.cos(1.3 * b))


# ---------------------------------------------------------------------------
# Long-expression / capacity stress suite
# ---------------------------------------------------------------------------
# Both targets have exactly 39 binary expression-tree nodes when written in
# their compact canonical form. Two scaled terms add four nodes to the 35-node
# unscaled skeleton, putting the targets immediately below PySR's maxsize=40
# while remaining below Operon's max_length=50. ``long_additive_12`` also exactly
# saturates the shared RuleKAN symbolic budget of 12 rules.


def f_long_additive_12(x):
    z = [_col(x, i) for i in range(12)]
    return (
        0.8 * torch.sin(z[0]) + 1.2 * torch.cos(z[1]) + torch.tanh(z[2]) + torch.exp(z[3])
        - torch.cos(z[4]) + torch.sin(z[5]) - torch.exp(z[6]) + torch.tanh(z[7])
        - torch.sin(z[8]) - torch.tanh(z[9]) + torch.cos(z[10]) - torch.exp(z[11])
    )


def f_long_product_6(x):
    z = [_col(x, i) for i in range(12)]
    return (
        0.8 * torch.sin(z[0]) * torch.exp(z[1])
        + 1.2 * torch.cos(z[2]) * torch.tanh(z[3])
        + torch.sin(z[4]) * torch.cos(z[5])
        + torch.exp(z[6]) * torch.tanh(z[7])
        + torch.cos(z[8]) * torch.exp(z[9])
        + torch.sin(z[10]) * torch.tanh(z[11])
    )


LONG_EXPRESSION_TASKS: Dict[str, TaskSpec] = {
    "long_additive_12": TaskSpec(
        "long_additive_12", "expression_length_stress",
        "12-term additive expression at the RuleKAN rule-budget ceiling (39 binary tree nodes)",
        n_var=12, ranges=[[-1.0, 1.0]] * 12, fn=f_long_additive_12,
        formula=(
            "0.8*sin(x0)+1.2*cos(x1)+tanh(x2)+exp(x3)-cos(x4)+sin(x5)-"
            "exp(x6)+tanh(x7)-sin(x8)-tanh(x9)+cos(x10)-exp(x11)"
        ),
        max_factors=1,
        expected_structures=tuple((i,) for i in range(12)),
        representability="long_expression_in_class",
        target_terms=12, target_tree_nodes=39,
    ),
    "long_product_6": TaskSpec(
        "long_product_6", "expression_length_stress",
        "six two-factor rules / twelve analytic factors (39 binary tree nodes)",
        n_var=12, ranges=[[-1.0, 1.0]] * 12, fn=f_long_product_6,
        formula=(
            "0.8*sin(x0)*exp(x1)+1.2*cos(x2)*tanh(x3)+sin(x4)*cos(x5)+"
            "exp(x6)*tanh(x7)+cos(x8)*exp(x9)+sin(x10)*tanh(x11)"
        ),
        max_factors=2,
        expected_structures=((0,1),(2,3),(4,5),(6,7),(8,9),(10,11)),
        representability="long_expression_in_class",
        target_terms=6, target_tree_nodes=39,
    ),
}


# ---------------------------------------------------------------------------
# Fuzzy if/then rule-recovery suite
# ---------------------------------------------------------------------------
# A fuzzy conditional with membership g in [0,1],
#     IF g THEN f_if ELSE f_else,
# is represented as (1-g)*f_else + g*f_if.  All gate variables below are
# sampled directly in [0,1], so no synthetic off-domain membership samples are
# required. Nested rules are evaluated in their natural tree form but their
# expected symbolic structure is scored in the equivalent expanded sum-product
# (DNF) form.


def f_fuzzy_ite_cross(x):
    g, a, b = _col(x, 0), _col(x, 1), _col(x, 2)
    return (1.0 - g) * (0.70 * torch.sin(2.1 * a)) + g * (1.10 * torch.exp(-0.75 * b))


def f_fuzzy_ite_same_variable(x):
    g = _col(x, 0)
    return (1.0 - g) * (0.65 * torch.sin(2.4 * g)) + g * (0.85 * torch.tanh(2.0 * g))


def f_fuzzy_ite_shared_branch_variable(x):
    g, z = _col(x, 0), _col(x, 1)
    return (1.0 - g) * (0.80 * torch.cos(1.8 * z)) + g * (0.55 * torch.exp(-0.60 * z))


def f_fuzzy_nested_tree(x):
    g0, g1, a, b, c = [_col(x, i) for i in range(5)]
    inner = (1.0 - g1) * (0.60 * torch.cos(1.7 * a)) + g1 * (0.90 * torch.exp(-0.80 * b))
    return (1.0 - g0) * inner + g0 * (0.50 * torch.tanh(1.9 * c))


def f_fuzzy_two_independent_rules(x):
    g0, g1, a, b, c, d = [_col(x, i) for i in range(6)]
    r0 = (1.0 - g0) * torch.sin(2.0 * a) + g0 * torch.exp(-0.70 * b)
    r1 = (1.0 - g1) * torch.cos(1.5 * c) + g1 * torch.tanh(1.8 * d)
    return 0.70 * r0 + 0.40 * r1


def f_fuzzy_product_branches(x):
    g, a, b, c, d = [_col(x, i) for i in range(5)]
    else_branch = 0.80 * torch.exp(-0.45 * a) * torch.sin(2.2 * b)
    if_branch = 0.60 * torch.sqrt(1.0 + (1.15 * c).square()) * torch.cos(1.55 * d)
    return (1.0 - g) * else_branch + g * if_branch


def f_fuzzy_same_gate_product_branches(x):
    g, a, b = [_col(x, i) for i in range(3)]
    else_branch = 0.70 * torch.exp(-0.55 * g) * torch.sin(2.0 * a)
    if_branch = 0.65 * torch.tanh(2.1 * g) * torch.cos(1.45 * b)
    return (1.0 - g) * else_branch + g * if_branch


def _gate(j: int, orientation: int) -> FuzzyFactorSpec:
    return FuzzyFactorSpec(j, "x", "gate", int(orientation))


def _branch(j: int, operator: str) -> FuzzyFactorSpec:
    return FuzzyFactorSpec(j, operator, "branch", 0)


FUZZY_TASKS: Dict[str, TaskSpec] = {
    "fuzzy_ite_cross": TaskSpec(
        "fuzzy_ite_cross", "fuzzy_rules", "single fuzzy if/else with gate and branches on different variables",
        n_var=3, ranges=[[0, 1], [-1.5, 1.5], [-1.5, 1.5]], fn=f_fuzzy_ite_cross,
        formula="(1-x0)*(0.70*sin(2.1*x1)) + x0*(1.10*exp(-0.75*x2))",
        max_factors=2, expected_structures=((0, 1), (0, 2)), representability="fuzzy_in_class",
        fuzzy_rules=(
            FuzzyRuleSpec("else", (_gate(0, -1), _branch(1, "sin"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(2, "exp"))),
        ),
    ),
    "fuzzy_ite_same_variable": TaskSpec(
        "fuzzy_ite_same_variable", "fuzzy_rules", "fuzzy if/else whose gate and both branch functions share the same variable",
        n_var=1, ranges=[[0, 1]], fn=f_fuzzy_ite_same_variable,
        formula="(1-x0)*(0.65*sin(2.4*x0)) + x0*(0.85*tanh(2*x0))",
        max_factors=2, expected_structures=((0, 0),), representability="fuzzy_same_variable",
        fuzzy_rules=(
            FuzzyRuleSpec("else", (_gate(0, -1), _branch(0, "sin"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(0, "tanh"))),
        ),
    ),
    "fuzzy_ite_shared_branch_variable": TaskSpec(
        "fuzzy_ite_shared_branch_variable", "fuzzy_rules", "fuzzy if/else with both branches acting on the same non-gate variable",
        n_var=2, ranges=[[0, 1], [-1.5, 1.5]], fn=f_fuzzy_ite_shared_branch_variable,
        formula="(1-x0)*(0.80*cos(1.8*x1)) + x0*(0.55*exp(-0.60*x1))",
        max_factors=2, expected_structures=((0, 1),), representability="fuzzy_in_class",
        fuzzy_rules=(
            FuzzyRuleSpec("else", (_gate(0, -1), _branch(1, "cos"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(1, "exp"))),
        ),
    ),
    "fuzzy_nested_tree": TaskSpec(
        "fuzzy_nested_tree", "fuzzy_rules", "depth-two fuzzy if/then tree, scored after exact DNF expansion",
        n_var=5, ranges=[[0, 1], [0, 1], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], fn=f_fuzzy_nested_tree,
        formula="(1-x0)*((1-x1)*(0.60*cos(1.7*x2)) + x1*(0.90*exp(-0.80*x3))) + x0*(0.50*tanh(1.9*x4))",
        max_factors=3, expected_structures=((0, 1, 2), (0, 1, 3), (0, 4)), representability="fuzzy_nested_dnf",
        fuzzy_rules=(
            FuzzyRuleSpec("else_else", (_gate(0, -1), _gate(1, -1), _branch(2, "cos"))),
            FuzzyRuleSpec("else_if", (_gate(0, -1), _gate(1, +1), _branch(3, "exp"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(4, "tanh"))),
        ),
    ),
    "fuzzy_two_independent_rules": TaskSpec(
        "fuzzy_two_independent_rules", "fuzzy_rules", "sum of two independent fuzzy if/else rules",
        n_var=6, ranges=[[0, 1], [0, 1], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], fn=f_fuzzy_two_independent_rules,
        formula="0.70*((1-x0)*sin(2*x2)+x0*exp(-0.70*x3)) + 0.40*((1-x1)*cos(1.5*x4)+x1*tanh(1.8*x5))",
        max_factors=2, expected_structures=((0, 2), (0, 3), (1, 4), (1, 5)), representability="fuzzy_multi_rule",
        fuzzy_rules=(
            FuzzyRuleSpec("r0_else", (_gate(0, -1), _branch(2, "sin"))),
            FuzzyRuleSpec("r0_if", (_gate(0, +1), _branch(3, "exp"))),
            FuzzyRuleSpec("r1_else", (_gate(1, -1), _branch(4, "cos"))),
            FuzzyRuleSpec("r1_if", (_gate(1, +1), _branch(5, "tanh"))),
        ),
    ),
    "fuzzy_product_branches": TaskSpec(
        "fuzzy_product_branches", "fuzzy_rules", "fuzzy if/else whose two branches are themselves products",
        n_var=5, ranges=[[0, 1], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]], fn=f_fuzzy_product_branches,
        formula="(1-x0)*(0.80*exp(-0.45*x1)*sin(2.2*x2)) + x0*(0.60*sqrt(1+(1.15*x3)^2)*cos(1.55*x4))",
        max_factors=3, expected_structures=((0, 1, 2), (0, 3, 4)), representability="fuzzy_product_branches",
        fuzzy_rules=(
            FuzzyRuleSpec("else", (_gate(0, -1), _branch(1, "exp"), _branch(2, "sin"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(3, "sqrt1p_sq"), _branch(4, "cos"))),
        ),
    ),
    "fuzzy_same_gate_product_branches": TaskSpec(
        "fuzzy_same_gate_product_branches", "fuzzy_rules", "fuzzy routing with same-variable gate/branch products plus a second branch variable",
        n_var=3, ranges=[[0, 1], [-1.5, 1.5], [-1.5, 1.5]], fn=f_fuzzy_same_gate_product_branches,
        formula="(1-x0)*(0.70*exp(-0.55*x0)*sin(2*x1)) + x0*(0.65*tanh(2.1*x0)*cos(1.45*x2))",
        max_factors=3, expected_structures=((0, 0, 1), (0, 0, 2)), representability="fuzzy_same_variable_product",
        fuzzy_rules=(
            FuzzyRuleSpec("else", (_gate(0, -1), _branch(0, "exp"), _branch(1, "sin"))),
            FuzzyRuleSpec("if", (_gate(0, +1), _branch(0, "tanh"), _branch(2, "cos"))),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Feynman-style generated problems
# ---------------------------------------------------------------------------

def f_feynman_kinetic(x):
    m, v = _col(x, 0), _col(x, 1)
    return 0.5 * m * v.square()


def f_feynman_gravity(x):
    m1, m2, r = _col(x, 0), _col(x, 1), _col(x, 2)
    return m1 * m2 / r.square()


def f_feynman_gaussian(x):
    theta = _col(x, 0)
    return torch.exp(-0.5 * theta.square()) / math.sqrt(2.0 * math.pi)


def f_feynman_lorentz_mix(x):
    q, e, v, b = [_col(x, i) for i in range(4)]
    return q * e + q * v * b


def f_feynman_resonance(x):
    w, w0, gamma = _col(x, 0), _col(x, 1), _col(x, 2)
    return 1.0 / ((w0.square() - w.square()).square() + (gamma * w).square())


SYNTHETIC_TASKS: Dict[str, TaskSpec] = {
    "same_var_exp_sin": TaskSpec(
        "same_var_exp_sin", "synthetic_core", "same-variable product of two analytic functions",
        n_var=1, ranges=[[-1.5, 1.5]], fn=f_same_var_exp_sin,
        formula="0.8*exp(-0.7*x0)*sin(2.4*x0)", max_factors=2,
        expected_structures=((0, 0),), representability="in_class",
    ),
    "cross_log_cos": TaskSpec(
        "cross_log_cos", "synthetic_core", "cross-variable nonlinear product",
        n_var=2, ranges=[[-2, 2], [-math.pi, math.pi]], fn=f_cross_log_cos,
        formula="0.75*log(1+x0^2)*cos(1.7*x1)", max_factors=2,
        expected_structures=((0, 1),), representability="in_class",
    ),
    "mixed_rank4": TaskSpec(
        "mixed_rank4", "synthetic_core", "two cross-products + unary + same-variable product",
        n_var=5,
        ranges=[[0, 2], [-math.pi, math.pi], [-2, 2], [-math.pi, math.pi], [-1.5, 1.5]],
        fn=f_mixed_rank4,
        formula="1.3*exp(-0.9*x0)*sin(2.2*x1)+0.75*log(1+x2^2)*cos(1.6*x3)+0.25*tanh(2*x4)+0.4*exp(-0.65*x4)*sin(2.35*x4)",
        max_factors=2, expected_structures=((0, 1), (2, 3), (4,), (4, 4)),
        representability="in_class",
    ),
    "three_way_product": TaskSpec(
        "three_way_product", "synthetic_core", "three-variable multiplicative rule",
        n_var=3, ranges=[[-1.5, 1.5]] * 3, fn=f_three_way_product,
        formula="0.9*sin(2*x0)*exp(0.55*x1)*sqrt(1+(1.2*x2)^2)", max_factors=3,
        expected_structures=((0, 1, 2),), representability="in_class",
    ),
    "two_same_var_mechanisms": TaskSpec(
        "two_same_var_mechanisms", "synthetic_core", "unary and same-variable product sharing one input",
        n_var=1, ranges=[[-1.5, 1.5]], fn=f_two_same_var_mechanisms,
        formula="0.3*tanh(2*x0)+0.55*exp(-0.6*x0)*sin(2.5*x0)", max_factors=2,
        expected_structures=((0,), (0, 0)), representability="in_class",
    ),
    "power_inv_bilinear": TaskSpec(
        "power_inv_bilinear", "power_expression_stress", "reciprocal of a bilinear sum-product subexpression",
        n_var=2, ranges=[[-0.8, 0.8], [-0.8, 0.8]], fn=f_power_inv_bilinear,
        formula="(1+0.8*x0*x1)^-1", max_factors=2, expected_structures=((0, 1),),
        representability="expression_power_exact",
    ),
    "power_inv_quadratic_sum": TaskSpec(
        "power_inv_quadratic_sum", "power_expression_stress", "reciprocal of an additive quadratic subexpression",
        n_var=2, ranges=[[-1.0, 1.0], [-1.0, 1.0]], fn=f_power_inv_quadratic_sum,
        formula="(1+0.6*x0^2+0.4*x1^2)^-1", max_factors=2, expected_structures=((0,), (1,)),
        representability="expression_power_exact",
    ),
    "power_square_affine_sum": TaskSpec(
        "power_square_affine_sum", "power_expression_stress", "positive integer power applied to an affine sum",
        n_var=2, ranges=[[-1.0, 1.0], [-1.0, 1.0]], fn=f_power_square_affine_sum,
        formula="(0.8+0.4*x0+0.3*x1)^2", max_factors=1, expected_structures=((0,), (1,)),
        representability="expression_power_exact",
    ),
    "power_ratio_product": TaskSpec(
        "power_ratio_product", "power_expression_stress", "positive-power numerator multiplied by a reciprocal denominator block",
        n_var=2, ranges=[[-0.8, 0.8], [-0.8, 0.8]], fn=f_power_ratio_product,
        formula="(0.7+0.4*x0)*(1+0.6*x0*x1)^-1", max_factors=2,
        expected_structures=((0,), (0, 1)), representability="expression_power_exact",
    ),
    "rational_product_mix": TaskSpec(
        "rational_product_mix", "synthetic_core", "rational, sqrt, tanh and trigonometric product mixture",
        n_var=5, ranges=[[-1.5, 1.5]] * 5, fn=f_rational_product_mix,
        formula="0.9*cos(2*x0)/(1+1.5*x1^2)+0.55*sqrt(1+x2^2)*tanh(1.7*x3)+0.2*sin(3*x4)",
        max_factors=2, expected_structures=((0, 1), (2, 3), (4,)), representability="in_class",
    ),
    "nested_same_sin_exp": TaskSpec(
        "nested_same_sin_exp", "nested_stress", "same-variable composition sin(exp(x))",
        n_var=1, ranges=[[-1.5, 1.5]], fn=f_nested_same_sin_exp,
        formula="sin(exp(0.7*x0))", max_factors=2, representability="nested_unary",
    ),
    "nested_same_tanh_sin": TaskSpec(
        "nested_same_tanh_sin", "nested_stress", "same-variable composition tanh(sin(x))",
        n_var=1, ranges=[[-1.5, 1.5]], fn=f_nested_same_tanh_sin,
        formula="tanh(1.6*sin(2.2*x0))", max_factors=2, representability="nested_unary",
    ),
    "nested_cross_sin_product": TaskSpec(
        "nested_cross_sin_product", "nested_stress", "outer sine applied to a cross-variable product",
        n_var=3, ranges=[[-1.5, 1.5]] * 3, fn=f_nested_cross_sin_product,
        formula="sin(1.3*x0*x1)+0.2*x2^2", max_factors=2, representability="nested_cross",
    ),
    "mixed_nested_products": TaskSpec(
        "mixed_nested_products", "nested_stress", "flat products mixed with nested same-variable composition",
        n_var=5, ranges=[[-1.5, 1.5]] * 5, fn=f_mixed_nested_products,
        formula="0.7*exp(-0.5*x0)*sin(2.1*x1)+0.4*tanh(1.4*sin(2*x2))/(1+x3^2)+0.35*exp(-0.7*x4)*sin(2.4*x4)",
        max_factors=2, representability="mixed",
    ),
    "nested_cross_oscillator": TaskSpec(
        "nested_cross_oscillator", "nested_stress", "outer sine of a sum of functions on different inputs",
        n_var=2, ranges=[[-math.pi, math.pi]] * 2, fn=f_nested_cross_oscillator,
        formula="sin(2.5*sin(1.6*x0)+0.7*cos(1.3*x1))", max_factors=2,
        representability="nested_cross",
    ),
    "pykan_exp_sin_square": TaskSpec(
        "pykan_exp_sin_square", "kan_canonical", "canonical pykan function-fitting example",
        n_var=2, ranges=[[-1, 1], [-1, 1]], fn=f_pykan_exp_sin_square,
        formula="exp(sin(pi*x0)+x1^2)", max_factors=2, representability="nested_factorizable",
        source="pykan_example",
    ),
    "pykan_singularity": TaskSpec(
        "pykan_singularity", "kan_canonical", "canonical pykan singular/log composition",
        n_var=2, ranges=[[0.2, 5.0], [0.2, 5.0]], fn=f_pykan_singularity,
        formula="sin(2*(log(x0)+log(x1)))", max_factors=2, representability="nested_cross",
        source="pykan_example",
    ),
    "pykan_radial": TaskSpec(
        "pykan_radial", "kan_canonical", "canonical pykan radial square-root example",
        n_var=2, ranges=[[-1, 1], [-1, 1]], fn=f_pykan_radial,
        formula="sqrt(x0^2+x1^2)", max_factors=2, representability="nested_cross",
        source="pykan_example",
    ),
    "feynman_kinetic": TaskSpec(
        "feynman_kinetic", "feynman", "kinetic-energy style equation",
        n_var=2, ranges=[[0.2, 3.0], [-2.0, 2.0]], fn=f_feynman_kinetic,
        formula="0.5*x0*x1^2", max_factors=2, expected_structures=((0, 1),),
        representability="in_class", source="feynman_style",
    ),
    "feynman_gravity": TaskSpec(
        "feynman_gravity", "feynman", "inverse-square multiplicative law",
        n_var=3, ranges=[[0.5, 2.0], [0.5, 2.0], [0.5, 2.0]], fn=f_feynman_gravity,
        formula="x0*x1/x2^2", max_factors=3, representability="product_rational",
        source="feynman_style",
    ),
    "feynman_gaussian": TaskSpec(
        "feynman_gaussian", "feynman", "Gaussian density factor",
        n_var=1, ranges=[[-3.0, 3.0]], fn=f_feynman_gaussian,
        formula="exp(-x0^2/2)/sqrt(2*pi)", max_factors=2, representability="nested_unary",
        source="feynman_style",
    ),
    "feynman_lorentz_mix": TaskSpec(
        "feynman_lorentz_mix", "feynman", "sum of pair and three-way products",
        n_var=4, ranges=[[-2, 2]] * 4, fn=f_feynman_lorentz_mix,
        formula="x0*x1+x0*x2*x3", max_factors=3,
        expected_structures=((0, 1), (0, 2, 3)), representability="in_class",
        source="feynman_style",
    ),
    "feynman_resonance": TaskSpec(
        "feynman_resonance", "feynman", "nested rational resonance denominator",
        n_var=3, ranges=[[0.2, 2.0], [0.5, 2.0], [0.1, 1.0]], fn=f_feynman_resonance,
        formula="1/((x1^2-x0^2)^2+(x2*x0)^2)", max_factors=3,
        representability="nested_rational", source="feynman_style",
    ),
}


# ---------------------------------------------------------------------------
# Real/tabular tasks
# ---------------------------------------------------------------------------

def _split_standardize(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_samples: int,
    task_type: str,
    feature_names: Optional[Sequence[str]] = None,
) -> BenchmarkData:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(np.float64)
    rng = np.random.default_rng(seed)
    if max_samples > 0 and len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    strat = y if task_type == "binary" and len(np.unique(y)) == 2 else None
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.35, random_state=seed, stratify=strat
    )
    strat_tmp = y_tmp if task_type == "binary" and len(np.unique(y_tmp)) == 2 else None
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.57, random_state=seed + 1, stratify=strat_tmp
    )
    scaler = StandardScaler().fit(X_tr)
    X_trn, X_van, X_ten = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    y_mean, y_std = 0.0, 1.0
    if task_type == "regression":
        y_mean = float(np.mean(y_tr))
        y_std = float(np.std(y_tr)) or 1.0
        y_tr = (y_tr - y_mean) / y_std
        y_va = (y_va - y_mean) / y_std
        y_te = (y_te - y_mean) / y_std
    def tx(a): return torch.as_tensor(a, dtype=torch.float32)
    def ty(a): return torch.as_tensor(a, dtype=torch.float32).reshape(-1, 1)
    return BenchmarkData(
        tx(X_trn), ty(y_tr), tx(X_van), ty(y_va), tx(X_ten), ty(y_te),
        input_mean=torch.as_tensor(scaler.mean_, dtype=torch.float32),
        input_std=torch.as_tensor(scaler.scale_, dtype=torch.float32),
        y_mean=y_mean, y_std=y_std, task_type=task_type,
        feature_names=list(feature_names) if feature_names is not None else None,
    )


def load_sklearn_diabetes(seed: int, max_samples: int) -> BenchmarkData:
    d = load_diabetes()
    return _split_standardize(d.data, d.target, seed=seed, max_samples=max_samples,
                              task_type="regression", feature_names=d.feature_names)


def load_sklearn_breast_cancer(seed: int, max_samples: int) -> BenchmarkData:
    d = load_breast_cancer()
    return _split_standardize(d.data, d.target, seed=seed, max_samples=max_samples,
                              task_type="binary", feature_names=d.feature_names)


def _load_uci(uci_id: int, seed: int, max_samples: int, task_type: str) -> BenchmarkData:
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as exc:
        raise RuntimeError("ucimlrepo is required for UCI tasks") from exc
    ds = fetch_ucirepo(id=uci_id)
    Xdf = ds.data.features.copy()
    ydf = ds.data.targets.copy()
    # Drop rows with missing data; these benchmark variants are numerical/binary.
    import pandas as pd
    all_df = pd.concat([Xdf, ydf], axis=1).replace("?", np.nan).dropna()
    Xdf = all_df[Xdf.columns]
    ydf = all_df[ydf.columns]
    # Preserve numerical columns as numbers and one-hot encode only genuinely
    # categorical columns.  One-hot encoding all columns would explode the
    # dimensionality of mixed datasets such as Adult.
    cat_cols = [c for c in Xdf.columns if not pd.api.types.is_numeric_dtype(Xdf[c])]
    X = pd.get_dummies(Xdf, columns=cat_cols, drop_first=False, dtype=float)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="raise")
    y_raw = ydf.iloc[:, 0]
    if task_type == "binary":
        vals = list(y_raw.dropna().unique())
        if len(vals) != 2:
            raise RuntimeError(f"UCI {uci_id} is not binary after preprocessing: labels={vals[:10]}")
        mapping = {vals[0]: 0.0, vals[1]: 1.0}
        y = y_raw.map(mapping).to_numpy(dtype=float)
    else:
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
    return _split_standardize(X.to_numpy(dtype=float), y, seed=seed, max_samples=max_samples,
                              task_type=task_type, feature_names=list(X.columns))



def load_uci_adult(seed: int, max_samples: int) -> BenchmarkData:
    return _load_uci(2, seed, max_samples, "binary")

def load_uci_spambase(seed: int, max_samples: int) -> BenchmarkData:
    return _load_uci(94, seed, max_samples, "binary")


def load_uci_gamma(seed: int, max_samples: int) -> BenchmarkData:
    return _load_uci(159, seed, max_samples, "binary")


def load_uci_diabetes_binary(seed: int, max_samples: int) -> BenchmarkData:
    return _load_uci(891, seed, max_samples, "binary")


REAL_TASKS: Dict[str, TaskSpec] = {
    "sklearn_diabetes": TaskSpec(
        "sklearn_diabetes", "real_small", "scikit-learn diabetes regression",
        task_type="regression", source="sklearn", loader=load_sklearn_diabetes,
    ),
    "sklearn_breast_cancer": TaskSpec(
        "sklearn_breast_cancer", "real_small", "Wisconsin breast-cancer binary classification",
        task_type="binary", source="sklearn", loader=load_sklearn_breast_cancer,
    ),
    "uci_adult": TaskSpec(
        "uci_adult", "uci_binary", "UCI Adult income binary classification",
        task_type="binary", source="uci", loader=load_uci_adult,
    ),
    "uci_spambase": TaskSpec(
        "uci_spambase", "uci_binary", "UCI Spambase binary classification",
        task_type="binary", source="uci", loader=load_uci_spambase,
    ),
    "uci_gamma": TaskSpec(
        "uci_gamma", "uci_binary", "UCI MAGIC Gamma Telescope binary classification",
        task_type="binary", source="uci", loader=load_uci_gamma,
    ),
    "uci_diabetes_binary": TaskSpec(
        "uci_diabetes_binary", "uci_binary", "UCI CDC diabetes binary classification",
        task_type="binary", source="uci", loader=load_uci_diabetes_binary,
    ),
}

TASKS: Dict[str, TaskSpec] = {**SYNTHETIC_TASKS, **LONG_EXPRESSION_TASKS, **FUZZY_TASKS, **REAL_TASKS}


def list_suites() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, spec in TASKS.items():
        out.setdefault(spec.suite, []).append(name)
    for values in out.values():
        values.sort()
    return dict(sorted(out.items()))


def make_synthetic_data(
    spec: TaskSpec,
    *,
    seed: int,
    train_n: int,
    val_n: int,
    test_n: int,
) -> BenchmarkData:
    if not spec.synthetic or spec.n_var is None or spec.ranges is None or spec.fn is None:
        raise ValueError(f"{spec.name} is not a synthetic task")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) + 104729)
    ranges = torch.as_tensor(spec.ranges, dtype=torch.float32)
    lo, hi = ranges[:, 0], ranges[:, 1]
    total = int(train_n + val_n + test_n)
    raw = torch.rand((total, spec.n_var), generator=g) * (hi - lo) + lo
    with torch.no_grad():
        y = spec.fn(raw).reshape(-1, 1).to(torch.float32)
    tr_raw = raw[:train_n]
    mean = tr_raw.mean(dim=0)
    std = tr_raw.std(dim=0).clamp_min(1e-6)
    x = (raw - mean) / std
    i1, i2 = train_n, train_n + val_n
    return BenchmarkData(
        x[:i1], y[:i1], x[i1:i2], y[i1:i2], x[i2:], y[i2:],
        input_mean=mean, input_std=std, y_mean=0.0, y_std=1.0,
        task_type="regression", feature_names=[f"x{i}" for i in range(spec.n_var)],
    )


def load_task_data(
    spec: TaskSpec,
    *,
    seed: int,
    train_n: int,
    val_n: int,
    test_n: int,
    max_real_samples: int,
) -> BenchmarkData:
    if spec.synthetic:
        return make_synthetic_data(spec, seed=seed, train_n=train_n, val_n=val_n, test_n=test_n)
    if spec.loader is None:
        raise RuntimeError(f"task {spec.name} has no loader")
    return spec.loader(seed, max_real_samples)
