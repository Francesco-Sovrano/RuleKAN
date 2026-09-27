import copy
import torch

from rulekan import (
    SumProductKAN,
    SumProductTrainingStage,
    SumProductRegularization,
    fit_sum_product_kan,
    project_constrained_numeric_to_symbolic,
)


def test_symbolic_manifold_distance_is_finite_and_differentiable():
    torch.manual_seed(0)
    m = SumProductKAN(in_dim=2, n_rules=3, max_factors=2, grid=4, seed=0)
    m.symbolic_enabled = False
    m.initialize_symbolic_manifold_seeds_()
    x = torch.randn(48, 2)
    _, details = m(x, return_details=True)
    c, info = m.symbolic_manifold_distance(x, details, max_samples=24)
    assert c.ndim == 0
    assert torch.isfinite(c)
    assert info["operator_errors"].shape == (3, 2, m.n_ops)
    c.backward()
    assert m.symbolic_affine.grad is not None
    assert torch.isfinite(m.symbolic_affine.grad).all()
    numeric_grads = [p.grad for p in m.numeric_edges.parameters() if p.grad is not None]
    assert numeric_grads
    assert all(torch.isfinite(g).all() for g in numeric_grads)


def test_manifold_stage_keeps_numeric_forward_and_trains_auxiliary_symbols():
    torch.manual_seed(1)
    m = SumProductKAN(in_dim=2, n_rules=3, max_factors=2, grid=4, seed=1)
    m.initialize_symbolic_manifold_seeds_()
    x = torch.randn(64, 2)
    y = (0.4 * x[:, [0]] + torch.sin(1.3 * x[:, [1]]))
    before = m.operator_logits.detach().clone()
    stage = SumProductTrainingStage(
        name="tiny manifold",
        steps=4,
        lr=1e-3,
        structure_hardening_start=1.0,
        structure_hardening_end=1.0,
        factor_hardening_start=1.0,
        factor_hardening_end=1.0,
        rule_hardening_start=1.0,
        rule_hardening_end=1.0,
        symbolic_hardening_start=0.0,
        symbolic_hardening_end=0.5,
        regularization=SumProductRegularization(),
        train_structure=False,
        train_variable_gates=False,
        train_rank_gates=False,
        train_symbolic=True,
        symbolic_enabled=False,
        symbolic_manifold_dual_init=1e-4,
        symbolic_manifold_rho=1e-4,
        symbolic_manifold_tolerance=0.02,
        symbolic_manifold_max_samples=32,
    )
    hist = fit_sum_product_kan(
        m, x, y, x, y, stages=[stage], log_every=9999,
        restore_best_each_stage=False, show_progress=False,
    )
    assert m.symbolic_enabled is False
    assert torch.isfinite(m(x)).all()
    assert not torch.equal(before, m.operator_logits.detach())
    assert "symbolic_manifold_dual" in hist[0]


def test_direct_manifold_projection_is_fully_symbolic_and_preserves_structure():
    torch.manual_seed(2)
    m = SumProductKAN(in_dim=2, n_rules=2, max_factors=2, grid=4, seed=2)
    m.symbolic_enabled = False
    m.initialize_symbolic_manifold_seeds_()
    x = torch.randn(40, 2)
    y = (x[:, [0]] * torch.sin(x[:, [1]]))
    m.discretize(force_symbolic=False, freeze_gates=True)
    hard_vars = m.hard_variable_choice.detach().clone()
    sm = project_constrained_numeric_to_symbolic(
        m, x, y, x, y, final_steps=0, final_lbfgs_steps=0,
    )
    assert sm.symbolic_enabled is True
    assert bool(sm.discretized.item())
    assert torch.equal(hard_vars, sm.hard_variable_choice)
    assert not bool(sm.hard_spline_choice.any())
    assert torch.isfinite(sm(x)).all()
