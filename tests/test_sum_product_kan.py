import torch

from symbolic_kan.sum_product_kan import (
    SumProductKAN,
    SumProductRegularization,
)


def _hard_identity_symbolic_model():
    model = SumProductKAN(
        in_dim=2,
        n_rules=1,
        max_factors=2,
        grid=4,
        k=2,
        symbolic_library=("x",),
        min_order=1,
        seed=0,
    )
    with torch.no_grad():
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_operator_choice.zero_()
        model.hard_spline_choice.zero_()
        model.symbolic_affine.zero_()
        model.symbolic_affine[..., 0] = 1.0
        model.symbolic_affine[..., 1] = 1.0
        model.rule_scale[:] = 2.0
        model.bias.zero_()
        model.discretized.fill_(True)
    return model


def test_hard_symbolic_rule_is_exact_product():
    model = _hard_identity_symbolic_model()
    x = torch.tensor([[2.0, 3.0], [-1.5, 4.0], [0.25, -2.0]])
    y = model(x)
    expected = 2.0 * x[:, [0]] * x[:, [1]]
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_identity_slot_reduces_product_to_unary():
    model = _hard_identity_symbolic_model()
    with torch.no_grad():
        model.hard_variable_choice[0, 1] = model.in_dim
        model.rule_scale[:] = 3.0
    x = torch.tensor([[2.0, 9.0], [-1.5, 4.0]])
    y = model(x)
    assert torch.allclose(y, 3.0 * x[:, [0]], atol=1e-6, rtol=1e-6)


def test_whole_rule_gate_removes_rule():
    model = _hard_identity_symbolic_model()
    with torch.no_grad():
        model.hard_rule_choice[:] = False
        model.bias[:] = 0.7
    x = torch.randn(7, 2)
    y = model(x)
    assert torch.allclose(y, torch.full((7, 1), 0.7), atol=1e-6)


def test_soft_regularizers_backpropagate_to_structure():
    model = SumProductKAN(
        in_dim=3,
        n_rules=4,
        max_factors=2,
        grid=4,
        k=2,
        symbolic_library=("x", "sin", "tanh"),
        seed=2,
    )
    x = torch.randn(32, 3)
    y, details = model(x, return_details=True)
    reg, parts = model.regularization(
        details,
        SumProductRegularization(
            rule_l0=1e-3,
            factor_l0=1e-3,
            variable_entropy=1e-3,
            operator_entropy=1e-3,
            spline_penalty=1e-3,
            contribution_group_lasso=1e-3,
            diversity=1e-3,
        ),
    )
    loss = y.square().mean() + reg
    loss.backward()
    assert model.rule_gate.log_alpha.grad is not None
    assert model.variable_logits.grad is not None
    assert model.operator_logits.grad is not None
    assert model.spline_logits.grad is not None
    assert torch.isfinite(loss)
    assert "rule_l0" in parts and "factor_l0" in parts


def test_formula_contains_exact_product_structure():
    model = _hard_identity_symbolic_model()
    expr = model.symbolic_formula(["u", "v"], digits=4)
    text = str(expr)
    assert "u" in text and "v" in text
    # SymPy may reorder factors but the expression should evaluate exactly.
    import sympy as sp
    u, v = sp.symbols("u v")
    assert sp.simplify(expr - 2 * u * v) == 0


def test_forward_is_finite_for_symbolic_library():
    model = SumProductKAN(
        in_dim=2,
        n_rules=3,
        max_factors=2,
        grid=4,
        k=2,
        symbolic_library=("x", "exp", "log1p_sq", "inv1p_sq", "sin"),
        seed=1,
    )
    x = torch.randn(24, 2) * 3
    y = model(x)
    assert torch.isfinite(y).all()


def test_conformal_pruning_can_remove_exactly_zero_rule():
    from symbolic_kan.sum_product_kan import conformal_prune_sum_product_rules
    model = SumProductKAN(
        in_dim=1,
        n_rules=2,
        max_factors=1,
        grid=3,
        k=2,
        symbolic_library=("x",),
        seed=0,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[:, 0] = 0
        model.hard_operator_choice.zero_()
        model.hard_spline_choice.zero_()
        model.symbolic_affine.zero_()
        model.symbolic_affine[..., 0] = 1.0
        model.symbolic_affine[..., 1] = 1.0
        model.rule_scale[:] = torch.tensor([1.0, 0.0])
        model.bias.zero_()
    x = torch.linspace(-1, 1, 80).unsqueeze(1)
    y = x.clone()
    out = conformal_prune_sum_product_rules(
        model, x, y, alpha=0.1, relative_tolerance=0.0,
        absolute_tolerance=1e-8, min_rules=1,
    )
    assert out["active_rules"] == 1
    assert 1 in out["accepted"]


def test_factor_presence_gate_controls_order_differentiably():
    model = SumProductKAN(
        in_dim=2, n_rules=1, max_factors=2, grid=4, k=2,
        symbolic_library=("x",), min_order=1, seed=4,
    )
    model.structure_hardening = 1.0
    model.factor_hardening = 0.0
    with torch.no_grad():
        model.variable_logits.fill_(-8.0)
        model.variable_logits[0, 0, 0] = 8.0
        model.variable_logits[0, 1, 1] = 8.0
        # Put optional factor in the differentiable interior.
        model.factor_gate.log_alpha[1] = 0.0
    x = torch.randn(16, 2)
    y = model(x).sum()
    y.backward()
    assert model.factor_gate.log_alpha.grad is not None
    assert torch.isfinite(model.factor_gate.log_alpha.grad).all()


def test_conformal_refit_pruning_removes_zero_rule():
    from symbolic_kan.sum_product_kan import conformal_refit_prune_sum_product_rules
    model = _hard_identity_symbolic_model()
    # Expand to a 2-rule model with one exactly zero contribution.
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=0,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[:, 0] = 0
        model.hard_operator_choice.zero_()
        model.hard_spline_choice.zero_()
        model.symbolic_affine.zero_()
        model.symbolic_affine[..., 0] = 1.0
        model.symbolic_affine[..., 1] = 1.0
        model.rule_scale[:] = torch.tensor([1.0, 0.0])
        model.bias.zero_()
    for p in (model.variable_logits, model.operator_logits, model.spline_logits,
              model.rule_gate.log_alpha, model.factor_gate.log_alpha):
        p.requires_grad_(False)
    x = torch.linspace(-1, 1, 120).unsqueeze(1)
    y = x.clone()
    out = conformal_refit_prune_sum_product_rules(
        model, x[:80], y[:80], x[80:100], y[80:100], x[100:], y[100:],
        alpha=0.1, relative_tolerance=0.0, absolute_tolerance=1e-7,
        validation_rel_budget=0.0, refit_steps=0, min_rules=1,
    )
    assert out["active_rules"] == 1
    assert 1 in out["accepted"]


def test_discretize_is_function_preserving_at_hard_endpoint():
    model = SumProductKAN(
        in_dim=3, n_rules=5, max_factors=2, grid=5, k=2,
        symbolic_library=("x", "sin"), seed=11,
    )
    model.symbolic_enabled = False
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    x = torch.randn(48, 3)
    with torch.no_grad():
        before = model(x).clone()
        model.discretize(force_symbolic=False, freeze_gates=False)
        model.symbolic_enabled = False
        after = model(x).clone()
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_joint_rule_gsr_recovers_exact_identity_product():
    from symbolic_kan.sum_product_kan import in_context_symbolic_rule_gsr
    model = SumProductKAN(
        in_dim=2, n_rules=1, max_factors=2, grid=4, k=2,
        symbolic_library=("x",), min_order=1, seed=5,
    )
    with torch.no_grad():
        model.numeric_edges.scale_base.zero_()
        model.numeric_edges.scale_sp.zero_()
        model.numeric_edges.scale_base[0, 0] = 1.0
        model.numeric_edges.scale_base[1, 1] = 1.0
        model.rule_scale[:] = 1.0
        model.bias.zero_()
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_spline_choice[:] = True
        model.discretized.fill_(True)
        model.symbolic_enabled = False
    x = torch.randn(96, 2)
    y = x[:, [0]] * x[:, [1]]
    sym, hist = in_context_symbolic_rule_gsr(
        model, x[:64], y[:64], x[64:], y[64:],
        library=("x",), local_topk=1, max_rule_candidates=1,
        trial_steps=0, lr=1e-4, global_rel_mse_budget=1.0,
        require_complete=True, verbose=False,
    )
    assert hist and hist[0]["accepted"]
    assert not bool(sym.hard_spline_choice[0, 0, 0])
    assert not bool(sym.hard_spline_choice[0, 1, 1])
    with torch.no_grad():
        pred = sym(x)
    assert torch.sqrt(torch.mean((pred - y) ** 2)) < 3e-3


def test_symbolic_copy_compaction_removes_zero_contribution_rule():
    from symbolic_kan.sum_product_kan import compact_sumproduct_for_symbolic
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=6,
    )
    with torch.no_grad():
        model.numeric_edges.scale_base.zero_()
        model.numeric_edges.scale_sp.zero_()
        model.numeric_edges.scale_base[0, 0] = 1.0
        model.numeric_edges.scale_base[0, 1] = 1.0
        model.rule_scale[:] = torch.tensor([1.0, 0.0])
        model.bias.zero_()
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[:, 0] = 0
        model.hard_spline_choice[:] = True
        model.discretized.fill_(True)
        model.symbolic_enabled = False
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = x.clone()
    compact, hist = compact_sumproduct_for_symbolic(
        model, x[:70], y[:70], x[70:], y[70:],
        min_rules=1, relative_mse_budget=0.0,
        refit_steps=0, lr=1e-4, candidate_pool=2, verbose=False,
    )
    assert int(compact.hard_rule_choice.sum()) == 1
    assert any(z["accepted"] for z in hist)


def test_mandatory_symbolic_matching_pursuit_has_zero_splines():
    from symbolic_kan.sum_product_kan import mandatory_symbolic_matching_pursuit
    torch.manual_seed(0)
    model = SumProductKAN(
        in_dim=2, n_rules=1, max_factors=2, grid=6, k=2,
        symbolic_library=("x", "sin"), min_order=1, seed=0,
    )
    with torch.no_grad():
        model.variable_logits.fill_(-10.0)
        model.variable_logits[0, 0, 0] = 10.0
        model.variable_logits[0, 1, 1] = 10.0
        model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_spline_choice[:] = True
        model.discretized.fill_(True)
    # Train the numerical KAN edge bank briefly on the simple product so its
    # local curves are informative enough to shortlist the identity atom.
    x = torch.rand(180, 2) * 2.0 - 1.0
    y = x[:, [0]] * x[:, [1]]
    for p in (model.variable_logits, model.rule_gate.log_alpha,
              model.factor_gate.log_alpha, model.operator_logits,
              model.spline_logits, model.symbolic_affine):
        p.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3
    )
    for _ in range(120):
        opt.zero_grad(); loss = ((model(x[:120]) - y[:120]) ** 2).mean(); loss.backward(); opt.step()
    sym, hist = mandatory_symbolic_matching_pursuit(
        model, x[:120], y[:120], x[120:], y[120:],
        library=("x", "sin"), local_topk=2, max_rule_candidates=4,
        max_symbolic_rules=1, min_symbolic_rules=1, beam_width=2,
        shortlist_refine_steps=20, trial_steps=30, final_steps=60,
        final_lbfgs_steps=20, target_val_rmse=5e-3, verbose=False,
    )
    assert int(sym.hard_rule_choice.sum()) == 1
    r = int(torch.nonzero(sym.hard_rule_choice, as_tuple=False)[0])
    for s in range(sym.max_factors):
        j = int(sym.hard_variable_choice[r, s])
        if j != sym.in_dim:
            assert not bool(sym.hard_spline_choice[r, s, j])
    formula = str(sym.symbolic_formula(["u", "v"], simplify=True))
    assert "Spline_" not in formula


def test_numeric_grid_refinement_is_finite_and_denser():
    from symbolic_kan.sum_product_kan import refine_sumproduct_numeric_grid
    model = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=2, grid=6, k=2,
        symbolic_library=("x",), seed=3,
    )
    model.symbolic_enabled = False
    x = torch.randn(80, 2).clamp(-1.8, 1.8)
    refined = refine_sumproduct_numeric_grid(model, x, 10)
    assert refined.numeric_edges.num == 10
    assert torch.isfinite(refined(x)).all()


def test_diverse_initialization_covers_same_input_products():
    model = SumProductKAN(
        in_dim=5, n_rules=15, max_factors=2, grid=4, k=2,
        symbolic_library=("x",), allow_self_products=True, seed=17,
    )
    model.structure_hardening = 1.0
    pi = model.variable_probabilities().detach()
    choices = torch.argmax(pi[..., : model.in_dim], dim=-1)
    pairs = {tuple(sorted((int(row[0]), int(row[1])))) for row in choices}
    for j in range(model.in_dim):
        assert (j, j) in pairs
    # C(5+2-1,2)=15: every unordered pair with replacement is covered once.
    assert len(pairs) == 15


def test_two_symbolic_functions_can_multiply_on_same_input():
    model = SumProductKAN(
        in_dim=1, n_rules=1, max_factors=2, grid=4, k=2,
        symbolic_library=("exp", "sin"), allow_self_products=True, seed=3,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.hard_rule_choice[:] = True
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 0
        model.hard_spline_choice.zero_()
        model.hard_operator_choice[0, 0, 0] = 0  # exp
        model.hard_operator_choice[0, 1, 0] = 1  # sin
        model.symbolic_affine.zero_()
        model.symbolic_affine[..., 0] = 1.0
        model.symbolic_affine[..., 1] = 1.0
        model.rule_scale[:] = 1.0
        model.bias.zero_()
    x = torch.linspace(-1.0, 1.0, 21).unsqueeze(1)
    y = model(x)
    expected = torch.exp(x) * torch.sin(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
    formula = str(model.symbolic_formula(["x"], digits=6))
    assert "exp" in formula and "sin" in formula
    assert "Spline_" not in formula


def test_numeric_init_can_forbid_self_products_while_symbolic_bank_keeps_them():
    from symbolic_kan.sum_product_kan import symbolic_structure_bank
    model = SumProductKAN(
        in_dim=5, n_rules=15, max_factors=2, grid=4, k=2,
        symbolic_library=("x",), allow_self_products=False, seed=19,
    )
    model.structure_hardening = 1.0
    pi = model.variable_probabilities().detach()
    choices = torch.argmax(pi[..., : model.in_dim], dim=-1)
    fopen = model.factor_open_probabilities().detach()
    numeric_pairs = [
        (int(choices[r, 0]), int(choices[r, 1]))
        for r in range(model.n_rules) if float(fopen[r, 1]) >= 0.5
    ]
    assert numeric_pairs
    assert all(a != b for a, b in numeric_pairs)

    symbolic = symbolic_structure_bank(5, 2, allow_self_products=True)
    for j in range(5):
        assert (j, j) in symbolic


def test_gmp_preselection_prunes_symbol_tuples_and_finds_self_product_atoms():
    from symbolic_kan.sum_product_kan import gmp_symbolic_operator_preselection
    torch.manual_seed(0)
    x = torch.linspace(-1.5, 1.5, 320).unsqueeze(1)
    y = 0.4 * torch.exp(-0.65 * x) * torch.sin(2.35 * x)
    result, stats = gmp_symbolic_operator_preselection(
        x, y, [(0, 0)], ("x", "exp", "sin", "cos", "tanh"),
        topk=2, steps=260, lr=2e-2,
        temperature_start=1.5, temperature_end=0.25,
        entropy_weight=2e-4, show_progress=False,
    )
    assert stats["naive_operator_tuples"] == 25.0
    assert stats["gmp_operator_tuples"] == 4.0
    assert stats["tuple_reduction_fraction"] >= 0.80
    assert result
    names = [{z["name"] for z in shortlist} for shortlist in result[0]["factor_shortlists"]]
    # Slot ordering is exchangeable, so exp and sin only need to appear across
    # the two retained factor shortlists, not in a prescribed slot.
    assert any("exp" in z for z in names)
    assert any("sin" in z for z in names)
    for shortlist in result[0]["factor_shortlists"]:
        for cand in shortlist:
            a, b, c, d = cand["affine"]
            assert abs(a - 1.0) < 1e-12
            assert abs(d) < 1e-12


def test_example_target_contains_nontrivial_same_variable_product_term():
    # This regression test protects the benchmark requirement introduced in v20.
    from examples.example_sum_product_kan import target
    x = torch.zeros(4, 5)
    x[:, 4] = torch.tensor([-1.0, -0.3, 0.6, 1.2])
    # Remove all terms except those depending on control by evaluating their
    # known control-only expression directly. The new damped sinusoid is not
    # representable by the old tanh-only control term.
    expected_control = (
        0.25 * torch.tanh(2.0 * x[:, [4]])
        + 0.40 * torch.exp(-0.65 * x[:, [4]]) * torch.sin(2.35 * x[:, [4]])
    )
    # At phase=0, concentration=0, angle=0 the two cross-variable terms vanish.
    assert torch.allclose(target(x), expected_control, atol=1e-7, rtol=1e-7)


def test_fully_symbolic_template_canonicalizes_factor_output_affine():
    from symbolic_kan.sum_product_kan import _install_symbolic_template
    model = SumProductKAN(
        in_dim=1, n_rules=1, max_factors=2, grid=4, k=2,
        symbolic_library=("exp", "sin"), allow_self_products=True, seed=23,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.hard_spline_choice.zero_()
    combo = (
        {"name": "exp", "affine": (3.0, -0.7, 0.2, 4.0)},
        {"name": "sin", "affine": (-2.0, 2.3, -0.1, -5.0)},
    )
    _install_symbolic_template(model, 0, ((0, 0), (1, 0)), combo, initial_scale=1.0)
    for slot, op in ((0, 0), (1, 1)):
        a, b, c, d = model.symbolic_affine[0, slot, 0, op].detach().tolist()
        assert abs(a - 1.0) < 1e-12
        assert abs(d) < 1e-12
    formula = str(model.symbolic_formula(["x"], simplify=False))
    assert "exp" in formula and "sin" in formula


def test_protected_log_sympy_export_matches_logabs_semantics():
    import numpy as np
    import sympy as sp
    from symbolic_kan.utils import SYMBOLIC_LIB

    sx = sp.symbols("x", real=True)
    expr = SYMBOLIC_LIB["log"][1](sx)
    assert expr == sp.log(sp.Abs(sx))

    # Stay away from zero, where the runtime epsilon smoothing intentionally
    # differs slightly from the ideal log(abs(x)) symbolic operator.
    values = torch.tensor([-8.0, -2.0, -0.25, 0.25, 2.0, 8.0], dtype=torch.float64)
    expected = SYMBOLIC_LIB["log"][0](values).detach().cpu().numpy()
    fn = sp.lambdify(sx, expr, modules="numpy")
    actual = np.asarray(fn(values.cpu().numpy()), dtype=float)
    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_composition_export_keeps_abs_inside_protected_log():
    import sympy as sp
    from symbolic_kan.composition_rulekan import Depth2CompositionAtom

    atom = Depth2CompositionAtom("cos", [[(0, "log")]], dtype=torch.float64)
    with torch.no_grad():
        atom.beta[0] = -5.0649
        atom.gamma[0] = -9.5181
    expr = atom.symbolic_expression(["z0"], digits=8)
    assert expr.has(sp.Abs)
    assert "log(Abs(" in str(expr)


def test_sumproduct_symbolic_library_audit_passes_default_atoms():
    from symbolic_kan.sum_product_kan import audit_sumproduct_symbolic_library
    report = audit_sumproduct_symbolic_library()
    assert report
    assert all(bool(info.get("ok", False)) for info in report.values())
    assert report["exp"]["overflow_guard_finite"] is True


def test_factor_l0_tracks_optional_factor_gate_not_masked_identity_channel():
    model = SumProductKAN(
        in_dim=2, n_rules=1, max_factors=2, min_order=1, grid=4, k=2,
        symbolic_library=("x",), allow_self_products=False, seed=13,
    )
    x = torch.randn(32, 2)
    with torch.no_grad():
        model.rule_gate.log_alpha.fill_(8.0)
        model.factor_gate.log_alpha.reshape(1, 2)[0, 1] = 8.0
    _, det = model(x, return_details=True)
    _, parts_open = model.regularization(det, SumProductRegularization(factor_l0=1.0))
    with torch.no_grad():
        model.factor_gate.log_alpha.reshape(1, 2)[0, 1] = -8.0
    _, det = model(x, return_details=True)
    _, parts_closed = model.regularization(det, SumProductRegularization(factor_l0=1.0))
    assert float(parts_open["factor_l0"].detach()) > float(parts_closed["factor_l0"].detach()) + 0.5


def test_prune_to_stability_removes_dead_rule_and_persists_mask():
    from symbolic_kan.sum_product_kan import prune_numeric_structure_to_stability
    torch.manual_seed(5)
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), allow_self_products=False, seed=5,
    )
    model.symbolic_enabled = False
    with torch.no_grad():
        model.rule_gate.log_alpha.fill_(8.0)
        model.rule_scale[0] = 1.0
        model.rule_scale[1] = 0.0
    x = torch.linspace(-1.0, 1.0, 96).unsqueeze(1)
    y = model(x).detach()
    pruned, info = prune_numeric_structure_to_stability(
        model, x, y, x, y,
        refit_steps=0, stabilize_rounds=0,
        local_rel_mse_budget=0.0, global_rel_mse_budget=0.0,
        min_rules=1, verbose=False, show_progress=False,
    )
    assert int(pruned.rule_alive_mask.sum()) == 1
    assert not bool(pruned.rule_alive_mask[1])
    before = pruned.rule_alive_mask.clone()
    # Optimizer steps cannot resurrect the physical mask.
    opt = torch.optim.Adam([p for p in pruned.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(set_to_none=True)
    torch.mean((pruned(x) - y) ** 2).backward()
    opt.step()
    assert torch.equal(before, pruned.rule_alive_mask)
    assert any(z["rule"] == 1 and z["accepted"] for z in info["accepted"])


def test_prune_to_stability_rolls_back_destructive_deletion():
    from symbolic_kan.sum_product_kan import prune_numeric_structure_to_stability
    model = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), allow_self_products=False, seed=17,
    )
    model.symbolic_enabled = False
    with torch.no_grad():
        model.rule_gate.log_alpha.fill_(8.0)
        model.variable_logits.fill_(-6.0)
        model.variable_logits[0, 0, 0] = 6.0
        model.variable_logits[1, 0, 1] = 6.0
        model.rule_scale[:] = 1.0
    x = torch.randn(128, 2)
    y = model(x).detach()
    pruned, info = prune_numeric_structure_to_stability(
        model, x, y, x, y,
        refit_steps=0, stabilize_rounds=0,
        local_rel_mse_budget=0.0, global_rel_mse_budget=0.0,
        min_rules=1, verbose=False, show_progress=False,
    )
    assert int(pruned.rule_alive_mask.sum()) == 2
    assert len(info["accepted"]) == 0
    assert len(info["rejected"]) >= 1


def test_gsr_affine_rmse_projection_handles_nonzero_candidate_mean():
    from symbolic_kan.sum_product_kan import _best_affine_residual_rmse
    x = torch.linspace(-2.0, 2.0, 101)
    h = 3.0 + torch.sin(1.7 * x)
    residual = -1.2 + 2.5 * h
    score, scale, offset = _best_affine_residual_rmse(h, residual)
    assert score < 1e-6
    assert abs(scale - 2.5) < 1e-5
    assert abs(offset + 1.2) < 1e-5


def test_default_symbolic_formula_matches_torch_forward_for_each_atom():
    import numpy as np
    import sympy as sp
    from symbolic_kan.sum_product_kan import _DEFAULT_SYMBOLIC_LIBRARY
    x = torch.linspace(-1.5, 1.5, 121).unsqueeze(1)
    sx = sp.symbols("x")
    for name in _DEFAULT_SYMBOLIC_LIBRARY:
        model = SumProductKAN(
            in_dim=1, n_rules=1, max_factors=1, grid=4, k=2,
            symbolic_library=(name,), allow_self_products=False, seed=31,
        )
        with torch.no_grad():
            model.discretized.fill_(True)
            model.hard_rule_choice.fill_(True)
            model.hard_variable_choice[0, 0] = 0
            model.hard_operator_choice[0, 0, 0] = 0
            model.hard_spline_choice.zero_()
            model.symbolic_affine[0, 0, 0, 0] = torch.tensor([1.0, 1.3, 0.2, 0.0])
            model.rule_scale[0] = 0.7
            model.bias.fill_(0.2)
        expected = model(x).detach().squeeze().cpu().numpy()
        expr = model.symbolic_formula(["x"], digits=12, simplify=False)
        fn = sp.lambdify(sx, expr, modules="numpy")
        actual = np.asarray(fn(x.squeeze().cpu().numpy()), dtype=float)
        assert np.all(np.isfinite(actual)), name
        assert np.allclose(actual, expected, atol=2e-6, rtol=2e-6), name


def test_fully_symbolic_shell_reopens_numerically_pruned_masks():
    from symbolic_kan.sum_product_kan import _make_fully_symbolic_shell
    model = SumProductKAN(
        in_dim=2, n_rules=3, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "sin"), allow_self_products=False, seed=41,
    )
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    model.discretize(force_symbolic=False, freeze_gates=True)
    with torch.no_grad():
        model.rule_alive_mask[0] = False
        model.factor_alive_mask[1, 1] = False
    shell = _make_fully_symbolic_shell(model)
    assert bool(shell.rule_alive_mask.all())
    assert bool(shell.factor_alive_mask.all())
    assert not bool(shell.hard_rule_choice.any())


def test_fast_discrete_symbolic_forward_matches_dense_forward():
    model = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=2, grid=4, k=2,
        symbolic_library=("exp", "sin", "tanh"), allow_self_products=True, seed=43,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = torch.tensor([True, True])
        model.hard_spline_choice.zero_()
        model.hard_variable_choice.fill_(model.in_dim)
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_variable_choice[1, 0] = 1
        model.hard_operator_choice[0, 0, 0] = 0  # exp
        model.hard_operator_choice[0, 1, 1] = 1  # sin
        model.hard_operator_choice[1, 0, 1] = 2  # tanh
        model.symbolic_affine[0, 0, 0, 0] = torch.tensor([1.0, -0.7, 0.1, 0.0])
        model.symbolic_affine[0, 1, 1, 1] = torch.tensor([1.0, 2.2, -0.2, 0.0])
        model.symbolic_affine[1, 0, 1, 2] = torch.tensor([1.0, 1.5, 0.05, 0.0])
        model.rule_scale[:] = torch.tensor([0.8, -0.3])
        model.bias.fill_(0.12)
    x = torch.randn(64, 2)
    fast = model(x)
    dense, _ = model(x, return_details=True)
    assert torch.allclose(fast, dense, atol=1e-6, rtol=1e-6)


def test_drop_negligible_symbolic_rules_removes_tiny_term_without_hurting_fit():
    from symbolic_kan.sum_product_kan import drop_negligible_symbolic_rules
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x", "sin"), allow_self_products=False, seed=47,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.hard_spline_choice.zero_()
        model.hard_variable_choice[:, 0] = 0
        model.hard_operator_choice[0, 0, 0] = 0
        model.hard_operator_choice[1, 0, 0] = 1
        model.symbolic_affine[0, 0, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        model.symbolic_affine[1, 0, 0, 1] = torch.tensor([1.0, 1.3, 0.1, 0.0])
        model.rule_scale[:] = torch.tensor([1.0, 3.8e-5])
        model.bias.zero_()
    x = torch.linspace(-1.0, 1.0, 121).unsqueeze(1)
    y = x.clone()
    before = torch.sqrt(torch.mean((model(x) - y) ** 2)).item()
    cleaned, mse, removed = drop_negligible_symbolic_rules(
        model, x, y, current_mse=before**2,
        scale_threshold=1e-4, contribution_rmse_threshold=1e-4,
        relative_mse_budget=1.0, min_rules=1, verbose=False,
    )
    assert removed == [1]
    assert not bool(cleaned.hard_rule_choice[1])
    assert float(cleaned.rule_scale[1].detach()) == 0.0
    after = torch.sqrt(torch.mean((cleaned(x) - y) ** 2)).item()
    assert after <= 1e-7
    assert abs(after - mse**0.5) <= 1e-7
    formula = str(cleaned.symbolic_formula(["x"], simplify=False))
    assert "sin" not in formula


def test_rulekan_fast_rbf_edge_bank_forward_and_gradients_are_finite():
    from symbolic_kan.sum_product_kan import FastRBFEdgeBank
    bank = FastRBFEdgeBank(
        in_dim=3, out_dim=8, num=9, grid_range=(-2.0, 2.0),
        base_fun=torch.nn.Identity(), device="cpu",
    )
    x = torch.randn(64, 3, requires_grad=True)
    y, pre, post, basis = bank(x)
    assert y.shape == (64, 8)
    assert pre.shape == (64, 8, 3)
    assert post.shape == (64, 8, 3)
    assert basis.shape == (64, 8, 3)
    assert torch.isfinite(y).all()
    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert bank.coef.grad is not None and torch.isfinite(bank.coef.grad).all()
    assert bank.basis.log_denominator.grad is not None
    assert torch.isfinite(bank.basis.log_denominator.grad).all()


def test_sumproduct_rbf_basis_uses_same_rulekan_structure_api():
    from symbolic_kan.sum_product_kan import FastRBFEdgeBank
    model = SumProductKAN(
        in_dim=2, n_rules=4, max_factors=2, grid=7, k=3,
        numeric_basis="rbf", allow_self_products=False, seed=53,
    )
    assert model.numeric_basis == "rbf"
    assert isinstance(model.numeric_edges, FastRBFEdgeBank)
    x = torch.randn(48, 2)
    y, details = model(x, return_details=True)
    assert y.shape == (48, 1)
    assert details["factor_values"].shape[:3] == (48, 4, 2)
    assert torch.isfinite(y).all()


def test_rulekan_fast_rbf_refinement_preserves_function_on_training_points():
    from symbolic_kan.sum_product_kan import refine_sumproduct_numeric_grid
    model = SumProductKAN(
        in_dim=2, n_rules=3, max_factors=1, grid=6, k=3,
        numeric_basis="rbf", allow_self_products=False, seed=59,
    )
    x = torch.linspace(-1.5, 1.5, 180).unsqueeze(1)
    x = torch.cat([x, torch.sin(x)], dim=1)
    model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
    model.discretize(force_symbolic=False, freeze_gates=True)
    with torch.no_grad():
        before = model(x).clone()
    refined = refine_sumproduct_numeric_grid(model, x, 12)
    assert refined.numeric_basis == "rbf"
    assert refined.numeric_edges.num == 12
    with torch.no_grad():
        after = refined(x)
    # Least-squares transfer is approximate but should preserve the learned
    # low-resolution RBF functions closely on the transfer points.
    assert torch.sqrt(torch.mean((before - after) ** 2)).item() < 2e-3


def test_rule_contribution_redundancy_detects_duplicate_rules():
    from symbolic_kan.sum_product_kan import rule_contribution_redundancy
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=61,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.hard_spline_choice.zero_()
        model.hard_variable_choice[:, 0] = 0
        model.hard_operator_choice[:, 0, 0] = 0
        model.symbolic_affine[:, 0, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        model.rule_scale[:] = torch.tensor([1.0, -0.4])
        model.bias.zero_()
    x = torch.linspace(-1.0, 1.0, 121).unsqueeze(1)
    diag = rule_contribution_redundancy(model, x)
    assert diag["max_abs_corr"] > 0.9999
    assert diag["span_r2"][0] > 0.9999
    assert diag["span_r2"][1] > 0.9999


def test_orthogonal_rule_scale_refit_solves_joint_coefficients():
    from symbolic_kan.sum_product_kan import orthogonal_rule_scale_refit
    model = SumProductKAN(
        in_dim=2, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=67,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.hard_spline_choice.zero_()
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[1, 0] = 1
        model.hard_operator_choice[:, 0, :] = 0
        model.symbolic_affine[0, 0, 0, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        model.symbolic_affine[1, 0, 1, 0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        model.rule_scale[:] = 0.1
        model.bias.zero_()
    x = torch.randn(200, 2)
    y = 2.5 * x[:, [0]] - 0.75 * x[:, [1]] + 0.3
    refit, mse = orthogonal_rule_scale_refit(model, x[:140], y[:140], x[140:], y[140:])
    assert mse < 1e-10
    assert abs(float(refit.rule_scale[0].detach()) - 2.5) < 1e-4
    assert abs(float(refit.rule_scale[1].detach()) + 0.75) < 1e-4
    assert abs(float(refit.bias.detach()) - 0.3) < 1e-4


def test_contribution_graph_regularizer_penalizes_duplicate_contributions():
    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=71,
    )
    with torch.no_grad():
        model.rule_scale[:] = 1.0
        model.rule_gate.log_alpha.fill_(8.0)
    x = torch.randn(128, 1)
    _, det = model(x, return_details=True)
    # Make the regularizer test independent of the current random edge bank by
    # supplying exactly duplicated contribution columns.
    det = dict(det)
    base = torch.linspace(-1.0, 1.0, 128).unsqueeze(1)
    det["contributions"] = torch.cat([base, 2.0 * base], dim=1)
    reg, parts = model.regularization(
        det,
        SumProductRegularization(graph_redundancy=1.0, contribution_correlation_threshold=0.95),
    )
    assert float(parts["graph_redundancy"].detach()) > 0.1
    assert float(reg.detach()) > 0.1


def test_prune_to_stability_can_be_bounded_to_one_accepted_deletion():
    from symbolic_kan.sum_product_kan import prune_numeric_structure_to_stability
    model = SumProductKAN(
        in_dim=1, n_rules=3, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), seed=83,
    )
    # Make all rules numerically negligible so at least one deletion is safe.
    with torch.no_grad():
        model.rule_scale[:] = torch.tensor([1.0, 0.0, 0.0])
        model.rule_gate.log_alpha.fill_(8.0)
        model.set_hardening(structure=1.0, factor=1.0, rule=1.0, symbolic=0.0)
        model.discretize(force_symbolic=False, freeze_gates=True)
    x = torch.linspace(-1,1,80).unsqueeze(1)
    with torch.no_grad():
        y = model(x).detach()
    pruned, info = prune_numeric_structure_to_stability(
        model, x[:60], y[:60], x[60:], y[60:],
        refit_steps=0, stabilize_rounds=0, recovery_probe_count=1,
        max_candidates_per_pass=3, local_rel_mse_budget=1.0,
        global_rel_mse_budget=1.0, max_accepted_deletions=1,
        verbose=False, show_progress=False,
    )
    assert len(info["accepted"]) == 1


def test_gmp_screening_scales_with_library_size_and_preserves_explicit_overrides():
    from symbolic_kan.sum_product_kan import scaled_gmp_screening_sizes

    assert scaled_gmp_screening_sizes(11) == (3, 5, 4)
    assert scaled_gmp_screening_sizes(14) == (4, 7, 6)
    assert scaled_gmp_screening_sizes(26) == (8, 12, 10)
    # Explicit values are experimental overrides and must not be rescaled.
    assert scaled_gmp_screening_sizes(26, topk=3, unary_topk=5, self_product_topk=4) == (3, 5, 4)


def test_gmp_symbolic_rng_is_independent_of_global_torch_rng_state():
    from symbolic_kan.sum_product_kan import gmp_symbolic_operator_preselection

    x = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    y = (torch.sin(1.3 * x[:, 0]) + 0.2 * x[:, 0] ** 2).unsqueeze(1)
    kwargs = dict(
        structures=[(0,)],
        library=("x", "x^2", "sin", "cos", "exp"),
        steps=4,
        max_samples=64,
        seed=321,
        show_progress=False,
    )

    torch.manual_seed(1)
    _ = torch.randn(1000)  # consume unrelated global RNG state
    a, sa = gmp_symbolic_operator_preselection(x, y, **kwargs)

    torch.manual_seed(999)
    _ = torch.randn(17)  # consume a different amount of global RNG state
    b, sb = gmp_symbolic_operator_preselection(x, y, **kwargs)

    assert sa == sb
    assert [c["name"] for c in a[0]["factor_shortlists"][0]] == [
        c["name"] for c in b[0]["factor_shortlists"][0]
    ]
    pa = torch.tensor([c["gmp_prob"] for c in a[0]["factor_shortlists"][0]])
    pb = torch.tensor([c["gmp_prob"] for c in b[0]["factor_shortlists"][0]])
    assert torch.equal(pa, pb)


def test_gmp_structure_rng_is_invariant_to_structure_order():
    from symbolic_kan.sum_product_kan import gmp_symbolic_operator_preselection

    x0 = torch.linspace(-1.0, 1.0, 48)
    x = torch.stack([x0, torch.cos(1.7 * x0)], dim=1)
    y = (torch.sin(1.4 * x[:, 0]) + 0.3 * x[:, 1] ** 2).unsqueeze(1)
    kwargs = dict(
        library=("x", "x^2", "sin", "cos", "exp"),
        steps=4,
        max_samples=48,
        seed=777,
        show_progress=False,
    )
    a, _ = gmp_symbolic_operator_preselection(x, y, structures=[(0,), (1,)], **kwargs)
    b, _ = gmp_symbolic_operator_preselection(x, y, structures=[(1,), (0,)], **kwargs)
    amap = {tuple(z["structure"]): z for z in a}
    bmap = {tuple(z["structure"]): z for z in b}
    for key in ((0,), (1,)):
        an = [c["name"] for c in amap[key]["factor_shortlists"][0]]
        bn = [c["name"] for c in bmap[key]["factor_shortlists"][0]]
        assert an == bn
        ap = torch.tensor([c["gmp_prob"] for c in amap[key]["factor_shortlists"][0]])
        bp = torch.tensor([c["gmp_prob"] for c in bmap[key]["factor_shortlists"][0]])
        assert torch.equal(ap, bp)


def test_gmp_auto_policy_is_structure_and_budget_aware():
    from symbolic_kan.sum_product_kan import _resolve_gmp_local_policy

    assert _resolve_gmp_local_policy(
        (0, 1), steps=25, identity_chart="auto", atom_backward_normalization="auto"
    ) == ("data_dual", "none")
    assert _resolve_gmp_local_policy(
        (0, 1), steps=70, identity_chart="auto", atom_backward_normalization="auto"
    ) == ("data_dual", "rms")
    assert _resolve_gmp_local_policy(
        (0, 0), steps=70, identity_chart="auto", atom_backward_normalization="auto"
    ) == ("raw", "none")
    assert _resolve_gmp_local_policy(
        (0,), steps=70, identity_chart="auto", atom_backward_normalization="auto"
    ) == ("raw", "none")


def test_gmp_auto_raw_repeated_product_matches_explicit_raw_initialization():
    from symbolic_kan.sum_product_kan import gmp_symbolic_operator_preselection

    x = torch.linspace(-1.5, 1.5, 128).unsqueeze(1)
    y = 0.4 * torch.exp(-0.65 * x) * torch.sin(2.35 * x)
    kwargs = dict(
        structures=[(0, 0)],
        library=("x", "exp", "sin", "cos", "tanh"),
        topk=2, steps=12, lr=2e-2,
        temperature_start=1.5, temperature_end=0.25,
        entropy_weight=2e-4, show_progress=False, seed=7,
    )
    auto, _ = gmp_symbolic_operator_preselection(
        x, y, identity_chart="auto", atom_backward_normalization="auto", **kwargs
    )
    raw, _ = gmp_symbolic_operator_preselection(
        x, y, identity_chart="raw", atom_backward_normalization="none", **kwargs
    )
    assert len(auto) == len(raw) == 1
    for asl, rsl in zip(auto[0]["factor_shortlists"], raw[0]["factor_shortlists"]):
        assert [c["name"] for c in asl] == [c["name"] for c in rsl]
        ap = torch.tensor([c["gmp_prob"] for c in asl])
        rp = torch.tensor([c["gmp_prob"] for c in rsl])
        assert torch.equal(ap, rp)
        aa = torch.tensor([c["affine"] for c in asl])
        ra = torch.tensor([c["affine"] for c in rsl])
        assert torch.equal(aa, ra)


def test_gmp_data_dual_is_latent_and_collapsed_before_topk():
    from symbolic_kan.sum_product_kan import gmp_symbolic_operator_preselection

    torch.manual_seed(0)
    x = torch.randn(96, 2)
    y = (0.4 * x[:, 0] * torch.exp(-0.3 * x[:, 1])).reshape(-1, 1)
    result, stats = gmp_symbolic_operator_preselection(
        x, y,
        structures=[(0, 1)],
        library=("x", "sin", "exp", "tanh"),
        steps=2,
        topk=4,
        identity_chart="data_dual",
        atom_backward_normalization="none",
        seed=3,
        max_samples=96,
    )
    assert stats["library_size"] == 4.0
    assert stats["latent_chart_count"] == 5.0
    assert len(result) == 1
    for shortlist in result[0]["factor_shortlists"]:
        names = [cand["name"] for cand in shortlist]
        assert len(names) == len(set(names))
        assert names.count("x") <= 1


def test_identity_canonicalization_folds_output_affine_exactly():
    from symbolic_kan.sum_product_kan import _canonical_symbolic_affine

    # a * (b*x + c) + d == (a*b)*x + (a*c+d).  The old symbolic
    # installation path discarded a,d and therefore warped fitted fuzzy gates.
    got = _canonical_symbolic_affine((2.0, 3.0, 4.0, 5.0), "x")
    assert got == (1.0, 6.0, 13.0, 0.0)
    # Non-affine atoms intentionally remain amplitude/offset canonicalized.
    got_sin = _canonical_symbolic_affine((2.0, 3.0, 4.0, 5.0), "sin")
    assert got_sin == (1.0, 3.0, 4.0, 0.0)


def test_interaction_shape_screen_preserves_fuzzy_branch_families():
    from benchmarks.models import TARGET_CORE_SYMBOLIC_LIBRARY
    from benchmarks.specs import TASKS, make_synthetic_data
    from symbolic_kan.sum_product_kan import interaction_shape_symbolic_shortlists

    data = make_synthetic_data(TASKS["fuzzy_ite_cross"], seed=0, train_n=1200, val_n=200, test_n=200)
    else_shape = interaction_shape_symbolic_shortlists(
        data.train_x, data.train_y, (0, 1), TARGET_CORE_SYMBOLIC_LIBRARY, bins=8, topk=4
    )
    if_shape = interaction_shape_symbolic_shortlists(
        data.train_x, data.train_y, (0, 2), TARGET_CORE_SYMBOLIC_LIBRARY, bins=8, topk=4
    )
    assert else_shape is not None and if_shape is not None
    assert float(else_shape[0][0]["interaction_rank1_fraction"]) > 0.6
    assert float(if_shape[0][0]["interaction_rank1_fraction"]) > 0.9
    assert {z["name"] for z in else_shape[1]} & {"sin", "cos"}
    assert "exp" in {z["name"] for z in if_shape[1]}


def test_tiny_symbolic_cleanup_accepts_absolute_target_tolerance():
    import torch
    from symbolic_kan.sum_product_kan import SumProductKAN, drop_negligible_symbolic_rules

    model = SumProductKAN(
        in_dim=1, n_rules=2, max_factors=1, grid=4, k=2,
        symbolic_library=("x",), min_order=1, seed=0,
    )
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.rule_alive_mask[:] = True
        model.factor_alive_mask[:] = True
        model.hard_spline_choice[:] = False
        model.hard_variable_choice[:, 0] = 0
        model.hard_operator_choice[:, 0, 0] = 0
        model.symbolic_affine[..., 0].fill_(1.0)
        model.symbolic_affine[..., 1].fill_(1.0)
        model.symbolic_affine[..., 2].zero_()
        model.symbolic_affine[..., 3].zero_()
        model.rule_scale[0] = 1.0
        model.rule_scale[1] = 1e-8
        model.bias.zero_()
    x = torch.linspace(-1.0, 1.0, 101).unsqueeze(1)
    with torch.no_grad():
        y = model(x).clone()
        mse = float(torch.mean((model(x) - y) ** 2))
    cleaned, _, removed = drop_negligible_symbolic_rules(
        model, x, y, current_mse=mse,
        scale_threshold=1e-4, contribution_rmse_threshold=1e-5,
        relative_mse_budget=0.0, absolute_rmse_budget=1e-6,
    )
    assert removed == [1]
    assert int(cleaned.hard_rule_choice.sum()) == 1
    with torch.no_grad():
        rmse = float(torch.sqrt(torch.mean((cleaned(x) - y) ** 2)))
    assert rmse < 1e-6


def test_recursive_partition_rescue_recovers_depth2_fuzzy_tree():
    import torch
    from symbolic_kan.sum_product_kan import (
        SumProductKAN, _make_fully_symbolic_shell,
        recursive_partition_symbolic_rescue,
    )

    g = torch.Generator().manual_seed(7)
    ntr, nv = 220, 100
    raw = torch.empty(ntr + nv, 5)
    raw[:, 0:2] = torch.rand(ntr + nv, 2, generator=g)
    raw[:, 2:5] = -1.5 + 3.0 * torch.rand(ntr + nv, 3, generator=g)
    mean = raw[:ntr].mean(0)
    std = raw[:ntr].std(0).clamp_min(1e-6)
    z = (raw - mean) / std

    def target(x):
        g0, g1, a, b, c = [x[:, j] for j in range(5)]
        inner = (1-g1) * (0.60 * torch.cos(1.7*a)) + g1 * (0.90 * torch.exp(-0.80*b))
        return ((1-g0) * inner + g0 * (0.50 * torch.tanh(1.9*c))).unsqueeze(1)

    y = target(raw)
    tx, vx = z[:ntr], z[ntr:]
    ty, vy = y[:ntr], y[ntr:]
    lib = ("x", "cos", "exp", "tanh")
    numeric = SumProductKAN(
        in_dim=5, n_rules=6, max_factors=3, grid=4, k=2,
        symbolic_library=lib, seed=0,
    )
    numeric.discretized.fill_(True)
    incumbent = _make_fully_symbolic_shell(numeric)
    with torch.no_grad():
        incumbent.bias.fill_(float(ty.mean()))

    rescued, meta = recursive_partition_symbolic_rescue(
        numeric, incumbent, tx, ty, vx, vy,
        input_mean=mean, input_std=std,
        allowed_supports=((0,1,2), (0,1,3), (0,4)),
        library=lib,
        leaf_topk=4, coarse_topk=8, max_samples=ntr,
        shallow_topk=6, shallow_steps=80, shallow_lbfgs_steps=15,
        deep_topk=3, deep_steps=260, deep_lbfgs_steps=60,
        min_improvement_rel=1e-4,
    )
    assert meta["selected"] is True
    assert meta["outer_gate"] == 0
    assert meta["inner_gate"] == 1
    assert set(meta["leaf_operators"]) == {"cos", "exp", "tanh"}
    assert {tuple(z) for z in meta["supports"]} == {(0,1,2), (0,1,3), (0,4)}
    with torch.no_grad():
        nrmse = torch.sqrt(torch.mean((rescued(vx)-vy)**2)) / vy.std()
    assert float(nrmse) < 2e-3


def test_complementary_two_rule_rescue_recovers_same_variable_fuzzy_rule():
    import torch
    from symbolic_kan.sum_product_kan import (
        SumProductKAN, _make_fully_symbolic_shell,
        complementary_two_rule_symbolic_rescue,
    )

    g = torch.Generator().manual_seed(19)
    ntr, nv = 180, 90
    raw = torch.rand(ntr + nv, 1, generator=g)
    mean = raw[:ntr].mean(0)
    std = raw[:ntr].std(0).clamp_min(1e-6)
    z = (raw - mean) / std
    x = raw[:, 0]
    y = ((1.0 - x) * (0.65 * torch.sin(2.4 * x))
         + x * (0.85 * torch.tanh(2.0 * x))).unsqueeze(1)
    tx, vx = z[:ntr], z[ntr:]
    ty, vy = y[:ntr], y[ntr:]

    lib = ("x", "sin", "tanh", "sqrt")
    numeric = SumProductKAN(
        in_dim=1, n_rules=4, max_factors=2, grid=4, k=2,
        symbolic_library=lib, seed=0,
    )
    numeric.discretized.fill_(True)
    incumbent = _make_fully_symbolic_shell(numeric)
    with torch.no_grad():
        incumbent.bias.fill_(float(ty.mean()))

    rescued, meta = complementary_two_rule_symbolic_rescue(
        numeric, incumbent, tx, ty, vx, vy,
        input_mean=mean, input_std=std,
        structure_candidates=((0,), (0, 0)),
        allowed_supports=((0,),),
        library=lib,
        max_input_dim=2, max_samples=ntr,
        seed_restarts=3, local_polish_steps=24, local_polish_lr=1e-2,
        family_final_topk=8, final_steps=180, final_lbfgs_steps=40,
        min_improvement_rel=1e-4,
    )
    assert meta["attempted"] is True
    assert meta["selected"] is True
    assert meta["gate_variable"] == 0
    assert {meta["operator_a"], meta["operator_b"]} == {"sin", "tanh"}
    assert int(meta["families_screened"]) >= len(lib) ** 2
    assert int(meta["families_joint_refined"]) >= len(lib)
    with torch.no_grad():
        nrmse = torch.sqrt(torch.mean((rescued(vx) - vy) ** 2)) / vy.std()
    assert float(nrmse) < 2e-3
