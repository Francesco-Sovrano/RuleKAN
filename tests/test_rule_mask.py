import tempfile

import torch

from symbolic_kan import KAN, RuleMaskProduct


def test_rule_mask_product_forward_and_backward():
    layer = RuleMaskProduct(in_dim=3, out_dim=2, init_prob=0.2, min_order=1)
    layer.set_hard_mask(torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.float32))

    edge_values = torch.tensor([[[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]])
    out, hard = layer(edge_values, return_mask=True)

    assert torch.equal(hard, torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    assert torch.allclose(out, torch.tensor([[6.0, 143.0]]))

    out.sum().backward()
    assert layer.logits.grad is not None
    assert torch.isfinite(layer.logits.grad).all()


def test_rule_mask_uses_neutral_identity_for_unselected_edges():
    layer = RuleMaskProduct(in_dim=3, out_dim=1, init_prob=0.2, min_order=1)
    layer.set_hard_mask(torch.tensor([[0, 1, 0]], dtype=torch.float32))
    edge_values = torch.tensor([[[100.0, 4.0, -20.0]]])
    out = layer(edge_values)
    assert torch.allclose(out, torch.tensor([[4.0]]))


def test_multkan_rule_mask_forward_and_symbolic_formula():
    model = KAN(
        width=[2, [1, 0], 1],
        grid=3,
        k=2,
        seed=0,
        auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={"init_prob": 0.3, "min_order": 1, "max_order": 2},
    )
    model.set_rule_mask(0, torch.tensor([[1.0, 1.0]]), freeze=True)

    model.fix_symbolic(0, 0, 0, "x", verbose=False, log_history=False)
    model.fix_symbolic(0, 1, 0, "x", verbose=False, log_history=False)
    model.fix_symbolic(1, 0, 0, "x", verbose=False, log_history=False)

    formula, _ = model.symbolic_formula(simplify=True, compact=False)
    assert str(formula[0]) in {"1.0*x_1*x_2", "1.0*x_2*x_1", "x_1*x_2", "x_2*x_1"}

    x = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    y = model(x)
    assert y.shape == (2, 1)


def test_rule_mask_checkpoint_roundtrip():
    model = KAN(
        width=[2, [2, 0], 1],
        grid=3,
        k=2,
        seed=0,
        auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={"init_prob": 0.3, "min_order": 1, "max_order": 2},
    )
    x = torch.randn(5, 2)
    y = model(x).detach()

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/ckpt"
        model.saveckpt(path)
        loaded = KAN.loadckpt(path)
        y_loaded = loaded(x).detach()

    assert loaded.rule_mask_layers == [True, False]
    assert torch.allclose(y, y_loaded)


def test_rule_mask_exact_order_bank_and_slot_constraint():
    layer = RuleMaskProduct(
        in_dim=5,
        out_dim=3,
        init_prob=0.5,
        min_order=1,
        max_order=3,
        rule_orders=[1, 2, 3],
    )
    hard = layer.hard_mask(stochastic=False)
    assert torch.equal(hard.sum(dim=-1), torch.tensor([1.0, 2.0, 3.0]))

    p = layer.slot_probabilities()
    identity = 5
    # Exact-order rules force all slots above their order to identity.
    assert torch.allclose(p[0, 1:, identity], torch.ones(2))
    assert torch.allclose(p[1, 2:, identity], torch.ones(1))
    # Required slots cannot choose identity.
    assert p[0, 0, identity] == 0
    assert torch.all(p[1, :2, identity] == 0)
    assert torch.all(p[2, :3, identity] == 0)


def test_rule_mask_repeated_node_pruning_and_checkpoint():
    """RuleMask metadata must track compact KANs across prune/refit rounds."""
    model = KAN(
        width=[3, [4, 0], 1],
        grid=3,
        k=2,
        seed=0,
        auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={
            "init_prob": 0.3,
            "min_order": 1,
            "max_order": 2,
            "rule_orders": [1, 2, 1, 2],
        },
    )
    x = torch.randn(32, 3)
    model.get_act(x)

    pruned = model.prune_node(
        mode="manual",
        active_neurons_id=[[0, 2]],
        log_history=False,
    )
    assert pruned.width[1] == [2, 0]
    assert pruned.rule_masks[0].rule_orders.tolist() == [1, 1]
    assert pruned(x).shape == (32, 1)

    # A second compacting pass used to fail because rule_mask_config still
    # contained the original four rule orders.
    pruned.get_act(x)
    pruned2 = pruned.prune_node(
        mode="manual",
        active_neurons_id=[[1]],
        log_history=False,
    )
    assert pruned2.width[1] == [1, 0]
    assert pruned2.rule_masks[0].rule_orders.tolist() == [1]
    y = pruned2(x).detach()

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/pruned"
        pruned2.saveckpt(path)
        loaded = KAN.loadckpt(path)
        y_loaded = loaded(x).detach()

    assert loaded.width[1] == [1, 0]
    assert loaded.rule_masks[0].rule_orders.tolist() == [1]
    assert torch.allclose(y, y_loaded)


def test_create_dataset_preserves_normalization_metadata():
    from symbolic_kan import create_dataset

    f = lambda x: (x[:, [0]] + 2.0 * x[:, [1]])
    data = create_dataset(
        f,
        n_var=2,
        ranges=[[-2.0, 1.0], [3.0, 7.0]],
        train_num=128,
        test_num=32,
        normalize_input=True,
        seed=4,
    )
    assert "input_mean" in data and "input_std" in data
    assert data["input_mean"].shape == (2,)
    assert data["input_std"].shape == (2,)
    assert torch.allclose(data["train_input"].mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(data["train_input"].std(dim=0), torch.ones(2), atol=1e-5)


def test_gmp_progressive_topk_and_discretize():
    """GMP shortlist pruning must be monotone and discretisation must be exact."""
    layer = RuleMaskProduct(
        in_dim=5,
        out_dim=2,
        init_prob=0.6,
        min_order=1,
        max_order=2,
        rule_orders=[1, 2],
        gate_mode="gmp",
        asinh_scale=3.0,
    )

    layer.prune_candidates_topk(3)
    assert torch.all(layer.candidate_mask[..., :5].sum(dim=-1) <= 3)
    before = layer.candidate_mask.clone()

    layer.prune_candidates_topk(2)
    assert torch.all(layer.candidate_mask[..., :5].sum(dim=-1) <= 2)
    # Progressive pruning must never resurrect a previously removed candidate.
    assert torch.all(~layer.candidate_mask | before)

    x = torch.randn(16, 2, 5)
    soft_y = layer(x)
    assert torch.isfinite(soft_y).all()

    mask = layer.discretize(freeze=True)
    assert bool(layer.discretized.item())
    assert not layer.logits.requires_grad
    assert torch.equal(mask.sum(dim=-1), torch.tensor([1.0, 2.0]))
    # Every slot has exactly one surviving categorical choice after discretisation.
    assert torch.all(layer.candidate_mask.sum(dim=-1) == 1)
    hard_y = layer(x)
    assert torch.isfinite(hard_y).all()


def test_gmp_checkpoint_preserves_shortlist_and_state():
    model = KAN(
        width=[3, [3, 0], 1],
        grid=3,
        k=2,
        seed=0,
        auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={
            "init_prob": 0.4,
            "min_order": 1,
            "max_order": 2,
            "rule_orders": [1, 2, 2],
            "gate_mode": "gmp",
            "asinh_scale": 2.5,
        },
    )
    rm = model.rule_masks[0]
    rm.prune_candidates_topk(2)
    rm.discretize(freeze=True)
    x = torch.randn(12, 3)
    y = model(x).detach()
    mask = rm.candidate_mask.detach().clone()

    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/gmp"
        model.saveckpt(path)
        loaded = KAN.loadckpt(path)
        y_loaded = loaded(x).detach()

    loaded_rm = loaded.rule_masks[0]
    assert loaded_rm.gate_mode == "gmp"
    assert bool(loaded_rm.discretized.item())
    assert torch.equal(mask, loaded_rm.candidate_mask)
    assert torch.allclose(y, y_loaded)

def test_hard_st_discretize_is_function_preserving():
    """Freezing an already-hard RuleMask must not change its numerical function."""
    layer = RuleMaskProduct(
        in_dim=4,
        out_dim=3,
        init_prob=0.5,
        min_order=1,
        max_order=2,
        rule_orders=[1, 2, 2],
        gate_mode="hard_st",
        stochastic=False,
    )
    x = torch.randn(32, 3, 4)
    with torch.no_grad():
        y_before = layer(x).clone()
        hard_before = layer.hard_mask(stochastic=False).clone()
        hard_after = layer.discretize(freeze=True)
        y_after = layer(x).clone()

    assert torch.equal(hard_before, hard_after)
    assert torch.allclose(y_before, y_after, atol=1e-7, rtol=1e-7)

def test_hard_st_does_not_add_unused_asinh_optimizer_parameter():
    hard = RuleMaskProduct(3, 2, gate_mode="hard_st", max_order=2)
    gmp = RuleMaskProduct(3, 2, gate_mode="gmp", max_order=2)
    assert "log_asinh_scale" not in dict(hard.named_parameters())
    assert "log_asinh_scale" in dict(gmp.named_parameters())


def test_fit_supports_best_validation_checkpoint_and_cosine_schedule():
    """Stable polishing options should run with a frozen RuleMask."""
    model = KAN(
        width=[2, [2, 0], 1], grid=3, k=2, seed=2, auto_save=False,
        rule_mask_layers=[True, False],
        rule_mask_config={
            "init_prob": 0.4, "min_order": 1, "max_order": 2,
            "rule_orders": [1, 2], "gate_mode": "hard_st",
        },
    )
    model.rule_masks[0].discretize(freeze=True)
    x = torch.randn(64, 2)
    y = (x[:, [0]] * x[:, [1]])
    data = {
        "train_input": x[:48], "train_label": y[:48],
        "test_input": x[48:], "test_label": y[48:],
    }
    res = model.fit(
        data, optimizer="Adam", steps=8, lr=1e-3, log=1000,
        validation_data=(x[48:], y[48:]), restore_best=True,
        validation_check_every=2, lr_schedule="cosine", min_lr=1e-4,
        grad_clip=0.5,
    )
    vals = [v for v in res["validation_loss"] if v is not None]
    assert vals
    with torch.no_grad():
        pred = model(x[48:])
    assert torch.isfinite(pred).all()


def test_multkan_unary_multiplication_node_is_well_defined():
    """A homogeneous multiplication node of arity one is an identity node."""
    model = KAN(
        width=[1, [0, 1], 1],
        grid=3,
        k=2,
        seed=0,
        auto_save=False,
        mult_arity=1,
    )
    x = torch.randn(16, 1)
    y = model(x)
    assert y.shape == (16, 1)
    assert torch.isfinite(y).all()


def test_auto_prune_keeps_one_hidden_node_when_threshold_removes_everything():
    model = KAN(width=[2, [2, 0], 1], grid=3, k=2, seed=0, auto_save=False)
    x = torch.randn(24, 2)
    model.get_act(x)
    pruned = model.prune_node(threshold=1e9, mode="auto", log_history=False)
    assert pruned.width[1] == [1, 0]
    assert torch.isfinite(pruned(x)).all()


def test_prune_rolls_back_if_edge_threshold_would_empty_graph():
    model = KAN(width=[2, [2, 0], 1], grid=3, k=2, seed=0, auto_save=False)
    x = torch.randn(24, 2)
    model.get_act(x)
    masks = [layer.mask.detach().clone() for layer in model.act_fun]
    pruned = model.prune(node_th=0, edge_th=1e9)
    assert pruned is model
    assert pruned.n_edge > 0
    for before, layer in zip(masks, model.act_fun):
        assert torch.equal(before, layer.mask)
    assert torch.isfinite(pruned(x)).all()
