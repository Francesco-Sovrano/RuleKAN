import math
import torch
from torch import nn

from benchmarks.specs import TASKS, make_synthetic_data
from symbolic_kan import SumProductKAN
from symbolic_kan.composition_rulekan import ComposedRuleKAN, depth2_composition_rescue


class _Zero(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)


def _numeric_shell(in_dim: int, seed: int = 0) -> SumProductKAN:
    model = SumProductKAN(
        in_dim=in_dim, n_rules=6, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "x^2", "exp", "sin", "cos", "tanh", "sqrt", "log1p_sq", "sqrt1p_sq", "inv1p_sq"),
        min_order=1, seed=seed,
    )
    model.discretize(force_symbolic=False, freeze_gates=True)
    return model


def test_depth2_composition_rescue_recovers_tanh_of_sin():
    spec = TASKS["nested_same_tanh_sin"]
    data = make_synthetic_data(spec, seed=0, train_n=260, val_n=90, test_n=100)
    num = _numeric_shell(1)
    result = depth2_composition_rescue(
        _Zero(), num,
        data.train_x, data.train_y, data.val_x, data.val_y,
        allowed_supports=[(0,)],
        library=num.symbolic_library,
        trigger_nrmse=0.0,
        max_families=80,
        shallow_steps=55,
        shallow_lbfgs_steps=12,
        deep_topk=5,
        deep_steps=150,
        deep_lbfgs_steps=50,
        correction_topk=0,
        seed=7,
    )
    assert result.selected
    assert isinstance(result.model, ComposedRuleKAN)
    assert result.metadata["composition_support"] == [0]
    with torch.no_grad():
        nrmse = float(torch.mean((result.model(data.test_x) - data.test_y) ** 2).sqrt() / data.test_y.std())
    assert nrmse < 2e-3


def test_depth2_composition_rescue_is_validation_gated_for_good_incumbent():
    spec = TASKS["nested_same_tanh_sin"]
    data = make_synthetic_data(spec, seed=1, train_n=80, val_n=30, test_n=30)

    class Exact(nn.Module):
        def forward(self, x):
            raw = x[:, 0:1] * data.input_std[0] + data.input_mean[0]
            return torch.tanh(1.6 * torch.sin(2.2 * raw))

    incumbent = Exact()
    num = _numeric_shell(1, seed=1)
    result = depth2_composition_rescue(
        incumbent, num,
        data.train_x, data.train_y, data.val_x, data.val_y,
        allowed_supports=[(0,)],
        library=num.symbolic_library,
        trigger_nrmse=1e-3,
        seed=9,
    )
    assert not result.selected
    assert result.model is incumbent
    assert result.metadata["reason"] == "incumbent_below_trigger"


def test_depth2_composition_rescue_recovers_nested_product_plus_flat_correction():
    spec = TASKS["nested_cross_sin_product"]
    data = make_synthetic_data(spec, seed=0, train_n=260, val_n=90, test_n=100)
    num = _numeric_shell(3)
    result = depth2_composition_rescue(
        _Zero(), num,
        data.train_x, data.train_y, data.val_x, data.val_y,
        allowed_supports=[(0, 1), (2,)],
        library=num.symbolic_library,
        trigger_nrmse=0.0,
        max_families=120,
        shallow_steps=45,
        shallow_lbfgs_steps=10,
        deep_topk=6,
        deep_steps=120,
        deep_lbfgs_steps=35,
        correction_topk=8,
        seed=5,
    )
    assert result.selected
    assert isinstance(result.model, ComposedRuleKAN)
    assert result.metadata["pattern_kind"] == "prod2"
    assert result.metadata["has_flat_correction"] is True
    assert result.metadata["correction_variable"] == 2
    assert result.metadata["correction_operator"] == "x^2"
    with torch.no_grad():
        nrmse = float(torch.mean((result.model(data.test_x) - data.test_y) ** 2).sqrt() / data.test_y.std())
    assert nrmse < 5e-4


def test_depth2_composition_rescue_recovers_nested_cross_oscillator_with_periodic_seed():
    spec = TASKS["nested_cross_oscillator"]
    data = make_synthetic_data(spec, seed=0, train_n=220, val_n=80, test_n=120)
    num = SumProductKAN(
        in_dim=2, n_rules=6, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "sin", "cos"), min_order=1, seed=0,
    )
    num.discretize(force_symbolic=False, freeze_gates=True)
    result = depth2_composition_rescue(
        _Zero(), num,
        data.train_x, data.train_y, data.val_x, data.val_y,
        allowed_supports=[(0,), (1,), (0, 1)],
        library=num.symbolic_library,
        trigger_nrmse=0.0,
        coarse_global_topk=12,
        coarse_family_topk=3,
        max_families=90,
        shallow_steps=40,
        shallow_lbfgs_steps=10,
        deep_topk=5,
        deep_steps=130,
        deep_lbfgs_steps=40,
        correction_topk=4,
        seed=17,
    )
    assert result.selected
    assert result.metadata["pattern_kind"] == "sum2"
    assert set(result.metadata["composition_support"]) == {0, 1}
    with torch.no_grad():
        nrmse = float(torch.mean((result.model(data.test_x) - data.test_y) ** 2).sqrt() / data.test_y.std())
    assert nrmse < 5e-4
