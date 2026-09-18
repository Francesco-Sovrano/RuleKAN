import inspect

import torch

from benchmarks.anfis import CompactANFIS, fit_compact_anfis
from benchmarks.models import model_supports_task, resolve_shared_benchmark_config, train_anfis
from benchmarks.specs import TASKS, make_synthetic_data


def test_anfis_hybrid_fit_is_train_validation_only_and_predictive():
    data = make_synthetic_data(TASKS["fuzzy_ite_cross"], seed=7, train_n=180, val_n=60, test_n=80)
    model = CompactANFIS(in_dim=data.train_x.shape[1], n_rules=8, device="cpu")
    model.initialize_from_data(data.train_x, seed=7, n_init=3)
    result = fit_compact_anfis(
        model, data.train_x, data.train_y, data.val_x, data.val_y,
        epochs=40, lr=2e-2, ridge=1e-4, patience=12,
    )
    with torch.no_grad():
        pred = result.model(data.test_x)
        nrmse = torch.mean((pred - data.test_y) ** 2).sqrt() / data.test_y.std().clamp_min(1e-12)
    assert torch.isfinite(pred).all()
    assert float(nrmse) < 0.15
    assert "test_x" not in inspect.signature(fit_compact_anfis).parameters
    assert "test_y" not in inspect.signature(fit_compact_anfis).parameters


def test_anfis_benchmark_adapter_and_shared_capacity():
    spec = TASKS["fuzzy_ite_cross"]
    shared = {
        "enabled": True,
        "capacity": {"policy": "fixed", "width": 6, "mult_units": 2},
        "max_product_order": "task",
        "grid": 4,
        "symbolic_library": "target_core",
    }
    cfg, meta = resolve_shared_benchmark_config("anfis", spec, {}, shared)
    assert cfg["anfis_rules"] == 6
    assert meta["shared_capacity_width"] == 6

    data = make_synthetic_data(spec, seed=3, train_n=100, val_n=30, test_n=40)
    run = train_anfis(spec, data, 3, {"anfis_rules": 5, "epochs": 20, "patience": 8, "kmeans_n_init": 2})
    assert run.model_name == "anfis"
    assert run.extras["fuzzy_system"] == "first_order_tsk_anfis"
    assert run.extras["anfis_rules"] == 5
    assert torch.isfinite(torch.tensor(run.metrics["test_nrmse"]))


def test_anfis_and_power_rulekan_are_not_scheduled_on_binary_tasks():
    binary = TASKS["sklearn_breast_cancer"]
    assert not model_supports_task("anfis", binary)
    assert not model_supports_task("power_rulekan", binary)
