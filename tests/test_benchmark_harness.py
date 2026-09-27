from pathlib import Path
import json

import pandas as pd

from benchmarks.aggregate import aggregate
from benchmarks.specs import TASKS, list_suites, make_synthetic_data


def test_capability_registry_has_products_nesting_and_kan_tracks():
    suites = list_suites()
    assert "synthetic_core" in suites
    assert "nested_stress" in suites
    assert "kan_canonical" in suites
    assert "feynman" in suites
    assert "real_small" in suites
    assert TASKS["mixed_rank4"].expected_structures
    assert TASKS["nested_cross_sin_product"].representability == "nested_cross"
    assert TASKS["pykan_exp_sin_square"].source == "pykan_example"


def test_synthetic_split_is_reproducible_and_normalized():
    spec = TASKS["same_var_exp_sin"]
    a = make_synthetic_data(spec, seed=3, train_n=80, val_n=20, test_n=30)
    b = make_synthetic_data(spec, seed=3, train_n=80, val_n=20, test_n=30)
    assert a.train_x.shape == (80, 1)
    assert (a.train_x - b.train_x).abs().max().item() == 0.0
    assert abs(float(a.train_x.mean())) < 1e-5


def test_aggregator_writes_pdf_only_figures(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    figures = tmp_path / "figures"
    figures.mkdir()
    # Stale raster artifacts are removed: benchmark figures are vector PDF only.
    (figures / "stale.png").write_bytes(b"not an image")
    rows = [
        {"status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression","model":"rulekan","seed":0,"test_rmse":0.01,"test_nrmse":0.02,"test_r2":0.99,"elapsed_seconds":1.2,"symbolic_seconds":0.7,"parameters":100,"finished_unix":1.0,"symbolic_test_rmse":0.015,"symbolic_test_nrmse":0.03},
        {"status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression","model":"autosym","paper_pipeline":"baseline","seed":0,"test_rmse":0.04,"test_nrmse":0.08,"numeric_test_rmse":0.02,"numeric_test_nrmse":0.04,"elapsed_seconds":1.4,"symbolic_seconds":0.8,"parameters":120,"finished_unix":2.0},
        {"status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression","model":"mlp","seed":0,"test_rmse":0.02,"test_nrmse":0.04,"test_r2":0.96,"elapsed_seconds":0.5,"parameters":200,"finished_unix":3.0},
    ]
    for i, row in enumerate(rows):
        (runs / f"r{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "symbolic_runs.csv").exists()
    assert (tmp_path / "symbolic_summary.csv").exists()
    assert (tmp_path / "summary.md").exists()
    pdf_names = {p.name for p in (tmp_path / "figures").glob("*.pdf")}
    assert "symbolic_rmse_by_task.pdf" in pdf_names
    assert "symbolic_runtime_by_task.pdf" in pdf_names
    assert "symbolic_accuracy_runtime.pdf" in pdf_names
    assert "numeric_vs_symbolic_rmse.pdf" in pdf_names
    assert not list((tmp_path / "figures").glob("*.png"))
    assert not list((tmp_path / "figures").glob("*.jpg"))

    symbolic = pd.read_csv(tmp_path / "symbolic_runs.csv")
    rk = symbolic[symbolic.model.eq("rulekan")].iloc[0]
    au = symbolic[symbolic.model.eq("autosym")].iloc[0]
    # Stage reconciliation: RuleKAN's explicit symbolic field is used, whereas
    # paper pipelines store the final symbolic model in test_nrmse.
    assert abs(rk.symbolic_nrmse - 0.03) < 1e-12
    assert abs(rk.numeric_nrmse - 0.02) < 1e-12
    assert abs(au.symbolic_nrmse - 0.08) < 1e-12
    assert abs(au.numeric_nrmse - 0.04) < 1e-12
    # Numeric-only MLP is intentionally excluded from symbolic-regression plots.
    assert "mlp" not in set(symbolic.model)


def test_attached_paper_baselines_are_registered():
    from benchmarks.models import TRAINERS, PAPER_SYMBOLIC_LIBRARY
    expected = {"autosym", "fastkan_autosym", "gsr", "fastkan_gsr", "gmp"}
    assert expected.issubset(TRAINERS)
    assert len(PAPER_SYMBOLIC_LIBRARY) == 25
    for name in ["tan", "tanh", "arctan", "arcsin", "arccos", "arctanh", "gaussian"]:
        assert name in PAPER_SYMBOLIC_LIBRARY


def test_paper_gsr_edge_policy_aliases():
    from rulekan.MultKAN import _norm_policy
    assert _norm_policy("importance_desc") == "best"
    assert _norm_policy("importance_asc") == "worst"
    assert _norm_policy("highest") == "best"
    assert _norm_policy("lowest") == "worst"


def test_evolutionary_sr_dependencies_and_primary_profiles_are_registered():
    import yaml
    from benchmarks.models import TRAINERS
    root = Path(__file__).resolve().parents[1]
    req = (root / "benchmarks" / "requirements-benchmark.txt").read_text()
    setup = (root / "benchmarks" / "setup.sh").read_text()
    assert "pysr==2.2.1" in req or "pysr==2.2.1" in setup
    assert "pyoperon==0.6.1" not in req
    assert "PyOperon" in req  # benchmarks/setup.sh handles the source build / macOS rpath repair
    assert "pyoperon" in (root / "benchmarks" / "setup.sh").read_text().lower()
    assert {"pysr", "operon"}.issubset(TRAINERS)
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    for profile in ("quick", "standard", "full", "research", "fuzzy"):
        assert {"pysr", "operon"}.issubset(set(cfg["profiles"][profile]["models"]))
    # The paper profile remains an exact replication matrix.
    assert "pysr" not in cfg["profiles"]["paper"]["models"]
    assert "operon" not in cfg["profiles"]["paper"]["models"]


def test_benchmark_requirements_include_pytest():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    req = (root / "benchmarks" / "requirements-benchmark.txt").read_text().splitlines()
    assert "pytest" in {x.strip() for x in req if x.strip() and not x.startswith("#")}


def test_rulekan_fast_is_registered_and_profiles_compare_both_rulekan_bases():
    import yaml
    from benchmarks.models import TRAINERS
    assert "rulekan" in TRAINERS
    assert "rulekan_fast" in TRAINERS
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    for profile in ("quick", "standard", "paper", "full"):
        assert "rulekan" in cfg["profiles"][profile]["models"]
        assert "rulekan_fast" in cfg["profiles"][profile]["models"]
        assert "rulekan_fast" in cfg["profiles"][profile]["model_config"]



def test_fuzzy_rule_suite_has_same_cross_nested_and_product_cases():
    suites = list_suites()
    assert "fuzzy_rules" in suites
    names = set(suites["fuzzy_rules"])
    assert {
        "fuzzy_ite_cross", "fuzzy_ite_same_variable", "fuzzy_nested_tree",
        "fuzzy_two_independent_rules", "fuzzy_product_branches",
        "fuzzy_same_gate_product_branches",
    }.issubset(names)
    assert len(TASKS["fuzzy_nested_tree"].fuzzy_rules) == 3
    assert TASKS["fuzzy_nested_tree"].max_factors == 3
    assert TASKS["fuzzy_ite_same_variable"].expected_structures == ((0, 0),)


def test_fuzzy_tasks_generate_finite_membership_routing_targets():
    import torch
    for name in list_suites()["fuzzy_rules"]:
        spec = TASKS[name]
        data = make_synthetic_data(spec, seed=7, train_n=64, val_n=16, test_n=20)
        assert torch.isfinite(data.train_y).all()
        assert torch.isfinite(data.test_y).all()
        # Every fuzzy benchmark uses at least one observed membership variable
        # sampled directly in [0,1]; no off-domain synthetic gate points are used.
        raw = data.train_x * data.input_std + data.input_mean
        gate_vars = sorted({f.variable for r in spec.fuzzy_rules for f in r.factors if f.role == "gate"})
        assert gate_vars
        for j in gate_vars:
            assert float(raw[:, j].min()) >= -1e-6
            assert float(raw[:, j].max()) <= 1.0 + 1e-6


def test_fuzzy_recovery_scorer_recognizes_exact_if_else_rule_pair():
    import torch
    from benchmarks.models import fuzzy_rule_recovery_scores
    from rulekan.sum_product_kan import SumProductKAN

    spec = TASKS["fuzzy_ite_cross"]
    data = make_synthetic_data(spec, seed=11, train_n=96, val_n=24, test_n=32)
    model = SumProductKAN(
        in_dim=3, n_rules=2, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "sin", "exp"), min_order=1, seed=3,
    )
    lib = {name: i for i, name in enumerate(model.symbolic_library)}
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.rule_alive_mask[:] = True
        model.factor_alive_mask[:] = True
        model.hard_spline_choice[:] = False
        # ELSE: (1-x0) * sin(x1)
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_operator_choice[0, 0, 0] = lib["x"]
        model.hard_operator_choice[0, 1, 1] = lib["sin"]
        # IF: x0 * exp(x2)
        model.hard_variable_choice[1, 0] = 0
        model.hard_variable_choice[1, 1] = 2
        model.hard_operator_choice[1, 0, 0] = lib["x"]
        model.hard_operator_choice[1, 1, 2] = lib["exp"]
        # Exact gate membership/complement in normalized coordinates.
        std0 = float(data.input_std[0])
        mean0 = float(data.input_mean[0])
        model.symbolic_affine[..., 0].fill_(1.0)
        model.symbolic_affine[..., 3].zero_()
        model.symbolic_affine[0, 0, 0, lib["x"], 1] = -std0
        model.symbolic_affine[0, 0, 0, lib["x"], 2] = 1.0 - mean0
        model.symbolic_affine[1, 0, 0, lib["x"], 1] = std0
        model.symbolic_affine[1, 0, 0, lib["x"], 2] = mean0
    scores = fuzzy_rule_recovery_scores(model, spec, data)
    assert scores["fuzzy_rule_f1"] == 1.0
    assert scores["fuzzy_gate_recall"] == 1.0
    assert scores["fuzzy_branch_recall"] == 1.0
    assert scores["fuzzy_exact_structure_recovery"] == 1.0
    assert scores["fuzzy_gate_rmse_max"] < 1e-5
    assert scores["fuzzy_gate_nrmse_max"] < 1e-5


def test_fuzzy_profile_and_live_outputs_are_registered(tmp_path: Path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    assert "fuzzy" in cfg["profiles"]
    assert cfg["profiles"]["fuzzy"]["suites"] == ["fuzzy_rules"]
    compact = {
        "rulekan", "rulekan_fast", "power_rulekan",
        "rulekan_omp_full", "rulekan_fast_omp_full",
        "autosym", "fastkan_autosym", "gsr", "fastkan_gsr", "gmp",
        "pysr", "operon", "anfis",
    }
    assert set(cfg["profiles"]["fuzzy"]["models"]) == compact
    # The historical full extractor matrix is still available explicitly.
    assert "fuzzy_exhaustive" in cfg["profiles"]
    assert "fuzzy_omp_ablation" in cfg["profiles"]

    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {"status":"completed","suite":"fuzzy_rules","task":"fuzzy_ite_cross","task_type":"regression","model":"rulekan","seed":0,
         "test_rmse":0.02,"test_nrmse":0.03,"symbolic_test_rmse":0.01,"symbolic_test_nrmse":0.015,"symbolic_seconds":1.0,
         "fuzzy_rule_f1":1.0,"fuzzy_gate_recall":1.0,"fuzzy_branch_recall":1.0,"fuzzy_gate_nrmse_median":0.01,
         "fuzzy_exact_structure_recovery":1.0,"fuzzy_expected_rules":2,"fuzzy_found_rules":2,"finished_unix":1.0},
        {"status":"completed","suite":"fuzzy_rules","task":"fuzzy_ite_cross","task_type":"regression","model":"autosym","paper_pipeline":"baseline","seed":0,
         "test_rmse":0.2,"test_nrmse":0.3,"numeric_test_rmse":0.1,"numeric_test_nrmse":0.15,"symbolic_seconds":1.2,"finished_unix":2.0},
    ]
    for i,row in enumerate(rows):
        (runs/f"f{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path/"fuzzy_runs.csv").exists()
    assert (tmp_path/"fuzzy_summary.csv").exists()
    assert (tmp_path/"fuzzy_summary.md").exists()
    assert (tmp_path/"figures"/"fuzzy_rule_recovery.pdf").exists()
    assert not list((tmp_path/"figures").glob("*.png"))


def test_fuzzy_tasks_are_scheduled_before_all_other_suites():
    import yaml
    from benchmarks.run_benchmark import _select_tasks
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    for profile_name in ("quick", "standard", "research", "full"):
        profile = cfg["profiles"][profile_name]
        tasks = _select_tasks(profile, None, None)
        fuzzy_positions = [i for i,t in enumerate(tasks) if TASKS[t].suite == "fuzzy_rules"]
        other_positions = [i for i,t in enumerate(tasks) if TASKS[t].suite != "fuzzy_rules"]
        assert fuzzy_positions
        if other_positions:
            assert max(fuzzy_positions) < min(other_positions)


def test_primary_symbolic_outputs_use_raw_rmse_names(tmp_path: Path):
    runs=tmp_path/"runs"; runs.mkdir()
    row={"status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression",
         "model":"rulekan","seed":0,"test_rmse":0.02,"test_nrmse":0.2,"symbolic_test_rmse":0.03,
         "symbolic_test_nrmse":0.3,"symbolic_seconds":1.0,"finished_unix":1.0}
    (runs/"a.json").write_text(json.dumps(row))
    aggregate(tmp_path,quiet=True)
    assert (tmp_path/"figures"/"symbolic_rmse_by_task.pdf").exists()
    assert (tmp_path/"figures"/"numeric_vs_symbolic_rmse.pdf").exists()
    md=(tmp_path/"summary.md").read_text()
    assert "symbolic RMSE" in md
    assert "numeric RMSE" in md
    assert "NRMSE =" not in md


def test_rulekan_component_ablation_models_and_profiles_are_registered():
    import yaml
    from benchmarks.models import TRAINERS
    from benchmarks.run_benchmark import _select_tasks
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    required = {
        "rulekan_no_product", "rulekan_no_pruning", "rulekan_no_gmp",
        "rulekan_no_product_no_pruning", "rulekan_no_product_no_gmp",
        "rulekan_no_pruning_no_gmp", "rulekan_no_product_no_pruning_no_gmp",
        "rulekan_one_shot_prune", "rulekan_no_backfit", "rulekan_no_self_product",
    }
    assert required.issubset(TRAINERS)
    for profile_name in ("ablation_quick", "ablation"):
        profile = cfg["profiles"][profile_name]
        assert "rulekan" in profile["models"]
        assert required.issubset(set(profile["models"]))
        tasks = _select_tasks(profile, None, None)
        assert tasks
        assert TASKS[tasks[0]].suite == "fuzzy_rules"
        first_nonfuzzy = next((i for i,t in enumerate(tasks) if TASKS[t].suite != "fuzzy_rules"), len(tasks))
        assert all(TASKS[t].suite == "fuzzy_rules" for t in tasks[:first_nonfuzzy])


def test_rulekan_ablation_helper_preserves_baseline_and_applies_override(monkeypatch):
    import benchmarks.models as bm
    captured = {}

    def fake_train(model_name, spec, data, seed, cfg, *, numeric_basis, pursuit_mode="gsr"):
        captured.update({
            "model_name": model_name,
            "cfg": dict(cfg),
            "numeric_basis": numeric_basis,
            "pursuit_mode": pursuit_mode,
        })
        return "ok"

    monkeypatch.setattr(bm, "_train_rulekan_family", fake_train)
    base = {"stage_scale": 0.24, "numeric_lbfgs_steps": 40, "use_gmp_preselection": True}
    out = bm.train_rulekan_no_gmp(None, None, 7, base)
    assert out == "ok"
    assert captured["model_name"] == "rulekan_no_gmp"
    assert captured["numeric_basis"] == "spline"
    assert captured["cfg"]["stage_scale"] == 0.24
    assert captured["cfg"]["numeric_lbfgs_steps"] == 40
    assert captured["cfg"]["use_gmp_preselection"] is False
    assert base["use_gmp_preselection"] is True


def test_ablation_aggregator_writes_factorial_and_paired_outputs(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    factorial = {
        "rulekan": (1,1,1),
        "rulekan_no_product": (0,1,1),
        "rulekan_no_pruning": (1,0,1),
        "rulekan_no_gmp": (1,1,0),
        "rulekan_no_product_no_pruning": (0,0,1),
        "rulekan_no_product_no_gmp": (0,1,0),
        "rulekan_no_pruning_no_gmp": (1,0,0),
        "rulekan_no_product_no_pruning_no_gmp": (0,0,0),
    }
    for i,(model,(s,p,g)) in enumerate(factorial.items()):
        # Enabling each core component improves symbolic RMSE in this synthetic fixture.
        rmse = 0.2 * (0.5 if s else 1.0) * (0.7 if p else 1.0) * (0.6 if g else 1.0)
        row = {
            "status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression",
            "model":model,"seed":0,"test_rmse":0.01,"test_nrmse":0.02,
            "symbolic_test_rmse":rmse,"symbolic_test_nrmse":rmse,"symbolic_seconds":1.0 + i/10,
            "elapsed_seconds":2.0+i/10,"finished_unix":float(i+1),
        }
        (runs/f"a{i}.json").write_text(json.dumps(row))
    # Extra paired variant.
    (runs/"backfit.json").write_text(json.dumps({
        "status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression",
        "model":"rulekan_no_backfit","seed":0,"test_rmse":0.01,"test_nrmse":0.02,
        "symbolic_test_rmse":0.09,"symbolic_test_nrmse":0.09,"symbolic_seconds":0.8,
        "elapsed_seconds":1.5,"finished_unix":20.0,
    }))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path/"ablation_pairs.csv").exists()
    assert (tmp_path/"ablation_factorial_effects.csv").exists()
    assert (tmp_path/"ablation_component_summary.csv").exists()
    assert (tmp_path/"ablation_summary.md").exists()
    assert (tmp_path/"figures"/"ablation_component_effects.pdf").exists()
    comp = pd.read_csv(tmp_path/"ablation_component_summary.csv")
    assert set(comp.component) == {"SumProduct", "Iterative pruning", "GMP"}
    assert (comp.rmse_multiplier_median < 1.0).all()
    assert not list((tmp_path/"figures").glob("*.png"))


def test_deep_multkan_late_multiplication_baselines_are_registered():
    import yaml
    from benchmarks.models import TRAINERS, DEEP_MULTKAN_PIPELINES
    required = {
        "multkan_deep_autosym", "fast_multkan_deep_autosym",
        "multkan_deep_gsr", "fast_multkan_deep_gsr", "multkan_deep_gmp",
    }
    assert required == set(DEEP_MULTKAN_PIPELINES)
    assert required.issubset(TRAINERS)
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    assert "deep_multkan" in cfg["profiles"]
    assert required.issubset(set(cfg["profiles"]["deep_multkan"]["models"]))
    # The compact research profile keeps only one representative deep baseline;
    # the full Cartesian product lives in explicit appendix/ablation profiles.
    assert "multkan_deep_gsr" in set(cfg["profiles"]["research"]["models"])
    assert required.issubset(set(cfg["profiles"]["research_exhaustive"]["models"]))
    assert required.issubset(set(cfg["profiles"]["deep_ablation"]["models"]))
    assert required.issubset(set(cfg["profiles"]["full"]["models"]))
    assert "deep_multkan_quick" in cfg["profiles"]
    assert {"multkan_deep_gsr", "fast_multkan_deep_gsr", "multkan_deep_gmp"}.issubset(
        set(cfg["profiles"]["deep_multkan_quick"]["models"])
    )


def test_deep_multkan_architecture_multiplies_after_first_hidden_transform():
    import torch
    from rulekan import KAN
    model = KAN(
        width=[2, [3, 0], [2, 1], 1],
        grid=3, k=2, seed=0, auto_save=False, save_act=False, device="cpu",
    )
    assert model.depth == 3
    assert model.width[1] == [3, 0]       # first hidden layer: no multiplication
    assert model.width[2] == [2, 1]       # second hidden layer: multiplication node
    assert model.width_in == [2, 3, 3, 1]
    assert model.width_out[2] == 4         # 2 additive subnodes + 2 inputs to one product
    x = torch.randn(11, 2)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (11, 1)
    assert torch.isfinite(y).all()


def test_multkan_depth_comparison_outputs_are_generated(tmp_path: Path):
    import json
    from benchmarks.aggregate import aggregate
    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {"status":"completed","suite":"nested_stress","task":"nested_cross_sin_product","task_type":"regression","model":"gsr","seed":0,
         "test_rmse":0.12,"test_nrmse":0.2,"numeric_test_rmse":0.08,"numeric_test_nrmse":0.13,"symbolic_seconds":2.0,"elapsed_seconds":3.0,"finished_unix":1.0},
        {"status":"completed","suite":"nested_stress","task":"nested_cross_sin_product","task_type":"regression","model":"multkan_deep_gsr","seed":0,
         "test_rmse":0.03,"test_nrmse":0.05,"numeric_test_rmse":0.05,"numeric_test_nrmse":0.08,"symbolic_seconds":4.0,"elapsed_seconds":5.0,"finished_unix":2.0},
    ]
    for i,row in enumerate(rows):
        (runs/f"r{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path/"multkan_depth_pairs.csv").exists()
    assert (tmp_path/"figures"/"multkan_depth_comparison.pdf").exists()
    import pandas as pd
    x = pd.read_csv(tmp_path/"multkan_depth_pairs.csv")
    assert len(x) == 1
    assert abs(float(x.iloc[0]["deep_over_shallow_rmse"]) - 0.25) < 1e-12


def test_research_srkan_uses_matched_target_core_library():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    assert cfg["profiles"]["research"]["model_config"]["srkan"]["functions"] == ["target_core"]


def test_shared_capacity_resolver_uses_fixed_total_width_consistently():
    import yaml
    from benchmarks.models import (
        resolve_shared_benchmark_config, RESEARCH_SYMBOLIC_LIBRARY,
        TARGET_CORE_SYMBOLIC_LIBRARY, MEDIUM_SYMBOLIC_LIBRARY,
    )
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    shared = cfg["profiles"]["research"]["shared_settings"]
    spec2 = TASKS["same_var_exp_sin"]
    spec3 = TASKS["three_way_product"]
    rk2, meta2 = resolve_shared_benchmark_config("rulekan", spec2, {}, shared)
    gsr2, _ = resolve_shared_benchmark_config("gsr", spec2, {}, shared)
    deep3, meta3 = resolve_shared_benchmark_config("multkan_deep_gsr", spec3, {}, shared)
    sr2, srmeta = resolve_shared_benchmark_config("srkan", spec2, {"functions": ["all"]}, shared)
    assert meta2["shared_capacity_width"] == 12
    assert rk2["n_rules"] == 12
    assert gsr2["width_additive"] == 8 and gsr2["mult_units"] == 4
    assert gsr2["width_additive"] + gsr2["mult_units"] == 12
    assert rk2["max_factors_override"] == 3
    assert gsr2["mult_arity"] == 3
    assert meta2["shared_max_product_order"] == 3
    assert meta2["shared_task_max_factors"] == spec2.max_factors
    assert meta3["shared_capacity_width"] == 12
    assert deep3["deep_width_1"] == 12
    assert deep3["deep_width_2"] + deep3["deep_mult_units"] == 12
    assert deep3["deep_mult_arity"] == 3
    assert rk2["grid"] == gsr2["grid"] == 12
    assert srmeta["shared_capacity_width"] == 12
    assert sr2["functions"] == ["target_core"]
    assert tuple(rk2["symbolic_library"]) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert tuple(gsr2["symbolic_library"]) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert len(TARGET_CORE_SYMBOLIC_LIBRARY) == 10
    assert len(MEDIUM_SYMBOLIC_LIBRARY) == 16
    assert set(TARGET_CORE_SYMBOLIC_LIBRARY).issubset(MEDIUM_SYMBOLIC_LIBRARY)
    assert set(MEDIUM_SYMBOLIC_LIBRARY).issubset(RESEARCH_SYMBOLIC_LIBRARY)


def test_paper_profile_preserves_reference_capacity_while_controlled_profiles_share_capacity():
    import yaml
    from benchmarks.config_utils import resolve_profile
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    assert cfg["profiles"]["paper"]["shared_settings"]["enabled"] is False
    assert cfg["profiles"]["paper"]["model_config"]["gsr"]["width_additive"] == 5
    assert cfg["profiles"]["paper"]["model_config"]["gsr"]["mult_units"] == 2
    for name in ("standard", "research", "fuzzy", "deep_multkan", "ablation", "full"):
        sh = resolve_profile(cfg, name)["shared_settings"]
        assert sh["enabled"] is True
        assert sh["capacity"]["policy"] == "fixed"
        assert sh["capacity"]["width"] == 12
        assert sh["capacity"]["mult_units"] == 4
        assert sh["symbolic_library"] == "target_core"
        assert sh["grid"] == 12


def test_ablation_profile_inherits_research_rulekan_baseline_exactly():
    import yaml
    from benchmarks.config_utils import resolve_profile
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    research = resolve_profile(cfg, "research")
    ablation = resolve_profile(cfg, "ablation")
    assert cfg["profiles"]["ablation"]["extends"] == "research"
    assert ablation["seeds"] == research["seeds"]
    assert ablation["data"] == research["data"]
    assert ablation["timeout_seconds"] == research["timeout_seconds"]
    assert ablation["shared_settings"] == research["shared_settings"]
    assert ablation["model_config"]["rulekan"] == research["model_config"]["rulekan"]
    assert ablation["model_config"]["rulekan_fast"] == research["model_config"]["rulekan_fast"]


def test_library_sensitivity_profile_uses_nested_libraries():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    p = cfg["profiles"]["library_sensitivity_quick"]
    assert p["library_values"] == ["core10", "medium16", "research26"]
    assert p["shared_settings"]["capacity"]["width"] == 12
    assert p["models"] == ["rulekan", "autosym", "gsr", "gmp"]


def test_width_sensitivity_profiles_use_common_width_values():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    assert cfg["profiles"]["width_sensitivity_quick"]["width_values"] == [6, 8, 10, 12]
    assert cfg["profiles"]["width_sensitivity"]["width_values"] == [6, 8, 10, 12, 16, 24]
    for p in ("width_sensitivity_quick", "width_sensitivity"):
        assert cfg["profiles"][p]["shared_settings"]["enabled"] is True
        assert "rulekan" in cfg["profiles"][p]["models"]
        assert "gsr" in cfg["profiles"][p]["models"]
        assert "multkan_deep_gsr" in cfg["profiles"][p]["models"]


def test_width_sensitivity_aggregator_outputs_csv_and_pdf(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    rows=[]
    for width, rmse in [(5,0.2),(10,0.1)]:
        for model, factor in [("rulekan",1.0),("gsr",1.4),("multkan_deep_gsr",0.9)]:
            rows.append({
                "status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression",
                "model":model,"seed":0,"shared_capacity_width":width,
                "test_rmse":rmse*factor,"test_nrmse":rmse*factor,
                "symbolic_test_rmse":rmse*factor if model=="rulekan" else None,
                "symbolic_test_nrmse":rmse*factor if model=="rulekan" else None,
                "paper_pipeline":"greedy_matching_pursuit" if model!="rulekan" else None,
                "symbolic_seconds":float(width)/10,"finished_unix":float(width),
            })
    for i,row in enumerate(rows):
        (runs/f"w{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path/"width_sensitivity.csv").exists()
    assert (tmp_path/"figures"/"width_sensitivity_rmse.pdf").exists()
    assert (tmp_path/"figures"/"width_sensitivity_runtime.pdf").exists()
    assert (tmp_path/"width_sensitivity_robustness.csv").exists()
    assert (tmp_path/"width_sensitivity_summary.md").exists()
    w = pd.read_csv(tmp_path/"width_sensitivity.csv")
    assert set(w.shared_capacity_width) == {5,10}


def test_library_sensitivity_aggregator_outputs_csv_and_pdf(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    rows=[]
    for size, rmse, sec, name in [(10,0.10,1.0,"core10"),(16,0.12,1.5,"medium16"),(26,0.16,2.2,"research26")]:
        rows.append({
            "status":"completed","suite":"synthetic_core","task":"same_var_exp_sin","task_type":"regression",
            "model":"rulekan","seed":0,"shared_symbolic_library_size":size,"shared_symbolic_library":name,
            "test_rmse":0.08,"test_nrmse":0.1,"symbolic_test_rmse":rmse,"symbolic_test_nrmse":rmse,
            "symbolic_seconds":sec,"finished_unix":float(size),
        })
    for i,row in enumerate(rows):
        (runs/f"l{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path/"library_sensitivity.csv").exists()
    assert (tmp_path/"library_sensitivity_summary.md").exists()
    assert (tmp_path/"figures"/"library_sensitivity_rmse.pdf").exists()
    assert (tmp_path/"figures"/"library_sensitivity_runtime.pdf").exists()
    x=pd.read_csv(tmp_path/"library_sensitivity.csv")
    assert set(x.shared_symbolic_library_size)=={10,16,26}


def test_shared_rulekan_symbolic_rule_budget_is_at_least_numeric_width():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    for name, profile in cfg["profiles"].items():
        sh = profile.get("shared_settings", {})
        if sh.get("enabled") is True:
            assert sh.get("symbolic_rule_budget_at_least_width") is True, name


def test_controlled_core_library_is_elementary_and_keeps_reciprocals():
    from benchmarks.models import TARGET_CORE_SYMBOLIC_LIBRARY
    required = {"x", "x^2", "1/x", "1/x^2", "sqrt", "log", "exp", "sin", "cos", "tanh"}
    forbidden = {"gaussian", "log1p_sq", "sqrt1p_sq", "inv1p_sq"}
    assert set(TARGET_CORE_SYMBOLIC_LIBRARY) == required
    assert forbidden.isdisjoint(TARGET_CORE_SYMBOLIC_LIBRARY)


def test_rulekan_shared_resolver_sets_symbolic_max_rules_to_width():
    import yaml
    from benchmarks.models import resolve_shared_benchmark_config
    from benchmarks.specs import TASKS
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    profile = cfg["profiles"]["controlled_compare_quick"]
    base = dict(profile["model_config"]["rulekan"])
    out, meta = resolve_shared_benchmark_config(
        "rulekan", TASKS["mixed_rank4"], base, profile["shared_settings"]
    )
    assert out["n_rules"] == 12
    assert out["symbolic_max_rules"] == 12
    assert meta["shared_symbolic_max_rules"] == 12


def test_research_profile_and_research_alias_use_core10_by_default():
    import yaml
    from benchmarks.models import resolve_shared_benchmark_config, TARGET_CORE_SYMBOLIC_LIBRARY
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    shared = cfg["profiles"]["research"]["shared_settings"]
    assert shared["symbolic_library"] == "target_core"
    resolved, meta = resolve_shared_benchmark_config("rulekan", TASKS["same_var_exp_sin"], {}, shared)
    assert tuple(resolved["symbolic_library"]) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert meta["shared_symbolic_library"] == "core10"

    # Plain "research" means the controlled elementary default; the wider
    # distractor library must be requested explicitly as "research26".
    resolved2, meta2 = resolve_shared_benchmark_config(
        "rulekan", TASKS["same_var_exp_sin"], {}, shared, library_override="research"
    )
    assert tuple(resolved2["symbolic_library"]) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert meta2["shared_symbolic_library"] == "core10"


def test_power_rulekan_replaces_rational_and_fast_factorial_registration():
    import yaml
    from benchmarks.models import TRAINERS, resolve_shared_benchmark_config
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    required = {
        "rulekan_fast_omp_linear", "rulekan_fast_omp_nonlinear",
        "rulekan_fast_omp_full", "rulekan_fast_graph", "power_rulekan",
    }
    assert required.issubset(TRAINERS)
    assert "rational_rulekan" not in TRAINERS and "rational_rulekan_fast" not in TRAINERS
    research_models = set(cfg["profiles"]["research"]["models"])
    assert {"rulekan_adaptive", "power_rulekan"}.issubset(research_models)
    fuzzy_models = set(cfg["profiles"]["fuzzy"]["models"])
    assert {"rulekan_fast_omp_full", "power_rulekan"}.issubset(fuzzy_models)
    assert cfg["profiles"]["research"]["shared_settings"]["grid"] == 12
    assert cfg["profiles"]["fuzzy"]["shared_settings"]["grid"] == 12
    assert required.issubset(set(cfg["profiles"]["research_exhaustive"]["models"]))
    assert required.issubset(set(cfg["profiles"]["fuzzy_exhaustive"]["models"]))
    spec = TASKS["fuzzy_ite_cross"]
    pcfg, meta = resolve_shared_benchmark_config(
        "power_rulekan", spec, {}, cfg["profiles"]["research"]["shared_settings"]
    )
    assert pcfg["grid"] == 12
    assert pcfg["n_rules"] == 12
    assert meta["shared_grid"] == 12

def test_paper_rulekan_grid_and_symbolic_capacity_match_numeric_width():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    paper = cfg["profiles"]["paper"]
    assert paper["shared_settings"]["enabled"] is False
    for name in ("rulekan", "rulekan_fast"):
        mc = paper["model_config"][name]
        assert mc["grid"] == 12
        assert mc["n_rules"] == 15
        assert mc["symbolic_max_rules"] == 15


def test_v51_accuracy_profiles_use_longer_numeric_schedule_without_touching_paper():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    for profile_name in ("standard", "research", "fuzzy"):
        profile = cfg["profiles"][profile_name]
        for model_name, mc in profile.get("model_config", {}).items():
            if model_name.startswith("rulekan"):
                assert mc.get("stage_scale") == 0.24, (profile_name, model_name)
                assert mc.get("plateau_rounds") == 4, (profile_name, model_name)
                assert mc.get("plateau_patience") == 2, (profile_name, model_name)
                assert mc.get("numeric_lbfgs_steps") == 40, (profile_name, model_name)

    # The paper profile keeps its numerical schedule, while symbolic capacity
    # is no longer artificially smaller than numerical rule width.
    paper = cfg["profiles"]["paper"]["model_config"]
    assert paper["rulekan"]["stage_scale"] == 0.12
    assert paper["rulekan_fast"]["stage_scale"] == 0.12
    assert "numeric_lbfgs_steps" not in paper["rulekan"]
    assert "numeric_lbfgs_steps" not in paper["rulekan_fast"]


def test_v51_quick_profile_remains_a_smoke_budget():
    import yaml
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    quick = cfg["profiles"]["quick"]["model_config"]
    assert quick["rulekan"]["stage_scale"] == 0.04
    assert quick["rulekan_fast"]["stage_scale"] == 0.04
    assert "numeric_lbfgs_steps" not in quick["rulekan"]
    assert "numeric_lbfgs_steps" not in quick["rulekan_fast"]


def test_v53_research_fuzzy_use_residual_edge_sparsification_but_paper_does_not():
    import yaml
    from benchmarks.models import resolve_shared_benchmark_config
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    for profile_name in ("standard", "research", "fuzzy"):
        sh = cfg["profiles"][profile_name]["shared_settings"]
        assert sh["symbolic_hybrid_hard_screening"] is True
        assert sh["symbolic_residual_structure_topk"] == 1
        out, meta = resolve_shared_benchmark_config(
            "rulekan", TASKS["fuzzy_ite_cross"], {}, sh
        )
        assert out["symbolic_hybrid_hard_screening"] is True
        assert out["symbolic_residual_structure_topk"] == 1
        assert meta["shared_symbolic_residual_structure_topk"] == 1
    assert cfg["profiles"]["paper"]["shared_settings"]["enabled"] is False


def test_v83_compact_profiles_and_profile_inheritance():
    import yaml
    from benchmarks.config_utils import resolve_profile
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    research = resolve_profile(cfg, "research")
    main = resolve_profile(cfg, "main")
    assert main["models"] == research["models"]
    assert len(research["models"]) == 19
    assert {
        "rulekan", "rulekan_comp", "rulekan_adaptive", "rulekan_fast", "rulekan_omp_full",
        "sisp", "sisp_comp", "power_rulekan", "power_rulekan_comp", "autosym", "fastkan_autosym",
        "gsr", "fastkan_gsr", "gmp", "srkan", "pysr", "operon", "anfis", "multkan_deep_gsr",
    } == set(research["models"])
    assert "sisp_fast" not in research["models"]
    exhaustive = resolve_profile(cfg, "research_exhaustive")
    assert len(exhaustive["models"]) > len(research["models"])
    assert {"rulekan_omp_linear", "rulekan_omp_nonlinear", "rulekan_graph"}.issubset(exhaustive["models"])
    omp = resolve_profile(cfg, "omp_ablation")
    assert omp["data"] == research["data"]
    assert omp["shared_settings"] == research["shared_settings"]
    assert "rulekan_omp_linear" in omp["models"] and "autosym" not in omp["models"]


def test_v55_redundancy_report_flags_equivalent_pairs(tmp_path: Path):
    import pandas as pd
    from benchmarks.aggregate import _build_method_redundancy, aggregate
    rows = []
    # A and B differ by ~2%, always within the 0.05 dex (~12%) equivalence margin.
    # C differs by 2x and must not be considered redundant.
    for task_i, task in enumerate(["t1", "t2"]):
        for seed in range(3):
            base = 0.01 * (1.0 + 0.1 * task_i + 0.02 * seed)
            for model, rmse in [("rulekan", base), ("rulekan_omp_full", base * 1.02), ("gsr", base * 2.0)]:
                rows.append({
                    "status": "completed", "suite": "synthetic_core", "task": task,
                    "task_type": "regression", "model": model, "seed": seed,
                    "test_rmse": base * 0.8, "test_nrmse": base * 0.8,
                    "symbolic_test_rmse": rmse, "symbolic_test_nrmse": rmse,
                    "symbolic_seconds": 1.0, "finished_unix": 1.0,
                })
    symbolic = pd.DataFrame(rows)
    symbolic = symbolic.assign(symbolic_rmse=symbolic["symbolic_test_rmse"])
    report = _build_method_redundancy(symbolic, threshold_dex=0.05, min_pairs=3)
    ab = report[((report.model_a == "rulekan") & (report.model_b == "rulekan_omp_full")) |
                ((report.model_b == "rulekan") & (report.model_a == "rulekan_omp_full"))].iloc[0]
    assert bool(ab.redundant)
    assert int(ab.n_paired) == 6
    ac = report[((report.model_a == "rulekan") & (report.model_b == "gsr")) |
                ((report.model_b == "rulekan") & (report.model_a == "gsr"))].iloc[0]
    assert not bool(ac.redundant)

    runs = tmp_path / "runs"; runs.mkdir()
    for i, row in enumerate(rows):
        (runs / f"r{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)
    assert (tmp_path / "method_redundancy.csv").exists()
    assert (tmp_path / "redundant_method_pairs.csv").exists()
    assert (tmp_path / "method_redundancy.md").exists()


def test_no_legacy_rational_rulekan_is_registered():
    import yaml
    from benchmarks.models import TRAINERS
    cfg = yaml.safe_load((Path(__file__).parents[1] / "benchmarks" / "configs" / "default.yaml").read_text())
    assert "rational_rulekan" not in TRAINERS
    assert "rational_rulekan_fast" not in TRAINERS
    for profile in cfg["profiles"].values():
        models = profile.get("models", []) if isinstance(profile, dict) else []
        assert "rational_rulekan" not in models
        assert "rational_rulekan_fast" not in models

def test_fuzzy_recovery_scorer_is_gate_gauge_and_sincos_phase_invariant():
    import math
    import torch
    from benchmarks.models import fuzzy_rule_recovery_scores
    from benchmarks.specs import TASKS, make_synthetic_data
    from rulekan.sum_product_kan import SumProductKAN

    spec = TASKS["fuzzy_ite_cross"]
    data = make_synthetic_data(spec, seed=13, train_n=96, val_n=24, test_n=32)
    model = SumProductKAN(
        in_dim=3, n_rules=2, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "sin", "cos", "exp"), min_order=1, seed=4,
    )
    lib = {name: i for i, name in enumerate(model.symbolic_library)}
    with torch.no_grad():
        model.discretized.fill_(True)
        model.symbolic_enabled = True
        model.hard_rule_choice[:] = True
        model.rule_alive_mask[:] = True
        model.factor_alive_mask[:] = True
        model.hard_spline_choice[:] = False
        # ELSE uses the algebraically equivalent -(1-x0) gate and cos(t-pi/2).
        model.hard_variable_choice[0, 0] = 0
        model.hard_variable_choice[0, 1] = 1
        model.hard_operator_choice[0, 0, 0] = lib["x"]
        model.hard_operator_choice[0, 1, 1] = lib["cos"]
        # IF uses x0 * exp(x2).
        model.hard_variable_choice[1, 0] = 0
        model.hard_variable_choice[1, 1] = 2
        model.hard_operator_choice[1, 0, 0] = lib["x"]
        model.hard_operator_choice[1, 1, 2] = lib["exp"]
        model.symbolic_affine[..., 0].fill_(1.0)
        model.symbolic_affine[..., 3].zero_()
        std0 = float(data.input_std[0]); mean0 = float(data.input_mean[0])
        # raw-1 == -(1-raw): opposite slope sign from the expected complement,
        # but exactly equivalent after product-rule gauge scaling.
        model.symbolic_affine[0, 0, 0, lib["x"], 1] = std0
        model.symbolic_affine[0, 0, 0, lib["x"], 2] = mean0 - 1.0
        model.symbolic_affine[1, 0, 0, lib["x"], 1] = std0
        model.symbolic_affine[1, 0, 0, lib["x"], 2] = mean0
        # cos(theta-pi/2) == sin(theta); scorer should compare family modulo phase.
        model.symbolic_affine[0, 1, 1, lib["cos"], 1] = 1.0
        model.symbolic_affine[0, 1, 1, lib["cos"], 2] = -math.pi / 2.0
    scores = fuzzy_rule_recovery_scores(model, spec, data)
    assert scores["fuzzy_rule_f1"] == 1.0
    assert scores["fuzzy_gate_recall"] == 1.0
    assert scores["fuzzy_branch_recall"] == 1.0
    assert scores["fuzzy_exact_structure_recovery"] == 1.0


def test_v82_adaptive_rescue_is_strictly_structure_conditioned(monkeypatch):
    import benchmarks.models as bm

    captured = {}

    def fake_fit(model, tx, ty, vx, vy, bank, classes, cfg, **kwargs):
        captured["bank"] = list(bank)
        captured["supports"] = [tuple(c["support"]) for c in classes]
        return object(), []

    monkeypatch.setattr(bm, "_fit_rulekan_learned_support_gsr", fake_fit)
    classes = [{"support": (0, 1)}, {"support": (2,)}]
    bank = [(0, 1), (0, 0, 1), (2,), (2, 2)]
    bm._fit_validation_structure_rescue_symbolic_gsr(
        None, None, None, None, None, classes, bank, {},
        n_rules=8, seed=0, pursuit_mode="gsr",
    )
    allowed = {frozenset(s) for s in captured["supports"]}
    assert captured["bank"] == bank
    assert all(frozenset(z) in allowed for z in captured["bank"])

    bad_bank = list(bank) + [(0, 2)]
    try:
        bm._fit_validation_structure_rescue_symbolic_gsr(
            None, None, None, None, None, classes, bad_bank, {},
            n_rules=8, seed=0, pursuit_mode="gsr",
        )
    except ValueError as exc:
        assert "unsupported structure" in str(exc)
    else:
        raise AssertionError("RuleKAN rescue accepted an unsupported variable support")


def test_v82_power_rulekan_uses_validation_selected_effective_support_contract():
    import benchmarks.models as bm

    run = bm.ModelRun(
        "rulekan",
        extras={
            # Deliberately stale primary supports: PowerRuleKAN must ignore these.
            "symbolic_learned_support_classes": [{"support": [0]}],
            "symbolic_learned_support_bank": [[0], [0, 0]],
            # Canonical validation-selected rescue supports.
            "symbolic_effective_support_classes": [
                {"support": [1, 2], "members": [], "active_members": [],
                 "score": 1.0, "structure_probability": 1.0, "numeric_strength": 1.0}
            ],
            "symbolic_effective_support_bank": [[1, 2], [1, 1, 2], [1, 2, 2]],
            "symbolic_effective_support_source": "validation_selected_high_recall_numeric_supports",
        },
    )
    bank, supports, source = bm._rulekan_effective_supports(run)
    assert supports == [(1, 2)]
    assert bank == [(1, 2), (1, 1, 2), (1, 2, 2)]
    assert source == "validation_selected_high_recall_numeric_supports"
    assert all(frozenset(z) == frozenset((1, 2)) for z in bank)


def test_v82_effective_support_contract_rejects_unsupported_power_structure():
    import benchmarks.models as bm

    run = bm.ModelRun(
        "rulekan",
        extras={
            "symbolic_effective_support_classes": [{"support": [1, 2]}],
            "symbolic_effective_support_bank": [[1, 2], [0, 2]],
            "symbolic_effective_support_source": "test",
        },
    )
    try:
        bm._rulekan_effective_supports(run)
    except ValueError as exc:
        assert "unsupported structure" in str(exc)
    else:
        raise AssertionError("PowerRuleKAN effective-support contract accepted an unsupported support")


def test_v82_power_rulekan_cannot_read_stale_primary_support_fields_directly():
    import inspect
    import benchmarks.models as bm

    src = inspect.getsource(bm._train_power_rulekan)
    assert "_rulekan_effective_supports(baseline)" in src
    assert "symbolic_learned_support_bank" not in src
    assert "symbolic_learned_support_classes" not in src
    # Both transformed-power fitting and ratio fitting must receive the same
    # canonical bank/support pair.  Future branches should reuse this pair.
    assert src.count("structure_bank=bank,allowed_supports=supports") >= 2
    # Validation-gated structure rescue remains the default precursor policy.
    assert 'cfg.get("power_baseline_validation_rescue", True)' in src


def test_power_rulekan_fuzzy_recovery_scores_final_p1_structure_and_rejects_powered_credit():
    import torch
    from benchmarks.models import fuzzy_rule_recovery_scores
    from rulekan.sum_product_kan import SumProductKAN
    from rulekan.power_rulekan import PowerRuleKAN

    spec = TASKS["fuzzy_ite_cross"]
    data = make_synthetic_data(spec, seed=23, train_n=96, val_n=24, test_n=32)
    base = SumProductKAN(
        in_dim=3, n_rules=2, max_factors=2, grid=4, k=2,
        symbolic_library=("x", "sin", "exp"), min_order=1, seed=5,
    )
    lib = {name: i for i, name in enumerate(base.symbolic_library)}
    with torch.no_grad():
        base.discretized.fill_(True)
        base.symbolic_enabled = True
        base.hard_rule_choice[:] = True
        base.rule_alive_mask[:] = True
        base.factor_alive_mask[:] = True
        base.hard_spline_choice[:] = False
        # ELSE: (1-x0) * sin(x1)
        base.hard_variable_choice[0, 0] = 0
        base.hard_variable_choice[0, 1] = 1
        base.hard_operator_choice[0, 0, 0] = lib["x"]
        base.hard_operator_choice[0, 1, 1] = lib["sin"]
        # IF: x0 * exp(x2)
        base.hard_variable_choice[1, 0] = 0
        base.hard_variable_choice[1, 1] = 2
        base.hard_operator_choice[1, 0, 0] = lib["x"]
        base.hard_operator_choice[1, 1, 2] = lib["exp"]
        std0 = float(data.input_std[0])
        mean0 = float(data.input_mean[0])
        base.symbolic_affine[..., 0].fill_(1.0)
        base.symbolic_affine[..., 3].zero_()
        base.symbolic_affine[0, 0, 0, lib["x"], 1] = -std0
        base.symbolic_affine[0, 0, 0, lib["x"], 2] = 1.0 - mean0
        base.symbolic_affine[1, 0, 0, lib["x"], 1] = std0
        base.symbolic_affine[1, 0, 0, lib["x"], 2] = mean0

    ordinary = PowerRuleKAN([base], [[(0, 1)]], scales=[1.0], bias=0.0)
    scores = fuzzy_rule_recovery_scores(ordinary, spec, data)
    assert scores["fuzzy_rule_f1"] == 1.0
    assert scores["fuzzy_gate_recall"] == 1.0
    assert scores["fuzzy_branch_recall"] == 1.0
    assert scores["fuzzy_exact_structure_recovery"] == 1.0

    powered = PowerRuleKAN([base], [[(0, 2)]], scales=[1.0], bias=0.0)
    powered_scores = fuzzy_rule_recovery_scores(powered, spec, data)
    assert powered_scores["fuzzy_rule_f1"] == 0.0
    assert powered_scores["fuzzy_gate_recall"] == 0.0
    assert powered_scores["fuzzy_exact_structure_recovery"] == 0.0


def test_all_benchmark_figure_model_labels_have_publication_casing():
    import yaml
    from benchmarks.aggregate import MODEL_LABELS, _label

    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    configured = set()
    for profile in cfg["profiles"].values():
        configured.update(profile.get("models", []) or [])
    assert configured.issubset(MODEL_LABELS)
    assert _label("power_rulekan") == "PowerRuleKAN"
    assert _label("rulekan_adaptive") == "RuleKAN Adaptive"
    assert _label("sisp") == "SISP"
    assert _label("sisp_fast") == "SISP Fast"


def test_main_benchmark_entrypoints_share_canonical_defaults():
    """Keep the public shell and Python benchmark launchers on the same defaults."""
    import yaml

    shell = Path("run_rulekan_benchmark.sh").read_text()
    runner = Path("benchmarks/run_benchmark.py").read_text()
    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())

    assert 'PROFILE="${1:-standard}"' in shell
    assert 'DEVICE="${RULEKAN_DEVICE:-cpu}"' in shell
    assert 'ap.add_argument("--profile", default="standard")' in runner
    assert 'ap.add_argument("--config", default="benchmarks/configs/default.yaml")' in runner
    assert '"--device", default="cpu"' in runner
    assert '"--cpu-fraction", type=float, default=0.5' in runner
    assert 'benchmarks/configs/default.yaml' in shell
    assert "standard" in cfg["profiles"]


def test_cpu_parallelism_defaults_to_half_logical_cpus():
    from benchmarks.device_utils import benchmark_workers, default_cpu_workers

    assert default_cpu_workers(0.5, cpu_count=8) == 4
    assert default_cpu_workers(0.5, cpu_count=5) == 2
    assert default_cpu_workers(0.5, cpu_count=1) == 1
    assert benchmark_workers("cpu", cpu_fraction=0.5, workers=None) >= 1
    assert benchmark_workers("cuda", cpu_fraction=0.5, workers=None) == 1
    assert benchmark_workers("mps", cpu_fraction=0.5, workers=None) == 1
    assert benchmark_workers("cpu", workers=3, cpu_fraction=0.5) == 3


def test_aggregate_separates_failed_running_skipped_and_incompatible(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    common = {"suite": "synthetic_core", "task": "same_var_exp_sin", "task_type": "regression", "seed": 0}
    rows = [
        {**common, "model": "rulekan", "status": "failed", "error": "timeout after 10s"},
        {**common, "model": "sisp", "status": "running", "started_unix": 1.0},
        {**common, "model": "pysr", "status": "skipped", "error": "manual skip"},
    ]
    for i, row in enumerate(rows):
        (runs / f"r{i}.json").write_text(json.dumps(row))
    (tmp_path / "skipped_incompatible_jobs.json").write_text(json.dumps([
        {"task": "x", "model": "m", "reason": "incompatible"},
        {"task": "y", "model": "m", "reason": "incompatible"},
    ]))

    aggregate(tmp_path, quiet=True)
    status = (tmp_path / "STATUS.txt").read_text()
    assert "completed: 0" in status
    assert "failed: 1" in status
    assert "skipped records: 1" in status
    assert "running: 1" in status
    assert "incompatible jobs not scheduled: 2" in status
    incomplete = pd.read_csv(tmp_path / "incomplete_runs.csv")
    assert set(incomplete.status) == {"failed", "running", "skipped"}
    assert (tmp_path / "failure_summary.csv").exists()


def test_reuse_completed_hides_build_hashes_and_summarizes_resume(tmp_path, monkeypatch, capsys):
    """Normal resume output should not dump one fingerprint mismatch per job."""
    import json
    import sys
    from benchmarks import run_benchmark as rb

    run_dir = tmp_path / "run"
    runs = run_dir / "runs"
    runs.mkdir(parents=True)
    result = runs / rb._job_name("fuzzy_ite_cross", "rulekan", 0)
    result.write_text(json.dumps({
        "status": "completed",
        "task": "fuzzy_ite_cross",
        "model": "rulekan",
        "seed": 0,
        "benchmark_build_fingerprint": "deadbeefdeadbeef",
    }))

    monkeypatch.setattr(rb, "benchmark_build_fingerprint", lambda *args, **kwargs: "cafebabecafebabe")
    monkeypatch.setattr(rb, "code_version", lambda *args, **kwargs: "test")
    monkeypatch.setattr(rb, "aggregate", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.run_benchmark",
        "--profile", "ablation_quick",
        "--tasks", "fuzzy_ite_cross",
        "--models", "rulekan",
        "--seeds", "0",
        "--run-dir", str(run_dir),
        "--reuse-completed",
    ])

    assert rb.main() == 0
    out = capsys.readouterr().out
    assert "deadbeef" not in out
    assert "cafebabe" not in out
    assert "stale-build" not in out
    assert "reused 1 completed despite code/configuration changes" in out


def test_external_dependency_preflight_imports_operon_wrapper(monkeypatch):
    from benchmarks import run_benchmark as rb

    calls = []

    def fake_import(name):
        calls.append(name)
        if name == "pyoperon.sklearn":
            raise OSError("missing shared library")
        return object()

    monkeypatch.setattr(rb.importlib, "import_module", fake_import)
    missing = rb._missing_optional_model_dependencies(["operon"])
    assert calls == ["pyoperon.sklearn"]
    assert len(missing) == 1
    assert missing[0][0] == "operon"
    assert "OSError: missing shared library" in missing[0][3]


def test_optional_dependency_preflight_does_not_import_pysr(monkeypatch):
    from benchmarks import run_benchmark as rb

    monkeypatch.setattr(rb.importlib.util, "find_spec", lambda name: object() if name == "pysr" else None)
    monkeypatch.setattr(rb.importlib, "import_module", lambda name: (_ for _ in ()).throw(AssertionError(name)))
    assert rb._missing_optional_model_dependencies(["pysr"]) == []


def test_srkan_preflight_rejects_unrelated_package(monkeypatch):
    from benchmarks import run_benchmark as rb

    fake = type("WrongSrkan", (), {})()
    monkeypatch.setattr(rb.importlib, "import_module", lambda name: fake)
    missing = rb._missing_optional_model_dependencies(["srkan"])
    assert len(missing) == 1
    assert missing[0][0] == "srkan"
    assert "not the official symbolic-regression SR-KAN API" in missing[0][3]


def test_srkan_preflight_accepts_official_api(monkeypatch):
    from benchmarks import run_benchmark as rb

    fake = type("OfficialSrkan", (), {"regressor": object(), "SympyEvaluator": object()})()
    monkeypatch.setattr(rb.importlib, "import_module", lambda name: fake)
    assert rb._missing_optional_model_dependencies(["srkan"]) == []


def test_fuzzy_predictive_summary_includes_anfis_without_fabricating_rule_recovery(tmp_path: Path):
    import json
    import pandas as pd
    from benchmarks.aggregate import aggregate

    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {
            "status": "completed", "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
            "task_type": "regression", "model": "anfis", "seed": 0,
            "test_rmse": 0.02, "test_nrmse": 0.03, "elapsed_seconds": 0.5,
            "finished_unix": 1.0,
        },
        {
            "status": "completed", "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
            "task_type": "regression", "model": "rulekan", "seed": 0,
            "test_rmse": 0.04, "test_nrmse": 0.06,
            "symbolic_test_rmse": 0.01, "symbolic_test_nrmse": 0.015,
            "symbolic_seconds": 0.4,
            "fuzzy_rule_f1": 1.0, "fuzzy_gate_recall": 1.0, "fuzzy_branch_recall": 1.0,
            "elapsed_seconds": 1.0, "finished_unix": 2.0,
        },
    ]
    for i, row in enumerate(rows):
        (runs / f"r{i}.json").write_text(json.dumps(row))
    aggregate(tmp_path, quiet=True)

    pred = pd.read_csv(tmp_path / "fuzzy_predictive_summary.csv")
    assert set(pred.model) == {"anfis", "rulekan"}
    anfis = pred[pred.model.eq("anfis")].iloc[0]
    assert abs(float(anfis.final_nrmse_median) - 0.03) < 1e-12
    structural = pd.read_csv(tmp_path / "fuzzy_summary.csv")
    assert "anfis" not in set(structural.model)
    assert (tmp_path / "figures" / "fuzzy_predictive_nrmse.pdf").exists()


def test_compositional_rulekan_variants_and_profiles_are_registered():
    import yaml
    from benchmarks.aggregate import MODEL_LABELS
    from benchmarks.models import TRAINERS

    assert {"rulekan_comp", "sisp_comp", "power_rulekan_comp"}.issubset(TRAINERS)
    assert MODEL_LABELS["rulekan_comp"] == "RuleKAN-Comp"
    assert MODEL_LABELS["sisp_comp"] == "SISP-Comp"
    assert MODEL_LABELS["power_rulekan_comp"] == "PowerRuleKAN-Comp"

    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    research = cfg["profiles"]["research"]
    assert {"rulekan_comp", "sisp_comp", "power_rulekan_comp"}.issubset(set(research["models"]))

    quick = cfg["profiles"]["composition_ablation_quick"]
    assert {"rulekan", "rulekan_comp", "sisp", "sisp_comp", "power_rulekan", "power_rulekan_comp", "pysr", "multkan_deep_gsr"}.issubset(set(quick["models"]))
    assert "nested_same_tanh_sin" in quick["tasks"]
    assert "same_var_exp_sin" in quick["tasks"]  # non-nested regression control


def test_statistical_tests_use_task_level_medians_and_write_holm_outputs(tmp_path: Path):
    import json
    import pandas as pd
    from benchmarks.aggregate import aggregate

    runs = tmp_path / "runs"
    runs.mkdir()
    models = {"rulekan": 1.0, "power_rulekan_comp": 0.55, "pysr": 1.8}
    k = 0
    for task_idx in range(10):
        for model, scale in models.items():
            for seed in range(3):
                nrmse = scale * (0.01 + 0.001 * task_idx) * (1.0 + 0.01 * seed)
                row = {
                    "status": "completed", "suite": "synthetic_core",
                    "task": f"stat_task_{task_idx}", "task_type": "regression",
                    "model": model, "seed": seed,
                    "test_rmse": nrmse, "test_nrmse": nrmse,
                    "symbolic_test_rmse": nrmse, "symbolic_test_nrmse": nrmse,
                    "symbolic_seconds": 0.1 + 0.01 * seed,
                    "elapsed_seconds": 0.2, "finished_unix": float(k + 1),
                }
                (runs / f"r{k}.json").write_text(json.dumps(row)); k += 1

    aggregate(tmp_path, quiet=True)
    pair = pd.read_csv(tmp_path / "statistical_final_nrmse_pairwise.csv")
    friedman = pd.read_csv(tmp_path / "statistical_final_nrmse_friedman.csv")
    ranks = pd.read_csv(tmp_path / "statistical_final_nrmse_ranks.csv")
    task_medians = pd.read_csv(tmp_path / "statistical_final_nrmse_task_medians.csv")

    assert set(pair.method_a).union(set(pair.method_b)) == set(models)
    assert pair["n_tasks"].min() == 10
    assert pair["p_holm"].notna().all()
    assert int(friedman.iloc[0].n_tasks) == 10
    assert set(ranks.model) == set(models)
    assert len(task_medians) == 30  # 10 tasks x 3 methods, seeds already collapsed
    assert (tmp_path / "statistical_tests.md").exists()
    assert (tmp_path / "figures" / "statistical_final_nrmse_ranks.pdf").exists()


def test_statistical_ranks_ignore_non_main_records_from_same_run_dir(tmp_path: Path):
    import json
    import pandas as pd
    import yaml
    from benchmarks.aggregate import aggregate

    runs = tmp_path / "runs"
    runs.mkdir()
    main = ["rulekan", "power_rulekan_comp", "pysr"]
    extras = ["sindy_unconstrained", "rulekan_no_product", "anfis", "pse"]
    snapshot = {
        "profile": "research_modern",
        "config": {"main_comparison_models": main},
    }
    (tmp_path / "benchmark_config_snapshot.yaml").write_text(yaml.safe_dump(snapshot))

    k = 0
    for task_idx in range(8):
        for model in [*main, *extras]:
            for seed in range(3):
                # Give extras absurdly good errors so leakage would visibly change ranks.
                scale = 1e-6 if model in extras else {
                    "rulekan": 1.0, "power_rulekan_comp": 0.5, "pysr": 1.5
                }[model]
                nrmse = scale * (0.01 + 0.001 * task_idx)
                row = {
                    "status": "completed", "suite": "synthetic_core",
                    "task": f"main_task_{task_idx}", "task_type": "regression",
                    "model": model, "seed": seed,
                    "test_rmse": nrmse, "test_nrmse": nrmse,
                    "symbolic_test_rmse": nrmse, "symbolic_test_nrmse": nrmse,
                    "symbolic_seconds": 0.05, "elapsed_seconds": 0.1,
                    "finished_unix": float(k + 1),
                }
                (runs / f"rank{k}.json").write_text(json.dumps(row)); k += 1

    aggregate(tmp_path, quiet=True)
    ranks = pd.read_csv(tmp_path / "statistical_final_nrmse_ranks.csv")
    pair = pd.read_csv(tmp_path / "statistical_final_nrmse_pairwise.csv")
    task = pd.read_csv(tmp_path / "statistical_final_nrmse_task_medians.csv")
    assert set(ranks.model) == set(main)
    assert set(pair.method_a).union(set(pair.method_b)) == set(main)
    assert set(task.model) == set(main)
    assert not set(extras).intersection(set(ranks.model))
    assert (tmp_path / "statistical_model_set.txt").read_text().splitlines() == main


def test_external_fuzzy_formula_backfill_unstandardizes_srkan_coordinates():
    """Expanded SR-KAN formulas must be scored in the raw fuzzy-rule gauge."""
    from benchmarks.aggregate import fuzzy_formula_recovery_scores, _row_input_standardization
    from benchmarks.specs import TASKS

    # Historical completed SR-KAN seed from fuzzy_ite_cross.  The expression is
    # written in the standardized coordinates seen by the external regressor;
    # after undoing the benchmark transform and collecting nearly identical
    # atoms it is exactly the two intended fuzzy rules.
    formula = (
        "-0.203880165628579*x_0*sin(1.85669631575588*x_1 + 0.00800693453575741) "
        "+ 0.323098851293211*x_0*exp(-0.629272504310598*x_2) "
        "+ 0.351395284064836*sin(1.85669631904184*x_1 + 0.00800691771775936) "
        "+ 0.552450924839964*exp(-0.629272483987078*x_2)"
    )
    row = {
        "task": "fuzzy_ite_cross", "model": "srkan", "seed": 1,
        "train_n": 1600, "val_n": 400, "test_n": 500,
    }
    spec = TASKS[row["task"]]
    mean, std = _row_input_standardization(row, spec)
    scores = fuzzy_formula_recovery_scores(
        formula, row["task"], input_mean=mean, input_std=std,
        formula_input_space="standardized",
    )
    assert scores["fuzzy_rule_f1"] == 1.0
    assert scores["fuzzy_gate_recall"] == 1.0
    assert scores["fuzzy_branch_recall"] == 1.0
    assert scores["fuzzy_exact_structure_recovery"] == 1.0
    assert scores["fuzzy_found_rules"] == 2.0


def test_external_fuzzy_dataframe_backfill_recovers_historical_input_scaling():
    import pandas as pd
    from benchmarks.aggregate import _backfill_fuzzy_formula_metrics

    formula = (
        "-0.203880165628579*x_0*sin(1.85669631575588*x_1 + 0.00800693453575741) "
        "+ 0.323098851293211*x_0*exp(-0.629272504310598*x_2) "
        "+ 0.351395284064836*sin(1.85669631904184*x_1 + 0.00800691771775936) "
        "+ 0.552450924839964*exp(-0.629272483987078*x_2)"
    )
    df = pd.DataFrame([{
        "suite": "fuzzy_rules", "task": "fuzzy_ite_cross", "task_type": "regression",
        "model": "srkan", "seed": 1, "formula": formula,
        "train_n": 1600, "val_n": 400, "test_n": 500,
    }])
    out = _backfill_fuzzy_formula_metrics(df)
    assert float(out.iloc[0]["fuzzy_rule_f1"]) == 1.0
    assert float(out.iloc[0]["fuzzy_exact_structure_recovery"]) == 1.0
    assert float(out.iloc[0]["fuzzy_formula_semantic_backfill"]) == 1.0


def test_srkan_output_transform_domain_guard_rejects_nested_tree_singular_transforms():
    from benchmarks.models import _safe_srkan_output_transforms
    from benchmarks.specs import TASKS, make_synthetic_data

    data = make_synthetic_data(
        TASKS["fuzzy_nested_tree"], seed=1, train_n=768, val_n=192, test_n=384
    )
    y = data.train_y.detach().cpu().numpy().reshape(-1)
    safe, dropped = _safe_srkan_output_transforms(
        y, ["inv", "square", "sqrt", "log"]
    )
    assert safe == []
    assert set(dropped) == {"inv", "square", "sqrt", "log"}


def test_srkan_output_transform_domain_guard_keeps_valid_positive_transforms():
    import numpy as np
    from benchmarks.models import _safe_srkan_output_transforms

    y = np.linspace(1.0, 2.0, 100)
    safe, dropped = _safe_srkan_output_transforms(
        y, ["inv", "square", "sqrt", "log"]
    )
    assert safe == ["inv", "square", "sqrt", "log"]
    assert dropped == []


def test_gate_aware_support_augmentation_requires_shared_branch_evidence():
    from benchmarks.models import _gate_aware_support_augmentation
    from benchmarks.specs import TASKS, make_synthetic_data

    data = make_synthetic_data(
        TASKS["fuzzy_nested_tree"], seed=0, train_n=400, val_n=100, test_n=100
    )
    classes = [
        {"support": (0, 3), "score": 1.0, "structure_probability": 1.0, "numeric_strength": 1.0},
        {"support": (1, 3), "score": 0.9, "structure_probability": 0.9, "numeric_strength": 0.8},
        {"support": (0, 2), "score": 0.8, "structure_probability": 0.8, "numeric_strength": 0.7},
        {"support": (1, 4), "score": 0.7, "structure_probability": 0.7, "numeric_strength": 0.6},
    ]
    out = _gate_aware_support_augmentation(classes, data, max_factors=3)
    supports = {tuple(c["support"]) for c in out}
    assert (0, 1, 3) in supports
    # Disjoint branch evidence must not be fused into a nested mechanism.
    assert (0, 1, 2) not in supports
    assert (0, 1, 4) not in supports


def test_long_expression_capacity_suite_is_registered_and_near_common_caps():
    import yaml
    from benchmarks.config_utils import resolve_profile
    suites = list_suites()
    assert set(suites["expression_length_stress"]) == {"long_additive_12", "long_product_6"}
    a = TASKS["long_additive_12"]
    p = TASKS["long_product_6"]
    assert a.n_var == p.n_var == 12
    assert a.target_terms == 12 and a.target_tree_nodes == 39 and a.max_factors == 1
    assert p.target_terms == 6 and p.target_tree_nodes == 39 and p.max_factors == 2
    assert len(a.expected_structures) == 12
    assert len(p.expected_structures) == 6

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    prof = resolve_profile(cfg, "long_expression")
    assert prof["suites"] == ["expression_length_stress"]
    assert prof["model_config"]["rulekan"]["symbolic_max_rules"] == 12
    assert prof["model_config"]["pysr"]["maxsize"] == 40
    assert prof["model_config"]["operon"]["max_length"] == 50


def test_long_expression_targets_are_finite_and_nontrivial():
    import torch
    for name in ("long_additive_12", "long_product_6"):
        data = make_synthetic_data(TASKS[name], seed=5, train_n=128, val_n=32, test_n=40)
        assert data.train_x.shape[1] == 12
        assert torch.isfinite(data.train_y).all()
        assert torch.isfinite(data.test_y).all()
        assert float(data.train_y.std()) > 0.1


def test_complementary_two_rule_rescue_recovers_same_variable_family_without_expanding_supports():
    import torch
    from rulekan.sum_product_kan import (
        SumProductKAN,
        _make_fully_symbolic_shell,
        complementary_two_rule_symbolic_rescue,
    )

    data = make_synthetic_data(
        TASKS["fuzzy_ite_same_variable"], seed=0, train_n=320, val_n=96, test_n=128
    )
    library = ("x", "sin", "tanh", "sqrt")
    numeric = SumProductKAN(
        in_dim=1, n_rules=4, max_factors=2, grid=6,
        symbolic_library=library, seed=0,
    )
    numeric.discretize()
    incumbent = _make_fully_symbolic_shell(numeric)
    with torch.no_grad():
        incumbent.rule_scale.zero_()
        incumbent.bias.fill_(float(data.train_y.mean()))

    rescued, meta = complementary_two_rule_symbolic_rescue(
        numeric, incumbent,
        data.train_x, data.train_y, data.val_x, data.val_y,
        input_mean=data.input_mean, input_std=data.input_std,
        structure_candidates=[(0, 0)],
        # RuleKAN's support contract contains only x0; the rescue is not allowed
        # to invent any other support even though it performs a richer family search.
        allowed_supports=[(0,)],
        library=library,
        max_samples=256,
        coarse_global_topk=16,
        coarse_final_topk=16,
        seed_restarts=3,
        local_polish_steps=20,
        local_lbfgs_topk=16,
        local_lbfgs_steps=30,
        family_final_topk=8,
        final_steps=80,
        final_lbfgs_steps=30,
    )
    assert meta["selected"] is True
    assert (meta["operator_a"], meta["operator_b"]) == ("sin", "tanh")
    assert meta["support_a"] == [0, 0]
    assert meta["support_b"] == [0, 0]
    with torch.no_grad():
        nrmse = float(torch.sqrt(torch.mean((rescued(data.test_x) - data.test_y) ** 2)) / data.test_y.std())
    assert nrmse < 5e-5

    # RuleSISP uses the same optimizer but deliberately passes no support
    # restriction.  The rescue must therefore remain structure-independent
    # rather than inheriting RuleKAN's learned-support contract.
    sisp_rescued, sisp_meta = complementary_two_rule_symbolic_rescue(
        numeric, incumbent,
        data.train_x, data.train_y, data.val_x, data.val_y,
        input_mean=data.input_mean, input_std=data.input_std,
        structure_candidates=[(0, 0)],
        allowed_supports=None,
        library=("x", "sin", "tanh"),
        max_samples=192,
        coarse_global_topk=9,
        coarse_final_topk=9,
        seed_restarts=2,
        local_polish_steps=16,
        local_lbfgs_topk=9,
        local_lbfgs_steps=24,
        family_final_topk=6,
        final_steps=60,
        final_lbfgs_steps=24,
    )
    assert sisp_meta["selected"] is True
    assert (sisp_meta["operator_a"], sisp_meta["operator_b"]) == ("sin", "tanh")



def test_research_modern_profile_adds_contemporary_symbolic_baselines():
    import yaml
    from benchmarks.config_utils import resolve_profile
    from benchmarks.models import TRAINERS
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "benchmarks" / "configs" / "default.yaml").read_text())
    modern = resolve_profile(cfg, "research_modern")
    assert {"symbolic_kan", "pse", "rils_rols", "sindy", "parfam", "eql"}.issubset(modern["models"])
    assert "udsr" not in modern["models"]  # separate legacy-Python environment only
    assert {"symbolic_kan", "pse", "rils_rols", "udsr", "sindy", "sindy_unconstrained", "parfam", "eql"}.issubset(TRAINERS)
    assert "sindy_unconstrained" not in modern["models"]
    assert modern["model_config"]["sindy"]["max_active_terms"] == 12
    assert "sindy_unconstrained" not in modern["main_comparison_models"]
    assert "pse" not in modern["main_comparison_models"]
    assert "anfis" not in modern["main_comparison_models"]
    assert len(modern["main_comparison_models"]) == 23
    assert modern["model_config"]["pse"]["n_symbol_layers"] == 3
    assert modern["model_config"]["rils_rols"]["max_fit_calls"] == 100000


def test_modern_sr_requirements_are_declared():
    root = Path(__file__).resolve().parents[1]
    req = (root / "benchmarks" / "requirements-modern-sr.txt").read_text()
    assert "psrn" in req
    assert "rils-rols" in req
    assert "deep-symbolic-optimization-pytorch" not in req
    assert "uDSR/DSO" in req
    assert "pybind11" in req
    assert "pysindy==2.1.0" in req
    assert "parfam==0.0.2" in req


def test_research_modern_vocabulary_matching_across_symbolic_baselines():
    import yaml
    from benchmarks.config_utils import resolve_profile
    from benchmarks.models import (
        OPERON_TARGET_CORE_SYMBOLS,
        PSE_TARGET_CORE_NATIVE,
        PYSR_TARGET_CORE_BINARY,
        PYSR_TARGET_CORE_UNARY,
        SYMBOLIC_KAN_TARGET_CORE_NATIVE,
        TARGET_CORE_SYMBOLIC_LIBRARY,
        UDSR_TARGET_CORE_NATIVE,
        SINDY_TARGET_CORE_NATIVE,
        PARFAM_TARGET_CORE_FUNCTIONS,
        EQL_TARGET_CORE_NATIVE,
        resolve_shared_benchmark_config,
    )

    cfg = yaml.safe_load(Path("benchmarks/configs/default.yaml").read_text())
    profile = resolve_profile(cfg, "research_modern")
    shared = profile["shared_settings"]
    spec = TASKS["same_var_exp_sin"]

    # KAN-derived methods and SR-KAN can use the exact conceptual core10 bank.
    gsr, gsr_meta = resolve_shared_benchmark_config(
        "gsr", spec, profile.get("model_config", {}).get("gsr", {}), shared
    )
    srkan, srkan_meta = resolve_shared_benchmark_config(
        "srkan", spec, profile.get("model_config", {}).get("srkan", {}), shared
    )
    assert tuple(gsr["symbolic_library"]) == tuple(TARGET_CORE_SYMBOLIC_LIBRARY)
    assert gsr_meta["shared_symbolic_native_exact_match"] is True
    assert srkan["functions"] == ["target_core"]
    assert srkan_meta["shared_symbolic_native_exact_match"] is True

    # Official Symbolic-KAN uses the closest native atoms.  Do not duplicate
    # identity (id + x), and do not give it the x^3 shortcut used previously.
    skan, skan_meta = resolve_shared_benchmark_config(
        "symbolic_kan", spec, profile["model_config"]["symbolic_kan"], shared
    )
    assert tuple(skan["lib"]) == tuple(SYMBOLIC_KAN_TARGET_CORE_NATIVE)
    assert "id" not in skan["lib"] and "x3" not in skan["lib"]
    assert skan_meta["shared_symbolic_library"] == "core10"
    assert skan_meta["shared_symbolic_library_size"] == 10
    assert skan_meta["shared_symbolic_native_exact_match"] is False
    assert "1/x^2" in skan_meta["shared_symbolic_native_note"]

    # General recursive SR systems are restricted to core-compatible direct
    # unary shortcuts; ordinary tree arithmetic remains available to compose
    # products, reciprocals, and inverse squares.
    pysr, pysr_meta = resolve_shared_benchmark_config(
        "pysr", spec, profile.get("model_config", {}).get("pysr", {}), shared
    )
    assert tuple(pysr["binary_operators"]) == tuple(PYSR_TARGET_CORE_BINARY)
    assert tuple(pysr["unary_operators"]) == tuple(PYSR_TARGET_CORE_UNARY)
    assert {"cube", "atan", "abs"}.isdisjoint(pysr["unary_operators"])
    assert pysr_meta["shared_symbolic_native_exact_match"] is False

    operon, operon_meta = resolve_shared_benchmark_config(
        "operon", spec, profile.get("model_config", {}).get("operon", {}), shared
    )
    assert tuple(operon["allowed_symbols"].split(",")) == tuple(OPERON_TARGET_CORE_SYMBOLS)
    assert {"atan", "abs"}.isdisjoint(operon["allowed_symbols"].split(","))
    assert operon_meta["shared_symbolic_native_exact_match"] is False

    # These baselines have method-native grammar constraints that prevent a
    # literal one-to-one core10 bank; the resolver records the exception rather
    # than silently claiming exact vocabulary matching.
    pse, pse_meta = resolve_shared_benchmark_config(
        "pse", spec, profile["model_config"]["pse"], shared
    )
    assert tuple(pse["operators"]) == tuple(PSE_TARGET_CORE_NATIVE)
    assert pse_meta["shared_symbolic_native_exact_match"] is False

    udsr, udsr_meta = resolve_shared_benchmark_config(
        "udsr", spec, profile["model_config"]["udsr"], shared
    )
    assert tuple(udsr["function_set"]) == tuple(UDSR_TARGET_CORE_NATIVE)
    assert "poly" in udsr["function_set"]
    assert udsr_meta["shared_symbolic_native_exact_match"] is False

    _, rils_meta = resolve_shared_benchmark_config(
        "rils_rols", spec, profile["model_config"]["rils_rols"], shared
    )
    assert rils_meta["shared_symbolic_native_exact_match"] is False
    assert "does not expose" in rils_meta["shared_symbolic_native_note"]

    sindy, sindy_meta = resolve_shared_benchmark_config(
        "sindy", spec, profile["model_config"]["sindy"], shared
    )
    assert tuple(sindy["primitive_library"]) == tuple(SINDY_TARGET_CORE_NATIVE)
    assert sindy["max_active_terms"] == 12
    assert sindy["max_interaction_order"] == min(3, int(sindy_meta["shared_max_product_order"]))
    assert sindy_meta["shared_symbolic_native_exact_match"] is True

    sindy_u, sindy_u_meta = resolve_shared_benchmark_config(
        "sindy_unconstrained", spec,
        {**profile["model_config"]["sindy"], "max_active_terms": None}, shared
    )
    assert tuple(sindy_u["primitive_library"]) == tuple(SINDY_TARGET_CORE_NATIVE)
    assert sindy_u["max_active_terms"] is None
    assert sindy_u["max_interaction_order"] == sindy["max_interaction_order"]
    assert sindy_u_meta["shared_symbolic_native_exact_match"] is True

    parfam, parfam_meta = resolve_shared_benchmark_config(
        "parfam", spec, profile["model_config"]["parfam"], shared
    )
    assert tuple(parfam["functions"]) == tuple(PARFAM_TARGET_CORE_FUNCTIONS)
    assert parfam_meta["shared_symbolic_native_exact_match"] is False

    eql, eql_meta = resolve_shared_benchmark_config(
        "eql", spec, profile["model_config"]["eql"], shared
    )
    assert tuple(eql["unary_library"]) == tuple(EQL_TARGET_CORE_NATIVE)
    assert eql["units_per_type"] == 10
    assert eql["total_layers"] == [2, 3, 4]
    assert eql["phase1_frac"] == 0.25
    assert eql["phase2_frac"] == 0.95
    assert eql_meta["shared_symbolic_native_exact_match"] is False



def test_setup_reconstructs_gitignored_external_symbolic_kan_checkout():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "benchmarks" / "setup.sh").read_text()
    gitignore = (root / ".gitignore").read_text()

    assert 'SYMBOLIC_KAN_COMMIT="${SYMBOLIC_KAN_COMMIT:-9481a82}"' in setup
    assert 'SYMBOLIC_KAN_REPO="${SYMBOLIC_KAN_REPO:-https://github.com/sfaroughi3/Pub_Symbolic_KANs.git}"' in setup
    assert 'install_symbolic_kan()' in setup
    assert 'git clone "$SYMBOLIC_KAN_REPO" "$SYMBOLIC_KAN_DIR"' in setup
    assert 'checkout --quiet --detach "$SYMBOLIC_KAN_COMMIT"' in setup
    assert 'Exp_reaction_diffusion/symKanTraining.py' in setup
    assert 'external/*' in gitignore


def test_setup_keeps_special_binary_baselines_out_of_plain_requirements():
    root = Path(__file__).resolve().parents[1]
    req = (root / "benchmarks" / "requirements-benchmark.txt").read_text()
    setup = (root / "benchmarks" / "setup.sh").read_text()

    requirements = [
        ln.strip()
        for ln in req.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]

    assert "rils-rols" not in requirements
    assert not any(line.startswith("pyoperon") for line in requirements)
    assert "deep-symbolic-optimization-pytorch" not in req

    # RILS-ROLS still requires its special installation path.
    assert "--no-build-isolation rils-rols" in setup

    # PyOperon is installed separately from the plain requirements file. Its
    # wheel/import is allowed to fail without aborting the remaining baseline
    # setup, and macOS repair is explicitly opt-in.
    assert 'PYOPERON_VERSION="${PYOPERON_VERSION:-0.6.1}"' in setup
    assert 'INSTALL_OPERON="${INSTALL_OPERON:-1}"' in setup
    assert 'OPERON_MACOS_FIX="${OPERON_MACOS_FIX:-0}"' in setup
    assert "install_pyoperon()" in setup
    assert "report_pyoperon_failure()" in setup
    assert "repair_pyoperon_macos()" in setup
    assert "--only-binary=:all:" in setup

    # The previous PyOperon source-build path is intentionally not used.
    assert "python script/dependencies.py" not in setup


def test_external_fuzzy_backfill_unscorable_formula_is_zero_not_nan():
    """Bounded-DNF overflow is a failed recovery, not missing structural data."""
    import pandas as pd
    from benchmarks.aggregate import _backfill_fuzzy_formula_metrics

    # More outer terms than the bounded structural parser intentionally permits.
    formula = " + ".join(f"sin({i + 1}*x0)" for i in range(300))
    df = pd.DataFrame([{
        "suite": "fuzzy_rules", "task": "fuzzy_ite_same_variable",
        "task_type": "regression", "model": "external_test", "seed": 0,
        "status": "completed", "formula": formula, "formula_input_space": "raw",
    }])
    out = _backfill_fuzzy_formula_metrics(df)
    row = out.iloc[0]
    assert float(row["fuzzy_rule_f1"]) == 0.0
    assert float(row["fuzzy_gate_recall"]) == 0.0
    assert float(row["fuzzy_branch_recall"]) == 0.0
    assert float(row["fuzzy_exact_structure_recovery"]) == 0.0
    assert float(row["fuzzy_formula_semantic_backfill_failed"]) == 1.0
    assert row["fuzzy_formula_semantic_backfill_reason"] == "unscorable_formula_or_dnf_limit"


def test_external_fuzzy_backfill_failed_run_is_zero_not_nan():
    import pandas as pd
    from benchmarks.aggregate import _backfill_fuzzy_formula_metrics

    df = pd.DataFrame([{
        "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
        "task_type": "regression", "model": "external_test", "seed": 0,
        "status": "failed", "error": "timeout",
    }])
    out = _backfill_fuzzy_formula_metrics(df)
    row = out.iloc[0]
    assert float(row["fuzzy_rule_f1"]) == 0.0
    assert float(row["fuzzy_exact_structure_recovery"]) == 0.0
    assert row["fuzzy_formula_semantic_backfill_reason"] == "run_status:failed"


def test_resume_reuses_changed_build_completed_by_default(tmp_path, monkeypatch, capsys):
    """Plain --resume/default invocation should actually behave like a cache."""
    import json
    import sys
    from benchmarks import run_benchmark as rb

    run_dir = tmp_path / "run"
    runs = run_dir / "runs"
    runs.mkdir(parents=True)
    result = runs / rb._job_name("fuzzy_ite_cross", "rulekan", 0)
    result.write_text(json.dumps({
        "status": "completed", "task": "fuzzy_ite_cross", "suite": "fuzzy_rules",
        "task_type": "regression", "model": "rulekan", "seed": 0,
        "benchmark_build_fingerprint": "oldoldoldoldold1",
    }))

    monkeypatch.setattr(rb, "benchmark_build_fingerprint", lambda *a, **k: "newnewnewnewnew2")
    monkeypatch.setattr(rb, "code_version", lambda *a, **k: "test")
    monkeypatch.setattr(rb, "aggregate", lambda *a, **k: None)
    monkeypatch.setattr(rb, "_validate_optional_model_dependencies", lambda models: (_ for _ in ()).throw(AssertionError("no runnable dependency preflight expected")))
    monkeypatch.setattr(sys, "argv", [
        "benchmarks.run_benchmark", "--profile", "ablation_quick",
        "--tasks", "fuzzy_ite_cross", "--models", "rulekan", "--seeds", "0",
        "--run-dir", str(run_dir),
    ])

    assert rb.main() == 0
    out = capsys.readouterr().out
    assert "reused 1 completed despite code/configuration changes" in out
    assert "runnable=1" not in out


def test_aggregate_fuzzy_structure_keeps_failed_external_run_in_denominator(tmp_path: Path):
    import json
    import pandas as pd
    from benchmarks.aggregate import aggregate

    runs = tmp_path / "runs"
    runs.mkdir()
    rows = [
        {
            "status": "completed", "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
            "task_type": "regression", "model": "rulekan", "seed": 0,
            "test_rmse": 0.1, "test_nrmse": 0.1,
            "symbolic_test_rmse": 0.1, "symbolic_test_nrmse": 0.1,
            "symbolic_seconds": 0.1, "elapsed_seconds": 0.2,
            "fuzzy_rule_f1": 1.0, "fuzzy_gate_recall": 1.0,
            "fuzzy_branch_recall": 1.0, "fuzzy_exact_structure_recovery": 1.0,
        },
        {
            "status": "failed", "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
            "task_type": "regression", "model": "sindy", "seed": 0,
            "error": "timeout",
        },
        {
            "status": "completed", "suite": "fuzzy_rules", "task": "fuzzy_ite_cross",
            "task_type": "regression", "model": "anfis", "seed": 0,
            "test_rmse": 0.2, "test_nrmse": 0.2,
        },
    ]
    for i, row in enumerate(rows):
        (runs / f"r{i}.json").write_text(json.dumps(row))

    aggregate(tmp_path, quiet=True)
    fs = pd.read_csv(tmp_path / "fuzzy_summary.csv")
    assert set(fs.model) == {"rulekan", "sindy"}
    sr = fs[fs.model.eq("sindy")].iloc[0]
    assert int(sr.n) == 1
    assert float(sr.fuzzy_rule_f1_median) == 0.0
    assert float(sr.fuzzy_exact_structure_recovery_median) == 0.0
