from __future__ import annotations

import copy
import inspect
import json
import math
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score

from symbolic_kan import KAN
from symbolic_kan.MLP import MLP
from symbolic_kan import (
    SumProductKAN,
    default_sum_product_schedule,
    fit_sum_product_kan,
    hard_numeric_plateau_polish,
    hard_numeric_precision_polish,
    mandatory_symbolic_matching_pursuit,
    learned_support_symbolic_gsr,
    capture_numeric_support_evidence,
    learned_numeric_support_classes,
    learned_structure_symbolic_bank,
    distill_numeric_structure_to_symbolic,
    project_constrained_numeric_to_symbolic,
    scaled_gmp_screening_sizes,
    symbolic_structure_bank,
    resolve_gmp_local_policy,
    prune_numeric_structure_to_stability,
    rule_contribution_redundancy,
    numeric_logic_diagnostics,
    compress_numeric_rule_bank,
    PowerRuleKAN,
    inverse_power_target,
    safe_integer_power,
    fit_affine_atom,
    harden_symbolic_base,
    hard_power_polish,
    hard_ratio_polish,
    low_dimensional_symbolic_family_rescue,
    complementary_two_rule_symbolic_rescue,
    product_partition_symbolic_rescue,
    recursive_partition_symbolic_rescue,
)

from .specs import BenchmarkData, TaskSpec
from .anfis import CompactANFIS, fit_compact_anfis
from .symbolic_kan_baseline import (
    fit_official_symbolic_kan, official_formula, official_hardened_predict,
)
from symbolic_kan.composition_rulekan import ComposedRuleKAN, depth2_composition_rescue
from symbolic_kan.sum_product_kan import _fully_symbolic_continuous_refit


# Exact 25-form operator library used by the paper-comparison baselines.
PAPER_SYMBOLIC_LIBRARY = (
    "0", "1", "x", "x^2", "x^3", "x^4", "x^5",
    "1/x", "1/x^2", "1/x^3", "sqrt", "1/sqrt(x)",
    "log", "exp", "sin", "cos", "tan", "tanh", "abs", "sgn",
    "arctan", "arcsin", "arccos", "arctanh", "gaussian",
)

# Broader sensitivity library.  The paper's constants 0/1 are excluded because
# RuleKAN already has an explicit global bias and per-rule amplitude.  This
# optional library deliberately retains compound shortcuts; the controlled
# benchmark below does not use them.
RESEARCH_SYMBOLIC_LIBRARY = tuple(
    [name for name in PAPER_SYMBOLIC_LIBRARY if name not in {"0", "1"}]
    + ["log1p_sq", "sqrt1p_sq", "inv1p_sq"]
)

COMPACT_SYMBOLIC_LIBRARY = (
    "x", "x^2", "x^3", "exp", "sin", "cos", "tanh", "arctan",
)

# Elementary target-complete vocabulary for controlled comparisons.  Compound
# shortcuts such as gaussian, log(1+x^2), sqrt(1+x^2), and 1/(1+x^2) are
# intentionally excluded and must be reconstructed by composition.
TARGET_CORE_SYMBOLIC_LIBRARY = (
    "x", "x^2", "1/x", "1/x^2", "sqrt", "log", "exp", "sin", "cos", "tanh",
)

# Native spellings/grammars used to align external symbolic-regression methods
# with ``core10`` as closely as their public APIs permit.  For recursive-tree
# methods, arithmetic operators are structural grammar rather than additional
# unary shortcuts, so e.g. ``1/x^2`` may be composed from division + square.
SYMBOLIC_KAN_TARGET_CORE_NATIVE = (
    "x", "x2", "inv", "sqrtx", "log", "exp", "sin", "cos", "tanh",
)
PYSR_TARGET_CORE_UNARY = (
    "square", "exp", "sin", "cos", "tanh", "sqrt", "log", "inv",
)
PYSR_TARGET_CORE_BINARY = ("+", "-", "*", "/")
OPERON_TARGET_CORE_SYMBOLS = (
    "add", "sub", "mul", "div", "constant", "variable", "square",
    "exp", "sin", "cos", "tanh", "sqrt", "log",
)
PSE_TARGET_CORE_NATIVE = (
    "Add", "Mul", "Sub", "Div", "Identity", "Sin", "Cos", "Exp", "Log", "Tanh",
)
UDSR_TARGET_CORE_NATIVE = (
    "add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly",
)

# SR-KAN accepts a configurable univariate extraction library, but four of the
# shared elementary atoms use different native names.
_SRKAN_TARGET_CORE_ALIASES = {
    "x": "linear",
    "x^2": "square",
    "1/x": "inv_x",
    "1/x^2": "inv_x2",
}

MEDIUM_SYMBOLIC_LIBRARY = TARGET_CORE_SYMBOLIC_LIBRARY + (
    "x^3", "x^4", "x^5", "1/x^3", "abs", "arctan",
)

assert set(TARGET_CORE_SYMBOLIC_LIBRARY).issubset(MEDIUM_SYMBOLIC_LIBRARY)
assert set(MEDIUM_SYMBOLIC_LIBRARY).issubset(RESEARCH_SYMBOLIC_LIBRARY)


def resolve_shared_benchmark_config(
    model_name: str,
    spec: TaskSpec,
    cfg: Dict[str, Any],
    shared: Optional[Dict[str, Any]] = None,
    *,
    width_override: Optional[int] = None,
    library_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve controlled neural capacity and symbolic-search vocabulary.

    ``W`` denotes *total hidden/rule width*, not product arity. RuleKAN receives
    W rule slots; shallow MultKAN receives additive+product units summing to W;
    each deep-MultKAN hidden layer has total width W; vanilla KAN receives W
    hidden units. Product arity is controlled independently through
    ``shared_settings.max_product_order``.

    This intentionally overcomplete substrate is designed for experiments in
    which the symbolic-regression/extraction strategy is the treatment.
    """
    out = dict(cfg)
    sh = dict(shared or {})
    if not bool(sh.get("enabled", bool(sh))):
        return out, {"shared_settings_enabled": False}

    cap = dict(sh.get("capacity", {}))
    if width_override is not None:
        width = int(width_override)
        policy = "override"
    else:
        policy = str(cap.get("policy", "fixed"))
        if policy == "fixed":
            width = int(cap.get("width", 12))
        elif policy == "task_max_factors":
            width = int(math.ceil(float(cap.get("multiplier", 2.0)) * max(1, int(spec.max_factors))))
            width = max(int(cap.get("min_width", 1)), width)
            if cap.get("max_width") is not None:
                width = min(int(cap["max_width"]), width)
        else:
            raise ValueError(f"unknown shared capacity policy={policy!r}")
    if width < 2:
        raise ValueError("shared total width must be >=2")

    is_rulekan = model_name.startswith("rulekan") or model_name.startswith("sisp") or model_name.startswith("power_rulekan")
    is_multkan = model_name in PAPER_PIPELINES or model_name in DEEP_MULTKAN_PIPELINES
    is_anfis = model_name == "anfis"
    is_srkan = model_name == "srkan"
    is_symbolic_kan = model_name == "symbolic_kan"
    is_pysr = model_name == "pysr"
    is_operon = model_name == "operon"
    is_pse = model_name == "pse"
    is_udsr = model_name == "udsr"
    is_rils_rols = model_name == "rils_rols"
    mult_units = 0
    additive_units = width
    if is_rulekan:
        out["n_rules"] = width
        if bool(sh.get("symbolic_rule_budget_at_least_width", sh.get("symbolic_max_rules_match_width", False))):
            requested = out.get("symbolic_max_rules")
            out["symbolic_max_rules"] = width if requested is None else max(int(requested), width)
    elif is_anfis:
        out["anfis_rules"] = width
    elif is_multkan:
        requested_mult = cap.get("mult_units")
        if requested_mult is None:
            frac = float(cap.get("mult_fraction", 1.0 / 3.0))
            requested_mult = max(int(cap.get("min_mult_units", 1)), int(round(width * frac)))
        mult_units = min(width - 1, max(1, int(requested_mult)))
        additive_units = width - mult_units
        out["width_additive"] = additive_units
        out["mult_units"] = mult_units
        # Deep model: first hidden layer is additive-only width W; second hidden
        # layer preserves the same total width split into additive/product units.
        out["deep_width_1"] = width
        out["deep_width_2"] = additive_units
        out["deep_mult_units"] = mult_units
    elif model_name == "vanilla_kan":
        out["hidden"] = width

    product_order_setting = sh.get("max_product_order", "task")
    if str(product_order_setting) == "task":
        resolved_product_order = int(spec.max_factors)
    else:
        try:
            resolved_product_order = int(product_order_setting)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "shared max_product_order must be 'task' or a positive integer"
            ) from exc
        if resolved_product_order < 1:
            raise ValueError("shared max_product_order must be >= 1")

    if is_multkan:
        out["mult_arity"] = resolved_product_order
        out["deep_mult_arity"] = resolved_product_order
    if is_rulekan and "max_factors_override" not in out:
        out["max_factors_override"] = resolved_product_order

    if sh.get("grid") is not None and (is_rulekan or is_multkan or model_name == "vanilla_kan"):
        out["grid"] = int(sh["grid"])
    if sh.get("symbolic_trial_steps") is not None and (is_rulekan or is_multkan):
        out["symbolic_trial_steps"] = int(sh["symbolic_trial_steps"])
    if is_rulekan and sh.get("symbolic_hybrid_hard_screening") is not None:
        out["symbolic_hybrid_hard_screening"] = bool(sh["symbolic_hybrid_hard_screening"])
    if is_rulekan and sh.get("symbolic_residual_structure_topk") is not None:
        out["symbolic_residual_structure_topk"] = int(sh["symbolic_residual_structure_topk"])

    lib_name = library_override if library_override is not None else sh.get("symbolic_library")
    resolved_lib_name = None
    shared_native_library = None
    shared_native_exact = None
    shared_native_note = None
    supports_shared_library = (
        is_rulekan or is_multkan or is_srkan or is_symbolic_kan or is_pysr
        or is_operon or is_pse or is_udsr or is_rils_rols
    )
    if lib_name is not None and supports_shared_library:
        if isinstance(lib_name, str):
            resolved_lib_name = lib_name
            if lib_name in {"research26", "full", "full26"}:
                # The 26-atom vocabulary is an explicit sensitivity/override
                # condition; it is not the default for the research profile.
                lib = list(RESEARCH_SYMBOLIC_LIBRARY)
                resolved_lib_name = "research26"
            elif lib_name in {"research", "target_core", "core", "core10", "core14"}:
                # Controlled research defaults to the elementary target-core
                # vocabulary.  Legacy ``core14`` remains accepted as an alias.
                lib = list(TARGET_CORE_SYMBOLIC_LIBRARY)
                resolved_lib_name = "core10"
            elif lib_name in {"medium", "medium16", "medium20"}:
                lib = list(MEDIUM_SYMBOLIC_LIBRARY)
                resolved_lib_name = "medium16"
            elif lib_name == "compact":
                lib = list(COMPACT_SYMBOLIC_LIBRARY)
            elif lib_name == "paper25":
                lib = list(PAPER_SYMBOLIC_LIBRARY)
            else:
                raise ValueError(f"unknown shared symbolic_library={lib_name!r}")
        else:
            lib = [str(x) for x in lib_name]
            resolved_lib_name = f"custom{len(lib)}"
        if is_rulekan or is_multkan:
            out["symbolic_library"] = lib
            shared_native_library = list(lib)
            shared_native_exact = True
        elif is_srkan:
            # Keep SR-KAN on exactly the same elementary target-core vocabulary
            # in the controlled benchmark.  Other shared-library overrides are
            # not silently remapped to SR-KAN's differently named native atoms.
            if resolved_lib_name == "core10":
                out["functions"] = ["target_core"]
                shared_native_library = [
                    _SRKAN_TARGET_CORE_ALIASES.get(name, name)
                    for name in TARGET_CORE_SYMBOLIC_LIBRARY
                ]
                shared_native_exact = True
            else:
                raise ValueError(
                    "SR-KAN shared-library matching currently supports only target_core/core10"
                )
        elif is_symbolic_kan:
            if resolved_lib_name != "core10":
                raise ValueError(
                    "Symbolic-KAN shared-library matching currently supports only target_core/core10"
                )
            # The official Symbolic-KAN code has no one-step inverse-square
            # primitive.  With the benchmark's two symbolic blocks it can form
            # 1/x^2 compositionally from x2+inv (or inv+x2).  Its sqrt/log/inv
            # implementations are protected versions of the corresponding
            # elementary atoms, consistent with the protected benchmark KAN
            # evaluations.  Do not modify the upstream primitive library.
            out["lib"] = list(SYMBOLIC_KAN_TARGET_CORE_NATIVE)
            shared_native_library = list(SYMBOLIC_KAN_TARGET_CORE_NATIVE)
            shared_native_exact = False
            shared_native_note = (
                "official Symbolic-KAN has no native 1/x^2 atom; it is constructible "
                "across two blocks from x2 and inv"
            )
        elif is_pysr:
            if resolved_lib_name != "core10":
                raise ValueError("PySR shared-library matching currently supports only target_core/core10")
            out["binary_operators"] = list(PYSR_TARGET_CORE_BINARY)
            out["unary_operators"] = list(PYSR_TARGET_CORE_UNARY)
            shared_native_library = [*PYSR_TARGET_CORE_BINARY, *PYSR_TARGET_CORE_UNARY]
            shared_native_exact = False
            shared_native_note = (
                "recursive tree grammar; identity is a variable leaf and 1/x^2 is composed from inv/square"
            )
        elif is_operon:
            if resolved_lib_name != "core10":
                raise ValueError("Operon shared-library matching currently supports only target_core/core10")
            out["allowed_symbols"] = ",".join(OPERON_TARGET_CORE_SYMBOLS)
            shared_native_library = list(OPERON_TARGET_CORE_SYMBOLS)
            shared_native_exact = False
            shared_native_note = (
                "recursive tree grammar; reciprocal and inverse-square are composed using division and square"
            )
        elif is_pse:
            if resolved_lib_name != "core10":
                raise ValueError("PSE shared-library matching currently supports only target_core/core10")
            out["operators"] = list(PSE_TARGET_CORE_NATIVE)
            shared_native_library = list(PSE_TARGET_CORE_NATIVE)
            shared_native_exact = False
            shared_native_note = (
                "official PSRN grammar has arithmetic/identity plus sin/cos/exp/log/tanh; "
                "square and reciprocal are composed, with no dedicated sqrt token in the configured public grammar"
            )
        elif is_udsr:
            if resolved_lib_name != "core10":
                raise ValueError("uDSR shared-library matching currently supports only target_core/core10")
            # Keep the LINEAR/poly token because removing it would turn the
            # baseline into DSO rather than uDSR.  Other tokens are restricted
            # to the elementary arithmetic/trig/exp/log set used by the public
            # uDSR configuration.
            out["function_set"] = list(UDSR_TARGET_CORE_NATIVE)
            shared_native_library = list(UDSR_TARGET_CORE_NATIVE)
            shared_native_exact = False
            shared_native_note = (
                "method-native uDSR exception: LINEAR/poly is retained; the public grammar has no direct tanh/sqrt atoms"
            )
        elif is_rils_rols:
            if resolved_lib_name != "core10":
                raise ValueError("RILS-ROLS shared-library matching currently supports only target_core/core10")
            # The public sklearn estimator does not expose its internal
            # operator set as a constructor argument.  Record this explicitly
            # rather than pretending it is vocabulary matched.
            shared_native_library = ["method-native fixed grammar"]
            shared_native_exact = False
            shared_native_note = (
                "public RILS-ROLS API does not expose an operator-library control; method-native grammar retained"
            )

    if sh.get("lr") is not None and (is_rulekan or is_multkan or model_name == "vanilla_kan"):
        out["lr"] = float(sh["lr"])

    meta = {
        "shared_settings_enabled": True,
        "shared_capacity_policy": policy,
        "shared_capacity_width": int(width),
        "shared_total_hidden_width": int(width),
        "shared_additive_units": int(additive_units) if is_multkan else None,
        "shared_mult_units": int(mult_units) if is_multkan else None,
        "shared_task_max_factors": int(spec.max_factors),
        "shared_max_product_order": int(resolved_product_order),
        "shared_grid": int(out["grid"]) if out.get("grid") is not None else None,
        "shared_symbolic_trial_steps": int(out["symbolic_trial_steps"]) if out.get("symbolic_trial_steps") is not None else None,
        "shared_symbolic_library": resolved_lib_name,
        "shared_symbolic_library_size": len(TARGET_CORE_SYMBOLIC_LIBRARY) if resolved_lib_name == "core10" else (
            len(out.get("symbolic_library", [])) if out.get("symbolic_library") is not None else None
        ),
        "shared_symbolic_native_library": shared_native_library,
        "shared_symbolic_native_library_size": len(shared_native_library) if shared_native_library is not None else None,
        "shared_symbolic_native_exact_match": shared_native_exact,
        "shared_symbolic_native_note": shared_native_note,
        "shared_symbolic_max_rules": int(out["symbolic_max_rules"]) if is_rulekan and out.get("symbolic_max_rules") is not None else None,
        "shared_symbolic_hybrid_hard_screening": bool(out.get("symbolic_hybrid_hard_screening", False)) if is_rulekan else None,
        "shared_symbolic_residual_structure_topk": int(out.get("symbolic_residual_structure_topk", 0)) if is_rulekan else None,
        "shared_mult_arity": int(out["mult_arity"]) if out.get("mult_arity") is not None else None,
    }
    return out, meta


PAPER_PIPELINES = {
    "autosym": "baseline",
    "fastkan_autosym": "fastkan_baseline",
    "gsr": "greedy_matching_pursuit",
    "fastkan_gsr": "fastkan_greedy_matching_pursuit",
    "gmp": "gated_greedy_matching_pursuit",
}

# Supplementary depth-controlled MultKAN baselines.  They reuse the same
# symbolic extraction algorithms as the paper pipelines but place multiplication
# nodes in a later hidden layer, after one learned KAN transformation.  This
# directly tests nested structures such as h(f(x) * g(y)).
DEEP_MULTKAN_PIPELINES = {
    "multkan_deep_autosym": "baseline",
    "fast_multkan_deep_autosym": "fastkan_baseline",
    "multkan_deep_gsr": "greedy_matching_pursuit",
    "fast_multkan_deep_gsr": "fastkan_greedy_matching_pursuit",
    "multkan_deep_gmp": "gated_greedy_matching_pursuit",
}


@dataclass
class ModelRun:
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    numeric_model: Optional[torch.nn.Module] = None
    symbolic_model: Optional[torch.nn.Module] = None


def _serialize_rulekan_support_classes(classes: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Serialize RuleKAN numerical support classes for benchmark diagnostics.

    The effective support fields are the canonical downstream contract.  Raw
    ``symbolic_learned_support_*`` fields are retained only as diagnostics for
    the primary/pre-rescue trajectory.
    """
    out = []
    for c in classes:
        out.append({
            "support": [int(v) for v in c["support"]],
            "members": [int(v) for v in c.get("members", ())],
            "active_members": [int(v) for v in c.get("active_members", ())],
            "score": float(c.get("score", 0.0)),
            "structure_probability": float(c.get("structure_probability", 0.0)),
            "numeric_strength": float(c.get("numeric_strength", 0.0)),
        })
    return out



def _downward_support_closure(
    classes: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Validation-rescue support closure without inventing new variables.

    If numerical RuleKAN licenses a support such as ``{x0,x1,x2}``, a rescue
    may test lower-dimensional mechanisms inside it (``{x0,x1}``, ``{x0,x2}``,
    ...).  This changes only the validation-gated rescue grammar; the primary
    learned-support trajectory remains untouched.
    """
    import itertools as _it
    out=[]; seen=set()
    # Preserve original classes first.
    for c in classes:
        support=tuple(sorted(set(int(v) for v in c.get("support",()))))
        if not support or support in seen:
            continue
        seen.add(support); cc=dict(c); cc["support"]=support; out.append(cc)
    originals=list(out)
    for c in originals:
        support=tuple(c["support"])
        for k in range(1,len(support)):
            for sub in _it.combinations(support,k):
                if sub in seen: continue
                seen.add(sub)
                out.append({
                    "support":tuple(sub),"members":(),"active_members":(),
                    # Keep closure proposals below their parent in diagnostics;
                    # these scores are not used as target-label information.
                    "score":float(c.get("score",0.0))*0.95,
                    "structure_probability":float(c.get("structure_probability",0.0))*0.95,
                    "numeric_strength":float(c.get("numeric_strength",0.0))*0.95,
                    "subset_closure_parent":support,
                })
    out.sort(key=lambda c:(len(c["support"]),tuple(c["support"])))
    return out


def _membership_gate_variables_from_data(
    data: BenchmarkData,
    *,
    lower: float = -0.08,
    upper: float = 1.08,
    min_span: float = 0.70,
) -> list[int]:
    """Infer unit-interval fuzzy membership coordinates from training data."""
    x = data.train_x.detach().cpu()
    mean = data.input_mean.detach().cpu().reshape(-1)
    std = data.input_std.detach().cpu().reshape(-1)
    if x.ndim != 2 or mean.numel() != x.shape[1] or std.numel() != x.shape[1]:
        return []
    raw = x * std[None, :] + mean[None, :]
    out=[]
    for j in range(raw.shape[1]):
        col=raw[:,j]; col=col[torch.isfinite(col)]
        if col.numel()<8: continue
        lo=float(col.min()); hi=float(col.max()); span=hi-lo
        if lo>=float(lower) and hi<=float(upper) and span>=float(min_span):
            out.append(int(j))
    return out


def _gate_aware_support_augmentation(
    classes: Sequence[Dict[str, Any]],
    data: BenchmarkData,
    *,
    max_factors: int,
) -> list[Dict[str, Any]]:
    """Recombine numerically linked gate/branch supports for rescue only.

    A nested fuzzy DNF may require ``{g0,g1,b}`` even when pruning leaves only
    ``{g0,b}`` and ``{g1,b}``.  Reconstruct that missing support *only* when two
    numerical support classes share the same non-gate branch variable(s) and
    contribute distinct membership-like gate variables.  This is deliberately
    narrower than blindly adding every unit-interval variable to every branch:
    independent fuzzy rules with disjoint branches therefore stay independent.
    """
    gates=set(_membership_gate_variables_from_data(data))
    out=[];seen=set()
    for c in classes:
        support=tuple(sorted(set(int(v) for v in c.get('support',()))))
        if not support or support in seen: continue
        cc=dict(c);cc['support']=support;out.append(cc);seen.add(support)
    if len(gates)<2 or int(max_factors)<3:
        return out
    originals=list(out)
    for i,a in enumerate(originals):
        A=set(int(v) for v in a['support'])
        for b in originals[i+1:]:
            B=set(int(v) for v in b['support'])
            shared_branch=(A & B) - gates
            if not shared_branch:
                continue
            gate_union=(A | B) & gates
            if len(gate_union)<2:
                continue
            U=A|B
            if len(U)>int(max_factors):
                continue
            aug=tuple(sorted(U))
            if aug in seen:
                continue
            seen.add(aug)
            out.append({
                'support':aug,'members':(),'active_members':(),
                'score':min(float(a.get('score',0.0)),float(b.get('score',0.0)))*0.95,
                'structure_probability':min(float(a.get('structure_probability',0.0)),float(b.get('structure_probability',0.0)))*0.95,
                'numeric_strength':min(float(a.get('numeric_strength',0.0)),float(b.get('numeric_strength',0.0)))*0.95,
                'gate_aware_parents':(tuple(a['support']),tuple(b['support'])),
                'gate_aware_shared_branch':tuple(sorted(shared_branch)),
            })
    out.sort(key=lambda c:(len(c['support']),tuple(c['support'])))
    return out


def _rulekan_effective_support_payload(
    classes: Sequence[Dict[str, Any]],
    bank: Sequence[Tuple[int, ...]],
    *,
    source: str,
) -> Dict[str, Any]:
    """Build and validate the canonical RuleKAN support payload.

    Centralizing publication here means future RuleKAN-derived models inherit
    the same structure-conditioned support contract instead of inventing their
    own interpretation of primary versus rescued supports.
    """
    serialized = _serialize_rulekan_support_classes(classes)
    supports = [tuple(int(v) for v in c["support"]) for c in serialized]
    allowed = {frozenset(s) for s in supports}
    if not allowed:
        raise RuntimeError("RuleKAN cannot publish an empty effective support set")
    clean_bank = [tuple(int(v) for v in z) for z in bank]
    for structure in clean_bank:
        if frozenset(structure) not in allowed:
            raise ValueError(
                f"RuleKAN cannot publish unsupported effective structure {structure}"
            )
    return {
        "symbolic_effective_support_classes": serialized,
        "symbolic_effective_support_bank": [list(z) for z in clean_bank],
        "symbolic_effective_support_source": str(source),
    }


def _rulekan_effective_supports(run: ModelRun) -> Tuple[list[Tuple[int, ...]], list[Tuple[int, ...]], str]:
    """Return the validation-selected RuleKAN support contract.

    Downstream RuleKAN-derived models (especially PowerRuleKAN) MUST consume
    this accessor rather than the primary ``symbolic_learned_support_*``
    diagnostics.  This makes validation-selected high-recall rescue supports
    automatically propagate to every later algebraic branch.
    """
    extras = run.extras
    raw_bank = extras.get("symbolic_effective_support_bank")
    raw_classes = extras.get("symbolic_effective_support_classes")
    if not raw_bank or not raw_classes:
        raise RuntimeError(
            "RuleKAN run is missing canonical symbolic_effective_support_* fields"
        )
    bank = [tuple(int(v) for v in z) for z in raw_bank]
    supports = [tuple(int(v) for v in c["support"]) for c in raw_classes]
    allowed = {frozenset(s) for s in supports}
    if not allowed:
        raise RuntimeError("RuleKAN effective support set is empty")
    for structure in bank:
        if frozenset(structure) not in allowed:
            raise ValueError(
                f"RuleKAN effective support bank contains unsupported structure {structure}"
            )
    source = str(extras.get("symbolic_effective_support_source", "unknown"))
    return bank, supports, source


def _param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _to_original_target(y: np.ndarray, data: BenchmarkData) -> np.ndarray:
    if data.task_type == "regression":
        return y * float(data.y_std) + float(data.y_mean)
    return y


def evaluate_predictions(pred: torch.Tensor, y: torch.Tensor, data: BenchmarkData) -> Dict[str, float]:
    yp = pred.detach().cpu().reshape(-1).numpy().astype(float)
    yt = y.detach().cpu().reshape(-1).numpy().astype(float)
    if data.task_type == "regression":
        yp0 = _to_original_target(yp, data)
        yt0 = _to_original_target(yt, data)
        err = yp0 - yt0
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ystd = float(np.std(yt0)) or 1.0
        try:
            r2 = float(r2_score(yt0, yp0))
        except Exception:
            r2 = float("nan")
        return {
            "test_rmse": rmse,
            "test_mae": mae,
            "test_nrmse": rmse / ystd,
            "test_r2": r2,
        }
    # All benchmark models expose one scalar.  They are trained against 0/1
    # labels with squared error; the scalar is interpreted as a score here.
    score = yp
    prob = 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    # If the model naturally learned outputs close to [0,1], use them directly.
    if np.nanmin(score) >= -0.1 and np.nanmax(score) <= 1.1:
        prob = np.clip(score, 0.0, 1.0)
    label = (prob >= 0.5).astype(int)
    yt_i = yt.astype(int)
    out = {
        "test_accuracy": float(accuracy_score(yt_i, label)),
        "test_f1": float(f1_score(yt_i, label, zero_division=0)),
        "test_brier_rmse": float(np.sqrt(np.mean((prob - yt_i) ** 2))),
    }
    try:
        out["test_roc_auc"] = float(roc_auc_score(yt_i, prob))
    except Exception:
        out["test_roc_auc"] = float("nan")
    return out


def _numeric_product_scale_diagnostics(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """Forward-scale diagnostics for multiplicative numerical rules.

    These are forward-only statistics for SumProductKAN-style models, used to
    distinguish search failure from a badly scaled numerical precursor.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _, details = model(x, return_details=True)

        if "factor_values" not in details or not hasattr(model, "rule_scale"):
            raise TypeError("numeric product-scale diagnostics require SumProductKAN-style forward details")
        detail_blocks = [details]
        scale_blocks = [model.rule_scale.detach()]

        factors = torch.cat([d["factor_values"].detach().abs().reshape(-1) for d in detail_blocks])
        raw = torch.cat([d["raw_rules"].detach().abs().reshape(-1) for d in detail_blocks])
        contrib = torch.cat([d["contributions"].detach().abs().reshape(-1) for d in detail_blocks])
        scales = torch.cat([v.abs().reshape(-1) for v in scale_blocks])

        def stat(v: torch.Tensor, q: float) -> float:
            if v.numel() == 0:
                return 0.0
            vf = v.float()
            qv = torch.tensor(q, device=v.device, dtype=vf.dtype)
            return float(torch.quantile(vf, qv).cpu())

        return {
            "numeric_factor_abs_p95": stat(factors, 0.95),
            "numeric_factor_abs_max": float(factors.max().cpu()) if factors.numel() else 0.0,
            "numeric_raw_rule_abs_p95": stat(raw, 0.95),
            "numeric_raw_rule_abs_max": float(raw.max().cpu()) if raw.numel() else 0.0,
            "numeric_contribution_abs_p95": stat(contrib, 0.95),
            "numeric_contribution_abs_max": float(contrib.max().cpu()) if contrib.numel() else 0.0,
            "numeric_rule_scale_abs_max": float(scales.max().cpu()) if scales.numel() else 0.0,
        }
    finally:
        model.train(was_training)


def _structure_set(model) -> set[Tuple[int, ...]]:
    if hasattr(model, "symbolic_structure_set"):
        return set(tuple(int(v) for v in z) for z in model.symbolic_structure_set())
    out: set[Tuple[int, ...]] = set()
    if not isinstance(model, SumProductKAN) or not bool(model.discretized.item()):
        return out
    active = torch.nonzero(model.hard_rule_choice & model.rule_alive_mask, as_tuple=False).squeeze(-1).tolist()
    for r in active:
        vars_: list[int] = []
        for s in range(model.max_factors):
            if not bool(model.factor_alive_mask[r, s]):
                continue
            j = int(model.hard_variable_choice[r, s])
            if j != model.in_dim:
                vars_.append(j)
        if vars_:
            out.add(tuple(vars_))
    return out


def _structure_scores(found: set[Tuple[int, ...]], expected: Sequence[Tuple[int, ...]]) -> Dict[str, float]:
    if not expected:
        return {}
    e = set(tuple(z) for z in expected)
    tp = len(found & e)
    precision = tp / max(1, len(found))
    recall = tp / max(1, len(e))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "structure_precision": float(precision),
        "structure_recall": float(recall),
        "structure_f1": float(f1),
    }


def _max_boolean_matching(matrix: list[list[bool]]) -> int:
    """Maximum cardinality matching for the small rule/factor sets used here."""
    if not matrix:
        return 0
    n_left = len(matrix)
    n_right = len(matrix[0]) if matrix[0] else 0
    best = 0
    used = [False] * n_right

    def dfs(i: int, count: int) -> None:
        nonlocal best
        if i >= n_left:
            best = max(best, count)
            return
        if count + (n_left - i) <= best:
            return
        dfs(i + 1, count)
        for j in range(n_right):
            if not used[j] and matrix[i][j]:
                used[j] = True
                dfs(i + 1, count + 1)
                used[j] = False

    dfs(0, 0)
    return int(best)


def _sumproduct_symbolic_rules(model: SumProductKAN) -> list[list[dict[str, float | int | str]]]:
    """Return active hard symbolic product rules from one SumProductKAN block."""
    rules: list[list[dict[str, float | int | str]]] = []
    if not bool(model.discretized.item()):
        return rules
    active = torch.nonzero(model.hard_rule_choice & model.rule_alive_mask, as_tuple=False).squeeze(-1).tolist()
    for r in active:
        factors: list[dict[str, float | int | str]] = []
        for slot in range(model.max_factors):
            if not bool(model.factor_alive_mask[r, slot]):
                continue
            j = int(model.hard_variable_choice[r, slot])
            if j == model.in_dim:
                continue
            if bool(model.hard_spline_choice[r, slot, j]):
                op = "spline"
                beta = float("nan")
                alpha = 1.0
                gamma = 0.0
                delta = 0.0
            else:
                k = int(model.hard_operator_choice[r, slot, j])
                op = str(model.symbolic_library[k])
                aff = model.symbolic_affine[r, slot, j, k].detach().cpu().reshape(-1)
                alpha, beta, gamma, delta = [float(v) for v in aff]
            factors.append({
                "variable": j, "operator": op, "beta": beta,
                "alpha": alpha, "gamma": gamma, "delta": delta,
            })
        if factors:
            rules.append(factors)
    return rules


_FUZZY_STRUCTURAL_ZERO_TOL = 5e-7


def _learned_symbolic_rules(model) -> list[list[dict[str, float | int | str]]]:
    """Return the final model's canonical expanded fuzzy-rule view.

    SumProductKAN is already an additive bank of product rules.  For
    PowerRuleKAN, an outer term consisting of exactly one base at power +1 is
    likewise additive, so its base rules can be scored directly.  Genuine
    powered, reciprocal, ratio, or product-of-bases terms are *not* silently
    credited from their RuleKAN precursor: each is represented by one opaque
    unmatched rule.  This makes fuzzy-rule recovery describe the final
    PowerRuleKAN structure rather than the easier p=1 fallback.
    """
    if isinstance(model, SumProductKAN):
        return _sumproduct_symbolic_rules(model)
    if isinstance(model, ComposedRuleKAN):
        rules: list[list[dict[str, float | int | str]]] = [[{
            "variable": -1, "operator": "__depth2_composition__",
            "beta": float("nan"), "alpha": 1.0, "gamma": 0.0, "delta": 0.0,
        }]]
        if model.correction is not None:
            rules.extend(_sumproduct_symbolic_rules(model.correction))
        return rules
    if isinstance(model, PowerRuleKAN):
        rules: list[list[dict[str, float | int | str]]] = []
        for term_idx, term in enumerate(model.terms):
            if term_idx < int(model.term_scale.numel()):
                # Structural scoring uses the same optimizer-noise floor as
                # formula canonicalization in benchmarks.aggregate. Terms below
                # this scale are numerical residue, not additional fuzzy rules.
                if abs(float(model.term_scale[term_idx].detach().cpu())) <= _FUZZY_STRUCTURAL_ZERO_TOL:
                    continue
            if len(term) == 1 and int(term[0][1]) == 1:
                base_idx = int(term[0][0])
                if 0 <= base_idx < len(model.bases) and isinstance(model.bases[base_idx], (SumProductKAN, ComposedRuleKAN)):
                    rules.extend(_learned_symbolic_rules(model.bases[base_idx]))
                    continue
            # A non-unit outer power/product is not an expanded fuzzy rule in
            # the canonical model representation. Count it as an unmatched
            # learned rule so precision/exact-recovery cannot be inflated.
            rules.append([{
                "variable": -1,
                "operator": "__powered_outer_term__",
                "beta": float("nan"),
                "alpha": 1.0,
                "gamma": 0.0,
                "delta": 0.0,
            }])
        return rules
    return []


def _fuzzy_factor_signature_match(expected, learned: dict[str, float | int | str]) -> bool:
    if int(learned["variable"]) != int(expected.variable):
        return False
    _lop=str(learned["operator"]); _eop=str(expected.operator)
    if not (_lop == _eop or {_lop,_eop} <= {"sin","cos"}):
        return False
    if expected.role == "gate":
        beta = float(learned.get("beta", float("nan")))
        if not np.isfinite(beta) or abs(beta) <= 1e-9:
            return False
        return (beta > 0) == (int(expected.orientation) > 0)
    return True


def _fuzzy_rule_signature_match(expected_rule, learned_rule: list[dict[str, float | int | str]]) -> bool:
    exp = list(expected_rule.factors)
    if len(exp) != len(learned_rule):
        return False
    matrix = [[_fuzzy_factor_signature_match(e, l) for l in learned_rule] for e in exp]
    return _max_boolean_matching(matrix) == len(exp)


def _fuzzy_factor_signature_match_on_data(expected, learned: dict[str, float | int | str], data: BenchmarkData) -> bool:
    """Gauge/phase-invariant fuzzy factor match on observed test coordinates."""
    if int(learned["variable"]) != int(expected.variable): return False
    lop=str(learned["operator"]); eop=str(expected.operator)
    if expected.role != "gate":
        return lop == eop or {lop,eop} <= {"sin","cos"}
    if lop != "x": return False
    j=int(expected.variable); z=data.test_x[:,j].detach().cpu()
    mean=float(data.input_mean.detach().cpu().reshape(-1)[j]); std=float(data.input_std.detach().cpu().reshape(-1)[j])
    raw=z*std+mean; target=raw if int(expected.orientation)>0 else (1.0-raw)
    pred=float(learned.get("alpha",1.0))*(float(learned.get("beta",1.0))*z+float(learned.get("gamma",0.0)))+float(learned.get("delta",0.0))
    denom=float(torch.sum(pred*pred).clamp_min(1e-12)); scale=float(torch.sum(pred*target))/denom
    err=float(torch.sqrt(torch.mean((scale*pred-target)**2))); sd=float(target.std().clamp_min(1e-8))
    return err/sd <= 0.08


def _fuzzy_rule_signature_match_on_data(expected_rule, learned_rule, data: BenchmarkData) -> bool:
    exp=list(expected_rule.factors)
    if len(exp)!=len(learned_rule): return False
    matrix=[[_fuzzy_factor_signature_match_on_data(e,l,data) for l in learned_rule] for e in exp]
    return _max_boolean_matching(matrix)==len(exp)


def fuzzy_rule_recovery_scores(model, spec: TaskSpec, data: BenchmarkData) -> Dict[str, float]:
    """Score recovery of an expanded fuzzy if/then rule system.

    These metrics are deliberately structural.  A fuzzy gate is expected to be
    an identity symbolic factor on the correct variable, with the sign of its
    affine slope distinguishing membership ``x`` from complement ``1-x``.
    Branch factors are matched by variable and operator family.  Continuous
    gate quality is additionally measured as NRMSE against the actual observed
    membership values on the held-out test points; no off-domain samples are
    generated.
    """
    expected_rules = list(spec.fuzzy_rules)
    if not expected_rules:
        return {}
    learned_rules = _learned_symbolic_rules(model)
    rule_matrix = [[_fuzzy_rule_signature_match_on_data(e, l, data) for l in learned_rules] for e in expected_rules]
    matched_rules = _max_boolean_matching(rule_matrix)
    p = matched_rules / max(1, len(learned_rules))
    r = matched_rules / max(1, len(expected_rules))
    f1 = 2.0 * p * r / max(1e-12, p + r)

    exp_gate = [f for rr in expected_rules for f in rr.factors if f.role == "gate"]
    exp_branch = [f for rr in expected_rules for f in rr.factors if f.role == "branch"]
    learned_factors = [f for rr in learned_rules for f in rr]
    learned_gate = [f for f in learned_factors if str(f["operator"]) == "x"]
    learned_branch = [f for f in learned_factors if str(f["operator"]) != "x"]

    gate_matrix = [[_fuzzy_factor_signature_match_on_data(e, l, data) for l in learned_gate] for e in exp_gate]
    branch_matrix = [[_fuzzy_factor_signature_match_on_data(e, l, data) for l in learned_branch] for e in exp_branch]
    gate_hits = _max_boolean_matching(gate_matrix)
    branch_hits = _max_boolean_matching(branch_matrix)
    gate_precision = gate_hits / max(1, len(learned_gate))
    gate_recall = gate_hits / max(1, len(exp_gate))
    branch_precision = branch_hits / max(1, len(learned_branch))
    branch_recall = branch_hits / max(1, len(exp_branch))

    # Continuous gate fidelity on observed held-out points only. Raw RMSE is the
    # primary reported quantity; normalized RMSE is retained only for backward
    # compatibility with older result files.
    gate_errors: list[float] = []
    gate_errors_normalized: list[float] = []
    xnorm = data.test_x.detach().cpu()
    mean = data.input_mean.detach().cpu().reshape(-1)
    std = data.input_std.detach().cpu().reshape(-1)
    for e in exp_gate:
        j = int(e.variable)
        raw = xnorm[:, j] * std[j] + mean[j]
        target = raw if int(e.orientation) > 0 else (1.0 - raw)
        target_sd = float(target.std().clamp_min(1e-8))
        candidates = []
        for lf in learned_gate:
            if int(lf["variable"]) != j:
                continue
            beta = float(lf["beta"])
            if not np.isfinite(beta): continue
            pred = (float(lf["alpha"]) * (beta * xnorm[:, j] + float(lf["gamma"])) + float(lf["delta"]))
            scl = float(torch.sum(pred*target) / torch.sum(pred*pred).clamp_min(1e-12))
            raw_err = float(torch.sqrt(torch.mean((scl*pred - target) ** 2)))
            candidates.append((raw_err, raw_err / target_sd))
        if candidates:
            best = min(candidates, key=lambda z: z[0])
            gate_errors.append(best[0])
            gate_errors_normalized.append(best[1])

    out = {
        "fuzzy_expected_rules": float(len(expected_rules)),
        "fuzzy_found_rules": float(len(learned_rules)),
        "fuzzy_rule_precision": float(p),
        "fuzzy_rule_recall": float(r),
        "fuzzy_rule_f1": float(f1),
        "fuzzy_gate_precision": float(gate_precision),
        "fuzzy_gate_recall": float(gate_recall),
        "fuzzy_branch_precision": float(branch_precision),
        "fuzzy_branch_recall": float(branch_recall),
        "fuzzy_exact_structure_recovery": float(matched_rules == len(expected_rules) == len(learned_rules)),
        "fuzzy_rule_count_error": float(len(learned_rules) - len(expected_rules)),
        "fuzzy_rule_count_abs_error": float(abs(len(learned_rules) - len(expected_rules))),
    }
    if gate_errors:
        out["fuzzy_gate_rmse_median"] = float(np.median(gate_errors))
        out["fuzzy_gate_rmse_max"] = float(np.max(gate_errors))
    if gate_errors_normalized:
        out["fuzzy_gate_nrmse_median"] = float(np.median(gate_errors_normalized))
        out["fuzzy_gate_nrmse_max"] = float(np.max(gate_errors_normalized))
    return out


def _scaled_numeric_stages(base_lr: float, scale: float):
    stages = default_sum_product_schedule(base_lr=base_lr, symbolic=False)
    for stage in stages:
        stage.steps = max(1, int(round(stage.steps * float(scale))))
    return stages




def _symbolicize_rational_branch(
    branch: SumProductKAN,
    tx: torch.Tensor,
    target_train: torch.Tensor,
    vx: torch.Tensor,
    target_val: torch.Tensor,
    cfg: Dict[str, Any],
    *,
    symbolic_seed: int,
    pursuit_mode: str = "gsr",
) -> SumProductKAN:
    """Symbolically distill one learned rational branch with an isolated RNG stream."""
    symbolic_model, _ = mandatory_symbolic_matching_pursuit(
        branch, tx, target_train, vx, target_val,
        max_symbolic_rules=int(cfg.get("symbolic_max_rules", branch.n_rules)),
        min_symbolic_rules=int(cfg.get("symbolic_min_rules", 1)),
        allow_symbolic_self_products=bool(cfg.get("allow_symbolic_self_products", True)),
        use_gmp_preselection=bool(cfg.get("use_gmp_preselection", True)),
        gmp_steps=int(cfg.get("symbolic_gmp_steps", 60)),
        gmp_topk=(int(cfg["symbolic_gmp_topk"]) if "symbolic_gmp_topk" in cfg else None),
        gmp_unary_topk=(int(cfg["symbolic_gmp_unary_topk"]) if "symbolic_gmp_unary_topk" in cfg else None),
        gmp_self_product_topk=(int(cfg["symbolic_gmp_self_topk"]) if "symbolic_gmp_self_topk" in cfg else None),
        gmp_relaxation_mode=str(cfg.get("symbolic_gmp_relaxation_mode", "soft")),
        gmp_atom_backward_normalization=str(cfg.get("symbolic_gmp_atom_backward_normalization", "none")),
        gmp_gumbel_noise_scale=float(cfg.get("symbolic_gmp_gumbel_noise_scale", 1.0)),
        gmp_complexity_weight=float(cfg.get("symbolic_gmp_complexity_weight", 0.0)),
        gmp_complexity_logit_prior=float(cfg.get("symbolic_gmp_complexity_logit_prior", 0.0)),
        gmp_nonlinearity_logit_prior=float(cfg.get("symbolic_gmp_nonlinearity_logit_prior", 0.0)),
        gmp_curvature_weight=float(cfg.get("symbolic_gmp_curvature_weight", 0.0)),
        gmp_nonlinearity_weight=float(cfg.get("symbolic_gmp_nonlinearity_weight", 0.0)),
        gmp_identity_chart=str(cfg.get("symbolic_gmp_identity_chart", "raw")),
        gmp_tuple_refine_steps=int(cfg.get("symbolic_tuple_steps", 20)),
        interaction_shape_screening=bool(cfg.get("symbolic_interaction_shape_screening", True)),
        interaction_shape_topk=int(cfg.get("symbolic_interaction_shape_topk", 0)),
        interaction_shape_bins=int(cfg.get("symbolic_interaction_shape_bins", 8)),
        interaction_shape_min_rank1=float(cfg.get("symbolic_interaction_shape_min_rank1", 0.65)),
        interaction_shape_require_multiple=bool(cfg.get("symbolic_interaction_shape_require_multiple", True)),
        initial_block_pursuit=bool(cfg.get("symbolic_initial_block_pursuit", True)),
        initial_block_pool=int(cfg.get("symbolic_initial_block_pool", 28)),
        initial_block_pair_beam=int(cfg.get("symbolic_initial_block_pair_beam", 10)),
        initial_block_refit_steps=int(cfg.get("symbolic_initial_block_refit_steps", 100)),
        initial_block_lbfgs_steps=int(cfg.get("symbolic_initial_block_lbfgs_steps", 20)),
        initial_block_consolidate_steps=int(cfg.get("symbolic_initial_block_consolidate_steps", 200)),
        initial_block_consolidate_lbfgs_steps=int(cfg.get("symbolic_initial_block_consolidate_lbfgs_steps", 60)),
        hybrid_hard_screening=bool(cfg.get("symbolic_hybrid_hard_screening", False)),
        hard_screen_beam_width=int(cfg.get("symbolic_hard_screen_beam", 64)),
        hard_screen_top_tuples=int(cfg.get("symbolic_hard_screen_top_tuples", 12)),
        hard_screen_max_samples=int(cfg.get("symbolic_hard_screen_samples", 512)),
        hard_screen_start_step=int(cfg.get("symbolic_hard_screen_start_step", 1)),
        hard_screen_matching_steps=int(cfg.get("symbolic_hard_screen_matching_steps", 4)),
        hard_screen_global_candidates=int(cfg.get("symbolic_hard_screen_global_candidates", 2)),
        hard_screen_residual_rescue=bool(cfg.get("symbolic_hard_screen_residual_rescue", False)),
        beam_width=int(cfg.get("symbolic_beam", 4)),
        structure_diverse_beam=bool(cfg.get("symbolic_structure_diverse_beam", False)),
        beam_max_per_structure=int(cfg.get("symbolic_beam_max_per_structure", 1)),
        residual_structure_topk=int(cfg.get("symbolic_residual_structure_topk", 0)),
        residual_structure_mass=float(cfg.get("symbolic_residual_structure_mass", 0.0)),
        residual_structure_min=int(cfg.get("symbolic_residual_structure_min", 1)),
        residual_structure_max=int(cfg.get("symbolic_residual_structure_max", 0)),
        residual_structure_gap_rel=float(cfg.get("symbolic_residual_structure_gap_rel", 0.0)),
        joint_scale_refit_each_commit=bool(cfg.get("symbolic_joint_scale_refit", False)),
        trial_steps=int(cfg.get("symbolic_trial_steps", 50)),
        commit_refit_steps=int(cfg.get("symbolic_commit_steps", 80)),
        commit_lbfgs_steps=int(cfg.get("symbolic_commit_lbfgs", 10)),
        backfit=bool(cfg.get("symbolic_backfit", True)),
        backfit_beam_width=int(cfg.get("symbolic_backfit_beam", 3)),
        backfit_steps=int(cfg.get("symbolic_backfit_steps", 40)),
        backfit_lbfgs_steps=int(cfg.get("symbolic_backfit_lbfgs", 5)),
        final_steps=int(cfg.get("symbolic_final_steps", 120)),
        final_lbfgs_steps=int(cfg.get("symbolic_final_lbfgs", 20)),
        target_val_rmse=float(cfg.get("symbolic_target_rmse", 1e-4)),
        pursuit_mode=str(cfg.get("pursuit_mode", pursuit_mode)),
        omp_extra_steps=int(cfg.get("omp_extra_steps", 60)),
        redundancy_cleanup=bool(cfg.get("redundancy_cleanup", True)),
        redundancy_span_r2_threshold=float(cfg.get("redundancy_span_r2", 0.995)),
        redundancy_corr_threshold=float(cfg.get("redundancy_corr", 0.995)),
        debug_topk=0,
        cleanup_max_seconds=float(cfg.get("symbolic_cleanup_seconds", 20.0)),
        cleanup_max_trials=int(cfg.get("symbolic_cleanup_trials", 10)),
        symbolic_seed=int(symbolic_seed),
        verbose=False, show_progress=False,
    )
    return symbolic_model



def _fit_rulekan_learned_support_gsr(
    model: SumProductKAN,
    tx: torch.Tensor,
    ty: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    learned_support_bank: Sequence[Tuple[int, ...]],
    learned_support_classes: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    n_rules: int,
    seed: int,
    pursuit_mode: str,
    max_symbolic_rules: Optional[int] = None,
):
    """Run RuleKAN's learned-support GSR with one explicit rank cap."""
    cap = int(cfg.get("symbolic_max_rules", n_rules) if max_symbolic_rules is None else max_symbolic_rules)
    return learned_support_symbolic_gsr(
        model, tx, ty, vx, vy,
        structure_candidates=learned_support_bank,
        allowed_supports=[c["support"] for c in learned_support_classes],
        max_symbolic_rules=cap,
        min_symbolic_rules=int(cfg.get("symbolic_min_rules", 1)),
        max_rules_per_structure=int(cfg.get("symbolic_max_rules_per_structure", 2)),
        allow_symbolic_self_products=True,
        use_gmp_preselection=bool(cfg.get("use_gmp_preselection", True)),
        gmp_steps=int(cfg.get("symbolic_gmp_steps", 60)),
        gmp_topk=(int(cfg["symbolic_gmp_topk"]) if "symbolic_gmp_topk" in cfg else None),
        gmp_unary_topk=(int(cfg["symbolic_gmp_unary_topk"]) if "symbolic_gmp_unary_topk" in cfg else None),
        gmp_self_product_topk=(int(cfg["symbolic_gmp_self_topk"]) if "symbolic_gmp_self_topk" in cfg else None),
        gmp_relaxation_mode=str(cfg.get("symbolic_gmp_relaxation_mode", "soft")),
        gmp_atom_backward_normalization=str(cfg.get("symbolic_gmp_atom_backward_normalization", "none")),
        gmp_gumbel_noise_scale=float(cfg.get("symbolic_gmp_gumbel_noise_scale", 1.0)),
        gmp_complexity_weight=float(cfg.get("symbolic_gmp_complexity_weight", 0.0)),
        gmp_complexity_logit_prior=float(cfg.get("symbolic_gmp_complexity_logit_prior", 0.0)),
        gmp_nonlinearity_logit_prior=float(cfg.get("symbolic_gmp_nonlinearity_logit_prior", 0.0)),
        gmp_curvature_weight=float(cfg.get("symbolic_gmp_curvature_weight", 0.0)),
        gmp_nonlinearity_weight=float(cfg.get("symbolic_gmp_nonlinearity_weight", 0.0)),
        gmp_identity_chart=str(cfg.get("symbolic_gmp_identity_chart", "raw")),
        gmp_tuple_refine_steps=int(cfg.get("symbolic_tuple_steps", 20)),
        interaction_shape_screening=bool(cfg.get("symbolic_interaction_shape_screening", True)),
        interaction_shape_topk=int(cfg.get("symbolic_interaction_shape_topk", 8)),
        interaction_shape_bins=int(cfg.get("symbolic_interaction_shape_bins", 8)),
        interaction_shape_min_rank1=float(cfg.get("symbolic_interaction_shape_min_rank1", 0.65)),
        interaction_shape_require_multiple=False,
        hard_proposal_union=bool(cfg.get("symbolic_hard_proposal_union", True)),
        hard_proposal_top_tuples=int(cfg.get("symbolic_hard_proposal_top_tuples", 8)),
        hard_proposal_beam_width=int(cfg.get("symbolic_hard_proposal_beam", 48)),
        hard_proposal_max_samples=int(cfg.get("symbolic_hard_proposal_samples", 384)),
        hard_proposal_max_per_factor=int(cfg.get("symbolic_hard_proposal_max_per_factor", 6)),
        initial_block_pursuit=bool(cfg.get("symbolic_initial_block_pursuit", True)),
        initial_block_pool=int(cfg.get("symbolic_initial_block_pool", 28)),
        initial_block_pair_beam=int(cfg.get("symbolic_initial_block_pair_beam", 10)),
        initial_block_refit_steps=int(cfg.get("symbolic_initial_block_refit_steps", 100)),
        initial_block_lbfgs_steps=int(cfg.get("symbolic_initial_block_lbfgs_steps", 20)),
        initial_block_consolidate_steps=int(cfg.get("symbolic_initial_block_consolidate_steps", 200)),
        initial_block_consolidate_lbfgs_steps=int(cfg.get("symbolic_initial_block_consolidate_lbfgs_steps", 60)),
        hybrid_hard_screening=bool(cfg.get("symbolic_residual_hard_proposal_refresh", True)),
        hard_screen_start_step=int(cfg.get("symbolic_hard_screen_start_step", 2)),
        hard_screen_matching_steps=int(cfg.get("symbolic_hard_screen_matching_steps", 3)),
        hard_screen_global_candidates=int(cfg.get("symbolic_hard_screen_global_candidates", 2)),
        hard_screen_structure_topk=int(cfg.get("symbolic_hard_screen_structure_topk", 1)),
        hard_screen_residual_rescue=bool(cfg.get("symbolic_hard_residual_operator_rescue", True)),
        hard_screen_beam_width=int(cfg.get("symbolic_hard_screen_beam", 48)),
        hard_screen_top_tuples=int(cfg.get("symbolic_hard_screen_top_tuples", 8)),
        hard_screen_max_samples=int(cfg.get("symbolic_hard_screen_samples", 384)),
        beam_width=int(cfg.get("symbolic_beam", 4)),
        structure_diverse_beam=bool(cfg.get("symbolic_structure_diverse_beam", True)),
        beam_max_per_structure=int(cfg.get("symbolic_beam_max_per_structure", 1)),
        residual_structure_topk=int(cfg.get("symbolic_residual_structure_topk", 0)),
        residual_structure_mass=float(cfg.get("symbolic_residual_structure_mass", 0.0)),
        residual_structure_min=int(cfg.get("symbolic_residual_structure_min", 1)),
        residual_structure_max=int(cfg.get("symbolic_residual_structure_max", 0)),
        residual_structure_gap_rel=float(cfg.get("symbolic_residual_structure_gap_rel", 0.0)),
        joint_scale_refit_each_commit=bool(cfg.get("symbolic_joint_scale_refit", True)),
        trial_steps=int(cfg.get("symbolic_trial_steps", 50)),
        commit_refit_steps=int(cfg.get("symbolic_commit_steps", 80)),
        commit_lbfgs_steps=int(cfg.get("symbolic_commit_lbfgs", 10)),
        backfit=bool(cfg.get("symbolic_backfit", True)),
        backfit_beam_width=int(cfg.get("symbolic_backfit_beam", 3)),
        backfit_steps=int(cfg.get("symbolic_backfit_steps", 40)),
        backfit_lbfgs_steps=int(cfg.get("symbolic_backfit_lbfgs", 5)),
        final_steps=int(cfg.get("symbolic_final_steps", 120)),
        final_lbfgs_steps=int(cfg.get("symbolic_final_lbfgs", 20)),
        target_val_rmse=float(cfg.get("symbolic_target_rmse", 1e-4)),
        min_rule_improvement_rel=float(cfg.get("symbolic_min_rule_improvement_rel", 2e-3)),
        elimination_rel_mse_budget=float(cfg.get("symbolic_elimination_rel_mse_budget", 2e-2)),
        pursuit_mode=str(cfg.get("pursuit_mode", pursuit_mode)),
        redundancy_cleanup=bool(cfg.get("redundancy_cleanup", True)),
        redundancy_span_r2_threshold=float(cfg.get("redundancy_span_r2", 0.995)),
        redundancy_corr_threshold=float(cfg.get("redundancy_corr", 0.995)),
        debug_topk=0,
        cleanup_max_seconds=float(cfg.get("symbolic_cleanup_seconds", 20.0)),
        cleanup_max_trials=int(cfg.get("symbolic_cleanup_trials", 10)),
        affine_partition_rescue=bool(cfg.get("symbolic_affine_partition_rescue", True)),
        affine_partition_family_beam=int(cfg.get("symbolic_affine_partition_family_beam", 18)),
        affine_partition_seed_topk=int(cfg.get("symbolic_affine_partition_seed_topk", 0)),
        affine_partition_max_samples=int(cfg.get("symbolic_affine_partition_max_samples", 384)),
        affine_partition_max_support_pairs=int(cfg.get("symbolic_affine_partition_max_support_pairs", 12)),
        affine_partition_refine_steps=int(cfg.get("symbolic_affine_partition_refine_steps", 120)),
        affine_partition_refine_lr=float(cfg.get("symbolic_affine_partition_refine_lr", 1.0e-3)),
        affine_partition_lbfgs_steps=int(cfg.get("symbolic_affine_partition_lbfgs_steps", 20)),
        affine_partition_final_polish_topk=int(cfg.get("symbolic_affine_partition_final_polish_topk", 8)),
        affine_partition_final_polish_steps=int(cfg.get("symbolic_affine_partition_final_polish_steps", 500)),
        affine_partition_final_polish_lbfgs_steps=int(cfg.get("symbolic_affine_partition_final_polish_lbfgs_steps", 80)),
        affine_partition_equivalence_rel_mse=float(cfg.get("symbolic_affine_partition_equivalence_rel_mse", 3.0)),
        affine_partition_equivalence_nrmse=float(cfg.get("symbolic_affine_partition_equivalence_nrmse", 5e-4)),
        affine_partition_cancellation_weight=float(cfg.get("symbolic_affine_partition_cancellation_weight", 1.0)),
        affine_partition_complexity_weight=float(cfg.get("symbolic_affine_partition_complexity_weight", 0.02)),
        affine_partition_min_improvement_rel=float(cfg.get("symbolic_affine_partition_min_improvement_rel", 1e-4)),
        symbolic_seed=int(cfg.get("symbolic_seed", seed + 100_003)),
        verbose=False, show_progress=False,
    )



def _fit_validation_structure_rescue_symbolic_gsr(
    model: SumProductKAN,
    tx: torch.Tensor,
    ty: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    support_classes: Sequence[Dict[str, Any]],
    structure_bank: Sequence[Tuple[int, ...]],
    cfg: Dict[str, Any],
    *,
    n_rules: int,
    seed: int,
    pursuit_mode: str,
):
    """Validation-gated rescue that remains strictly RuleKAN-structure-conditioned.

    The rescue may spend more search budget and may use a higher-recall set of
    numerical support classes captured before destructive pruning, but every
    symbolic structure must stay inside a support licensed by the numerical
    RuleKAN.  It never enumerates the full variable grammar and never creates a
    cross-variable support that was absent from the numerical support evidence.
    """
    if not support_classes or not structure_bank:
        raise RuntimeError("RuleKAN structural rescue requires learned numerical supports")
    allowed_supports = {
        frozenset(int(v) for v in c.get("support", ()))
        for c in support_classes
        if c.get("support")
    }
    for structure in structure_bank:
        support = frozenset(int(v) for v in structure)
        if support not in allowed_supports:
            raise ValueError(
                f"RuleKAN structural rescue received unsupported structure {tuple(structure)}"
            )

    local = dict(cfg)
    # Rescue is a deeper search over the learned structure manifold, not a new
    # hypothesis class.  Map the rescue budgets onto the ordinary constrained
    # GSR controls while preserving all support checks in learned_support_symbolic_gsr.
    local["symbolic_gmp_steps"] = int(cfg.get(
        "symbolic_validation_rescue_gmp_steps",
        max(int(cfg.get("symbolic_gmp_steps", 60)), 80),
    ))
    local["symbolic_tuple_steps"] = int(cfg.get(
        "symbolic_validation_rescue_tuple_steps",
        max(int(cfg.get("symbolic_tuple_steps", 20)), 30),
    ))
    local["symbolic_trial_steps"] = int(cfg.get(
        "symbolic_validation_rescue_trial_steps",
        max(int(cfg.get("symbolic_trial_steps", 50)), 70),
    ))
    local["symbolic_commit_steps"] = int(cfg.get(
        "symbolic_validation_rescue_commit_steps",
        max(int(cfg.get("symbolic_commit_steps", 80)), 100),
    ))
    local["symbolic_commit_lbfgs"] = int(cfg.get(
        "symbolic_validation_rescue_commit_lbfgs",
        max(int(cfg.get("symbolic_commit_lbfgs", 10)), 15),
    ))
    local["symbolic_backfit_steps"] = int(cfg.get(
        "symbolic_validation_rescue_backfit_steps",
        max(int(cfg.get("symbolic_backfit_steps", 40)), 55),
    ))
    local["symbolic_backfit_lbfgs"] = int(cfg.get(
        "symbolic_validation_rescue_backfit_lbfgs",
        max(int(cfg.get("symbolic_backfit_lbfgs", 5)), 8),
    ))
    local["symbolic_final_steps"] = int(cfg.get(
        "symbolic_validation_rescue_final_steps",
        max(int(cfg.get("symbolic_final_steps", 120)), 160),
    ))
    local["symbolic_final_lbfgs"] = int(cfg.get(
        "symbolic_validation_rescue_final_lbfgs",
        max(int(cfg.get("symbolic_final_lbfgs", 20)), 30),
    ))
    local["symbolic_initial_block_pursuit"] = bool(cfg.get(
        "symbolic_validation_rescue_initial_block", True
    ))
    local["symbolic_initial_block_pool"] = int(cfg.get(
        "symbolic_validation_rescue_initial_pool",
        max(int(cfg.get("symbolic_initial_block_pool", 28)), 36),
    ))
    local["symbolic_initial_block_pair_beam"] = int(cfg.get(
        "symbolic_validation_rescue_pair_beam",
        max(int(cfg.get("symbolic_initial_block_pair_beam", 10)), 12),
    ))
    local["symbolic_hard_screen_beam"] = int(cfg.get(
        "symbolic_validation_rescue_hard_beam",
        max(int(cfg.get("symbolic_hard_screen_beam", 48)), 64),
    ))
    local["symbolic_hard_screen_top_tuples"] = int(cfg.get(
        "symbolic_validation_rescue_hard_tuples",
        max(int(cfg.get("symbolic_hard_screen_top_tuples", 8)), 12),
    ))
    local["symbolic_hard_residual_operator_rescue"] = True
    local["symbolic_residual_hard_proposal_refresh"] = True
    local["symbolic_joint_scale_refit"] = True
    local["symbolic_structure_diverse_beam"] = True
    local["symbolic_seed"] = int(cfg.get("symbolic_seed", seed + 100_003)) + 17_003

    max_rules = min(
        int(cfg.get("symbolic_max_rules", n_rules)),
        int(cfg.get("symbolic_validation_rescue_max_rules", max(8, int(cfg.get("symbolic_max_rules", n_rules))))),
    )
    return _fit_rulekan_learned_support_gsr(
        model, tx, ty, vx, vy, structure_bank, support_classes, local,
        n_rules=n_rules, seed=seed, pursuit_mode=pursuit_mode,
        max_symbolic_rules=max(1, max_rules),
    )

def _train_rulekan_family(
    model_name: str,
    spec: TaskSpec,
    data: BenchmarkData,
    seed: int,
    cfg: Dict[str, Any],
    *,
    numeric_basis: str,
    pursuit_mode: str = "gsr",
) -> ModelRun:
    device = str(cfg.get("device", "cpu"))
    in_dim = int(data.train_x.shape[1])
    n_rules = int(cfg.get("n_rules", max(10, 2 * in_dim + 4)))
    max_factors = int(max(1, cfg.get("max_factors_override", spec.max_factors)))
    model = SumProductKAN(
        in_dim=in_dim,
        n_rules=n_rules,
        max_factors=max_factors,
        grid=int(cfg.get("grid", 12)),
        k=int(cfg.get("k", 3)),
        grid_range=tuple(cfg.get("grid_range", [-2.5, 2.5])),
        numeric_basis=str(numeric_basis),
        rbf_train_grid=bool(cfg.get("rbf_train_grid", True)),
        rbf_train_width=bool(cfg.get("rbf_train_width", True)),
        rbf_width_scale=float(cfg.get("rbf_width_scale", 1.0)),
        symbolic_library=tuple(cfg.get("symbolic_library", (
            "x", "x^2", "x^3", "exp", "sin", "cos", "tanh", "arctan",
        ))),
        min_order=1,
        symbolic_gradient_scale=float(cfg.get("symbolic_gradient_scale", 4.0)),
        symbolic_product_gradient_scale=float(cfg.get("symbolic_product_gradient_scale", 2.0)),
        numeric_factor_gradient_scale=float(cfg.get("numeric_factor_gradient_scale", 4.0)),
        numeric_product_gradient_scale=float(cfg.get("numeric_product_gradient_scale", 2.0)),
        numeric_product_gradient_power=float(cfg.get("numeric_product_gradient_power", 1.0)),
        numeric_product_gradient_max_gain=float(cfg.get("numeric_product_gradient_max_gain", 8.0)),
        init_factor_open_prob=float(cfg.get("init_factor_open_prob", 0.95)),
        allow_self_products=False,
        seed=int(seed),
        device=device,
    )
    tx = data.train_x.to(device)
    ty = data.train_y.to(device)
    vx = data.val_x.to(device)
    vy = data.val_y.to(device)
    qx = data.test_x.to(device)
    qy = data.test_y.to(device)

    t0 = time.perf_counter()
    stages = _scaled_numeric_stages(float(cfg.get("lr", 2e-3)), float(cfg.get("stage_scale", 0.15)))
    if bool(cfg.get("numeric_symbolic_manifold", False)):
        if "manifold_warmup_scale" in cfg and stages:
            stages[0].steps = max(1, int(round(1200 * float(cfg.get("manifold_warmup_scale", 0.5)))))
        # Free numerical warmup first. During compression/consolidation, train
        # auxiliary symbolic family parameters while keeping the actual forward
        # model purely numerical. The augmented Lagrangian attracts each active
        # selected spline/RBF factor to one analytic family before pruning.
        model.initialize_symbolic_manifold_seeds_()
        if len(stages) > 1:
            s = stages[1]
            s.train_symbolic = True
            s.symbolic_enabled = False
            s.symbolic_hardening_start = float(cfg.get("manifold_hardening_start", 0.0))
            s.symbolic_hardening_end = float(cfg.get("manifold_hardening_mid", 0.65))
            s.operator_temperature_start = float(cfg.get("manifold_temperature_start", 1.2))
            s.operator_temperature_end = float(cfg.get("manifold_temperature_mid", 0.55))
            s.symbolic_lr_scale = float(cfg.get("manifold_symbolic_lr_scale", 3.0))
            s.symbolic_manifold_dual_init = float(cfg.get("manifold_dual_init", 3e-4))
            s.symbolic_manifold_rho = float(cfg.get("manifold_rho", 3e-4))
            s.symbolic_manifold_tolerance = float(cfg.get("manifold_tolerance", 0.02))
            s.symbolic_manifold_dual_update_every = int(cfg.get("manifold_dual_every", 50))
            s.symbolic_manifold_dual_max = float(cfg.get("manifold_dual_max", 0.03))
            s.symbolic_manifold_max_samples = int(cfg.get("manifold_max_samples", 256))
        if len(stages) > 2:
            s = stages[2]
            s.train_symbolic = True
            s.symbolic_enabled = False
            s.symbolic_hardening_start = float(cfg.get("manifold_hardening_mid", 0.65))
            s.symbolic_hardening_end = float(cfg.get("manifold_hardening_end", 1.0))
            s.operator_temperature_start = float(cfg.get("manifold_temperature_mid", 0.55))
            s.operator_temperature_end = float(cfg.get("manifold_temperature_end", 0.25))
            s.symbolic_lr_scale = float(cfg.get("manifold_symbolic_lr_scale", 3.0))
            s.symbolic_manifold_dual_init = float(cfg.get("manifold_dual_hard_init", 8e-4))
            s.symbolic_manifold_rho = float(cfg.get("manifold_hard_rho", 8e-4))
            s.symbolic_manifold_tolerance = float(cfg.get("manifold_tolerance", 0.02))
            s.symbolic_manifold_dual_update_every = int(cfg.get("manifold_dual_every", 50))
            s.symbolic_manifold_dual_max = float(cfg.get("manifold_dual_max", 0.03))
            s.symbolic_manifold_max_samples = int(cfg.get("manifold_max_samples", 256))
    # Optional compression pressure. These multipliers act only on the existing
    # compression/consolidation penalties; warmup remains an unconstrained
    # function-fitting stage so the model can first discover useful mechanisms.
    rule_l0_mult = float(cfg.get("numeric_rule_l0_multiplier", 1.0))
    factor_l0_mult = float(cfg.get("numeric_factor_l0_multiplier", 1.0))
    group_mult = float(cfg.get("numeric_group_lasso_multiplier", 1.0))
    if rule_l0_mult != 1.0 or factor_l0_mult != 1.0 or group_mult != 1.0:
        for stage in stages[1:]:
            stage.regularization.rule_l0 *= rule_l0_mult
            stage.regularization.factor_l0 *= factor_l0_mult
            stage.regularization.contribution_group_lasso *= group_mult
    corr_w = float(cfg.get("contribution_correlation", 0.0))
    graph_w = float(cfg.get("graph_redundancy", 0.0))
    corr_thr = float(cfg.get("contribution_correlation_threshold", 0.95))
    if corr_w or graph_w:
        # Apply only to compression/consolidation, never representation warmup.
        for stage in stages[1:]:
            stage.regularization.contribution_correlation = corr_w
            stage.regularization.graph_redundancy = graph_w
            stage.regularization.contribution_correlation_threshold = corr_thr
    pre = stages[:-1]
    hard = stages[-1:]
    fit_sum_product_kan(
        model, tx, ty, vx, vy, stages=pre,
        log_every=max(100000, int(cfg.get("log_every", 100000))),
        restore_best_each_stage=True, show_progress=bool(cfg.get("progress", False)),
    )
    # High-recall support evidence is captured before destructive pruning.  It
    # remains a numerical prior only: later symbolic search may change support
    # rank, factor multiplicity and operator identity inside these supports.
    pre_prune_support_evidence = capture_numeric_support_evidence(
        model, vx,
        factor_open_threshold=float(cfg.get("symbolic_support_factor_open_threshold", 0.20)),
    )
    prune_info_1 = None
    pruning_mode = str(cfg.get("pruning_mode", "iterative" if bool(cfg.get("iterative_pruning", True)) else "none"))
    if pruning_mode not in {"iterative", "one_shot", "none"}:
        raise ValueError(f"unknown pruning_mode={pruning_mode!r}")
    if pruning_mode != "none":
        model, prune_info_1 = prune_numeric_structure_to_stability(
            model, tx, ty, vx, vy,
            refit_steps=int(cfg.get("prune_refit_steps", 60)),
            refit_lr=float(cfg.get("prune_lr", 2e-4)),
            stabilize_rounds=int(cfg.get("prune_stabilize_rounds", 2)),
            stabilize_steps=int(cfg.get("prune_stabilize_steps", 30)),
            recovery_probe_count=int(cfg.get("prune_recovery_probes", 1)),
            max_candidates_per_pass=int(cfg.get("prune_candidates", 4)),
            local_rel_mse_budget=float(cfg.get("prune_local_budget", 0.01)),
            global_rel_mse_budget=float(cfg.get("prune_global_budget", 0.03)),
            min_rules=1, prune_optional_factors=True,
            max_accepted_deletions=(1 if pruning_mode == "one_shot" else 0),
            verbose=False, show_progress=False,
        )
    fit_sum_product_kan(
        model, tx, ty, vx, vy, stages=hard,
        log_every=max(100000, int(cfg.get("log_every", 100000))),
        restore_best_each_stage=True, show_progress=bool(cfg.get("progress", False)),
    )
    prune_info_2 = None
    if pruning_mode == "iterative":
        model, prune_info_2 = prune_numeric_structure_to_stability(
            model, tx, ty, vx, vy,
            refit_steps=int(cfg.get("prune_refit_steps", 60)),
            refit_lr=float(cfg.get("prune_lr", 2e-4)),
            stabilize_rounds=int(cfg.get("prune_stabilize_rounds", 2)),
            stabilize_steps=int(cfg.get("prune_stabilize_steps", 30)),
            recovery_probe_count=int(cfg.get("prune_recovery_probes", 1)),
            max_candidates_per_pass=int(cfg.get("prune_candidates", 4)),
            local_rel_mse_budget=float(cfg.get("prune_local_budget", 0.01)),
            global_rel_mse_budget=float(cfg.get("prune_global_budget", 0.03)),
            min_rules=1, prune_optional_factors=True, max_accepted_deletions=0,
            verbose=False, show_progress=False,
        )

    model.discretize(force_symbolic=False, freeze_gates=True)
    plateau_history = []
    if int(cfg.get("plateau_rounds", 1)) > 0:
        model, plateau_history = hard_numeric_plateau_polish(
            model, tx, ty, vx, vy,
            rounds=int(cfg.get("plateau_rounds", 1)),
            steps_per_round=int(cfg.get("plateau_steps", 80)),
            lr=float(cfg.get("plateau_lr", 3e-4)),
            min_relative_improvement=float(cfg.get("plateau_min_improvement", 1e-3)),
            patience=int(cfg.get("plateau_patience", 2)),
            verbose=False, show_progress=False,
        )

    # Optional fixed-structure second-order polish.  This deliberately keeps the
    # spline/RBF resolution and every discrete gate unchanged: only the already
    # selected numerical edge functions and continuous readout parameters move.
    # hard_numeric_precision_polish keeps the pre-LBFGS checkpoint unless the
    # validation RMSE improves, so enabling this cannot knowingly degrade the
    # held-out numerical precursor.
    lbfgs_steps = int(cfg.get("numeric_lbfgs_steps", 0))
    lbfgs_before_val = None
    lbfgs_after_val = None
    if lbfgs_steps > 0:
        model.eval()
        with torch.no_grad():
            lbfgs_before_val = float(torch.sqrt(torch.mean((model(vx) - vy) ** 2)).cpu())
        model, lbfgs_history = hard_numeric_precision_polish(
            model, tx, ty, vx, vy,
            target_rmse=float(cfg.get("numeric_target_rmse", 0.0)),
            grid_schedule=(),
            adam_steps_per_grid=0,
            lbfgs_steps=lbfgs_steps,
            extra_final_rounds=0,
            verbose=False, show_progress=False,
        )
        model.eval()
        with torch.no_grad():
            lbfgs_after_val = float(torch.sqrt(torch.mean((model(vx) - vy) ** 2)).cpu())
    logic_compression_info = None
    if bool(cfg.get("numeric_logic_compression", False)):
        model, logic_compression_info = compress_numeric_rule_bank(
            model, tx, ty, vx, vy,
            min_rules=int(cfg.get("numeric_logic_min_rules", 1)),
            max_deletions=int(cfg.get("numeric_logic_max_deletions", 0)),
            candidate_trials=int(cfg.get("numeric_logic_candidate_trials", 4)),
            lbfgs_steps=int(cfg.get("numeric_logic_lbfgs_steps", 24)),
            relative_rmse_tolerance=float(cfg.get("numeric_logic_relative_rmse_tolerance", 0.25)),
            nrmse_tolerance=float(cfg.get("numeric_logic_nrmse_tolerance", 3e-3)),
            verbose=False,
        )

    numeric_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        num_pred = model(qx)
    metrics = evaluate_predictions(num_pred, qy, data)
    metrics["numeric_seconds"] = float(numeric_seconds)
    numeric_scale_diag = _numeric_product_scale_diagnostics(model, vx)
    resolved_gmp_topk, resolved_gmp_unary_topk, resolved_gmp_self_topk = scaled_gmp_screening_sizes(
        len(model.symbolic_library),
        topk=(int(cfg["symbolic_gmp_topk"]) if "symbolic_gmp_topk" in cfg else None),
        unary_topk=(int(cfg["symbolic_gmp_unary_topk"]) if "symbolic_gmp_unary_topk" in cfg else None),
        self_product_topk=(int(cfg["symbolic_gmp_self_topk"]) if "symbolic_gmp_self_topk" in cfg else None),
    )

    learned_support_classes = learned_numeric_support_classes(
        model,
        evidence=pre_prune_support_evidence,
        max_supports=int(cfg.get("symbolic_learned_support_max", max(4, min(n_rules, 2 * in_dim + 2)))),
        include_all_active=True,
        score_importance_mix=float(cfg.get("symbolic_learned_support_importance_mix", 0.50)),
    )
    learned_support_bank = learned_structure_symbolic_bank(
        learned_support_classes, max_factors=max_factors,
    )

    symbolic_structure_candidates = None
    if bool(cfg.get("symbolic_use_compressed_numeric_structures", False)):
        logic_diag = numeric_logic_diagnostics(model, vx)
        raw_structures = [tuple(int(v) for v in z) for z in logic_diag.get("active_structures", [])]
        max_structures = int(cfg.get("symbolic_numeric_structure_max", 0))
        if raw_structures and (max_structures <= 0 or len(raw_structures) <= max_structures):
            symbolic_structure_candidates = raw_structures

    requested_gmp_chart = str(cfg.get("symbolic_gmp_identity_chart", "raw"))
    requested_gmp_norm = str(cfg.get("symbolic_gmp_atom_backward_normalization", "none"))
    requested_gmp_steps = int(cfg.get("symbolic_gmp_steps", 60))
    diagnostic_structures = []
    if bool(cfg.get("symbolic", False)) and spec.synthetic and data.task_type == "regression":
        diagnostic_structures = list(symbolic_structure_candidates) if symbolic_structure_candidates else symbolic_structure_bank(
            in_dim, max_factors, allow_self_products=bool(cfg.get("allow_symbolic_self_products", True))
        )
    effective_gmp_policy_examples = []
    policy_counts: Dict[str, int] = {}
    for structure in diagnostic_structures:
        chart, norm = resolve_gmp_local_policy(
            structure, steps=requested_gmp_steps, identity_chart=requested_gmp_chart,
            atom_backward_normalization=requested_gmp_norm,
        )
        key = f"{chart}+{norm}"
        policy_counts[key] = policy_counts.get(key, 0) + 1
        if len(effective_gmp_policy_examples) < 12:
            effective_gmp_policy_examples.append({
                "structure": list(structure), "identity_chart": chart, "atom_normalization": norm,
            })
    effective_gmp_policy_summary = "; ".join(f"{k}:{v}" for k, v in sorted(policy_counts.items()))

    extras: Dict[str, Any] = {
        "parameters": _param_count(model),
        "active_numeric_rules": int((model.hard_rule_choice & model.rule_alive_mask).sum().item()),
        # Primary/pre-rescue diagnostics.  Downstream RuleKAN-derived models
        # must use symbolic_effective_support_* instead.
        "symbolic_learned_support_classes": _serialize_rulekan_support_classes(learned_support_classes),
        "symbolic_learned_support_bank": [list(z) for z in learned_support_bank],
        # Canonical downstream contract.  This starts as the primary support
        # bank and is replaced atomically if validation selects the high-recall
        # structure-conditioned rescue.
        **_rulekan_effective_support_payload(
            learned_support_classes, learned_support_bank,
            source="primary_learned_numeric_supports",
        ),
        "numeric_logic_compression": bool(cfg.get("numeric_logic_compression", False)),
        "numeric_logic_deletions": len(logic_compression_info["accepted_deletions"]) if logic_compression_info else 0,
        "numeric_logic_before_active_rules": int(len(logic_compression_info["before"]["active_rules"])) if logic_compression_info else None,
        "numeric_logic_after_active_rules": int(len(logic_compression_info["after"]["active_rules"])) if logic_compression_info else None,
        "numeric_logic_before_cancellation_index": float(logic_compression_info["before"]["cancellation_index"]) if logic_compression_info else None,
        "numeric_logic_after_cancellation_index": float(logic_compression_info["after"]["cancellation_index"]) if logic_compression_info else None,
        "numeric_logic_before_max_span_r2": float(logic_compression_info["before"]["max_span_r2"]) if logic_compression_info else None,
        "numeric_logic_after_max_span_r2": float(logic_compression_info["after"]["max_span_r2"]) if logic_compression_info else None,
        "numeric_logic_active_structures": [list(z) for z in logic_compression_info["after"]["active_structures"]] if logic_compression_info else [],
        "prune1_deletions": len(prune_info_1["accepted"]) if prune_info_1 else 0,
        "prune2_deletions": len(prune_info_2["accepted"]) if prune_info_2 else 0,
        "numeric_basis": str(numeric_basis),
        "pursuit_mode": str(cfg.get("pursuit_mode", pursuit_mode)),
        "contribution_correlation": float(cfg.get("contribution_correlation", 0.0)),
        "numeric_rule_l0_multiplier": float(cfg.get("numeric_rule_l0_multiplier", 1.0)),
        "numeric_factor_l0_multiplier": float(cfg.get("numeric_factor_l0_multiplier", 1.0)),
        "numeric_group_lasso_multiplier": float(cfg.get("numeric_group_lasso_multiplier", 1.0)),
        "graph_redundancy": float(cfg.get("graph_redundancy", 0.0)),
        "pruning_mode": pruning_mode,
        "sumproduct_enabled": bool(max_factors > 1),
        "gmp_preselection": bool(cfg.get("use_gmp_preselection", True)),
        "symbolic_self_products": bool(cfg.get("allow_symbolic_self_products", True)),
        "symbolic_backfit": bool(cfg.get("symbolic_backfit", True)),
        "resolved_n_rules": int(n_rules),
        "resolved_max_factors": int(max_factors),
        "resolved_grid": int(cfg.get("grid", 12)),
        "numeric_stage_scale": float(cfg.get("stage_scale", 0.15)),
        "symbolic_gradient_scale": float(cfg.get("symbolic_gradient_scale", 4.0)),
        "symbolic_product_gradient_scale": float(cfg.get("symbolic_product_gradient_scale", 2.0)),
        "numeric_factor_gradient_scale": float(cfg.get("numeric_factor_gradient_scale", 4.0)),
        "numeric_product_gradient_scale": float(cfg.get("numeric_product_gradient_scale", 2.0)),
        "numeric_product_gradient_power": float(cfg.get("numeric_product_gradient_power", 1.0)),
        "numeric_product_gradient_max_gain": float(cfg.get("numeric_product_gradient_max_gain", 8.0)),
        **numeric_scale_diag,
        "numeric_plateau_rounds_completed": max(0, len(plateau_history) - 1),
        "numeric_lbfgs_steps": int(lbfgs_steps),
        "numeric_lbfgs_before_val_rmse": lbfgs_before_val,
        "numeric_lbfgs_after_val_rmse": lbfgs_after_val,
        "resolved_symbolic_library_size": int(len(model.symbolic_library)),
        "symbolic_seed": int(cfg.get("symbolic_seed", seed + 100_003)),
        "resolved_symbolic_gmp_topk": int(resolved_gmp_topk),
        "resolved_symbolic_gmp_unary_topk": int(resolved_gmp_unary_topk),
        "resolved_symbolic_gmp_self_topk": int(resolved_gmp_self_topk),
        "symbolic_gmp_relaxation_mode": str(cfg.get("symbolic_gmp_relaxation_mode", "soft")),
        "symbolic_gmp_atom_backward_normalization": requested_gmp_norm,
        "symbolic_gmp_effective_policy": effective_gmp_policy_summary,
        "symbolic_gmp_effective_policy_examples": effective_gmp_policy_examples,
        "symbolic_gmp_complexity_weight": float(cfg.get("symbolic_gmp_complexity_weight", 0.0)),
        "symbolic_gmp_complexity_logit_prior": float(cfg.get("symbolic_gmp_complexity_logit_prior", 0.0)),
        "symbolic_gmp_nonlinearity_logit_prior": float(cfg.get("symbolic_gmp_nonlinearity_logit_prior", 0.0)),
        "symbolic_gmp_curvature_weight": float(cfg.get("symbolic_gmp_curvature_weight", 0.0)),
        "symbolic_gmp_nonlinearity_weight": float(cfg.get("symbolic_gmp_nonlinearity_weight", 0.0)),
        "symbolic_gmp_identity_chart": requested_gmp_chart,
        "symbolic_interaction_shape_screening": bool(cfg.get("symbolic_interaction_shape_screening", True)),
        "symbolic_interaction_shape_bins": int(cfg.get("symbolic_interaction_shape_bins", 8)),
        "symbolic_interaction_shape_min_rank1": float(cfg.get("symbolic_interaction_shape_min_rank1", 0.65)),
        "symbolic_interaction_shape_require_multiple": bool(cfg.get("symbolic_interaction_shape_require_multiple", True)),
        "symbolic_initial_block_pursuit": bool(cfg.get("symbolic_initial_block_pursuit", True)),
        "symbolic_initial_block_pool": int(cfg.get("symbolic_initial_block_pool", 28)),
        "symbolic_initial_block_pair_beam": int(cfg.get("symbolic_initial_block_pair_beam", 10)),
        "symbolic_initial_block_consolidate_steps": int(cfg.get("symbolic_initial_block_consolidate_steps", 200)),
        "symbolic_initial_block_consolidate_lbfgs_steps": int(cfg.get("symbolic_initial_block_consolidate_lbfgs_steps", 60)),
        "symbolic_hybrid_hard_screening": bool(cfg.get("symbolic_hybrid_hard_screening", False)),
        "symbolic_residual_structure_topk": int(cfg.get("symbolic_residual_structure_topk", 0)),
        "symbolic_residual_structure_gap_rel": float(cfg.get("symbolic_residual_structure_gap_rel", 0.0)),
        "symbolic_joint_scale_refit": bool(cfg.get("symbolic_joint_scale_refit", False)),
        "symbolic_hard_screen_beam": int(cfg.get("symbolic_hard_screen_beam", 64)),
        "symbolic_hard_screen_top_tuples": int(cfg.get("symbolic_hard_screen_top_tuples", 12)),
        "symbolic_hard_screen_start_step": int(cfg.get("symbolic_hard_screen_start_step", 1)),
        "symbolic_hard_screen_matching_steps": int(cfg.get("symbolic_hard_screen_matching_steps", 4)),
        "symbolic_hard_screen_global_candidates": int(cfg.get("symbolic_hard_screen_global_candidates", 2)),
        "symbolic_hard_screen_residual_rescue": bool(cfg.get("symbolic_hard_screen_residual_rescue", False)),
        "symbolic_use_compressed_numeric_structures": bool(cfg.get("symbolic_use_compressed_numeric_structures", False)),
        "symbolic_numeric_structure_candidates": [list(z) for z in symbolic_structure_candidates] if symbolic_structure_candidates else [],
    }

    symbolic_model = None
    if bool(cfg.get("symbolic", False)) and spec.synthetic and data.task_type == "regression":
        st = time.perf_counter()
        try:
            symbolic_takeover = str(cfg.get(
                "symbolic_takeover",
                "learned_support_gsr" if model_name.startswith("rulekan") else "matching_pursuit",
            )).strip().lower()
            if symbolic_takeover in {"distill", "numeric_distill", "warm_start"}:
                symbolic_model, distill_history = distill_numeric_structure_to_symbolic(
                    model, tx, ty, vx, vy,
                    gmp_steps=int(cfg.get("symbolic_gmp_steps", 60)),
                    gmp_topk=(int(cfg["symbolic_gmp_topk"]) if "symbolic_gmp_topk" in cfg else None),
                    gmp_unary_topk=(int(cfg["symbolic_gmp_unary_topk"]) if "symbolic_gmp_unary_topk" in cfg else None),
                    gmp_self_product_topk=(int(cfg["symbolic_gmp_self_topk"]) if "symbolic_gmp_self_topk" in cfg else None),
                    gmp_relaxation_mode=str(cfg.get("symbolic_gmp_relaxation_mode", "soft")),
                    gmp_atom_backward_normalization=requested_gmp_norm,
                    gmp_identity_chart=requested_gmp_chart,
                    tuple_refine_steps=int(cfg.get("symbolic_tuple_steps", 20)),
                    max_rule_candidates=int(cfg.get("symbolic_max_rule_candidates", 16)),
                    final_steps=int(cfg.get("symbolic_final_steps", 120)),
                    final_lbfgs_steps=int(cfg.get("symbolic_final_lbfgs", 20)),
                    symbolic_seed=int(cfg.get("symbolic_seed", seed + 100_003)),
                    verbose=False,
                )
                extras["symbolic_takeover"] = "numeric_distill"
                extras["distilled_numeric_rules"] = int(sum(1 for h in distill_history if "operators" in h))
            elif symbolic_takeover in {"manifold", "manifold_projection", "constrained_projection"}:
                symbolic_model = project_constrained_numeric_to_symbolic(
                    model, tx, ty, vx, vy,
                    final_steps=int(cfg.get("symbolic_final_steps", 120)),
                    final_lr=float(cfg.get("manifold_projection_lr", 2e-4)),
                    final_lbfgs_steps=int(cfg.get("symbolic_final_lbfgs", 20)),
                )
                extras["symbolic_takeover"] = "manifold_projection"
            elif symbolic_takeover in {"learned_support_gsr", "rulekan", "learned"}:
                if not learned_support_bank:
                    raise RuntimeError("RuleKAN learned no numerical support bank for symbolic GSR")
                symbolic_model, symbolic_history = _fit_rulekan_learned_support_gsr(
                    model, tx, ty, vx, vy, learned_support_bank, learned_support_classes, cfg,
                    n_rules=n_rules, seed=seed, pursuit_mode=pursuit_mode,
                )

                # Rank continuation explores sparse symbolic branches without
                # reducing the declared capacity. symbolic_max_rules remains the
                # maximum admissible rank; smaller caps are alternative search
                # trajectories selected by validation loss.
                rank_runs=[]
                with torch.no_grad():
                    _base_vmse=float(torch.mean((symbolic_model(vx)-vy)**2).cpu())
                rank_runs.append({"cap":int(cfg.get("symbolic_max_rules",n_rules)),"val_rmse":math.sqrt(max(_base_vmse,0.0))})
                if bool(cfg.get("symbolic_rank_continuation", True)):
                    max_cap=int(cfg.get("symbolic_max_rules",n_rules))
                    requested_caps=cfg.get("symbolic_rank_continuation_caps", [3,6])
                    caps=[]
                    for cc in requested_caps:
                        ci=max(int(cfg.get("symbolic_min_rules",1)),int(cc))
                        if ci<max_cap and ci not in caps: caps.append(ci)
                    best_model=symbolic_model;best_hist=symbolic_history;best_vmse=_base_vmse;best_cap=max_cap
                    for ci in caps:
                        cand,cand_hist=_fit_rulekan_learned_support_gsr(
                            model, tx, ty, vx, vy, learned_support_bank, learned_support_classes, cfg,
                            n_rules=n_rules, seed=seed, pursuit_mode=pursuit_mode, max_symbolic_rules=ci,
                        )
                        cand.eval()
                        with torch.no_grad(): cv=float(torch.mean((cand(vx)-vy)**2).cpu())
                        rank_runs.append({"cap":int(ci),"val_rmse":math.sqrt(max(cv,0.0))})
                        if math.isfinite(cv) and cv<best_vmse:
                            best_model,best_hist,best_vmse,best_cap=cand,cand_hist,cv,ci
                    symbolic_model,symbolic_history=best_model,best_hist
                    extras["symbolic_rank_continuation_selected_cap"]=int(best_cap)
                extras["symbolic_rank_continuation_runs"]=rank_runs
                # Surface affine-partition diagnostics from the validation-selected
                # symbolic trajectory.  These are observational only.
                _part = next((h for h in reversed(symbolic_history)
                              if h.get("affine_partition_rescue_attempted") is not None), None)
                if _part is not None:
                    extras["symbolic_affine_partition_attempted"] = bool(_part.get("affine_partition_rescue_attempted", False))
                    extras["symbolic_affine_partition_selected"] = bool(_part.get("affine_partition_rescue", False))
                    extras["symbolic_affine_partition_reason"] = str(_part.get("affine_partition_rescue_reason", "unknown"))
                    extras["symbolic_affine_partition_gate_variable"] = _part.get("gate_variable")
                    extras["symbolic_affine_partition_support_a"] = _part.get("support_a")
                    extras["symbolic_affine_partition_support_b"] = _part.get("support_b")
                    extras["symbolic_affine_partition_operators"] = _part.get("operators")
                    extras["symbolic_affine_partition_incumbent_validation_mse"] = _part.get("incumbent_validation_mse")
                    extras["symbolic_affine_partition_predictive_best_validation_mse"] = _part.get("predictive_best_validation_mse")
                    extras["symbolic_affine_partition_best_validation_mse"] = _part.get("best_validation_mse")
                    extras["symbolic_affine_partition_equivalence_cap"] = _part.get("validation_equivalence_cap")
                    extras["symbolic_affine_partition_partition_seconds"] = _part.get("partition_seconds")
                    extras["symbolic_affine_partition_incumbent_preference_score"] = _part.get("incumbent_preference_score")
                    extras["symbolic_affine_partition_best_partition_preference_validation_mse"] = _part.get("best_partition_preference_validation_mse")
                    extras["symbolic_affine_partition_best_partition_preference_cancellation_score"] = _part.get("best_partition_preference_cancellation_score")
                    extras["symbolic_affine_partition_best_partition_preference_description_complexity"] = _part.get("best_partition_preference_description_complexity")
                    extras["symbolic_affine_partition_best_partition_preference_preference_score"] = _part.get("best_partition_preference_preference_score")
                    extras["symbolic_affine_partition_best_partition_preference_operator_a"] = _part.get("best_partition_preference_operator_a")
                    extras["symbolic_affine_partition_best_partition_preference_operator_b"] = _part.get("best_partition_preference_operator_b")
                    extras["symbolic_affine_partition_best_partition_predictive_validation_mse"] = _part.get("best_partition_predictive_validation_mse")
                    extras["symbolic_affine_partition_best_partition_predictive_preference_score"] = _part.get("best_partition_predictive_preference_score")
                    extras["symbolic_affine_partition_best_partition_predictive_operator_a"] = _part.get("best_partition_predictive_operator_a")
                    extras["symbolic_affine_partition_best_partition_predictive_operator_b"] = _part.get("best_partition_predictive_operator_b")

                # Adaptive rescue: retain strict learned-support RuleKAN as
                # the primary trajectory, then optionally deepen the symbolic
                # search on a higher-recall bank of numerical supports captured
                # before pruning.  This remains structure-conditioned: no support
                # absent from RuleKAN's numerical evidence can enter the search.
                # The rescue is selected only when it improves validation MSE.
                rescue_enabled = bool(cfg.get("symbolic_validation_rescue", False))
                with torch.no_grad():
                    primary_vmse = float(torch.mean((symbolic_model(vx) - vy) ** 2).cpu())
                    numeric_vmse = float(torch.mean((model(vx) - vy) ** 2).cpu())
                    val_scale = float(torch.std(vy).cpu())
                val_scale = max(val_scale, 1e-12)
                primary_vrmse = math.sqrt(max(primary_vmse, 0.0))
                primary_vnrmse = primary_vrmse / val_scale
                numeric_vrmse = math.sqrt(max(numeric_vmse, 0.0))
                rescue_threshold = float(cfg.get("symbolic_validation_rescue_nrmse", 0.03))
                rescue_numeric_ratio = float(cfg.get("symbolic_validation_rescue_numeric_ratio", 1.35))
                rescue_triggered = bool(
                    rescue_enabled
                    and (
                        not math.isfinite(primary_vnrmse)
                        or (
                            primary_vnrmse > rescue_threshold
                            and (
                                not math.isfinite(numeric_vrmse)
                                or primary_vrmse > rescue_numeric_ratio * max(numeric_vrmse, 1e-12)
                                or primary_vnrmse > float(cfg.get("symbolic_validation_rescue_force_nrmse", 0.10))
                            )
                        )
                    )
                )
                extras["symbolic_validation_rescue_enabled"] = rescue_enabled
                extras["symbolic_validation_rescue_triggered"] = rescue_triggered
                extras["symbolic_validation_primary_val_rmse"] = primary_vrmse
                extras["symbolic_validation_primary_val_nrmse"] = primary_vnrmse
                extras["symbolic_validation_numeric_val_rmse"] = numeric_vrmse
                extras["symbolic_validation_rescue_selected"] = False
                if rescue_triggered:
                    try:
                        rescue_support_classes = learned_numeric_support_classes(
                            model,
                            evidence=pre_prune_support_evidence,
                            max_supports=int(cfg.get("symbolic_validation_rescue_support_max", 0)),
                            include_all_active=True,
                            score_importance_mix=float(cfg.get("symbolic_learned_support_importance_mix", 0.50)),
                        )
                        rescue_support_bank = learned_structure_symbolic_bank(
                            rescue_support_classes, max_factors=max_factors,
                        )
                        primary_supports = {tuple(int(v) for v in c["support"]) for c in learned_support_classes}
                        rescue_supports = {tuple(int(v) for v in c["support"]) for c in rescue_support_classes}
                        extras["symbolic_validation_rescue_support_classes"] = [
                            list(z) for z in sorted(rescue_supports, key=lambda z: (len(z), z))
                        ]
                        extras["symbolic_validation_rescue_added_support_classes"] = [
                            list(z) for z in sorted(rescue_supports - primary_supports, key=lambda z: (len(z), z))
                        ]
                        extras["symbolic_validation_rescue_structure_bank_size"] = int(len(rescue_support_bank))
                        rescue_model, rescue_history = _fit_validation_structure_rescue_symbolic_gsr(
                            model, tx, ty, vx, vy,
                            rescue_support_classes, rescue_support_bank, cfg,
                            n_rules=n_rules, seed=seed, pursuit_mode=pursuit_mode,
                        )
                        rescue_model.eval()
                        with torch.no_grad():
                            rescue_vmse = float(torch.mean((rescue_model(vx) - vy) ** 2).cpu())
                        rescue_vrmse = math.sqrt(max(rescue_vmse, 0.0))
                        extras["symbolic_validation_rescue_val_rmse"] = rescue_vrmse
                        min_rel = max(0.0, float(cfg.get("symbolic_validation_rescue_min_improvement_rel", 0.005)))
                        if math.isfinite(rescue_vmse) and (
                            not math.isfinite(primary_vmse)
                            or rescue_vmse < primary_vmse * (1.0 - min_rel)
                        ):
                            symbolic_model, symbolic_history = rescue_model, rescue_history
                            primary_vmse = rescue_vmse
                            extras["symbolic_validation_rescue_selected"] = True
                            # Atomically promote the validation-selected rescue
                            # bank to the canonical support contract.  Every
                            # downstream RuleKAN-derived model (PowerRuleKAN
                            # included) therefore inherits the same supports by
                            # default instead of reading stale primary fields.
                            extras.update(_rulekan_effective_support_payload(
                                rescue_support_classes, rescue_support_bank,
                                source="validation_selected_high_recall_numeric_supports",
                            ))
                    except Exception as rescue_exc:
                        extras["symbolic_validation_rescue_error"] = f"{type(rescue_exc).__name__}: {rescue_exc}"

                extras["symbolic_takeover"] = (
                    "adaptive_high_recall_learned_support_rescue"
                    if extras["symbolic_validation_rescue_selected"]
                    else "learned_support_gsr"
                )
                extras["symbolic_structure_source"] = (
                    "validation_selected_high_recall_numeric_supports"
                    if extras["symbolic_validation_rescue_selected"]
                    else "pre_prune_high_recall_numeric_supports"
                )
            elif symbolic_takeover in {"matching_pursuit", "gsr", "fresh"}:
                symbolic_model, symbolic_history = mandatory_symbolic_matching_pursuit(
                    model, tx, ty, vx, vy,
                max_symbolic_rules=int(cfg.get("symbolic_max_rules", 6)),
                min_symbolic_rules=int(cfg.get("symbolic_min_rules", 1)),
                allow_symbolic_self_products=bool(cfg.get("allow_symbolic_self_products", True)),
                structure_candidates=symbolic_structure_candidates,
                use_gmp_preselection=bool(cfg.get("use_gmp_preselection", True)),
                gmp_steps=int(cfg.get("symbolic_gmp_steps", 60)),
                gmp_topk=(int(cfg["symbolic_gmp_topk"]) if "symbolic_gmp_topk" in cfg else None),
                gmp_unary_topk=(int(cfg["symbolic_gmp_unary_topk"]) if "symbolic_gmp_unary_topk" in cfg else None),
                gmp_self_product_topk=(int(cfg["symbolic_gmp_self_topk"]) if "symbolic_gmp_self_topk" in cfg else None),
                gmp_relaxation_mode=str(cfg.get("symbolic_gmp_relaxation_mode", "soft")),
                gmp_atom_backward_normalization=str(cfg.get("symbolic_gmp_atom_backward_normalization", "none")),
                gmp_gumbel_noise_scale=float(cfg.get("symbolic_gmp_gumbel_noise_scale", 1.0)),
                gmp_complexity_weight=float(cfg.get("symbolic_gmp_complexity_weight", 0.0)),
                gmp_complexity_logit_prior=float(cfg.get("symbolic_gmp_complexity_logit_prior", 0.0)),
                gmp_nonlinearity_logit_prior=float(cfg.get("symbolic_gmp_nonlinearity_logit_prior", 0.0)),
                gmp_curvature_weight=float(cfg.get("symbolic_gmp_curvature_weight", 0.0)),
                gmp_nonlinearity_weight=float(cfg.get("symbolic_gmp_nonlinearity_weight", 0.0)),
                gmp_identity_chart=str(cfg.get("symbolic_gmp_identity_chart", "raw")),
                gmp_tuple_refine_steps=int(cfg.get("symbolic_tuple_steps", 20)),
                interaction_shape_screening=bool(cfg.get("symbolic_interaction_shape_screening", True)),
                interaction_shape_topk=int(cfg.get("symbolic_interaction_shape_topk", 0)),
                interaction_shape_bins=int(cfg.get("symbolic_interaction_shape_bins", 8)),
                interaction_shape_min_rank1=float(cfg.get("symbolic_interaction_shape_min_rank1", 0.65)),
                interaction_shape_require_multiple=bool(cfg.get("symbolic_interaction_shape_require_multiple", True)),
                initial_block_pursuit=bool(cfg.get("symbolic_initial_block_pursuit", True)),
                initial_block_pool=int(cfg.get("symbolic_initial_block_pool", 28)),
                initial_block_pair_beam=int(cfg.get("symbolic_initial_block_pair_beam", 10)),
                initial_block_refit_steps=int(cfg.get("symbolic_initial_block_refit_steps", 100)),
                initial_block_lbfgs_steps=int(cfg.get("symbolic_initial_block_lbfgs_steps", 20)),
                initial_block_consolidate_steps=int(cfg.get("symbolic_initial_block_consolidate_steps", 200)),
                initial_block_consolidate_lbfgs_steps=int(cfg.get("symbolic_initial_block_consolidate_lbfgs_steps", 60)),
                hybrid_hard_screening=bool(cfg.get("symbolic_hybrid_hard_screening", False)),
                hard_screen_beam_width=int(cfg.get("symbolic_hard_screen_beam", 64)),
                hard_screen_top_tuples=int(cfg.get("symbolic_hard_screen_top_tuples", 12)),
                hard_screen_max_samples=int(cfg.get("symbolic_hard_screen_samples", 512)),
                hard_screen_start_step=int(cfg.get("symbolic_hard_screen_start_step", 1)),
                hard_screen_matching_steps=int(cfg.get("symbolic_hard_screen_matching_steps", 4)),
                hard_screen_global_candidates=int(cfg.get("symbolic_hard_screen_global_candidates", 2)),
                hard_screen_residual_rescue=bool(cfg.get("symbolic_hard_screen_residual_rescue", False)),
                beam_width=int(cfg.get("symbolic_beam", 4)),
                structure_diverse_beam=bool(cfg.get("symbolic_structure_diverse_beam", False)),
                beam_max_per_structure=int(cfg.get("symbolic_beam_max_per_structure", 1)),
                residual_structure_topk=int(cfg.get("symbolic_residual_structure_topk", 0)),
                residual_structure_mass=float(cfg.get("symbolic_residual_structure_mass", 0.0)),
                residual_structure_min=int(cfg.get("symbolic_residual_structure_min", 1)),
                residual_structure_max=int(cfg.get("symbolic_residual_structure_max", 0)),
        residual_structure_gap_rel=float(cfg.get("symbolic_residual_structure_gap_rel", 0.0)),
                joint_scale_refit_each_commit=bool(cfg.get("symbolic_joint_scale_refit", False)),
                trial_steps=int(cfg.get("symbolic_trial_steps", 50)),
                commit_refit_steps=int(cfg.get("symbolic_commit_steps", 80)),
                commit_lbfgs_steps=int(cfg.get("symbolic_commit_lbfgs", 10)),
                backfit=bool(cfg.get("symbolic_backfit", True)),
                backfit_beam_width=int(cfg.get("symbolic_backfit_beam", 3)),
                backfit_steps=int(cfg.get("symbolic_backfit_steps", 40)),
                backfit_lbfgs_steps=int(cfg.get("symbolic_backfit_lbfgs", 5)),
                final_steps=int(cfg.get("symbolic_final_steps", 120)),
                final_lbfgs_steps=int(cfg.get("symbolic_final_lbfgs", 20)),
                target_val_rmse=float(cfg.get("symbolic_target_rmse", 1e-4)),
                pursuit_mode=str(cfg.get("pursuit_mode", pursuit_mode)),
                omp_extra_steps=int(cfg.get("omp_extra_steps", 60)),
                redundancy_cleanup=bool(cfg.get("redundancy_cleanup", True)),
                redundancy_span_r2_threshold=float(cfg.get("redundancy_span_r2", 0.995)),
                redundancy_corr_threshold=float(cfg.get("redundancy_corr", 0.995)),
                debug_topk=0,
                cleanup_max_seconds=float(cfg.get("symbolic_cleanup_seconds", 20.0)),
                cleanup_max_trials=int(cfg.get("symbolic_cleanup_trials", 10)),
                symbolic_seed=int(cfg.get("symbolic_seed", seed + 100_003)),
                    verbose=False, show_progress=False,
                )
                extras["symbolic_takeover"] = "matching_pursuit"
                extras["symbolic_block_initial_rules"] = int(sum(bool(h.get("block_initial", False)) for h in symbolic_history))
                extras["symbolic_block_initial_used"] = bool(extras["symbolic_block_initial_rules"] >= 2)
            else:
                raise ValueError(f"unknown symbolic_takeover={symbolic_takeover!r}")
            # SR-inspired validation-gated rescue portfolio.  These rescues run
            # once on the validation-selected symbolic trajectory (after rank
            # continuation / adaptive support rescue), avoiding the expensive
            # mistake of repeating an exhaustive search inside every rank cap.
            if symbolic_model is not None:
                if model_name.startswith("sisp"):
                    _rescue_structures = symbolic_structure_bank(
                        in_dim, max_factors,
                        allow_self_products=bool(cfg.get("allow_symbolic_self_products", True)),
                    )
                    _rescue_allowed_supports = None
                else:
                    _effective_classes = [
                        {**dict(c), "support": tuple(int(v) for v in c.get("support",()))}
                        for c in extras.get("symbolic_effective_support_classes", [])
                        if c.get("support")
                    ] or list(learned_support_classes)
                    if bool(cfg.get("symbolic_validation_rescue_support_subset_closure", True)):
                        _effective_classes = _downward_support_closure(_effective_classes)
                        extras["symbolic_rescue_support_subset_closure"] = True
                    else:
                        extras["symbolic_rescue_support_subset_closure"] = False
                    if bool(cfg.get("symbolic_gate_aware_support_augmentation", True)):
                        _effective_classes = _gate_aware_support_augmentation(
                            _effective_classes, data, max_factors=max_factors,
                        )
                        extras["symbolic_gate_aware_support_augmentation"] = True
                        extras["symbolic_gate_candidate_variables"] = _membership_gate_variables_from_data(data)
                    else:
                        extras["symbolic_gate_aware_support_augmentation"] = False
                        extras["symbolic_gate_candidate_variables"] = []
                    _rescue_allowed_supports = [tuple(int(v) for v in c["support"]) for c in _effective_classes]
                    _rescue_structures = learned_structure_symbolic_bank(
                        _effective_classes, max_factors=max_factors,
                    )
                    extras["symbolic_rescue_support_classes"] = [list(z) for z in _rescue_allowed_supports]

                symbolic_model.eval()
                with torch.no_grad():
                    _portfolio_vmse = float(torch.mean((symbolic_model(vx)-vy)**2).cpu())
                    _portfolio_scale = max(float(torch.std(vy).cpu()), 1e-12)
                _portfolio_vnrmse = math.sqrt(max(_portfolio_vmse, 0.0)) / _portfolio_scale

                extras["symbolic_low_dimensional_family_rescue_enabled"] = bool(
                    cfg.get("symbolic_low_dimensional_family_rescue", True)
                )
                extras["symbolic_complementary_two_rule_rescue_enabled"] = bool(
                    cfg.get("symbolic_complementary_two_rule_rescue", True)
                )
                extras["symbolic_product_partition_rescue_enabled"] = bool(
                    cfg.get("symbolic_product_partition_rescue", True)
                )
                extras["symbolic_recursive_partition_rescue_enabled"] = bool(
                    cfg.get("symbolic_recursive_partition_rescue", True)
                )

                _low_trigger = float(cfg.get("symbolic_low_dimensional_rescue_trigger_nrmse", 1e-5))
                if (
                    extras["symbolic_low_dimensional_family_rescue_enabled"]
                    and in_dim <= int(cfg.get("symbolic_low_dimensional_max_input_dim", 2))
                    and (_portfolio_vnrmse > _low_trigger or not math.isfinite(_portfolio_vnrmse))
                ):
                    _t_rescue=time.perf_counter()
                    _low_model,_low_meta = low_dimensional_symbolic_family_rescue(
                        model, symbolic_model, tx, ty, vx, vy,
                        structure_candidates=_rescue_structures,
                        allowed_supports=_rescue_allowed_supports,
                        library=tuple(model.symbolic_library),
                        max_input_dim=int(cfg.get("symbolic_low_dimensional_max_input_dim", 2)),
                        max_order=int(cfg.get("symbolic_low_dimensional_max_order", 2)),
                        family_refine_topk=int(cfg.get("symbolic_low_dimensional_family_refine_topk", 18)),
                        max_samples=int(cfg.get("symbolic_low_dimensional_max_samples", 512)),
                        refine_steps=int(cfg.get("symbolic_low_dimensional_refine_steps", 180)),
                        refine_lr=float(cfg.get("symbolic_low_dimensional_refine_lr", 8e-4)),
                        lbfgs_steps=int(cfg.get("symbolic_low_dimensional_lbfgs_steps", 40)),
                        final_topk=int(cfg.get("symbolic_low_dimensional_final_topk", 5)),
                        final_steps=int(cfg.get("symbolic_low_dimensional_final_steps", 500)),
                        final_lbfgs_steps=int(cfg.get("symbolic_low_dimensional_final_lbfgs_steps", 100)),
                        min_improvement_rel=float(cfg.get("symbolic_low_dimensional_min_improvement_rel", 1e-4)),
                        verbose=False,
                    )
                    extras["symbolic_low_dimensional_family_rescue_attempted"] = bool(_low_meta.get("attempted",False))
                    extras["symbolic_low_dimensional_family_rescue_selected"] = bool(_low_meta.get("selected",False))
                    extras["symbolic_low_dimensional_family_rescue_reason"] = str(_low_meta.get("reason","unknown"))
                    extras["symbolic_low_dimensional_family_rescue_operators"] = _low_meta.get("operators")
                    extras["symbolic_low_dimensional_family_rescue_structure"] = _low_meta.get("structure")
                    extras["symbolic_low_dimensional_family_rescue_families_screened"] = _low_meta.get("families_screened")
                    extras["symbolic_low_dimensional_family_rescue_seconds"] = float(time.perf_counter()-_t_rescue)
                    if bool(_low_meta.get("selected",False)):
                        symbolic_model=_low_model
                        _portfolio_vmse=float(_low_meta.get("best_validation_mse",_portfolio_vmse))
                        _portfolio_vnrmse=math.sqrt(max(_portfolio_vmse,0.0))/_portfolio_scale
                else:
                    extras["symbolic_low_dimensional_family_rescue_attempted"] = False
                    extras["symbolic_low_dimensional_family_rescue_selected"] = False

                # Exhaustive two-rule complementary search fills the gap between
                # single-rule low-dimensional rescue and higher-order product
                # partitions.  RuleKAN remains support-constrained; SISP passes
                # an unrestricted support contract as usual.
                _two_trigger = float(cfg.get("symbolic_complementary_two_rule_trigger_nrmse", 1e-5))
                if (
                    extras["symbolic_complementary_two_rule_rescue_enabled"]
                    and in_dim <= int(cfg.get("symbolic_complementary_two_rule_max_input_dim", 2))
                    and max_factors >= 2
                    and (_portfolio_vnrmse > _two_trigger or not math.isfinite(_portfolio_vnrmse))
                ):
                    _t_rescue=time.perf_counter()
                    _two_model,_two_meta = complementary_two_rule_symbolic_rescue(
                        model, symbolic_model, tx, ty, vx, vy,
                        input_mean=data.input_mean.to(tx),
                        input_std=data.input_std.to(tx),
                        structure_candidates=_rescue_structures,
                        allowed_supports=_rescue_allowed_supports,
                        library=tuple(model.symbolic_library),
                        max_input_dim=int(cfg.get("symbolic_complementary_two_rule_max_input_dim", 2)),
                        max_samples=int(cfg.get("symbolic_complementary_two_rule_max_samples", 384)),
                        max_support_pairs=int(cfg.get("symbolic_complementary_two_rule_max_support_pairs", 12)),
                        coarse_global_topk=int(cfg.get("symbolic_complementary_two_rule_coarse_global_topk", 16)),
                        coarse_final_topk=int(cfg.get("symbolic_complementary_two_rule_coarse_final_topk", 32)),
                        seed_restarts=int(cfg.get("symbolic_complementary_two_rule_seed_restarts", 3)),
                        local_polish_steps=int(cfg.get("symbolic_complementary_two_rule_local_polish_steps", 28)),
                        local_polish_lr=float(cfg.get("symbolic_complementary_two_rule_local_polish_lr", 1e-2)),
                        local_lbfgs_topk=int(cfg.get("symbolic_complementary_two_rule_local_lbfgs_topk", 32)),
                        local_lbfgs_steps=int(cfg.get("symbolic_complementary_two_rule_local_lbfgs_steps", 35)),
                        family_final_topk=int(cfg.get("symbolic_complementary_two_rule_final_topk", 10)),
                        final_steps=int(cfg.get("symbolic_complementary_two_rule_final_steps", 240)),
                        final_lbfgs_steps=int(cfg.get("symbolic_complementary_two_rule_final_lbfgs", 50)),
                        min_improvement_rel=float(cfg.get("symbolic_complementary_two_rule_min_improvement_rel", 1e-4)),
                        verbose=False,
                    )
                    extras["symbolic_complementary_two_rule_rescue_attempted"] = bool(_two_meta.get("attempted",False))
                    extras["symbolic_complementary_two_rule_rescue_selected"] = bool(_two_meta.get("selected",False))
                    extras["symbolic_complementary_two_rule_rescue_reason"] = str(_two_meta.get("reason","unknown"))
                    extras["symbolic_complementary_two_rule_gate_variable"] = _two_meta.get("gate_variable")
                    extras["symbolic_complementary_two_rule_support_a"] = _two_meta.get("support_a")
                    extras["symbolic_complementary_two_rule_support_b"] = _two_meta.get("support_b")
                    extras["symbolic_complementary_two_rule_operator_a"] = _two_meta.get("operator_a")
                    extras["symbolic_complementary_two_rule_operator_b"] = _two_meta.get("operator_b")
                    extras["symbolic_complementary_two_rule_families_screened"] = _two_meta.get("families_screened")
                    extras["symbolic_complementary_two_rule_families_joint_refined"] = _two_meta.get("families_joint_refined")
                    extras["symbolic_complementary_two_rule_families_lbfgs_refined"] = _two_meta.get("families_lbfgs_refined")
                    extras["symbolic_complementary_two_rule_coarse_families_promoted"] = _two_meta.get("coarse_families_promoted_to_full_refit")
                    extras["symbolic_complementary_two_rule_rescue_seconds"] = float(time.perf_counter()-_t_rescue)
                    if bool(_two_meta.get("selected",False)):
                        symbolic_model=_two_model
                        _portfolio_vmse=float(_two_meta.get("best_validation_mse",_portfolio_vmse))
                        _portfolio_vnrmse=math.sqrt(max(_portfolio_vmse,0.0))/_portfolio_scale
                        extras["symbolic_takeover"] = str(extras.get("symbolic_takeover","symbolic")) + "+two_rule_partition"
                else:
                    extras["symbolic_complementary_two_rule_rescue_attempted"] = False
                    extras["symbolic_complementary_two_rule_rescue_selected"] = False

                _prod_trigger = float(cfg.get("symbolic_product_partition_trigger_nrmse", 1e-5))
                if (
                    extras["symbolic_product_partition_rescue_enabled"]
                    and max_factors >= 3
                    and (_portfolio_vnrmse > _prod_trigger or not math.isfinite(_portfolio_vnrmse))
                ):
                    _t_rescue=time.perf_counter()
                    _prod_model,_prod_meta = product_partition_symbolic_rescue(
                        model, symbolic_model, tx, ty, vx, vy,
                        structure_candidates=_rescue_structures,
                        allowed_supports=_rescue_allowed_supports,
                        library=tuple(model.symbolic_library),
                        branch_family_pool=int(cfg.get("symbolic_product_partition_branch_pool", 72)),
                        pair_beam=int(cfg.get("symbolic_product_partition_pair_beam", 28)),
                        max_samples=int(cfg.get("symbolic_product_partition_max_samples", 384)),
                        max_support_pairs=int(cfg.get("symbolic_product_partition_max_support_pairs", 32)),
                        shallow_steps=int(cfg.get("symbolic_product_partition_shallow_steps", 100)),
                        shallow_lr=float(cfg.get("symbolic_product_partition_shallow_lr", 1e-3)),
                        shallow_lbfgs_steps=int(cfg.get("symbolic_product_partition_shallow_lbfgs", 16)),
                        deep_topk=int(cfg.get("symbolic_product_partition_deep_topk", 8)),
                        deep_steps=int(cfg.get("symbolic_product_partition_deep_steps", 420)),
                        deep_lbfgs_steps=int(cfg.get("symbolic_product_partition_deep_lbfgs", 70)),
                        min_improvement_rel=float(cfg.get("symbolic_product_partition_min_improvement_rel", 1e-4)),
                        verbose=False,
                    )
                    extras["symbolic_product_partition_rescue_attempted"] = bool(_prod_meta.get("attempted",False))
                    extras["symbolic_product_partition_rescue_selected"] = bool(_prod_meta.get("selected",False))
                    extras["symbolic_product_partition_rescue_reason"] = str(_prod_meta.get("reason","unknown"))
                    extras["symbolic_product_partition_gate_variable"] = _prod_meta.get("gate_variable")
                    extras["symbolic_product_partition_support_a"] = _prod_meta.get("support_a")
                    extras["symbolic_product_partition_support_b"] = _prod_meta.get("support_b")
                    extras["symbolic_product_partition_operators_a"] = _prod_meta.get("operators_a")
                    extras["symbolic_product_partition_operators_b"] = _prod_meta.get("operators_b")
                    extras["symbolic_product_partition_families_screened"] = _prod_meta.get("families_screened")
                    extras["symbolic_product_partition_rescue_seconds"] = float(time.perf_counter()-_t_rescue)
                    if bool(_prod_meta.get("selected",False)):
                        symbolic_model=_prod_model
                        _portfolio_vmse=float(_prod_meta.get("best_validation_mse",_portfolio_vmse))
                        _portfolio_vnrmse=math.sqrt(max(_portfolio_vmse,0.0))/_portfolio_scale
                else:
                    extras["symbolic_product_partition_rescue_attempted"] = False
                    extras["symbolic_product_partition_rescue_selected"] = False

                _recursive_trigger = float(cfg.get("symbolic_recursive_partition_trigger_nrmse", 1e-5))
                if (
                    extras["symbolic_recursive_partition_rescue_enabled"]
                    and max_factors >= 3
                    and (_portfolio_vnrmse > _recursive_trigger or not math.isfinite(_portfolio_vnrmse))
                ):
                    _t_rescue=time.perf_counter()
                    _rec_model,_rec_meta = recursive_partition_symbolic_rescue(
                        model, symbolic_model, tx, ty, vx, vy,
                        input_mean=data.input_mean.to(tx),
                        input_std=data.input_std.to(tx),
                        allowed_supports=_rescue_allowed_supports,
                        library=tuple(model.symbolic_library),
                        leaf_topk=int(cfg.get("symbolic_recursive_partition_leaf_topk", 10)),
                        coarse_topk=int(cfg.get("symbolic_recursive_partition_coarse_topk", 16)),
                        max_samples=int(cfg.get("symbolic_recursive_partition_max_samples", 384)),
                        shallow_topk=int(cfg.get("symbolic_recursive_partition_shallow_topk", 8)),
                        shallow_steps=int(cfg.get("symbolic_recursive_partition_shallow_steps", 90)),
                        shallow_lr=float(cfg.get("symbolic_recursive_partition_shallow_lr", 8e-4)),
                        shallow_lbfgs_steps=int(cfg.get("symbolic_recursive_partition_shallow_lbfgs", 14)),
                        deep_topk=int(cfg.get("symbolic_recursive_partition_deep_topk", 3)),
                        deep_steps=int(cfg.get("symbolic_recursive_partition_deep_steps", 500)),
                        deep_lbfgs_steps=int(cfg.get("symbolic_recursive_partition_deep_lbfgs", 100)),
                        min_improvement_rel=float(cfg.get("symbolic_recursive_partition_min_improvement_rel", 1e-4)),
                        verbose=False,
                    )
                    extras["symbolic_recursive_partition_rescue_attempted"] = bool(_rec_meta.get("attempted",False))
                    extras["symbolic_recursive_partition_rescue_selected"] = bool(_rec_meta.get("selected",False))
                    extras["symbolic_recursive_partition_rescue_reason"] = str(_rec_meta.get("reason","unknown"))
                    extras["symbolic_recursive_partition_outer_gate"] = _rec_meta.get("outer_gate")
                    extras["symbolic_recursive_partition_inner_gate"] = _rec_meta.get("inner_gate")
                    extras["symbolic_recursive_partition_leaf_variables"] = _rec_meta.get("leaf_variables")
                    extras["symbolic_recursive_partition_leaf_operators"] = _rec_meta.get("leaf_operators")
                    extras["symbolic_recursive_partition_supports"] = _rec_meta.get("supports")
                    extras["symbolic_recursive_partition_gate_candidates"] = _rec_meta.get("gate_candidates")
                    extras["symbolic_recursive_partition_frozen_gate_factors"] = _rec_meta.get("frozen_gate_factors",[])
                    extras["symbolic_recursive_partition_rescue_seconds"] = float(time.perf_counter()-_t_rescue)
                    if bool(_rec_meta.get("selected",False)):
                        symbolic_model=_rec_model
                        _portfolio_vmse=float(_rec_meta.get("best_validation_mse",_portfolio_vmse))
                        _portfolio_vnrmse=math.sqrt(max(_portfolio_vmse,0.0))/_portfolio_scale
                        extras["symbolic_takeover"] = str(extras.get("symbolic_takeover","symbolic")) + "+recursive_partition"
                else:
                    extras["symbolic_recursive_partition_rescue_attempted"] = False
                    extras["symbolic_recursive_partition_rescue_selected"] = False

            if bool(cfg.get("symbolic_composition_rescue", False)) and symbolic_model is not None:
                if model_name.startswith("sisp"):
                    _comp_supports = None
                else:
                    _comp_supports = [
                        tuple(int(v) for v in c["support"])
                        for c in extras.get("symbolic_effective_support_classes", [])
                    ]
                _comp = depth2_composition_rescue(
                    symbolic_model, model, tx, ty, vx, vy,
                    allowed_supports=_comp_supports,
                    library=tuple(model.symbolic_library),
                    trigger_nrmse=float(cfg.get("symbolic_composition_trigger_nrmse", 3e-3)),
                    min_improvement_rel=float(cfg.get("symbolic_composition_min_improvement_rel", 2e-3)),
                    coarse_global_topk=int(cfg.get("symbolic_composition_coarse_topk", 18)),
                    coarse_family_topk=int(cfg.get("symbolic_composition_family_topk", 4)),
                    max_families=int(cfg.get("symbolic_composition_max_families", 220)),
                    shallow_steps=int(cfg.get("symbolic_composition_shallow_steps", 80)),
                    shallow_lbfgs_steps=int(cfg.get("symbolic_composition_shallow_lbfgs", 20)),
                    deep_topk=int(cfg.get("symbolic_composition_deep_topk", 6)),
                    deep_steps=int(cfg.get("symbolic_composition_deep_steps", 250)),
                    deep_lbfgs_steps=int(cfg.get("symbolic_composition_deep_lbfgs", 80)),
                    correction_topk=int(cfg.get("symbolic_composition_correction_topk", 4)),
                    seed=int(cfg.get("symbolic_seed", seed + 100_003)) + 730_001,
                )
                extras["symbolic_composition_attempted"] = bool(_comp.metadata.get("attempted", False))
                extras["symbolic_composition_selected"] = bool(_comp.selected)
                extras["symbolic_composition_metadata"] = dict(_comp.metadata)
                if _comp.selected:
                    symbolic_model = _comp.model
                    extras["symbolic_takeover"] = str(extras.get("symbolic_takeover", "symbolic")) + "+depth2_composition"

            # Final high-precision constant polish.  Topology/operators/supports
            # are frozen; only symbolic affine constants and the readout move.
            # Keeping this after all discrete rescue selection avoids multiplying
            # float64 cost across search trajectories.
            extras["symbolic_precision_polish_enabled"] = bool(cfg.get("symbolic_precision_polish", True))
            extras["symbolic_precision_polish_selected"] = False
            if extras["symbolic_precision_polish_enabled"] and isinstance(symbolic_model, SumProductKAN):
                try:
                    symbolic_model.eval()
                    _p0_dtype = next(symbolic_model.parameters()).dtype
                    with torch.no_grad():
                        _p0 = float(torch.mean((symbolic_model(vx.to(dtype=_p0_dtype))-vy.to(dtype=_p0_dtype))**2).cpu())
                    _dm = copy.deepcopy(symbolic_model).to(dtype=torch.float64)
                    _dm, _p1 = _fully_symbolic_continuous_refit(
                        _dm, tx.to(dtype=torch.float64), ty.to(dtype=torch.float64),
                        vx.to(dtype=torch.float64), vy.to(dtype=torch.float64),
                        steps=int(cfg.get("symbolic_precision_polish_steps", 220)),
                        lr=float(cfg.get("symbolic_precision_polish_lr", 2e-4)),
                        lbfgs_steps=int(cfg.get("symbolic_precision_polish_lbfgs", 80)),
                        frozen_symbolic_factors=tuple(
                            tuple(int(v) for v in z)
                            for z in extras.get("symbolic_recursive_partition_frozen_gate_factors", [])
                        ),
                        show_progress=False,
                    )
                    extras["symbolic_precision_polish_before_val_mse"] = float(_p0)
                    extras["symbolic_precision_polish_after_val_mse"] = float(_p1)
                    if math.isfinite(float(_p1)) and float(_p1) <= float(_p0) * (1.0 + 1e-10):
                        symbolic_model = _dm
                        extras["symbolic_precision_polish_selected"] = True
                except Exception as _prec_exc:
                    extras["symbolic_precision_polish_error"] = f"{type(_prec_exc).__name__}: {_prec_exc}"

            symbolic_model.eval()
            _sym_dtype = next(symbolic_model.parameters()).dtype if any(True for _ in symbolic_model.parameters()) else qx.dtype
            with torch.no_grad():
                sym_pred = symbolic_model(qx.to(dtype=_sym_dtype))
            sym_metrics = evaluate_predictions(sym_pred, qy, data)
            metrics["symbolic_test_rmse"] = float(sym_metrics.get("test_rmse", float("nan")))
            metrics["symbolic_test_nrmse"] = float(sym_metrics.get("test_nrmse", float("nan")))
            metrics["symbolic_gap_rmse"] = metrics["symbolic_test_rmse"] - metrics.get("test_rmse", float("nan"))
            found = _structure_set(symbolic_model)
            metrics.update(_structure_scores(found, spec.expected_structures))
            if spec.fuzzy_rules:
                metrics.update(fuzzy_rule_recovery_scores(symbolic_model, spec, data))
            extras["symbolic_structures"] = sorted([list(z) for z in found])
            if hasattr(symbolic_model, "active_symbolic_rule_count"):
                extras["active_symbolic_rules"] = int(symbolic_model.active_symbolic_rule_count())
            else:
                extras["active_symbolic_rules"] = int((symbolic_model.hard_rule_choice & symbolic_model.rule_alive_mask).sum().item())
            if isinstance(symbolic_model, SumProductKAN):
                red = rule_contribution_redundancy(symbolic_model, vx.to(dtype=_sym_dtype))
                extras["symbolic_max_abs_rule_corr"] = float(red.get("max_abs_corr", 0.0))
                extras["symbolic_max_span_r2"] = float(red.get("max_span_r2", 0.0))
                extras["symbolic_rule_span_r2"] = red.get("span_r2", {})
            else:
                extras["symbolic_max_abs_rule_corr"] = 0.0
                extras["symbolic_max_span_r2"] = 0.0
                extras["symbolic_rule_span_r2"] = {}
            var_names = data.feature_names or [f"x{i}" for i in range(in_dim)]
            expr = symbolic_model.symbolic_formula(
                variable_names=var_names,
                input_mean=data.input_mean.to(qx), input_std=data.input_std.to(qx),
                digits=8, simplify=False,
            )
            extras["formula"] = str(expr)
            extras["formula_input_space"] = "raw"
            if isinstance(symbolic_model, SumProductKAN):
                extras["spline_factors_remaining"] = int(sum(
                    int(f.get("expert") == "spline")
                    for r in symbolic_model.hard_structure(variable_names=var_names)
                    for f in r.get("factors", [])
                ))
            else:
                extras["spline_factors_remaining"] = 0
        except Exception as exc:
            extras["symbolic_error"] = f"{type(exc).__name__}: {exc}"
        metrics["symbolic_seconds"] = float(time.perf_counter() - st)
    return ModelRun(model_name, metrics, extras, numeric_model=model, symbolic_model=symbolic_model)



def train_rulekan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Spline-basis RuleKAN."""
    return _train_rulekan_family("rulekan", spec, data, seed, cfg, numeric_basis="spline")


def train_rulekan_comp(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """RuleKAN with validation-gated depth-2 symbolic composition rescue."""
    local = dict(cfg)
    local["symbolic_composition_rescue"] = True
    return _train_rulekan_family("rulekan_comp", spec, data, seed, local, numeric_basis="spline")


def train_rulekan_distill(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    local = dict(cfg)
    local["symbolic_takeover"] = "numeric_distill"
    return _train_rulekan_family("rulekan_distill", spec, data, seed, local, numeric_basis="spline")


def train_rulekan_fast_distill(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    local = dict(cfg)
    local["symbolic_takeover"] = "numeric_distill"
    return _train_rulekan_family("rulekan_fast_distill", spec, data, seed, local, numeric_basis="rbf")


def train_rulekan_hybrid(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """RuleKAN with GMP unioned with deterministic hard multistart screening."""
    local = dict(cfg)
    local["symbolic_hybrid_hard_screening"] = True
    local.setdefault("symbolic_hard_screen_top_tuples", 4)
    local.setdefault("symbolic_hard_screen_beam", 48)
    local.setdefault("symbolic_hard_screen_samples", 256)
    local.setdefault("symbolic_hard_screen_start_step", 1)
    local.setdefault("symbolic_hard_screen_start_step", 1)
    local.setdefault("symbolic_hard_screen_matching_steps", 4)
    local.setdefault("symbolic_hard_screen_global_candidates", 2)
    local.setdefault("symbolic_hard_screen_residual_rescue", False)
    return _train_rulekan_family("rulekan_hybrid", spec, data, seed, local, numeric_basis="spline")


def train_rulekan_adaptive(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Learned-support RuleKAN with validation-gated high-recall support rescue."""
    local = dict(cfg)
    local["symbolic_validation_rescue"] = True
    local.setdefault("symbolic_validation_rescue_nrmse", 0.03)
    local.setdefault("symbolic_validation_rescue_force_nrmse", 0.10)
    local.setdefault("symbolic_validation_rescue_numeric_ratio", 1.35)
    local.setdefault("symbolic_validation_rescue_min_improvement_rel", 0.005)
    return _train_rulekan_family("rulekan_adaptive", spec, data, seed, local, numeric_basis="spline")


def train_rulekan_fast_hybrid(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Fast/RBF RuleKAN with the same hybrid symbolic proposal stage."""
    local = dict(cfg)
    local["symbolic_hybrid_hard_screening"] = True
    local.setdefault("symbolic_hard_screen_top_tuples", 4)
    local.setdefault("symbolic_hard_screen_beam", 48)
    local.setdefault("symbolic_hard_screen_samples", 256)
    local.setdefault("symbolic_hard_screen_matching_steps", 4)
    local.setdefault("symbolic_hard_screen_global_candidates", 2)
    local.setdefault("symbolic_hard_screen_residual_rescue", False)
    return _train_rulekan_family("rulekan_fast_hybrid", spec, data, seed, local, numeric_basis="rbf")


def train_rulekan_fast_adaptive(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """RBF counterpart of structure-conditioned adaptive RuleKAN."""
    local = dict(cfg)
    local["symbolic_validation_rescue"] = True
    local.setdefault("symbolic_validation_rescue_nrmse", 0.03)
    local.setdefault("symbolic_validation_rescue_force_nrmse", 0.10)
    local.setdefault("symbolic_validation_rescue_numeric_ratio", 1.35)
    local.setdefault("symbolic_validation_rescue_min_improvement_rel", 0.005)
    return _train_rulekan_family("rulekan_fast_adaptive", spec, data, seed, local, numeric_basis="rbf")


def train_rulekan_fast(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """RuleKAN with FastKAN-style Gaussian RBF numerical experts.

    The rule bank, gates, pruning, symbolic library, GMP/GSR extraction and
    readout are identical to RuleKAN. Only the numerical univariate basis is
    changed from cubic B-splines to Gaussian radial basis functions.
    """
    return _train_rulekan_family("rulekan_fast", spec, data, seed, cfg, numeric_basis="rbf")




def train_sisp(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Structure-Independent Symbolic Pursuit with the same spline precursor."""
    local = dict(cfg)
    local["symbolic_takeover"] = "matching_pursuit"
    local["symbolic_use_compressed_numeric_structures"] = False
    return _train_rulekan_family("sisp", spec, data, seed, local, numeric_basis="spline")


def train_sisp_fast(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """SISP with the same RBF numerical precursor used by RuleKAN-RBF."""
    local = dict(cfg)
    local["symbolic_takeover"] = "matching_pursuit"
    local["symbolic_use_compressed_numeric_structures"] = False
    return _train_rulekan_family("sisp_fast", spec, data, seed, local, numeric_basis="rbf")


def train_sisp_comp(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """SISP with the same validation-gated depth-2 composition rescue."""
    local = dict(cfg)
    local["symbolic_takeover"] = "matching_pursuit"
    local["symbolic_use_compressed_numeric_structures"] = False
    local["symbolic_composition_rescue"] = True
    return _train_rulekan_family("sisp_comp", spec, data, seed, local, numeric_basis="spline")



def _fit_power_base_on_transformed_target(
    numeric_model: SumProductKAN,
    tx: torch.Tensor,
    target_train: torch.Tensor,
    vx: torch.Tensor,
    target_val: torch.Tensor,
    *,
    structure_bank: Sequence[Tuple[int, ...]],
    allowed_supports: Sequence[Sequence[int]],
    cfg: Dict[str, Any],
    seed: int,
) -> torch.nn.Module:
    """Fit one support-licensed symbolic base for a fixed outer integer power."""
    model, _ = learned_support_symbolic_gsr(
        numeric_model, tx, target_train, vx, target_val,
        structure_candidates=structure_bank,
        allowed_supports=allowed_supports,
        max_symbolic_rules=int(cfg.get("power_base_max_rules", min(6, numeric_model.n_rules))),
        min_symbolic_rules=1,
        max_rules_per_structure=int(cfg.get("power_base_max_rules_per_structure", 2)),
        allow_symbolic_self_products=True,
        use_gmp_preselection=True,
        gmp_steps=int(cfg.get("power_base_gmp_steps", 18)),
        gmp_topk=int(cfg.get("power_base_gmp_topk", 4)),
        gmp_unary_topk=int(cfg.get("power_base_gmp_unary_topk", 6)),
        gmp_self_product_topk=int(cfg.get("power_base_gmp_self_topk", 5)),
        gmp_tuple_refine_steps=int(cfg.get("power_base_tuple_steps", 6)),
        interaction_shape_screening=True,
        interaction_shape_topk=int(cfg.get("power_base_interaction_topk", 6)),
        interaction_shape_bins=int(cfg.get("power_base_interaction_bins", 8)),
        interaction_shape_min_rank1=float(cfg.get("power_base_interaction_min_rank1", 0.65)),
        interaction_shape_require_multiple=False,
        hard_proposal_union=True,
        hard_proposal_top_tuples=int(cfg.get("power_base_hard_tuples", 8)),
        hard_proposal_beam_width=int(cfg.get("power_base_hard_beam", 48)),
        hard_proposal_max_samples=int(cfg.get("power_base_hard_samples", 384)),
        hard_proposal_max_per_factor=int(cfg.get("power_base_hard_per_factor", 6)),
        initial_block_pursuit=True,
        initial_block_pool=int(cfg.get("power_base_initial_pool", 24)),
        initial_block_pair_beam=int(cfg.get("power_base_pair_beam", 8)),
        initial_block_refit_steps=int(cfg.get("power_base_pair_steps", 35)),
        initial_block_lbfgs_steps=0,
        initial_block_consolidate_steps=int(cfg.get("power_base_consolidate_steps", 50)),
        initial_block_consolidate_lbfgs_steps=0,
        hybrid_hard_screening=False,
        hard_screen_residual_rescue=True,
        hard_screen_beam_width=int(cfg.get("power_base_hard_beam", 48)),
        hard_screen_top_tuples=int(cfg.get("power_base_hard_tuples", 8)),
        hard_screen_max_samples=int(cfg.get("power_base_hard_samples", 384)),
        beam_width=int(cfg.get("power_base_beam", 4)),
        structure_diverse_beam=True,
        beam_max_per_structure=1,
        joint_scale_refit_each_commit=True,
        trial_steps=int(cfg.get("power_base_trial_steps", 16)),
        commit_refit_steps=int(cfg.get("power_base_commit_steps", 24)),
        commit_lbfgs_steps=0,
        backfit=True,
        backfit_beam_width=2,
        backfit_steps=int(cfg.get("power_base_backfit_steps", 12)),
        backfit_lbfgs_steps=0,
        final_steps=int(cfg.get("power_base_final_steps", 30)),
        final_lbfgs_steps=0,
        residual_operator_rescue=True,
        residual_rescue_gmp_topk=int(cfg.get("power_base_rescue_topk", 5)),
        residual_rescue_gmp_steps=int(cfg.get("power_base_rescue_gmp_steps", 15)),
        residual_rescue_beam_width=3,
        residual_rescue_steps=int(cfg.get("power_base_rescue_steps", 20)),
        residual_rescue_lbfgs_steps=0,
        cleanup_max_seconds=float(cfg.get("power_base_cleanup_seconds", 3.0)),
        cleanup_max_trials=int(cfg.get("power_base_cleanup_trials", 4)),
        symbolic_seed=int(seed),
        verbose=False, show_progress=False,
    )
    if bool(cfg.get("power_composition_rescue", False)):
        licensed = sorted({
            tuple(sorted(set(int(v) for v in z)))
            for z in structure_bank if len(z) > 0
        })
        if not licensed:
            licensed = [tuple(int(v) for v in s) for s in allowed_supports if len(s) > 0]
        comp = depth2_composition_rescue(
            model, numeric_model, tx, target_train, vx, target_val,
            allowed_supports=licensed,
            library=tuple(numeric_model.symbolic_library),
            trigger_nrmse=float(cfg.get("power_composition_trigger_nrmse", cfg.get("symbolic_composition_trigger_nrmse", 3e-3))),
            min_improvement_rel=float(cfg.get("power_composition_min_improvement_rel", cfg.get("symbolic_composition_min_improvement_rel", 2e-3))),
            coarse_global_topk=int(cfg.get("power_composition_coarse_topk", cfg.get("symbolic_composition_coarse_topk", 18))),
            coarse_family_topk=int(cfg.get("power_composition_family_topk", cfg.get("symbolic_composition_family_topk", 4))),
            max_families=int(cfg.get("power_composition_max_families", cfg.get("symbolic_composition_max_families", 220))),
            shallow_steps=int(cfg.get("power_composition_shallow_steps", cfg.get("symbolic_composition_shallow_steps", 80))),
            shallow_lbfgs_steps=int(cfg.get("power_composition_shallow_lbfgs", cfg.get("symbolic_composition_shallow_lbfgs", 20))),
            deep_topk=int(cfg.get("power_composition_deep_topk", cfg.get("symbolic_composition_deep_topk", 6))),
            deep_steps=int(cfg.get("power_composition_deep_steps", cfg.get("symbolic_composition_deep_steps", 250))),
            deep_lbfgs_steps=int(cfg.get("power_composition_deep_lbfgs", cfg.get("symbolic_composition_deep_lbfgs", 80))),
            correction_topk=int(cfg.get("power_composition_correction_topk", cfg.get("symbolic_composition_correction_topk", 4))),
            seed=int(seed) + 910_003,
        )
        selected_model = comp.model if comp.selected else model
        setattr(selected_model, "_power_composition_metadata", dict(comp.metadata))
        return selected_model
    setattr(model, "_power_composition_metadata", {"attempted": False, "selected": False, "reason": "disabled"})
    return model



def _ratio_pilot_feature_matrix(
    numeric_model: SumProductKAN,
    x: torch.Tensor,
    structure_bank: Optional[Sequence[Tuple[int,...]]] = None,
    feature_specs: Optional[Sequence[Dict[str,Any]]] = None,
) -> Tuple[torch.Tensor,List[Dict[str,Any]]]:
    """Build cheap support-aware features for the linearized ratio pilot."""
    numeric_model.eval()
    with torch.no_grad():
        _,det=numeric_model(x,return_details=True)
        C=det["contributions"].detach()
    cols=[]; specs=[]
    if feature_specs is None:
        hs=numeric_model.hard_structure()
        for i in range(C.shape[1]):
            support=[]
            if i<len(hs):
                for f in hs[i].get("factors",[]):
                    if not f.get("identity",False) and f.get("variable_index") is not None:
                        support.append(int(f["variable_index"]))
            cols.append(C[:,i:i+1]);specs.append({"kind":"numeric_rule","index":int(i),"support":sorted(set(support))})
        seen=set()
        for z in structure_bank or ():
            zz=tuple(int(v) for v in z)
            if not zz or zz in seen: continue
            seen.add(zz)
            v=torch.ones((x.shape[0],1),device=x.device,dtype=x.dtype)
            for j in zz: v=v*x[:,j:j+1]
            # Standardize only for conditioning; support/ranking is unchanged.
            mu=v.mean(); sd=v.std().clamp_min(1e-6); v=(v-mu)/sd
            cols.append(v);specs.append({"kind":"monomial","structure":list(zz),"support":sorted(set(zz)),
                                         "mean":float(mu.detach().cpu()),"std":float(sd.detach().cpu())})
    else:
        for sp in feature_specs:
            if sp["kind"]=="numeric_rule":
                cols.append(C[:,int(sp["index"]):int(sp["index"])+1]);specs.append(dict(sp))
            elif sp["kind"]=="monomial":
                zz=tuple(int(v) for v in sp["structure"])
                v=torch.ones((x.shape[0],1),device=x.device,dtype=x.dtype)
                for j in zz: v=v*x[:,j:j+1]
                mu=float(sp.get("mean",0.0)); sd=max(float(sp.get("std",1.0)),1e-6)
                v=(v-mu)/sd
                cols.append(v);specs.append(dict(sp))
            else: raise ValueError(f"unknown ratio-pilot feature kind {sp['kind']!r}")
    if not cols: raise ValueError("ratio pilot has no features")
    return torch.cat(cols,dim=1),specs


def _linearized_ratio_pilot(
    numeric_model: SumProductKAN,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    structure_bank: Optional[Sequence[Tuple[int,...]]] = None,
    ridge: float = 1e-5,
    top_supports: int = 2,
) -> Dict[str, Any]:
    """Cheap linearized pilot for ``A(x)/B(x)`` using support-aware features.

    With denominator intercept fixed to one, ``A/B ~= y`` implies

        a0 + Phi a - y Phi b ~= y.

    ``Phi`` unions numerical RuleKAN contributions with simple monomials from
    the learned-support multiplicity bank. The monomials make support discovery
    robust when a flexible numerical rule hides an interaction shape that is
    poorly aligned with the final analytic denominator.
    """
    Phi,specs=_ratio_pilot_feature_matrix(numeric_model,x,structure_bank)
    yy=y.reshape(y.shape[0],1).to(Phi)
    ones=torch.ones((Phi.shape[0],1),device=Phi.device,dtype=Phi.dtype)
    X=torch.cat([ones,Phi,-yy*Phi],dim=1)
    eye=torch.eye(X.shape[1],device=X.device,dtype=X.dtype)
    beta=torch.linalg.solve(X.T@X+float(ridge)*eye,X.T@yy).reshape(-1)
    M=Phi.shape[1];a0=beta[0];avec=beta[1:1+M];bvec=beta[1+M:1+2*M]
    A=a0+Phi@avec[:,None];B=1.0+Phi@bvec[:,None]
    std=Phi.std(dim=0).clamp_min(1e-8)
    def rank_support(coef):
        agg={}
        for i,sc in enumerate((coef.abs()*std).detach().cpu().tolist()):
            sup=tuple(int(v) for v in specs[i].get("support",[]))
            if not sup: continue
            agg[sup]=agg.get(sup,0.0)+float(sc)
        return [list(k) for k,_ in sorted(agg.items(),key=lambda kv:kv[1],reverse=True)[:max(1,int(top_supports))]]
    return {
        "numerator_target":A.detach(),"denominator_target":B.detach(),
        "numerator_intercept":float(a0.detach().cpu()),
        "numerator_coefficients":avec.detach().cpu().tolist(),
        "denominator_coefficients":bvec.detach().cpu().tolist(),
        "feature_specs":specs,
        "numerator_supports":rank_support(avec),"denominator_supports":rank_support(bvec),
        "denominator_margin":float(B.abs().min().detach().cpu()),
    }


def _apply_ratio_pilot(numeric_model: SumProductKAN, x: torch.Tensor, pilot: Dict[str,Any]) -> Tuple[torch.Tensor,torch.Tensor]:
    Phi,_=_ratio_pilot_feature_matrix(numeric_model,x,feature_specs=pilot["feature_specs"])
    avec=torch.as_tensor(pilot["numerator_coefficients"],device=Phi.device,dtype=Phi.dtype)
    bvec=torch.as_tensor(pilot["denominator_coefficients"],device=Phi.device,dtype=Phi.dtype)
    A=float(pilot["numerator_intercept"])+Phi@avec[:,None]
    B=1.0+Phi@bvec[:,None]
    return A,B


def _bank_for_supports(bank: Sequence[Tuple[int,...]], supports: Sequence[Sequence[int]]) -> List[Tuple[int,...]]:
    allowed={frozenset(int(v) for v in s) for s in supports if len(s)>0}
    out=[tuple(z) for z in bank if frozenset(int(v) for v in z) in allowed]
    return out if out else [tuple(z) for z in bank]


def _linear_fit_columns(H: torch.Tensor, y: torch.Tensor, *, ridge: float = 1e-8) -> torch.Tensor:
    yy=y.reshape(y.shape[0],1).to(H)
    X=torch.cat([torch.ones((H.shape[0],1),device=H.device,dtype=H.dtype),H],dim=1)
    eye=torch.eye(X.shape[1],device=X.device,dtype=X.dtype)
    return torch.linalg.solve(X.T@X+float(ridge)*eye,X.T@yy).reshape(-1)


def _select_power_atoms(
    candidates: Sequence[Dict[str,Any]], tx: torch.Tensor, ty: torch.Tensor,
    vx: torch.Tensor, vy: torch.Tensor, *, max_outer_rules: int = 2,
) -> Tuple[List[int], torch.Tensor, float]:
    """Validation-scored OMP over complete powered-expression atoms."""
    valid=[]; Htr=[]; Hva=[]
    for i,c in enumerate(candidates):
        m=c.get("model")
        if m is None or not math.isfinite(float(c.get("val_rmse",float("inf")))): continue
        m.eval()
        with torch.no_grad():
            a=m(tx).reshape(tx.shape[0],-1).mean(dim=1,keepdim=True)
            b=m(vx).reshape(vx.shape[0],-1).mean(dim=1,keepdim=True)
        if torch.isfinite(a).all() and torch.isfinite(b).all():
            valid.append(i);Htr.append(a);Hva.append(b)
    if not valid: raise RuntimeError("no finite powered atoms for outer pursuit")
    Htr=torch.cat(Htr,dim=1);Hva=torch.cat(Hva,dim=1)
    selected=[]; best_sel=None; best_beta=None; best_val=float("inf")
    remaining=list(range(len(valid)))
    for _ in range(min(max(1,int(max_outer_rules)),len(remaining))):
        round_best=None
        for j in remaining:
            cols=selected+[j]
            beta=_linear_fit_columns(Htr[:,cols],ty)
            pv=beta[0]+Hva[:,cols]@beta[1:,None]
            vmse=float(torch.mean((pv-vy)**2).cpu())
            if math.isfinite(vmse) and (round_best is None or vmse<round_best[0]):
                round_best=(vmse,j,beta)
        if round_best is None: break
        _,j,_=round_best;selected.append(j);remaining.remove(j)
        beta=_linear_fit_columns(Htr[:,selected],ty)
        pv=beta[0]+Hva[:,selected]@beta[1:,None]
        vmse=float(torch.mean((pv-vy)**2).cpu())
        if vmse<best_val:
            best_val=vmse;best_sel=list(selected);best_beta=beta.detach().clone()
    if best_sel is None:
        raise RuntimeError("outer powered-atom pursuit failed")
    return [valid[j] for j in best_sel],best_beta,math.sqrt(max(best_val,0.0))


def _merge_power_atoms(
    candidates: Sequence[Dict[str,Any]], selected: Sequence[int], beta: torch.Tensor,
    *, reciprocal_epsilon: float,
) -> PowerRuleKAN:
    """Merge selected atoms into the full multi-outer-rule PowerRuleKAN form."""
    bases=[];terms=[];scales=[]
    bias=float(beta[0].detach().cpu())
    for coef,cidx in zip(beta[1:].detach().cpu().tolist(),selected):
        atom=candidates[int(cidx)]["model"]
        if isinstance(atom,PowerRuleKAN):
            if len(atom.terms)!=1:
                raise ValueError("outer pursuit expects one-term PowerRuleKAN atoms")
            off=len(bases)
            bases.extend(copy.deepcopy(list(atom.bases)))
            terms.append([(off+int(i),int(p)) for i,p in atom.terms[0]])
            scales.append(float(coef)*float(atom.term_scale[0].detach().cpu()))
            bias += float(coef)*float(atom.bias.detach().cpu().reshape(-1)[0])
        elif isinstance(atom,(SumProductKAN,ComposedRuleKAN)):
            off=len(bases);bases.append(copy.deepcopy(atom));terms.append([(off,1)]);scales.append(float(coef))
        else:
            raise TypeError(f"unsupported powered atom type {type(atom).__name__}")
    return PowerRuleKAN(bases,terms,scales=scales,bias=bias,reciprocal_epsilon=float(reciprocal_epsilon))


def _power_reciprocal_selection_margin(
    model: PowerRuleKAN,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
) -> float:
    """Return the reciprocal safety margin used for model selection.

    The test split is deliberately excluded. Reciprocal-domain admissibility is
    part of candidate selection, so it may depend on training and validation
    coordinates but never on test coordinates.
    """
    return float(model.reciprocal_domain_margin(torch.cat([train_x, val_x], dim=0)))


def _fit_ratio_candidate(
    numeric_model: SumProductKAN, tx: torch.Tensor, ty: torch.Tensor,
    vx: torch.Tensor, vy: torch.Tensor, *,
    structure_bank: Sequence[Tuple[int,...]], allowed_supports: Sequence[Sequence[int]],
    cfg: Dict[str,Any], seed: int,
) -> Tuple[Optional[PowerRuleKAN],Dict[str,Any]]:
    """Linearized-ratio pilot -> hard bases -> cross-multiplied -> quotient polish."""
    diag={"kind":"ratio"}
    try:
        pilot=_linearized_ratio_pilot(
            numeric_model,tx,ty,structure_bank=structure_bank,
            ridge=float(cfg.get("power_ratio_pilot_ridge",1e-5)),
            top_supports=int(cfg.get("power_ratio_pilot_supports",2)),
        )
        diag.update({k:v for k,v in pilot.items() if not k.endswith("_target")})
        num_bank=_bank_for_supports(structure_bank,pilot["numerator_supports"])
        den_bank=_bank_for_supports(structure_bank,pilot["denominator_supports"])
        ratio_cfg=dict(cfg)
        ratio_cfg["power_base_max_rules"]=int(cfg.get("power_ratio_base_max_rules",cfg.get("power_base_max_rules",5)))
        _,Bv_pilot=_apply_ratio_pilot(numeric_model,vx,pilot)
        B=_fit_power_base_on_transformed_target(
            numeric_model,tx,pilot["denominator_target"],vx,Bv_pilot,
            structure_bank=den_bank,allowed_supports=allowed_supports,cfg=ratio_cfg,seed=seed+11,
        )
        harden_symbolic_base(B)
        with torch.no_grad(): At=ty*B(tx); Av=vy*B(vx)
        A=_fit_power_base_on_transformed_target(
            numeric_model,tx,At,vx,Av,structure_bank=num_bank,
            allowed_supports=allowed_supports,cfg=ratio_cfg,seed=seed+23,
        )
        harden_symbolic_base(A)
        diag["denominator_composition"] = dict(getattr(B, "_power_composition_metadata", {}))
        diag["numerator_composition"] = dict(getattr(A, "_power_composition_metadata", {}))
        vr,aa,bb,margin=hard_ratio_polish(
            A,B,tx,ty,vx,vy,
            cross_steps=int(cfg.get("power_ratio_cross_steps",30)),
            quotient_steps=int(cfg.get("power_ratio_quotient_steps",30)),
            lr=float(cfg.get("power_ratio_polish_lr",2e-3)),
            reciprocal_epsilon=float(cfg.get("power_reciprocal_epsilon",1e-8)),
            reciprocal_margin=float(cfg.get("power_reciprocal_margin",2e-2)),
            reciprocal_barrier=float(cfg.get("power_ratio_barrier",2e-2)),
        )
        model=PowerRuleKAN([copy.deepcopy(A),copy.deepcopy(B)],[[(0,1),(1,-1)]],scales=[aa],bias=bb,
                           reciprocal_epsilon=float(cfg.get("power_reciprocal_epsilon",1e-8))).to(tx.device)
        margin_select=_power_reciprocal_selection_margin(model,tx,vx)
        diag.update({"val_rmse":float(vr),"margin":float(margin_select),
                     "margin_scope":"train+validation","cross_multiplied":True,
                     "numerator_bank":[list(z) for z in num_bank],"denominator_bank":[list(z) for z in den_bank]})
        if not math.isfinite(margin_select) or margin_select<float(cfg.get("power_reciprocal_margin",2e-2)):
            diag["rejected"]="reciprocal_margin";return None,diag
        return model,diag
    except Exception as exc:
        diag["error"]=f"{type(exc).__name__}: {exc}";return None,diag


def _train_power_rulekan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any], *, model_name: str) -> ModelRun:
    """Fit the full sum-of-products-of-powered-bases PowerRuleKAN.

    Candidate generation deliberately separates discrete algebra from continuous
    fitting:

    * ordinary RuleKAN is an explicit p=1 candidate;
    * fixed powers use transformed-target initialization ``y**(1/p)`` whenever
      that target is real and safe;
    * transformed bases are hard-discretized before original-target polish;
    * a ratio pilot solves the linearized cross-multiplied relation
      ``A - y B ~= 0`` to propose numerator/denominator supports, followed by
      hard cross-multiplied and quotient polish;
    * validation-scored outer OMP can select several complete powered atoms,
      giving ``b + sum_r a_r prod_t B_rt**p_rt`` rather than silently fixing one
      outer rule.

    Because the ordinary RuleKAN symbolic model is one of the outer atoms, the
    larger hypothesis class need not rediscover the p=1 solution through the
    harder recursive parameterization.
    """
    if data.task_type != "regression":
        raise ValueError(f"{model_name} currently supports regression only")
    t0=time.perf_counter()
    composition_enabled = bool(cfg.get("power_composition_rescue", False)) or model_name.endswith("_comp")
    base_cfg=dict(cfg);base_cfg["symbolic"]=True;base_cfg["symbolic_takeover"]="learned_support_gsr"
    base_cfg["symbolic_validation_rescue"] = bool(cfg.get("power_baseline_validation_rescue", True))
    base_cfg["symbolic_composition_rescue"] = bool(composition_enabled)
    base_cfg["power_composition_rescue"] = bool(composition_enabled)
    # PowerRuleKAN uses RuleKAN as a structural/numerical precursor. The helper
    # reports stage metrics on its ``test_*`` fields, but those metrics do not
    # participate in PowerRuleKAN fitting. Replace those reporting-only fields
    # with validation data so the true benchmark test split remains untouched
    # until the complete powered expression has been selected.
    baseline_data=replace(data,test_x=data.val_x,test_y=data.val_y)
    baseline=_train_rulekan_family("rulekan_comp" if composition_enabled else "rulekan",spec,baseline_data,seed,base_cfg,numeric_basis="spline")
    numeric_model=baseline.numeric_model
    if numeric_model is None:
        raise RuntimeError("PowerRuleKAN requires the ordinary RuleKAN numerical precursor")
    device=str(cfg.get("device","cpu"))
    tx,ty=data.train_x.to(device),data.train_y.to(device)
    vx,vy=data.val_x.to(device),data.val_y.to(device)
    # Canonical support propagation: PowerRuleKAN never reads the stale primary
    # learned-support diagnostics directly.  If validation selected RuleKAN's
    # high-recall structure-conditioned rescue, every powered/ratio branch sees
    # that exact effective bank automatically.
    bank, supports, effective_support_source = _rulekan_effective_supports(baseline)

    reciprocal_eps=float(cfg.get("power_reciprocal_epsilon",1e-8))
    min_margin=float(cfg.get("power_reciprocal_margin",2e-2))
    candidates:List[Dict[str,Any]]=[]

    # Guaranteed p=1 fallback: this is the already-fitted ordinary RuleKAN
    # symbolic model, not a fresh recursive optimization.
    if baseline.symbolic_model is not None:
        fallback=copy.deepcopy(baseline.symbolic_model).to(device); fallback.eval()
        with torch.no_grad():
            bval=float(torch.sqrt(torch.mean((fallback(vx)-vy)**2)).cpu())
        candidates.append({"power":1,"val_rmse":bval,"model":fallback,
                           "kind":"ordinary_rulekan","margin":float("inf")})

    power_values=tuple(int(p) for p in cfg.get("power_values",(-3,-2,-1,1,2,3)))
    for ci,pwr in enumerate(power_values):
        if int(pwr) in (0,1): continue
        zt=inverse_power_target(ty,int(pwr),eps=float(cfg.get("power_target_epsilon",1e-7)))
        zv=inverse_power_target(vy,int(pwr),eps=float(cfg.get("power_target_epsilon",1e-7)))
        if zt is None or zv is None:
            candidates.append({"power":int(pwr),"val_rmse":float("inf"),"model":None,
                               "kind":"transformed_power","rejected":"nonreal_or_unsafe_transformed_target"})
            continue
        try:
            base=_fit_power_base_on_transformed_target(
                numeric_model,tx,zt,vx,zv,structure_bank=bank,allowed_supports=supports,
                cfg=cfg,seed=int(seed)+1009*(ci+1),
            )
            # HARD-BEFORE-POLISH: never let the soft operator mixture drift away
            # from a good transformed-target symbolic solution.
            harden_symbolic_base(base)
            vr,aa,bb=hard_power_polish(
                base,int(pwr),tx,ty,vx,vy,
                steps=int(cfg.get("power_hard_polish_steps",32)),
                lr=float(cfg.get("power_hard_polish_lr",2e-3)),
                reciprocal_epsilon=reciprocal_eps,
                reciprocal_margin=min_margin,
                reciprocal_barrier=float(cfg.get("power_reciprocal_barrier",2e-2)),
            )
            wrapper=PowerRuleKAN([copy.deepcopy(base)],[[(0,int(pwr))]],scales=[aa],bias=bb,
                                 reciprocal_epsilon=reciprocal_eps).to(device)
            wrapper.eval();margin=_power_reciprocal_selection_margin(wrapper,tx,vx)
            if int(pwr)<0 and (not math.isfinite(margin) or margin<min_margin):
                candidates.append({"power":int(pwr),"val_rmse":float("inf"),"model":wrapper,
                                   "kind":"transformed_power","margin":margin,"rejected":"reciprocal_margin"})
                continue
            with torch.no_grad(): vr=float(torch.sqrt(torch.mean((wrapper(vx)-vy)**2)).cpu())
            candidates.append({"power":int(pwr),"val_rmse":vr,"model":wrapper,
                               "kind":"transformed_power","margin":margin,
                               "margin_scope":"train+validation",
                               "hard_before_polish":True,
                               "base_composition":dict(getattr(base, "_power_composition_metadata", {}))})
        except Exception as exc:
            candidates.append({"power":int(pwr),"val_rmse":float("inf"),"model":None,
                               "kind":"transformed_power","error":f"{type(exc).__name__}: {exc}"})

    # Ratio-aware initialization.  This adds the genuinely two-block atom A/B.
    ratio_diag=None
    if bool(cfg.get("power_ratio_enabled",True)):
        ratio_model,ratio_diag=_fit_ratio_candidate(
            numeric_model,tx,ty,vx,vy,structure_bank=bank,allowed_supports=supports,
            cfg=cfg,seed=int(seed)+50021,
        )
        if ratio_model is not None:
            ratio_model.eval()
            with torch.no_grad(): rvr=float(torch.sqrt(torch.mean((ratio_model(vx)-vy)**2)).cpu())
            candidates.append({"power":"ratio","val_rmse":rvr,"model":ratio_model,
                               "kind":"ratio_cross_multiplied","margin":float(ratio_diag.get("margin",float("inf"))),
                               "ratio_pilot":{k:v for k,v in ratio_diag.items() if k not in {"val_rmse"}}})
        else:
            candidates.append({"power":"ratio","val_rmse":float("inf"),"model":None,
                               "kind":"ratio_cross_multiplied",**({"ratio_pilot":ratio_diag} if ratio_diag else {})})

    viable=[c for c in candidates if c.get("model") is not None and math.isfinite(float(c.get("val_rmse",float("inf"))))]
    if not viable:
        raise RuntimeError("PowerRuleKAN produced no finite candidate")

    # Multiple outer powered rules: complete powered atoms are combined by a
    # cheap linear OMP/validation pass.  This makes the benchmark trainer match
    # the model class instead of restricting it to one outer powered rule.
    outer_cap=max(1,int(cfg.get("power_outer_rules",2)))
    selected,beta,outer_val=_select_power_atoms(candidates,tx,ty,vx,vy,max_outer_rules=outer_cap)
    model=_merge_power_atoms(candidates,selected,beta,reciprocal_epsilon=reciprocal_eps).to(device)
    model.eval()
    # The benchmark test split enters PowerRuleKAN only after the powered
    # expression and outer coefficients have been selected on train/validation.
    qx,qy=data.test_x.to(device),data.test_y.to(device)
    with torch.no_grad(): pred=model(qx)
    metrics=evaluate_predictions(pred,qy,data)
    metrics["symbolic_test_rmse"]=float(metrics["test_rmse"])
    metrics["symbolic_test_nrmse"]=float(metrics.get("test_nrmse",float("nan")))
    metrics["symbolic_gap_rmse"]=0.0
    if spec.fuzzy_rules:
        metrics.update(fuzzy_rule_recovery_scores(model, spec, data))
    metrics["numeric_seconds"]=float(baseline.metrics.get("numeric_seconds",0.0))
    metrics["symbolic_seconds"]=float(time.perf_counter()-t0-metrics["numeric_seconds"])
    metrics["train_seconds"]=float(time.perf_counter()-t0)

    var_names=data.feature_names or [f"x{i}" for i in range(int(spec.n_var or tx.shape[1]))]
    formula=str(model.symbolic_formula(variable_names=var_names,input_mean=data.input_mean.to(qx),
                                       input_std=data.input_std.to(qx),digits=8,simplify=False))
    extras=dict(baseline.extras)
    selected_info=[]
    for coeff,cidx in zip(beta[1:].detach().cpu().tolist(),selected):
        c=candidates[int(cidx)]
        selected_info.append({"candidate_index":int(cidx),"outer_coefficient":float(coeff),
                              "kind":str(c.get("kind")),"power":c.get("power"),
                              "candidate_val_rmse":float(c.get("val_rmse",float("inf")))})
    reciprocal_selection_margin=_power_reciprocal_selection_margin(model,tx,vx)
    reciprocal_test_margin=float(model.reciprocal_domain_margin(qx))
    reciprocal_test_valid=bool(not math.isnan(reciprocal_test_margin) and reciprocal_test_margin>=min_margin)
    extras.update({
        "formula":formula,"symbolic_formula":formula,"formula_input_space":"raw",
        "symbolic_takeover":"multi_outer_discrete_expression_power",
        "power_values":list(power_values),
        "power_outer_rules_max":int(outer_cap),
        "power_outer_rules_selected":int(len(selected)),
        "power_selected_atoms":selected_info,
        "power_outer_intercept":float(beta[0].detach().cpu()),
        "power_outer_val_rmse":float(outer_val),
        "power_candidates":[{k:v for k,v in c.items() if k!="model"} for c in candidates],
        "power_ratio_pilot":ratio_diag,
        "power_transformed_target_initialization":True,
        "power_ratio_aware_initialization":bool(cfg.get("power_ratio_enabled",True)),
        "power_linearized_ratio_pilot":bool(cfg.get("power_ratio_enabled",True)),
        "power_hard_before_polish":True,
        "power_ordinary_rulekan_candidate":baseline.symbolic_model is not None,
        "power_composition_rescue":bool(composition_enabled),
        "power_composition_baseline_selected":bool(baseline.extras.get("symbolic_composition_selected", False)),
        "power_composition_candidate_count":int(sum(
            bool((c.get("base_composition") or {}).get("selected", False))
            for c in candidates if isinstance(c, dict)
        )),
        "power_effective_support_source": effective_support_source,
        "power_effective_support_classes": [list(s) for s in supports],
        "power_effective_support_bank": [list(z) for z in bank],
        "chosen_powers":model.chosen_powers(),
        # Candidate admissibility and model selection use train+validation only.
        # The test margin is a post-selection generalization diagnostic.
        "reciprocal_domain_margin":float(reciprocal_selection_margin),
        "reciprocal_domain_margin_train_val":float(reciprocal_selection_margin),
        "reciprocal_domain_margin_test":float(reciprocal_test_margin),
        "reciprocal_domain_test_valid":bool(reciprocal_test_valid),
        "fully_symbolic":True,
    })
    return ModelRun(model_name,metrics,extras,numeric_model=numeric_model,symbolic_model=model)


def train_power_rulekan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    local = dict(cfg)
    local["power_composition_rescue"] = False
    return _train_power_rulekan(spec, data, seed, local, model_name="power_rulekan")


def train_power_rulekan_comp(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """PowerRuleKAN whose p=1, transformed-power and ratio bases may use one licensed depth-2 composition."""
    local = dict(cfg)
    local["power_composition_rescue"] = True
    return _train_power_rulekan(spec, data, seed, local, model_name="power_rulekan_comp")


def train_rulekan_omp_linear(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_omp_linear", spec, data, seed, cfg, numeric_basis="spline", pursuit_mode="omp_linear")

def train_rulekan_omp_nonlinear(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_omp_nonlinear", spec, data, seed, cfg, numeric_basis="spline", pursuit_mode="omp_nonlinear")

def train_rulekan_omp_full(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_omp_full", spec, data, seed, cfg, numeric_basis="spline", pursuit_mode="omp_full")

def train_rulekan_graph(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    local = dict(cfg)
    local.setdefault("graph_redundancy", 2e-5)
    local.setdefault("contribution_correlation_threshold", 0.97)
    return _train_rulekan_family("rulekan_graph", spec, data, seed, local, numeric_basis="spline", pursuit_mode="gsr")

def train_rulekan_fast_omp_linear(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_fast_omp_linear", spec, data, seed, cfg, numeric_basis="rbf", pursuit_mode="omp_linear")


def train_rulekan_fast_omp_nonlinear(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_fast_omp_nonlinear", spec, data, seed, cfg, numeric_basis="rbf", pursuit_mode="omp_nonlinear")

def train_rulekan_fast_omp_full(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    return _train_rulekan_family("rulekan_fast_omp_full", spec, data, seed, cfg, numeric_basis="rbf", pursuit_mode="omp_full")

def train_rulekan_fast_graph(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    local = dict(cfg)
    local.setdefault("graph_redundancy", 2e-5)
    local.setdefault("contribution_correlation_threshold", 0.97)
    return _train_rulekan_family("rulekan_fast_graph", spec, data, seed, local, numeric_basis="rbf", pursuit_mode="gsr")

def _rulekan_ablation(model_name, spec, data, seed, cfg, **overrides):
    """Run a spline RuleKAN ablation from the supplied baseline configuration."""
    local = dict(cfg)
    local.update(overrides)
    return _train_rulekan_family(model_name, spec, data, seed, local, numeric_basis="spline")


def train_rulekan_no_product(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_product", spec, data, seed, cfg, max_factors_override=1)

def train_rulekan_no_pruning(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_pruning", spec, data, seed, cfg, pruning_mode="none")

def train_rulekan_one_shot_prune(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_one_shot_prune", spec, data, seed, cfg, pruning_mode="one_shot")

def train_rulekan_no_gmp(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_gmp", spec, data, seed, cfg, use_gmp_preselection=False)

def train_rulekan_no_backfit(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_backfit", spec, data, seed, cfg, symbolic_backfit=False)

def train_rulekan_no_self_product(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_self_product", spec, data, seed, cfg, allow_symbolic_self_products=False)

def train_rulekan_no_product_no_pruning(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_product_no_pruning", spec, data, seed, cfg, max_factors_override=1, pruning_mode="none")

def train_rulekan_no_product_no_gmp(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_product_no_gmp", spec, data, seed, cfg, max_factors_override=1, use_gmp_preselection=False)

def train_rulekan_no_pruning_no_gmp(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_pruning_no_gmp", spec, data, seed, cfg, pruning_mode="none", use_gmp_preselection=False)

def train_rulekan_no_product_no_pruning_no_gmp(spec, data, seed, cfg):
    return _rulekan_ablation("rulekan_no_product_no_pruning_no_gmp", spec, data, seed, cfg, max_factors_override=1, pruning_mode="none", use_gmp_preselection=False)

def train_vanilla_kan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    device = str(cfg.get("device", "cpu"))
    d = int(data.train_x.shape[1])
    hidden = int(cfg.get("hidden", max(5, min(16, d * 2))))
    model = KAN(
        width=[d, hidden, 1], grid=int(cfg.get("grid", 5)), k=int(cfg.get("k", 3)),
        seed=int(seed), auto_save=False, save_act=False, device=device,
    )
    ds = {
        "train_input": data.train_x.to(device), "train_label": data.train_y.to(device),
        "test_input": data.val_x.to(device), "test_label": data.val_y.to(device),
    }
    t0 = time.perf_counter()
    model.fit(
        ds, optimizer="Adam", steps=int(cfg.get("steps", 300)), lr=float(cfg.get("lr", 2e-3)),
        lamb=0.0, log=max(100000, int(cfg.get("log_every", 100000))),
        validation_data=(data.val_x.to(device), data.val_y.to(device)),
        restore_best=True, validation_check_every=max(5, int(cfg.get("validation_every", 20))),
        lr_schedule="cosine", min_lr=float(cfg.get("min_lr", 1e-5)), grad_clip=1.0,
    )
    seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad(): pred = model(data.test_x.to(device))
    metrics = evaluate_predictions(pred, data.test_y.to(device), data)
    metrics["numeric_seconds"] = float(seconds)
    return ModelRun("vanilla_kan", metrics, {"parameters": _param_count(model)}, numeric_model=model)




def train_anfis(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Compact first-order Takagi-Sugeno ANFIS regression baseline.

    Gaussian premise memberships are trained by gradient descent and affine
    Sugeno consequents are refit by ridge least squares. Train data fit the
    parameters, validation data select the checkpoint, and the test split is
    evaluated only after training has finished.
    """
    if spec.task_type != "regression":
        raise ValueError("ANFIS baseline currently supports regression tasks only")
    device = str(cfg.get("device", "cpu"))
    tx, ty = data.train_x.to(device), data.train_y.to(device)
    vx, vy = data.val_x.to(device), data.val_y.to(device)
    qx, qy = data.test_x.to(device), data.test_y.to(device)
    n_rules = max(1, int(cfg.get("anfis_rules", cfg.get("n_rules", 12))))
    n_rules = min(n_rules, max(1, int(tx.shape[0])))
    model = CompactANFIS(
        int(tx.shape[1]), n_rules,
        min_sigma=float(cfg.get("min_sigma", 0.05)),
        max_sigma=float(cfg.get("max_sigma", 5.0)),
        device=device,
        dtype=tx.dtype,
    )
    model.initialize_from_data(tx, seed=int(seed), n_init=int(cfg.get("kmeans_n_init", 10)))
    result = fit_compact_anfis(
        model, tx, ty, vx, vy,
        epochs=int(cfg.get("epochs", 120)),
        lr=float(cfg.get("lr", 2e-2)),
        ridge=float(cfg.get("ridge", 1e-4)),
        patience=int(cfg.get("patience", 20)),
        min_delta_rel=float(cfg.get("min_delta_rel", 1e-5)),
        premise_l2=float(cfg.get("premise_l2", 1e-5)),
    )
    model = result.model
    with torch.no_grad():
        pred = model(qx)
    metrics = evaluate_predictions(pred, qy, data)
    metrics["numeric_seconds"] = float(result.seconds)
    diag = model.premise_diagnostics(vx)
    extras: Dict[str, Any] = {
        "parameters": _param_count(model),
        "fuzzy_system": "first_order_tsk_anfis",
        "anfis_rules": int(n_rules),
        "anfis_epochs_run": int(result.epochs_run),
        "anfis_best_val_rmse": math.sqrt(max(0.0, float(result.best_val_mse))),
        "anfis_membership": "gaussian_product_normalized",
        "anfis_consequent_order": 1,
        "anfis_hybrid_learning": True,
        **diag,
    }
    return ModelRun("anfis", metrics, extras, numeric_model=model)


def _external_sr_arrays(data: BenchmarkData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite float64 arrays for external symbolic-regression engines."""
    x_train = np.asarray(data.train_x.detach().cpu().numpy(), dtype=np.float64)
    y_train = np.asarray(data.train_y.detach().cpu().reshape(-1).numpy(), dtype=np.float64)
    x_test = np.asarray(data.test_x.detach().cpu().numpy(), dtype=np.float64)
    if not (np.isfinite(x_train).all() and np.isfinite(y_train).all() and np.isfinite(x_test).all()):
        raise ValueError("external symbolic-regression baselines require finite train/test arrays")
    return x_train, y_train, x_test


def _external_feature_names(data: BenchmarkData, d: int) -> list[str]:
    names = list(data.feature_names or [])
    if len(names) != int(d):
        names = [f"x{i}" for i in range(int(d))]
    # Keep names simple and valid for both PySR/SymbolicRegression.jl and Operon.
    return [str(name).replace(" ", "_").replace("-", "_") for name in names]




def _predict_sympy_formula(formula: str, names: Sequence[str], x: np.ndarray) -> np.ndarray:
    """Evaluate a returned symbolic expression on benchmark arrays.

    External engines use slightly different textual conventions.  This parser
    intentionally handles only ordinary analytic expressions; a backend whose
    exported expression cannot be parsed should expose its own ``predict`` API.
    """
    import sympy as _sp
    text = str(formula).strip().replace("^", "**")
    xs = [_sp.Symbol(str(n), real=True) for n in names]
    local = {str(n): xs[i] for i, n in enumerate(names)}
    local.update({
        "abs": _sp.Abs, "Abs": _sp.Abs, "sqrt": _sp.sqrt, "exp": _sp.exp,
        "log": _sp.log, "sin": _sp.sin, "cos": _sp.cos, "tan": _sp.tan,
        "tanh": _sp.tanh, "atan": _sp.atan, "arctan": _sp.atan,
    })
    expr = _sp.sympify(text, locals=local)
    fn = _sp.lambdify(xs, expr, modules="numpy")
    cols = [np.asarray(x[:, i], dtype=np.float64) for i in range(x.shape[1])]
    pred = np.asarray(fn(*cols), dtype=np.float64)
    if pred.ndim == 0:
        pred = np.full(x.shape[0], float(pred), dtype=np.float64)
    pred = np.broadcast_to(pred.reshape(-1), (x.shape[0],)).copy()
    if not np.isfinite(pred).all():
        raise ValueError("symbolic formula produced non-finite predictions")
    return pred


def train_symbolic_kan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Run the authors' official Symbolic-KAN implementation.

    The upstream training/selection/hardening/LBFGS routine is executed
    directly from the vendored ``sfaroughi3/Pub_Symbolic_KANs`` snapshot.
    The benchmark adapter only supplies the already-created train/validation
    arrays and serializes the trained discrete model for common formula scoring.
    """
    if spec.task_type != "regression":
        raise ValueError("Symbolic-KAN baseline currently supports regression tasks only")

    device = str(cfg.get("device", "cpu"))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    t0 = time.perf_counter()
    api, model, fit_diag = fit_official_symbolic_kan(
        data.train_x, data.train_y, data.val_x, data.val_y,
        seed=int(seed),
        device=device,
        repo_root=cfg.get("official_repo_root"),
        source_subdir=str(cfg.get("official_source_subdir", "Exp_reaction_diffusion")),
        config=cfg,
    )
    seconds = time.perf_counter() - t0

    x_test = np.asarray(data.test_x.detach().cpu().numpy(), dtype=np.float64)
    names = _external_feature_names(data, x_test.shape[1])
    formula = official_formula(model, names)

    # Score the authors' hardened/discrete symbolic network with their own
    # evaluator.  This is the closest possible measurement of the official
    # implementation: its numerical primitive code includes stability guards
    # (for example clipped exponentials) that the upstream human-readable
    # equation exporter intentionally omits.
    hard_pred = official_hardened_predict(api, model, data.test_x)
    metrics = evaluate_predictions(hard_pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)

    # Also evaluate the serialized human-readable formula when numerically
    # possible.  This is diagnostic/provenance, not the primary score, because
    # it can differ from the upstream evaluator only at those stability guards.
    export_finite = False
    export_error = None
    try:
        pred_np = _predict_sympy_formula(formula, names, x_test)
        pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
        export_metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
        metrics["exported_formula_test_nrmse"] = float(export_metrics.get("test_nrmse", float("nan")))
        if pred.numel() == hard_pred.numel():
            delta = pred.reshape(-1).double() - hard_pred.reshape(-1).double()
            metrics["symbolic_export_vs_official_rmse"] = float(torch.sqrt(torch.mean(delta.square())).cpu())
        export_finite = True
    except Exception as exc:
        metrics["exported_formula_test_nrmse"] = float("nan")
        metrics["symbolic_export_vs_official_rmse"] = float("nan")
        export_error = f"{type(exc).__name__}: {exc}"

    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "symbolic_kan_official_github",
        "symbolic_kan_reference": "Faroughi et al. 2026, JCP 566:115223",
        "symbolic_kan_official_repo": fit_diag.get("symbolic_kan_upstream_url"),
        "symbolic_kan_official_commit": fit_diag.get("symbolic_kan_upstream_commit"),
        "symbolic_kan_exact_official_code": True,
        "symbolic_kan_training_routine": "train_regression_onehot",
        "symbolic_kan_benchmark_adapter": "dataset injection + faithful formula serialization",
        "symbolic_prediction_source": "official_hardened_evaluator",
        "symbolic_export_finite": bool(export_finite),
        "symbolic_export_error": export_error,
        "parameters": _param_count(model),
        **fit_diag,
    }
    return ModelRun("symbolic_kan", metrics, extras, numeric_model=model, symbolic_model=model)

def train_pse(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Official PSE/PSRN symbolic-regression baseline (Ruan et al., 2026)."""
    if spec.task_type != "regression":
        raise ValueError("PSE baseline currently supports regression tasks only")
    try:
        from psrn import PSRN_Regressor
    except ImportError as exc:
        raise ImportError(
            "PSE baseline requires the official `psrn` package; install "
            "`python -m pip install psrn`."
        ) from exc

    x_train, y_train, x_test = _external_sr_arrays(data)
    names = _external_feature_names(data, x_train.shape[1])
    # PSE's official operator names. The default is deliberately elementary;
    # users can override it per profile when matching a different grammar.
    operators = list(cfg.get("operators", [
        "Add", "Mul", "Sub", "Div", "Identity", "Sin", "Cos", "Exp", "Log", "Tanh",
    ]))
    device_cfg = str(cfg.get("device", "cpu"))
    use_cpu = bool(cfg.get("use_cpu", not device_cfg.startswith("cuda")))
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(device_cfg if torch.cuda.is_available() else "cpu")
    n_inputs = int(cfg.get("n_inputs", max(x_train.shape[1] + int(cfg.get("extra_input_slots", 2)), x_train.shape[1])))
    regressor = PSRN_Regressor(
        variables=names,
        use_const=bool(cfg.get("use_constant", True)),
        n_symbol_layers=int(cfg.get("n_symbol_layers", 3)),
        device=device,
        token_generator_config={
            "base": {"has_const": bool(cfg.get("use_constant", True)), "tokens": operators}
        },
        stage_config={
            "default": {
                "operators": operators,
                "time_limit": int(cfg.get("time_limit", cfg.get("timeout_seconds", 600))),
                "n_psrn_inputs": n_inputs,
                "n_sample_variables": int(cfg.get("n_sample_variables", min(3, x_train.shape[1]))),
            },
            "stages": [{}],
        },
    )
    t0 = time.perf_counter()
    regressor.fit(
        x_train,
        y_train.reshape(-1, 1),
        n_down_sample=int(cfg.get("n_down_sample", min(256, len(x_train)))),
        use_threshold=bool(cfg.get("use_threshold", False)),
        threshold=float(cfg.get("threshold", 1e-12)),
        probe=None,
        prun_const=bool(cfg.get("prun_const", True)),
        prun_ndigit=int(cfg.get("prun_ndigit", 6)),
        top_k=int(cfg.get("top_k", 10)),
    )
    seconds = time.perf_counter() - t0
    table = regressor.display_expr_table(sort_by=str(cfg.get("sort_by", "mse")))
    if table is None or len(table) == 0:
        raise RuntimeError("PSE returned no symbolic expressions")
    best = table[0]
    formula = str(best[0] if isinstance(best, (tuple, list)) else best)
    pred_np = _predict_sympy_formula(formula, names, x_test)
    if not np.isfinite(pred_np).all():
        raise ValueError("PSE returned non-finite predictions")
    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "pse_psrn_official",
        "pse_operators": operators,
        "pse_n_symbol_layers": int(cfg.get("n_symbol_layers", 3)),
        "pse_n_inputs": n_inputs,
        "parameters": 0,
    }
    if isinstance(best, (tuple, list)):
        if len(best) > 1: extras["pse_reward"] = float(best[1])
        if len(best) > 2: extras["pse_loss"] = float(best[2])
        if len(best) > 3: extras["symbolic_model_length"] = float(best[3])
    return ModelRun("pse", metrics, extras)


def train_rils_rols(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Official RILS-ROLS symbolic-regression baseline."""
    if spec.task_type != "regression":
        raise ValueError("RILS-ROLS baseline currently supports regression tasks only")
    try:
        from rils_rols.rils_rols import RILSROLSRegressor
    except ImportError as exc:
        raise ImportError(
            "RILS-ROLS baseline requires `rils-rols`; on Linux install pybind11 first, then `pip install rils-rols`."
        ) from exc
    x_train, y_train, x_test = _external_sr_arrays(data)
    requested: Dict[str, Any] = {
        "max_fit_calls": int(cfg.get("max_fit_calls", 100000)),
        "max_seconds": int(cfg.get("max_seconds", cfg.get("timeout_seconds", 600))),
        "complexity_penalty": float(cfg.get("complexity_penalty", 1e-3)),
        "error_tolerance": float(cfg.get("error_tolerance", 1e-12)),
        "max_complexity": int(cfg.get("max_complexity", 40)),
        "sample_size": float(cfg.get("sample_size", 1.0)),
        "verbose": bool(cfg.get("verbose", False)),
        "random_state": int(seed),
    }
    # The public package has changed constructor options across releases. Pass
    # only parameters exposed by the installed version (unless it accepts
    # arbitrary **kwargs) so one adapter works across those releases.
    sig = inspect.signature(RILSROLSRegressor)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    kwargs = requested if accepts_kwargs else {k: v for k, v in requested.items() if k in sig.parameters}
    model = RILSROLSRegressor(**kwargs)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    seconds = time.perf_counter() - t0
    pred_np = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
    if not np.isfinite(pred_np).all():
        raise ValueError("RILS-ROLS returned non-finite predictions")
    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    if hasattr(model, "model_string"):
        value = getattr(model, "model_string")
        formula = str(value() if callable(value) else value)
    elif hasattr(model, "model_simp"):
        formula = str(getattr(model, "model_simp"))
    else:
        formula = str(getattr(model, "model", ""))
    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "rils_rols_official",
        "parameters": 0,
        "rils_rols_max_fit_calls": int(requested["max_fit_calls"]),
        "rils_rols_max_seconds": int(requested["max_seconds"]),
        "rils_rols_complexity_penalty": float(requested["complexity_penalty"]),
    }
    return ModelRun("rils_rols", metrics, extras)


def _normalize_dso_formula_variables(formula: str, names: Sequence[str]) -> str:
    """Map DSO's one-based x1,x2,... names to this benchmark's feature names."""
    text = str(formula).strip()
    # DSO examples and pretty-printer use x1, x2, ... . If an x0 token is
    # already present, treat the expression as zero-based and leave it alone.
    if re.search(r"\bx0\b", text):
        return text
    for i in reversed(range(len(names))):
        text = re.sub(rf"\bx{i + 1}\b", f"__dso_var_{i}__", text)
    for i, name in enumerate(names):
        text = text.replace(f"__dso_var_{i}__", str(name))
    return text


def train_udsr(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Official unified Deep Symbolic Regression (uDSR) baseline.

    Uses the DSO sklearn interface with the NeurIPS-2022 LINEAR/``poly`` token
    and GP-meld enabled, which are the defining additions of uDSR in the
    authors' public implementation.  The PyTorch refactor is preferred so this
    benchmark does not acquire a TensorFlow dependency.
    """
    if spec.task_type != "regression":
        raise ValueError("uDSR baseline currently supports regression tasks only")
    try:
        import dso
        from dso import DeepSymbolicRegressor
    except ImportError as exc:
        raise ImportError(
            "uDSR requires the official DSO package; install the PyTorch refactor "
            "from https://github.com/dso-org/deep-symbolic-optimization-pytorch "
            "(`pip install 'git+https://github.com/dso-org/deep-symbolic-optimization-pytorch.git#subdirectory=dso'`)."
        ) from exc

    x_train, y_train, x_test = _external_sr_arrays(data)
    names = _external_feature_names(data, x_train.shape[1])
    function_set = [str(v) for v in cfg.get(
        "function_set", ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly"]
    )]
    # Calling the method uDSR is only accurate when the LINEAR/poly token and
    # neural-guided GP meld from the 2022 release are actually enabled.
    if "poly" not in function_set:
        function_set.append("poly")

    dso_cfg: Dict[str, Any] = {
        "experiment": {"seed": int(seed)},
        "task": {
            "task_type": "regression",
            "function_set": function_set,
            "metric": str(cfg.get("metric", "inv_nrmse")),
            "metric_params": list(cfg.get("metric_params", [1.0])),
            "threshold": float(cfg.get("threshold", 1e-12)),
            "protected": bool(cfg.get("protected", False)),
            "poly_optimizer_params": {
                "degree": int(cfg.get("poly_degree", 3)),
                "coef_tol": float(cfg.get("poly_coef_tol", 1e-6)),
                "regressor": str(cfg.get("poly_regressor", "dso_least_squares")),
                "regressor_params": dict(cfg.get("poly_regressor_params", {})),
            },
        },
        "gp_meld": {
            "run_gp_meld": True,
            "population_size": int(cfg.get("gp_population_size", 100)),
            "generations": int(cfg.get("gp_generations", 20)),
            "parallel_eval": bool(cfg.get("gp_parallel_eval", False)),
        },
        "training": {
            "n_samples": int(cfg.get("n_samples", 20000)),
            "batch_size": int(cfg.get("batch_size", 500)),
            "epsilon": float(cfg.get("epsilon", 0.02)),
            "n_cores_batch": int(cfg.get("n_cores_batch", 1)),
        },
        "logging": {
            "save_summary": False,
            "save_all_iterations": False,
        },
    }

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rulekan_udsr_") as td:
        dso_cfg["experiment"]["logdir"] = td
        config_path = f"{td}/udsr.json"
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(dso_cfg, fh)
        # Public API: DeepSymbolicRegressor(<config JSON path>).  A keyword
        # fallback keeps the adapter compatible with small API refactors.
        try:
            model = DeepSymbolicRegressor(config_path)
        except TypeError:
            model = DeepSymbolicRegressor(config=config_path)
        model.fit(x_train, y_train)
        pred_np = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
        program = getattr(model, "program_", None)
        if program is None:
            raise RuntimeError("uDSR returned no best symbolic program")
        pretty = getattr(program, "pretty", None)
        formula = str(pretty() if callable(pretty) else program)
    seconds = time.perf_counter() - t0

    if not np.isfinite(pred_np).all():
        raise ValueError("uDSR returned non-finite predictions")
    formula = _normalize_dso_formula_variables(formula, names)
    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "udsr_dso_official",
        "udsr_repo": "https://github.com/dso-org/deep-symbolic-optimization-pytorch",
        "udsr_function_set": function_set,
        "udsr_poly_degree": int(cfg.get("poly_degree", 3)),
        "udsr_gp_meld": True,
        "udsr_n_samples": int(cfg.get("n_samples", 20000)),
        "dso_version": str(getattr(dso, "__version__", "unknown")),
        "parameters": 0,
    }
    return ModelRun("udsr", metrics, extras, symbolic_model=model)

def _safe_srkan_output_transforms(
    y_train: np.ndarray, requested: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Domain-gate optional SR-KAN target transformations.

    SR-KAN's ``manipulate_output`` transformations are search aids, not
    universally valid preprocessing. In particular, log/sqrt require a
    positive target, squaring loses the sign of a sign-changing target, and
    reciprocal fitting becomes ill-conditioned when observations approach
    zero. An invalid optional transformation must not be allowed to turn an
    otherwise valid identity SR run into SymPy ``zoo``/ComplexInfinity.
    """
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        return [], [str(t) for t in requested]

    lo = float(np.min(y))
    hi = float(np.max(y))
    scale = max(float(np.std(y)), float(np.max(np.abs(y))), 1e-12)
    domain_eps = max(1e-12, 1e-10 * scale)
    # Reciprocal transforms can be formally defined but catastrophically
    # conditioned near zero. A 1% scale margin is deliberately conservative:
    # the identity search remains available and is the fair fallback.
    inv_eps = max(1e-12, 1e-2 * scale)
    min_abs = float(np.min(np.abs(y)))
    strictly_positive = lo > domain_eps
    strictly_negative = hi < -domain_eps
    single_sign = strictly_positive or strictly_negative

    safe: list[str] = []
    dropped: list[str] = []
    for raw in requested:
        t = str(raw).strip().lower()
        ok = True
        if t in {"log", "sqrt"}:
            ok = strictly_positive
        elif t == "square":
            # y -> y^2 is not one-to-one if the observed target changes sign.
            ok = single_sign
        elif t in {"inv", "inverse", "reciprocal"}:
            ok = min_abs > inv_eps
        if ok:
            safe.append(str(raw))
        else:
            dropped.append(str(raw))
    return safe, dropped


def _srkan_invalid_symbolic_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(token in text for token in (
        "complexinfinity", "zoo", "non-finite", "nonfinite",
        "nan", "infinity", "unsupported sympy type",
    ))


def _resolve_srkan_functions(srkan_pkg, requested: Sequence[str]) -> List[str]:
    """Resolve the elementary benchmark ``target_core`` vocabulary for SR-KAN.

    The controlled library contains only elementary atoms.  Four use different
    names in SR-KAN, so aliases are installed in its public function dictionary.
    No compound benchmark-specific shortcuts are injected.
    """
    requested = list(requested)
    if requested != ["target_core"]:
        return requested

    try:
        from srkan.function_libraries.univariate import function_lib as srkan_function_lib
    except Exception as exc:
        raise ImportError(
            "SR-KAN target_core matching requires the official SR-KAN function library"
        ) from exc

    lib = srkan_function_lib.all_expr
    for benchmark_name, native_name in _SRKAN_TARGET_CORE_ALIASES.items():
        if native_name not in lib:
            raise RuntimeError(f"SR-KAN native function {native_name!r} is unavailable")
        lib[benchmark_name] = lib[native_name]

    missing = [name for name in TARGET_CORE_SYMBOLIC_LIBRARY if name not in lib]
    if missing:
        raise RuntimeError(f"SR-KAN target_core mapping is incomplete: {missing}")
    return list(TARGET_CORE_SYMBOLIC_LIBRARY)


def train_srkan(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Official SR-KAN baseline (Bühler & Guillén-Gosálbez, 2026).

    The official project is installed from GitHub.  The unrelated ``srkan``
    package on PyPI is intentionally rejected by checking the public API.
    """
    if spec.task_type != "regression":
        raise ValueError("SR-KAN baseline currently supports regression tasks only")
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        import srkan as srkan_pkg
        regressor = getattr(srkan_pkg, "regressor", None)
        evaluator_cls = getattr(srkan_pkg, "SympyEvaluator", None)
        if regressor is None or evaluator_cls is None:
            raise ImportError(
                "imported package named 'srkan' is not the official symbolic-regression SR-KAN API"
            )
    except Exception as exc:
        raise ImportError(
            "Official SR-KAN baseline requires the Bühler & Guillén-Gosálbez repository, not the "
            "unrelated PyPI package named srkan. Install with "
            "`python -m pip install 'git+https://github.com/marcobuhler/SR-KAN.git'`."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    x_train, y_train, x_test = _external_sr_arrays(data)
    x = jnp.asarray(x_train)
    y = jnp.asarray(y_train).reshape(-1, 1)
    kwargs: Dict[str, Any] = {
        "key": jr.key(int(seed)),
        "result_threshold": float(cfg.get("result_threshold", 1e-6)),
        "simpl_threshold": float(cfg.get("simpl_threshold", 1e-2)),
        "do_rounding": bool(cfg.get("do_rounding", True)),
        "scale_x": bool(cfg.get("scale_x", True)),
        "scale_y": bool(cfg.get("scale_y", True)),
        "unscale": bool(cfg.get("unscale", True)),
        "functions": _resolve_srkan_functions(
            srkan_pkg, cfg.get("functions", ["target_core"])
        ),
        "exclude_functions": list(cfg.get("exclude_functions", [])),
        "brute_force": bool(cfg.get("brute_force", True)),
        "simplifications": bool(cfg.get("simplifications", True)),
        "manipulate_output": list(cfg.get("manipulate_output", ["inv", "square", "sqrt", "log"])),
        "combination_kan_types": cfg.get(
            "combination_kan_types", [["sum", "sum"], ["sum", "mult"], ["mult", "mult"]]
        ),
        "use_adam": bool(cfg.get("use_adam", True)),
        "use_bfgs": bool(cfg.get("use_bfgs", True)),
        "regularization_params": list(cfg.get("regularization_params", [1e-4, 1.0, 2.0, 1.0])),
        "n_grids": list(cfg.get("n_grids", [5, 10])),
        "rand_constants": bool(cfg.get("rand_constants", True)),
        "random_iterations": int(cfg.get("random_iterations", 10)),
        "backward_elim": bool(cfg.get("backward_elim", True)),
        "verbosity": int(cfg.get("verbosity", 0)),
    }
    
    requested_output_transforms = list(kwargs["manipulate_output"])
    safe_output_transforms, dropped_output_transforms = _safe_srkan_output_transforms(
        y_train, requested_output_transforms
    )
    kwargs["manipulate_output"] = safe_output_transforms

    def _fit_and_predict(local_kwargs: Dict[str, Any]):
        local_model = regressor(**local_kwargs)
        local_expression = local_model.fit(x, y)
        local_evaluator = evaluator_cls(local_expression)
        local_pred = np.asarray(
            local_evaluator(jnp.asarray(x_test), None, mse=False), dtype=np.float64
        ).reshape(-1).copy()
        if not np.isfinite(local_pred).all():
            raise ValueError("SR-KAN returned non-finite predictions")
        return local_model, local_expression, local_pred

    t0 = time.perf_counter()
    retried_identity = False
    try:
        model, expression, pred_np = _fit_and_predict(kwargs)
    except Exception as exc:
        # Output manipulations are optional search accelerators.  If one yields
        # a SymPy singularity, retry the identical SR-KAN search with only the
        # untransformed target instead of marking the entire benchmark seed as
        # a method failure.
        if kwargs["manipulate_output"] and _srkan_invalid_symbolic_error(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["manipulate_output"] = []
            model, expression, pred_np = _fit_and_predict(retry_kwargs)
            retried_identity = True
            kwargs = retry_kwargs
        else:
            raise
    seconds = time.perf_counter() - t0
    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    extras: Dict[str, Any] = {
        "formula": str(expression),
        "formula_input_space": "standardized",
        "symbolic_backend": "srkan_official_github",
        "symbolic_backend_version": getattr(srkan_pkg, "__version__", "github"),
        "parameters": 0,
        "srkan_functions": kwargs["functions"],
        "srkan_brute_force": kwargs["brute_force"],
        "srkan_simplifications": kwargs["simplifications"],
        "srkan_output_transforms": kwargs["manipulate_output"],
        "srkan_output_transforms_requested": requested_output_transforms,
        "srkan_output_transforms_dropped_domain": dropped_output_transforms,
        "srkan_output_transform_identity_retry": bool(retried_identity),
        "srkan_combination_kan_types": kwargs["combination_kan_types"],
    }
    return ModelRun("srkan", metrics, extras)


def train_pysr(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Evolutionary symbolic-regression baseline using PySR/SymbolicRegression.jl."""
    if spec.task_type != "regression":
        raise ValueError("PySR baseline currently supports regression tasks only")
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise ImportError(
            "PySR baseline requires pysr==2.2.1; install benchmarks/requirements-benchmark.txt"
        ) from exc

    x_train, y_train, x_test = _external_sr_arrays(data)
    names = _external_feature_names(data, x_train.shape[1])
    binary = list(cfg.get("binary_operators", ["+", "-", "*", "/"]))
    unary = list(cfg.get(
        "unary_operators",
        ["square", "cube", "exp", "sin", "cos", "tanh", "atan", "abs", "sqrt", "log", "inv"],
    ))
    timeout = cfg.get("timeout_seconds", None)
    kwargs: Dict[str, Any] = {
        "niterations": int(cfg.get("niterations", 100)),
        "populations": int(cfg.get("populations", 8)),
        "population_size": int(cfg.get("population_size", 50)),
        "maxsize": int(cfg.get("maxsize", 40)),
        "binary_operators": binary,
        "unary_operators": unary,
        "model_selection": str(cfg.get("model_selection", "best")),
        "random_state": int(seed),
        "deterministic": bool(cfg.get("deterministic", True)),
        "parallelism": str(cfg.get("parallelism", "serial")),
        "verbosity": int(cfg.get("verbosity", 0)),
        "progress": bool(cfg.get("progress", False)),
        "update": bool(cfg.get("update", False)),
        "precision": int(cfg.get("precision", 64)),
    }
    if cfg.get("maxdepth") is not None:
        kwargs["maxdepth"] = int(cfg["maxdepth"])
    if timeout is not None:
        kwargs["timeout_in_seconds"] = float(timeout)

    with tempfile.TemporaryDirectory(prefix="rulekan_pysr_") as td:
        # A temporary run directory keeps repeated benchmark workers isolated.
        kwargs["output_directory"] = td
        model = PySRRegressor(**kwargs)
        t0 = time.perf_counter()
        model.fit(x_train, y_train, variable_names=names)
        seconds = time.perf_counter() - t0
        pred_np = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
        try:
            formula = str(model.sympy())
        except Exception:
            formula = str(getattr(model, "equations_", ""))

    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "pysr",
        "symbolic_backend_version": "2.2.1",
        "parameters": 0,
        "binary_operators": binary,
        "unary_operators": unary,
        "evolutionary_population_size": int(kwargs["population_size"]),
        "evolutionary_populations": int(kwargs["populations"]),
        "evolutionary_iterations": int(kwargs["niterations"]),
    }
    return ModelRun("pysr", metrics, extras)


def train_operon(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    """Genetic-programming symbolic-regression baseline using PyOperon."""
    if spec.task_type != "regression":
        raise ValueError("Operon baseline currently supports regression tasks only")
    try:
        from pyoperon.sklearn import SymbolicRegressor
    except ImportError as exc:
        raise ImportError(
            "Operon baseline requires pyoperon==0.6.1; install benchmarks/requirements-benchmark.txt"
        ) from exc

    x_train, y_train, x_test = _external_sr_arrays(data)
    # PyOperon's evaluator expects column-major input; forcing Fortran order also
    # enforces the column-major layout expected by PyOperon's evaluator.
    x_train = np.asfortranarray(x_train)
    x_test = np.asfortranarray(x_test)
    names = _external_feature_names(data, x_train.shape[1])
    allowed = str(cfg.get(
        "allowed_symbols",
        "add,sub,mul,div,constant,variable,square,exp,sin,cos,tanh,atan,abs,sqrt,log",
    ))
    objectives = list(cfg.get("objectives", ["rmse", "length"]))
    kwargs: Dict[str, Any] = {
        "allowed_symbols": allowed,
        "objectives": objectives,
        "optimizer": str(cfg.get("optimizer", "lm")),
        "optimizer_iterations": int(cfg.get("optimizer_iterations", 5)),
        "max_length": int(cfg.get("max_length", 50)),
        "max_depth": int(cfg.get("max_depth", 12)),
        "population_size": int(cfg.get("population_size", 1000)),
        "generations": int(cfg.get("generations", 250)),
        "max_evaluations": int(cfg.get("max_evaluations", 500_000)),
        "tournament_size": int(cfg.get("tournament_size", 5)),
        "model_selection_criterion": str(cfg.get("model_selection_criterion", "minimum_description_length")),
        "n_threads": int(cfg.get("n_threads", 1)),
        "random_state": int(seed),
    }
    if cfg.get("max_time") is not None:
        kwargs["max_time"] = float(cfg["max_time"])

    model = SymbolicRegressor(**kwargs)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    seconds = time.perf_counter() - t0
    pred_np = np.asarray(model.predict(x_test), dtype=np.float64).reshape(-1)
    try:
        formula = str(model.get_model_string(model.model_, precision=8, names=names))
    except Exception:
        formula = str(getattr(model, "model_", ""))

    pred = torch.as_tensor(pred_np, dtype=data.test_y.dtype).reshape(-1, 1)
    metrics = evaluate_predictions(pred, data.test_y.cpu(), data)
    metrics["symbolic_seconds"] = float(seconds)
    stats = getattr(model, "stats_", {}) or {}
    extras: Dict[str, Any] = {
        "formula": formula,
        "formula_input_space": "standardized",
        "symbolic_backend": "operon",
        "symbolic_backend_version": "0.6.1",
        "parameters": 0,
        "allowed_symbols": allowed,
        "evolutionary_population_size": int(kwargs["population_size"]),
        "evolutionary_generations": int(kwargs["generations"]),
        "evolutionary_max_evaluations": int(kwargs["max_evaluations"]),
    }
    for src, dst in (("model_length", "symbolic_model_length"), ("model_complexity", "symbolic_model_complexity")):
        if src in stats:
            try:
                extras[dst] = float(stats[src])
            except Exception:
                pass
    return ModelRun("operon", metrics, extras)


def train_mlp(spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    device = str(cfg.get("device", "cpu"))
    d = int(data.train_x.shape[1])
    width = cfg.get("width") or [d, 64, 64, 1]
    width = list(width)
    width[0] = d
    width[-1] = 1
    model = MLP(width=width, seed=int(seed), save_act=False, device=device)
    ds = {
        "train_input": data.train_x.to(device), "train_label": data.train_y.to(device),
        "test_input": data.val_x.to(device), "test_label": data.val_y.to(device),
    }
    t0 = time.perf_counter()
    model.fit(
        ds, opt="Adam", steps=int(cfg.get("steps", 500)), lr=float(cfg.get("lr", 2e-3)),
        lamb=0.0, log=max(100000, int(cfg.get("log_every", 100000))),
    )
    seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad(): pred = model(data.test_x.to(device))
    metrics = evaluate_predictions(pred, data.test_y.to(device), data)
    metrics["numeric_seconds"] = float(seconds)
    return ModelRun("mlp", metrics, {"parameters": _param_count(model)}, numeric_model=model)



def _paper_dataset(data: BenchmarkData, device: str) -> Dict[str, torch.Tensor]:
    return {
        "train_input": data.train_x.to(device),
        "train_label": data.train_y.to(device),
        # The paper's KAN fit API uses test_* during training diagnostics.  In
        # this benchmark we use the validation split here and reserve test_* for
        # final held-out evaluation.
        "test_input": data.val_x.to(device),
        "test_label": data.val_y.to(device),
    }


def _paper_formula(model: torch.nn.Module) -> str:
    try:
        formula_list, _ = model.symbolic_formula(simplify=False)
        if isinstance(formula_list, (list, tuple)) and formula_list:
            return str(formula_list[0])
        return str(formula_list)
    except Exception as exc:
        return f"<export_failed: {type(exc).__name__}: {exc}>"


def train_paper_pipeline(
    model_name: str,
    spec: TaskSpec,
    data: BenchmarkData,
    seed: int,
    cfg: Dict[str, Any],
    *,
    deep_late_multiplication: bool = False,
) -> ModelRun:
    """Run a configured MultKAN symbolic-regression pipeline.

    The implementation calls the repository's MultKAN symbolic-regression
    methods directly: baseline symbolic regression, greedy symbolic regression,
    gated symbolic layers, and FastKAN radial numerical atoms.
    """
    pipeline_map = DEEP_MULTKAN_PIPELINES if deep_late_multiplication else PAPER_PIPELINES
    if model_name not in pipeline_map:
        raise KeyError(model_name)
    if data.task_type != "regression":
        raise ValueError(f"{model_name} is a symbolic-regression pipeline; task_type={data.task_type}")

    method = pipeline_map[model_name]
    device = str(cfg.get("device", "cpu"))
    d = int(data.train_x.shape[1])
    width_additive = int(cfg.get("width_additive", 5))
    mult_units = int(cfg.get("mult_units", 2))
    if deep_late_multiplication:
        # First hidden layer is additive only. Multiplication occurs in the next
        # hidden layer, after learned KAN transforms have already mixed the raw
        # inputs into intermediate features. A final KAN layer can then apply a
        # univariate function to those multiplicative features.
        deep_width_1 = int(cfg.get("deep_width_1", width_additive))
        deep_width_2 = int(cfg.get("deep_width_2", width_additive))
        deep_mult_units = int(cfg.get("deep_mult_units", max(1, mult_units)))
        width = [d, [deep_width_1, 0], [deep_width_2, deep_mult_units], 1]
    else:
        width = [d, [width_additive, mult_units], 1]
    tx = data.train_x.to(device)
    vx = data.val_x.to(device)
    qx = data.test_x.to(device)
    qy = data.test_y.to(device)
    ds = _paper_dataset(data, device)

    x_min = float(torch.min(tx).detach().cpu())
    x_max = float(torch.max(tx).detach().cpu())
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min == x_max:
        x_min, x_max = -1.0, 1.0

    lib = [str(name) for name in cfg.get("symbolic_library", PAPER_SYMBOLIC_LIBRARY)]
    mult_arity = int(cfg.get("deep_mult_arity", cfg.get("mult_arity", 2))) if deep_late_multiplication else int(cfg.get("mult_arity", 2))
    kan_kwargs: Dict[str, Any] = dict(
        width=width,
        grid=int(cfg.get("grid", 20)),
        grid_range=[x_min, x_max],
        mult_arity=mult_arity,
        seed=int(seed),
        auto_save=False,
        save_act=True,
        device=device,
    )
    if method == "gated_greedy_matching_pursuit":
        kan_kwargs["atom_names"] = lib
    elif method.startswith("fastkan"):
        kan_kwargs["numeric_atom_configs"] = {
            "radial_bf": {"num_grids": int(cfg.get("grid", 20))}
        }
    else:
        kan_kwargs["numeric_atom_configs"] = {
            "bspline": {"num_grids": int(cfg.get("grid", 20)), "degree": 3}
        }

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = KAN(**kan_kwargs)
    steps = int(cfg.get("steps", 200))
    lr = float(cfg.get("lr", 1e-2))
    reg_metric = str(cfg.get("reg_metric", "edge_backward"))
    lamb = float(cfg.get("lamb", 1e-2))
    prune_iters = int(cfg.get("prune_iters", 3))
    node_th = float(cfg.get("node_th", 0.1))
    edge_th = float(cfg.get("edge_th", 0.0))
    topk_start = int(cfg.get("gate_top_k_start", 10))
    topk_final = int(cfg.get("top_k_gates", 5))

    training_options: Dict[str, Any] = {
        "optimizer": "Adam", "lr": lr, "steps": steps, "reg_metric": reg_metric,
    }
    if method == "gated_greedy_matching_pursuit":
        training_options["gating_entropy"] = float(cfg.get("gating_entropy", 1e-3))
        training_options["gating_l1"] = float(cfg.get("gating_l1", 1e-2))
    else:
        training_options["gating_entropy"] = 0.0
        training_options["gating_l1"] = 0.0

    t0 = time.perf_counter()
    model.fit(ds, **training_options)
    training_options["lamb"] = lamb
    if prune_iters > 0:
        delta = max(0, (topk_start - topk_final) // max(1, prune_iters))
        for i in range(prune_iters):
            topk = max(topk_final, topk_start - (i + 1) * delta)
            model = model.prune(node_th=node_th, edge_th=edge_th, gate_top_k=topk)
            model.fit(ds, **training_options)

    training_options["lamb"] = 0.0
    model.fit(ds, **training_options)
    numeric_seconds = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        numeric_pred = model(qx)
    metrics = evaluate_predictions(numeric_pred, qy, data)
    # Preserve pre-extraction accuracy as a separate field because all five
    # pipelines share a numeric-training stage but differ in symbolic conversion.
    metrics = {f"numeric_{k}": v for k, v in metrics.items()}
    metrics["numeric_seconds"] = float(numeric_seconds)

    st = time.perf_counter()
    if method in ("baseline", "fastkan_baseline"):
        model.baseline_symbolic_regression(
            lib=lib, weight_simple=0.0, r2_threshold=0.0, verbose=0,
        )
        edge_policy = "local_per_edge"
    else:
        symbolic_options = dict(training_options)
        symbolic_options["steps"] = int(cfg.get("symbolic_trial_steps", 100))
        symbolic_options["lamb"] = 0.0
        if method == "gated_greedy_matching_pursuit":
            # The post-GMP refinement policy is configured separately from the
            # ordinary GSR edge policy. `importance_asc` ranks the restricted
            # refinement edges in ascending importance.
            edge_policy = str(cfg.get("gmp_refinement_policy", "importance_asc"))
        else:
            edge_policy = str(cfg.get("edge_policy", "importance_desc"))
        model.greedy_symbolic_regression(
            ds,
            lib=lib,
            top_k_gates=topk_final,
            policy=edge_policy,
            verbose=0,
            **symbolic_options,
        )

    # Paper protocol uses a final non-regularized polish after extraction.
    model.fit(ds, **training_options)
    symbolic_seconds = time.perf_counter() - st
    model.eval()
    with torch.no_grad():
        pred = model(qx)
    final_metrics = evaluate_predictions(pred, qy, data)
    metrics.update(final_metrics)
    metrics["symbolic_seconds"] = float(symbolic_seconds)
    extras: Dict[str, Any] = {
        "parameters": _param_count(model),
        "paper_pipeline": method,
        "paper_operator_library_size": len(lib),
        "paper_operator_library": lib,
        "edge_policy": edge_policy,
        "formula": _paper_formula(model),
        "formula_input_space": "standardized",
        "multkan_depth": int(model.depth),
        "multkan_width": str(model.width),
        "width_additive": width_additive,
        "mult_units": mult_units,
        "multiplication_placement": "late_hidden" if deep_late_multiplication else "first_hidden",
        "mult_arity": mult_arity,
    }
    return ModelRun(model_name, metrics, extras, numeric_model=model, symbolic_model=model)


def train_autosym(spec, data, seed, cfg):
    return train_paper_pipeline("autosym", spec, data, seed, cfg)


def train_fastkan_autosym(spec, data, seed, cfg):
    return train_paper_pipeline("fastkan_autosym", spec, data, seed, cfg)


def train_gsr(spec, data, seed, cfg):
    return train_paper_pipeline("gsr", spec, data, seed, cfg)


def train_fastkan_gsr(spec, data, seed, cfg):
    return train_paper_pipeline("fastkan_gsr", spec, data, seed, cfg)


def train_gmp(spec, data, seed, cfg):
    return train_paper_pipeline("gmp", spec, data, seed, cfg)


def train_multkan_deep_autosym(spec, data, seed, cfg):
    return train_paper_pipeline("multkan_deep_autosym", spec, data, seed, cfg, deep_late_multiplication=True)

def train_fast_multkan_deep_autosym(spec, data, seed, cfg):
    return train_paper_pipeline("fast_multkan_deep_autosym", spec, data, seed, cfg, deep_late_multiplication=True)

def train_multkan_deep_gsr(spec, data, seed, cfg):
    return train_paper_pipeline("multkan_deep_gsr", spec, data, seed, cfg, deep_late_multiplication=True)

def train_fast_multkan_deep_gsr(spec, data, seed, cfg):
    return train_paper_pipeline("fast_multkan_deep_gsr", spec, data, seed, cfg, deep_late_multiplication=True)

def train_multkan_deep_gmp(spec, data, seed, cfg):
    return train_paper_pipeline("multkan_deep_gmp", spec, data, seed, cfg, deep_late_multiplication=True)


TRAINERS = {
    "rulekan": train_rulekan,
    "rulekan_comp": train_rulekan_comp,
    "sisp": train_sisp,
    "sisp_fast": train_sisp_fast,
    "sisp_comp": train_sisp_comp,
    "power_rulekan": train_power_rulekan,
    "power_rulekan_comp": train_power_rulekan_comp,
    "rulekan_hybrid": train_rulekan_hybrid,
    "rulekan_adaptive": train_rulekan_adaptive,
    "rulekan_distill": train_rulekan_distill,
    "rulekan_fast": train_rulekan_fast,
    "rulekan_fast_hybrid": train_rulekan_fast_hybrid,
    "rulekan_fast_adaptive": train_rulekan_fast_adaptive,
    "rulekan_fast_distill": train_rulekan_fast_distill,
    "rulekan_omp_linear": train_rulekan_omp_linear,
    "rulekan_omp_nonlinear": train_rulekan_omp_nonlinear,
    "rulekan_omp_full": train_rulekan_omp_full,
    "rulekan_graph": train_rulekan_graph,
    "rulekan_fast_omp_linear": train_rulekan_fast_omp_linear,
    "rulekan_fast_omp_nonlinear": train_rulekan_fast_omp_nonlinear,
    "rulekan_fast_omp_full": train_rulekan_fast_omp_full,
    "rulekan_fast_graph": train_rulekan_fast_graph,
    "rulekan_no_product": train_rulekan_no_product,
    "rulekan_no_pruning": train_rulekan_no_pruning,
    "rulekan_one_shot_prune": train_rulekan_one_shot_prune,
    "rulekan_no_gmp": train_rulekan_no_gmp,
    "rulekan_no_backfit": train_rulekan_no_backfit,
    "rulekan_no_self_product": train_rulekan_no_self_product,
    "rulekan_no_product_no_pruning": train_rulekan_no_product_no_pruning,
    "rulekan_no_product_no_gmp": train_rulekan_no_product_no_gmp,
    "rulekan_no_pruning_no_gmp": train_rulekan_no_pruning_no_gmp,
    "rulekan_no_product_no_pruning_no_gmp": train_rulekan_no_product_no_pruning_no_gmp,
    "autosym": train_autosym,
    "fastkan_autosym": train_fastkan_autosym,
    "gsr": train_gsr,
    "fastkan_gsr": train_fastkan_gsr,
    "gmp": train_gmp,
    "multkan_deep_autosym": train_multkan_deep_autosym,
    "fast_multkan_deep_autosym": train_fast_multkan_deep_autosym,
    "multkan_deep_gsr": train_multkan_deep_gsr,
    "fast_multkan_deep_gsr": train_fast_multkan_deep_gsr,
    "multkan_deep_gmp": train_multkan_deep_gmp,
    "srkan": train_srkan,
    "symbolic_kan": train_symbolic_kan,
    "pse": train_pse,
    "rils_rols": train_rils_rols,
    "udsr": train_udsr,
    "pysr": train_pysr,
    "operon": train_operon,
    "anfis": train_anfis,
    "vanilla_kan": train_vanilla_kan,
    "mlp": train_mlp,
}


def model_supports_task(model_name: str, spec: TaskSpec) -> bool:
    # The five paper pipelines are symbolic-regression methods.  RuleKAN, KAN
    # and MLP additionally serve as predictive baselines on classification data.
    if model_name in PAPER_PIPELINES or model_name in DEEP_MULTKAN_PIPELINES or model_name in {"pysr", "operon", "srkan", "symbolic_kan", "pse", "rils_rols", "udsr", "power_rulekan", "power_rulekan_comp", "anfis"}:
        return spec.task_type == "regression"
    return model_name in TRAINERS


def run_model(model_name: str, spec: TaskSpec, data: BenchmarkData, seed: int, cfg: Dict[str, Any]) -> ModelRun:
    if model_name not in TRAINERS:
        raise KeyError(f"unknown model {model_name}; choices={sorted(TRAINERS)}")
    if not model_supports_task(model_name, spec):
        raise ValueError(f"model {model_name} does not support task type {spec.task_type}")
    return TRAINERS[model_name](spec, data, seed, cfg)
