from ._version import __version__

from .MultKAN import *
from .utils import *
#torch.use_deterministic_algorithms(True)
from .rule_mask import RuleMaskProduct
from .sum_product_kan import (
    SumProductKAN,
    FastRBFEdgeBank,
    SumProductRegularization,
    SumProductTrainingStage,
    default_sum_product_schedule,
    fit_sum_product_kan,
    prune_numeric_structure_to_stability,
    audit_sumproduct_symbolic_library,
    drop_negligible_symbolic_rules,
    rule_contribution_redundancy,
    numeric_logic_diagnostics,
    compress_numeric_rule_bank,
    orthogonal_rule_scale_refit,
    redundancy_aware_symbolic_prune,
)
from .sum_product_kan import (
    conformal_prune_sum_product_rules,
    conformal_refit_prune_sum_product_rules,
    shortlist_symbolic_edge,
    in_context_symbolic_rule_gsr,
    hard_numeric_plateau_polish,
    hard_symbolic_plateau_polish,
    compact_sumproduct_for_symbolic,
)

from .sum_product_kan import hard_numeric_precision_polish, mandatory_symbolic_matching_pursuit, distill_numeric_structure_to_symbolic, project_constrained_numeric_to_symbolic, capture_numeric_support_evidence, learned_numeric_support_classes, learned_structure_symbolic_bank, learned_support_symbolic_gsr, low_dimensional_symbolic_family_rescue, complementary_two_rule_symbolic_rescue, product_partition_symbolic_rescue, recursive_partition_symbolic_rescue

from .sum_product_kan import symbolic_structure_bank, gmp_symbolic_operator_preselection, scaled_gmp_screening_sizes, resolve_gmp_local_policy


from .power_rulekan import (PowerRuleKAN, inverse_power_target, safe_integer_power, fit_affine_atom,
    harden_symbolic_base, hard_power_polish, hard_ratio_polish)

from .composition_rulekan import Depth2CompositionAtom, ComposedRuleKAN, depth2_composition_rescue
