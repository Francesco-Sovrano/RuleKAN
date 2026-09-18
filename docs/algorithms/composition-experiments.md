# Composition-rescue test coverage

The depth-2 composition rescue is exposed through `rulekan_comp`, `sisp_comp`, and `power_rulekan_comp`. Each method retains a separate flat counterpart so that composition depth is an explicit experimental factor.

The automated test suite covers the following behaviors:

- recovery of a same-variable nested target of the form `tanh(sin(x))`;
- rejection of composition rescue when the incumbent flat model already satisfies the validation criterion;
- recovery of a mixed target containing a composed interaction plus a flat correction rule;
- preservation of train/validation/test separation during composition selection;
- formula export and prediction consistency for composed symbolic models.

The `research` profile includes the flat and compositional RuleKAN, SISP, and PowerRuleKAN variants. `composition_ablation` restricts execution to nine nesting-sensitive tasks and eight methods while inheriting the research data, timeout, capacity, and symbolic-library settings.
