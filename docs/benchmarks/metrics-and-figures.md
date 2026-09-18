# Metrics and figures

## Predictive regression metrics

Regression evaluation reports RMSE and normalized RMSE. The benchmark stores numerical-stage and symbolic-stage values separately where a method has both stages. Aggregate code derives common `numeric_rmse`, `numeric_nrmse`, `symbolic_rmse` and `symbolic_nrmse` columns across method families.

For real regression tasks, target standardization is fitted on the training split; reported prediction metrics are computed on the standardized benchmark target representation stored in `BenchmarkData`.

## Binary metrics

Binary tasks use the prediction-evaluation path in `benchmarks/models.py`, which reports classification metrics including accuracy, F1 and ROC AUC where defined.

## Support recovery

For synthetic tasks with declared `expected_structures`, the benchmark compares sets of variable-multiplicity tuples and reports:

```text
structure_precision
structure_recall
structure_f1
```

This is distinct from fuzzy-rule recovery. Ordinary structure metrics ask whether the expected variable structures were recovered, not whether a gate has the correct orientation or a branch operator has the correct semantic role.

## Fuzzy-rule recovery

Fuzzy tasks define expected rules as factor signatures. Each factor specifies:

- variable;
- operator family;
- role: `gate` or `branch`;
- gate orientation for membership `x` or complement `1-x`.

Expected nested fuzzy trees are scored in their exact expanded sum-product/DNF form.

A learned gate must be an identity-family factor on the correct variable. Its affine chart is evaluated on the held-out standardized coordinates and compared with the corresponding raw membership `x` or complement `1-x`; an optimal scalar is allowed when measuring gauge-equivalent gate shape. Branch factors are matched by variable and operator family. `sin` and `cos` are treated as phase-equivalent families.

Maximum bipartite matching prevents one learned rule or factor from receiving credit for several expected rules.

Reported metrics are:

```text
fuzzy_expected_rules
fuzzy_found_rules
fuzzy_rule_precision
fuzzy_rule_recall
fuzzy_rule_f1
fuzzy_gate_precision
fuzzy_gate_recall
fuzzy_branch_precision
fuzzy_branch_recall
fuzzy_exact_structure_recovery
fuzzy_rule_count_error
fuzzy_rule_count_abs_error
fuzzy_gate_rmse_median
fuzzy_gate_rmse_max
fuzzy_gate_nrmse_median
fuzzy_gate_nrmse_max
```

The gate RMSE statistics use observed held-out test coordinates; the scorer does not synthesize off-domain membership points.

## Fuzzy predictive baseline comparison

`fuzzy_predictive_summary.csv` and `figures/fuzzy_predictive_nrmse.pdf` compare final predictive NRMSE across all completed fuzzy-task methods. Symbolic methods use their final symbolic predictor; numerical fuzzy systems such as ANFIS use their final numerical predictor. This table is separate from exact fuzzy-rule recovery because ANFIS and RuleKAN use different rule grammars.

## PowerRuleKAN fuzzy scoring

Fuzzy recovery describes the **final PowerRuleKAN expression**. An outer term containing exactly one RuleKAN base at power `+1` is expanded into that base's additive product rules and scored normally. A genuine powered term, reciprocal, ratio or product of several bases is represented as an unmatched opaque rule for fuzzy scoring.

This prevents a PowerRuleKAN model from receiving structural credit from an unused or structurally simpler `p=1` fallback when its selected outer expression is not itself an expanded fuzzy rule system.

## Formula export

RuleKAN formulas are exported by `SumProductKAN.symbolic_formula`. PowerRuleKAN recursively exports base formulas and integer powers. Input normalization can be supplied to formula export so affine charts are written in original input coordinates.

## Symbolic error figures

Symbolic RMSE/NRMSE figures use the final symbolic predictor for every method. RuleKAN-family rows read explicit `symbolic_test_*` fields. Methods whose final predictor is already symbolic, including AutoSym/GSR/GMP/PySR/Operon, use their final test error as the symbolic value. Numerical-only metrics are not substituted into symbolic figures.

Box/point figures aggregate seeds using the conventions stated in the generated figure footer: boxes represent Q1–Q3, the center/filled marker is the median, whiskers span min–max and open points show individual seeds.

## Reciprocal-domain diagnostics

For PowerRuleKAN, `reciprocal_domain_margin` and `reciprocal_domain_margin_train_val` report the minimum absolute negative-power base value on training plus validation inputs, which is the margin used for candidate admissibility. `reciprocal_domain_margin_test` reports the corresponding value on test inputs after model selection, and `reciprocal_domain_test_valid` indicates whether it clears the configured reciprocal margin. The test diagnostic does not participate in model selection.



## Statistical comparisons

Aggregation writes task-level inferential comparisons for `final_nrmse` and `symbolic_nrmse`. Seeds are first collapsed to the median within each task, so the benchmark task is the statistical unit rather than the individual seed. Pairwise method comparisons use two-sided Wilcoxon signed-rank tests on paired `log10(NRMSE)` values across common tasks, followed by Holm family-wise correction.

The omnibus comparison uses a Friedman test on the common-task panel of methods with at least 75% of the maximum task coverage. The report includes Kendall's W and average task ranks. Generated files include:

```text
statistical_final_nrmse_task_medians.csv
statistical_final_nrmse_pairwise.csv
statistical_final_nrmse_friedman.csv
statistical_final_nrmse_ranks.csv
statistical_symbolic_nrmse_task_medians.csv
statistical_symbolic_nrmse_pairwise.csv
statistical_symbolic_nrmse_friedman.csv
statistical_symbolic_nrmse_ranks.csv
statistical_tests.md
figures/statistical_final_nrmse_ranks.pdf
figures/statistical_symbolic_nrmse_ranks.pdf
```

Holm-adjusted significance is reported at `alpha=0.05`; raw and adjusted p-values are both retained. The rank PDFs mark methods that are significantly worse than the best-ranked method on the corresponding pairwise test.

### Coordinate gauge for formula-level fuzzy recovery

All benchmark learners receive standardized numerical inputs. Native RuleKAN
exports symbolic expressions in the original feature coordinates by applying the
inverse input transform during formula construction. External SR engines store
expressions in the standardized coordinates they were given. Formula-level fuzzy
recovery therefore first substitutes `z_j = (x_j - mean_j) / std_j` and only then
performs semantic DNF/gate matching. Historical synthetic runs reconstruct the
same training-set mean/std deterministically from task, seed, and split sizes.
This prevents a coordinate-system artifact from being counted as a structural
rule-recovery failure.
