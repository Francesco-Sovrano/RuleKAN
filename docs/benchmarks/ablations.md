# Ablations

RuleKAN provides two independent ablation mechanisms: the benchmark component-ablation profile and a standalone one-factor-at-a-time Feynman script.

## Benchmark component ablation

The component ablation is a profile in `benchmarks/configs/default.yaml` and runs through the normal benchmark harness.

```bash
./run_rulekan_benchmark.sh ablation
```

Equivalent direct invocation:

```bash
python -m benchmarks.run_benchmark \
  --config benchmarks/configs/default.yaml \
  --profile ablation \
  --run-dir benchmark_results/current
```

The profile extends `research`. It inherits seeds `0,1,2`, synthetic train/validation/test sizes `1600/400/500`, maximum real-data sample count `5000`, per-job timeout `2400` seconds, fixed shared width `12`, grid `12`, the `target_core` symbolic library, and the research RuleKAN configuration. The ablation profile replaces the method and task selections.

### Methods

The configured method matrix contains 14 variants:

```text
rulekan
rulekan_no_product
rulekan_no_pruning
rulekan_no_gmp
rulekan_no_product_no_pruning
rulekan_no_product_no_gmp
rulekan_no_pruning_no_gmp
rulekan_no_product_no_pruning_no_gmp
rulekan_one_shot_prune
rulekan_no_backfit
rulekan_no_self_product
rulekan_fast
rulekan_omp_linear
rulekan_graph
```

The first eight rows form a `2^3` factorial over product structure, iterative pruning, and GMP. The remaining variants isolate one-shot pruning, symbolic backfitting, repeated-variable symbolic products, Gaussian-RBF numerical edges, linear OMP pursuit, and graph-redundancy handling.

### Tasks

The profile contains 14 tasks:

```text
fuzzy_ite_cross
fuzzy_ite_same_variable
fuzzy_nested_tree
fuzzy_product_branches
same_var_exp_sin
two_same_var_mechanisms
cross_log_cos
mixed_rank4
nested_same_sin_exp
nested_cross_sin_product
pykan_exp_sin_square
pykan_singularity
feynman_kinetic
feynman_gaussian
```

With 14 methods, 14 tasks, and 3 seeds, the full matrix contains 588 scheduled conditions before compatibility filtering.

### Run a subset

Override any part of the matrix with the benchmark runner:

```bash
python -m benchmarks.run_benchmark \
  --profile ablation \
  --models rulekan,rulekan_no_product,rulekan_no_pruning,rulekan_no_gmp \
  --tasks same_var_exp_sin,mixed_rank4 \
  --seeds 0 \
  --run-dir benchmark_results/ablation_subset
```

Use `--device`, `--workers`, and `--cpu-threads-per-job` as with other benchmark profiles.

### Resume existing ablation records

The runner reuses completed records whose build fingerprint matches the active source/configuration fingerprint. To allow completed records with a different fingerprint to be reused:

```bash
./run_rulekan_benchmark.sh ablation --reuse-completed
```

This does not create missing records. Conditions absent from the run directory can still execute.

To rebuild only aggregate tables and figures from existing JSON records:

```bash
python -m benchmarks.aggregate --run-dir benchmark_results/current
```

No training jobs are launched by the aggregate command.

### Component-ablation outputs

When suitable completed symbolic rows exist, aggregation writes:

```text
ablation_pairs.csv
ablation_component_summary.csv
ablation_summary.md
figures/ablation_component_effects.pdf
```

The run directory also contains the normal benchmark outputs such as `runs.csv`, `summary.csv`, `symbolic_runs.csv`, `symbolic_summary.csv`, status files, per-condition JSON, and logs.

## Standalone Feynman OFAT ablation

`tools/ablation_ofat.py` is separate from the configured benchmark profile. It operates on local Feynman text files and compares five MultKAN symbolic-regression paths while changing one hyperparameter at a time.

Display all options:

```bash
python -m tools.ablation_ofat --help
```

### Dataset layout

The default root is:

```text
symbolic_kan/datasets/
```

The selected variant must be one of:

```text
Feynman_without_units
Feynman_with_units
bonus_without_units
bonus_with_units
```

The expected layout is:

```text
symbolic_kan/datasets/
├── FeynmanEquations.csv              optional formula metadata
└── Feynman_with_units/
    ├── I.10.7
    ├── I.12.1
    └── ...
```

Each dataset file is loaded with `numpy.loadtxt`. It must contain numeric columns without a required header. All columns except the last are inputs; the last column is the regression target.

CLI dataset names use the `feynman_` prefix and underscores in place of filename periods. For example:

```text
file:       I.10.7
CLI name:   feynman_I_10_7
```

If `--datasets` is omitted, the script enumerates the selected variant directory, shuffles names with `--dataset_select_seed`, and keeps the first `--max_datasets` entries.

### Default methods

Every OFAT configuration runs these five methods:

```text
baseline
fastkan_baseline
greedy_matching_pursuit
fastkan_greedy_matching_pursuit
gated_greedy_matching_pursuit
```

### Default OFAT center and grids

The baseline center is:

```text
width_mid       5,2
lamb            1e-2
prune_iters     3
seed            1
```

The default grids are:

```text
width_mid       5,2  10,2  20,2  50,2  100,2
lamb            1e-4  1e-3  1e-2  1e-1
prune_iters     1  3  5
seed            1  2  3
```

The script builds 15 OFAT configurations: five width values, four regularization values, three pruning counts, and three seeds. Because the baseline value appears in each factor grid, the baseline condition can occur more than once with different `ofat_factor` labels. Each configuration is evaluated with all five methods, giving 75 method-runs per dataset.

### Basic command

```bash
python -m tools.ablation_ofat \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --equations_csv symbolic_kan/datasets/FeynmanEquations.csv \
  --device cpu \
  --max_datasets 10 \
  --dataset_select_seed 123 \
  --output_csv benchmark_results/ofat_ablation/results.csv
```

The output parent directory is created automatically.

### Run explicit datasets

```bash
python -m tools.ablation_ofat \
  --datasets feynman_I_10_7 feynman_I_12_1 \
  --feynman_root symbolic_kan/datasets \
  --feynman_variant Feynman_with_units \
  --output_csv benchmark_results/ofat_ablation/selected.csv
```

Every explicit name must correspond to a file in the selected variant directory.

### Change the OFAT grids

```bash
python -m tools.ablation_ofat \
  --max_datasets 3 \
  --width_mid_grid 5,2 10,2 20,2 \
  --lamb_grid 0.001 0.01 \
  --prune_iters_grid 1 3 \
  --seed_grid 1 2 \
  --output_csv benchmark_results/ofat_ablation/custom.csv
```

The baseline values can be changed independently:

```text
--baseline_width_mid
--baseline_lamb
--baseline_prune_iters
--baseline_seed
```

Each factor sweep holds the other three values at these baseline settings.

### Sampling

The defaults are:

```text
--train_num 2000
--test_num 1000
--split_strategy random
```

`train_num` and `test_num` are caps. The loader allocates up to `train_num` rows to training first, then up to `test_num` rows from the remaining data. `split_strategy=random` uses a seeded permutation. `split_strategy=linspace` selects evenly spaced row indices.

### Training and pruning controls

The principal fixed controls are:

```text
--grid 20
--lr 0.01
--steps 200
--reg_metric edge_backward
--node_th 0.1
--edge_th 0.0
--gate_top_k_start 10
--top_k_gates 5
--gating_entropy 0.001
--gating_l1 0.01
--regression_policy worst
```

`gating_entropy`, `gating_l1`, and `regression_policy` apply to the gated matching-pursuit path as implemented by the script.

### Device and workers

Supported device values are:

```text
cpu
cuda
mps
auto
```

CPU execution uses process-level parallelism. If `--workers` is omitted, the worker count is derived from `--cpu_fraction`, whose default is `0.5`. `--cpu_threads_per_job` defaults to `1`.

Examples:

```bash
python -m tools.ablation_ofat --device cpu --workers 8 --cpu_threads_per_job 1
python -m tools.ablation_ofat --device cuda --workers 1
python -m tools.ablation_ofat --device mps --workers 1
```

### Output and append behavior

The default output file is:

```text
benchmark_results/ofat_ablation/results.csv
```

By default `--append` is enabled. Each completed or failed method-run is written immediately as one row. Existing CSV content is retained. Start a new file by using a new path or by passing `--no-append`; when `--no-append` is selected, an existing file at the output path is removed before execution starts.

Successful rows include identifiers, OFAT settings, training controls, formula text, train/test MSE, dataset sizes, total wall time, and per-stage timings when `--timing` is enabled. Failed rows retain the condition identifiers and an `error` field.

Formula simplification is disabled by default. Enable it with:

```bash
python -m tools.ablation_ofat --simplify
```
