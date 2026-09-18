# Hardware and parallelism

## Default execution policy

Benchmark execution defaults to **CPU**. When CPU is selected and `--workers` is not given, the orchestrator runs independent task/model/seed jobs in parallel using approximately 50% of the available logical CPUs:

```text
workers = max(1, floor(os.cpu_count() * 0.5))
```

Each CPU job is capped to one PyTorch/BLAS thread by default. This avoids oversubscription: four concurrent experiment processes should not each start a full-machine BLAS thread pool.

The relevant CLI controls are:

```text
--device cpu
--workers N
--cpu-fraction 0.5
--cpu-threads-per-job 1
--aggregate-every N
```

`--workers` overrides the automatic worker count. `--cpu-fraction` is used only when the resolved device is CPU and `--workers` is omitted. `--aggregate-every 0` means one aggregate refresh per worker wave; this avoids regenerating all tables and PDF figures after every individual subprocess when many jobs finish concurrently.

The shell launcher also defaults to CPU. Set `RULEKAN_DEVICE=cuda`, `RULEKAN_DEVICE=mps`, or pass a later `--device` argument to override it.

## CUDA

CUDA can accelerate the PyTorch-heavy numerical RuleKAN/MultKAN stages and larger batched symbolic refits. Use:

```bash
python -m benchmarks.run_benchmark --profile research --device cuda
```

or, for a specific visible device:

```bash
python -m benchmarks.run_benchmark --profile research --device cuda:0
```

An explicit CUDA request fails before scheduling jobs when the current PyTorch build cannot see CUDA. GPU execution defaults to one experiment subprocess at a time because concurrent jobs usually contend for the same accelerator memory. `--workers` can override this when the user deliberately manages multiple devices or has enough accelerator memory.

GPU speedups are workload-dependent. Numerical spline/RBF training and large dense tensor operations are the best candidates. Symbolic search contains Python control flow, many small candidate fits, scalar host synchronizations and validation decisions; those sections can be latency-bound and may see little speedup from a GPU.

## Apple MPS

Apple Silicon can be selected with:

```bash
python -m benchmarks.run_benchmark --profile research --device mps
```

The runner enables `PYTORCH_ENABLE_MPS_FALLBACK=1` for MPS child processes so unsupported PyTorch operators can execute on CPU. Symbolic proposal random seeds are generated on CPU and transferred to the target device, avoiding MPS-specific `torch.Generator` limitations.

MPS is therefore supported as an acceleration option, but mixed MPS/CPU fallback can reduce or eliminate the speed advantage for symbolic-search-heavy jobs. CPU remains the reproducible default.

## `auto`

`--device auto` chooses CUDA when available, then MPS, then CPU. It is opt-in; the default is still CPU.

## CPU subprocess environment

For CPU jobs the orchestrator sets the following per-child thread limits to `--cpu-threads-per-job`:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
RULEKAN_TORCH_THREADS
```

`benchmarks.run_one` applies `RULEKAN_TORCH_THREADS` with `torch.set_num_threads` and sets one inter-op thread. These limits apply to the independent benchmark subprocess; external systems may still manage their own workers internally according to their model-specific configuration.

## External symbolic-regression baselines

PySR and Operon are CPU-oriented external systems in this benchmark harness. They receive NumPy arrays on CPU and use their native runtimes. Selecting CUDA/MPS for a mixed benchmark therefore does not make every method GPU-backed.

## Timeouts and process cleanup

Each experiment runs in its own process group. On timeout the orchestrator terminates the process group before recording a failed result. This prevents child processes launched by external tools from surviving after the benchmark job has timed out.
