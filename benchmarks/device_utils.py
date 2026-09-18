from __future__ import annotations

import os
from typing import Mapping


def resolve_device(requested: str | None) -> str:
    """Resolve and validate a benchmark device string.

    CPU is the default. ``auto`` prefers CUDA, then Apple MPS, then CPU.
    Explicit accelerator requests fail early when the backend is unavailable.
    """
    req = str(requested or "cpu").strip().lower()
    import torch

    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if req == "cpu":
        return "cpu"
    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            built = bool(mps is not None and mps.is_built())
            raise RuntimeError(
                f"MPS was requested but is unavailable (built={built}). "
                "Use --device cpu or run with an MPS-enabled PyTorch build on Apple Silicon."
            )
        return "mps"
    if req == "cuda" or req.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        if req.startswith("cuda:"):
            try:
                idx = int(req.split(":", 1)[1])
            except Exception as exc:
                raise ValueError(f"invalid CUDA device: {requested!r}") from exc
            if idx < 0 or idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {idx} was requested but only {torch.cuda.device_count()} device(s) are visible."
                )
        return req
    raise ValueError(f"unknown device {requested!r}; expected cpu, cuda, cuda:N, mps, or auto")


def default_cpu_workers(cpu_fraction: float = 0.5, cpu_count: int | None = None) -> int:
    """Concurrent CPU experiments for the requested fraction of logical CPUs."""
    frac = float(cpu_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"cpu_fraction must be in (0, 1], got {cpu_fraction}")
    detected = (getattr(os, "process_cpu_count", lambda: None)() or os.cpu_count() or 1)
    ncpu = max(1, int(cpu_count if cpu_count is not None else detected))
    return max(1, int(ncpu * frac))


def benchmark_workers(device: str, *, workers: int | None = None, cpu_fraction: float = 0.5) -> int:
    """Choose job-level concurrency.

    CPU defaults to one independent experiment per worker on about 50% of the
    logical CPUs. Accelerators default to one experiment at a time to avoid GPU
    memory contention; ``workers`` can still override this explicitly.
    """
    if workers is not None:
        if int(workers) < 1:
            raise ValueError("workers must be >= 1")
        return int(workers)
    return default_cpu_workers(cpu_fraction) if device == "cpu" else 1


def child_process_env(
    device: str,
    *,
    cpu_threads_per_job: int = 1,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment for one benchmark subprocess."""
    env = dict(os.environ if base is None else base)
    if device == "cpu":
        n = max(1, int(cpu_threads_per_job))
        # Keep independent CPU experiments independent instead of allowing each
        # BLAS/PyTorch process to consume the whole machine.
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = str(n)
        env["RULEKAN_TORCH_THREADS"] = str(n)
    elif device == "mps":
        # PyTorch can fall back to CPU for operators not implemented by MPS.
        # This improves compatibility for the symbolic linear-algebra path.
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def configure_torch_threads(device: str, explicit_threads: int | None = None) -> None:
    """Apply the per-process CPU thread cap inside ``run_one``."""
    if device != "cpu":
        return
    raw = explicit_threads
    if raw is None:
        env = os.environ.get("RULEKAN_TORCH_THREADS")
        raw = int(env) if env else None
    if raw is None:
        return
    import torch

    n = max(1, int(raw))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits changing inter-op threads only before parallel work
        # starts. ``run_one`` calls this early, but keep the helper idempotent.
        pass
