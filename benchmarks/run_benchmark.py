from __future__ import annotations

import argparse
import importlib
import importlib.util
import csv
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import List

import yaml

from .aggregate import aggregate
from .models import model_supports_task
from .config_utils import resolve_profile
from .build_info import benchmark_build_fingerprint, code_version
from .device_utils import benchmark_workers, child_process_env, resolve_device
from .specs import TASKS, list_suites


def _csv_list(text: str | None) -> List[str] | None:
    if text is None:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _select_tasks(profile: dict, suites_override, tasks_override) -> List[str]:
    if tasks_override:
        names = tasks_override
    elif profile.get("tasks"):
        names = list(profile["tasks"])
    else:
        suites = suites_override or list(profile.get("suites", []))
        names = [name for name, spec in TASKS.items() if spec.suite in suites]
    missing = [x for x in names if x not in TASKS]
    if missing:
        raise KeyError(f"unknown tasks: {missing}")
    # Fuzzy if/then rule recovery is the primary RuleKAN capability test and
    # therefore always runs first when present. Preserve registry order inside
    # each suite instead of alphabetically reordering tasks.
    uniq = list(dict.fromkeys(names))
    return sorted(uniq, key=lambda n: (0 if TASKS[n].suite == "fuzzy_rules" else 1, uniq.index(n)))


def _job_name(task: str, model: str, seed: int, shared_width: int | None = None, shared_library: str | None = None) -> str:
    suffix = "" if shared_width is None else f"__width{int(shared_width)}"
    if shared_library is not None:
        suffix += f"__lib{shared_library}"
    return f"{task}__{model}__seed{seed}{suffix}.json"


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                proc.kill()
    except ProcessLookupError:
        pass


def _run_subprocess_job(
    *,
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    result_path: Path,
    timeout: int,
    failure_payload: dict,
) -> int:
    t0 = time.time()
    try:
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=(os.name != "nt"),
            )
            try:
                return int(proc.wait(timeout=timeout))
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                payload = dict(failure_payload)
                payload.update({
                    "status": "failed",
                    "error": f"timeout after {timeout}s",
                    "elapsed_seconds": time.time() - t0,
                })
                result_path.write_text(json.dumps(payload, indent=2))
                return 124
    except Exception as exc:
        payload = dict(failure_payload)
        payload.update({
            "status": "failed",
            "error": f"orchestrator: {type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - t0,
        })
        result_path.write_text(json.dumps(payload, indent=2))
        return 125


def _write_skipped_manifest(run_dir: Path, skipped_jobs: list[dict]) -> None:
    json_path = run_dir / "skipped_incompatible_jobs.json"
    csv_path = run_dir / "skipped_incompatible_jobs.csv"
    json_path.write_text(json.dumps(skipped_jobs, indent=2))
    columns = ["task", "suite", "model", "seed", "shared_capacity_width", "shared_symbolic_library", "reason"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: row.get(k) for k in columns} for row in skipped_jobs])


OPTIONAL_MODEL_DEPENDENCIES = {
    # PySR import can initialize its Julia bridge, so presence is checked without
    # importing here; the model adapter still reports a precise runtime error.
    "pysr": ("pysr", "pysr==2.2.1", False),
    # Operon's failure mode is commonly a broken/missing binary dependency even
    # when a package spec is present. Import the sklearn wrapper before launch.
    "operon": ("pyoperon.sklearn", "pyoperon==0.6.1", True),
    # Official SR-KAN is currently installed from its authors' GitHub repository.
    # PyPI's package named ``srkan`` is an unrelated spiking-network project.
    "srkan": ("srkan", "git+https://github.com/marcobuhler/SR-KAN.git", True),
}


def _missing_optional_model_dependencies(models: list[str]) -> list[tuple[str, str, str, str]]:
    """Return selected external baselines whose dependency cannot be imported.

    Importing, rather than only checking ``find_spec``, also catches broken
    binary wheels and missing shared-library dependencies before a benchmark
    matrix is launched.
    """
    missing=[]
    for model in sorted(set(models)):
        dep=OPTIONAL_MODEL_DEPENDENCIES.get(model)
        if dep is None:
            continue
        module, requirement, must_import=dep
        try:
            if must_import:
                imported = importlib.import_module(module)
                if model == "srkan" and (
                    not hasattr(imported, "regressor") or not hasattr(imported, "SympyEvaluator")
                ):
                    raise ImportError(
                        "package named srkan is present but it is not the official symbolic-regression SR-KAN API"
                    )
            elif importlib.util.find_spec(module) is None:
                raise ModuleNotFoundError(module)
        except Exception as exc:
            detail=f"{type(exc).__name__}: {exc}"
            missing.append((model,module,requirement,detail))
    return missing


def _validate_optional_model_dependencies(models: list[str]) -> None:
    missing=_missing_optional_model_dependencies(models)
    if not missing:
        return
    detail="; ".join(f"{model} -> {req} ({err})" for model,_,req,err in missing)
    extra = (
        " For SR-KAN use `python -m pip install -r benchmarks/requirements-srkan.txt`; "
        "do not install the unrelated PyPI package named `srkan`."
        if any(model == "srkan" for model, _, _, _ in missing) else ""
    )
    raise RuntimeError(
        "selected external benchmark baselines are not importable: " + detail + ". "
        "Install/repair the standard benchmark dependencies with "
        "`python -m pip install -r benchmarks/requirements-benchmark.txt`." + extra
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="RuleKAN benchmark orchestrator")
    ap.add_argument("--config", default="benchmarks/configs/default.yaml")
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--run-dir", default="benchmark_results/current")
    ap.add_argument("--models", default=None, help="comma-separated override")
    ap.add_argument("--suites", default=None, help="comma-separated override")
    ap.add_argument("--tasks", default=None, help="comma-separated override")
    ap.add_argument("--seeds", default=None, help="comma-separated integer override")
    ap.add_argument(
        "--device", default="cpu",
        help="compute device: cpu (default), cuda, cuda:N, mps, or auto",
    )
    ap.add_argument(
        "--workers", type=int, default=None,
        help="concurrent experiment subprocesses; default is ~50%% of logical CPUs on CPU and 1 on GPU",
    )
    ap.add_argument(
        "--cpu-fraction", type=float, default=0.5,
        help="fraction of logical CPUs used as concurrent CPU experiment workers when --workers is omitted (default: 0.5)",
    )
    ap.add_argument(
        "--cpu-threads-per-job", type=int, default=1,
        help="PyTorch/BLAS threads per CPU experiment subprocess (default: 1)",
    )
    ap.add_argument(
        "--aggregate-every", type=int, default=0,
        help="refresh aggregate tables/PDFs after N finished jobs; 0 means once per worker-wave",
    )
    ap.add_argument("--edge-policy", default=None,
                    choices=["importance_desc", "importance_asc", "random", "ltr", "rtl"],
                    help="override GSR edge-conversion order; paper GSR default is importance_desc")
    ap.add_argument("--gmp-refinement-policy", default=None,
                    choices=["importance_desc", "importance_asc", "random", "ltr", "rtl"],
                    help="override the restricted post-GMP GSR edge order")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--reuse-completed", action=argparse.BooleanOptionalAction, default=False,
        help="with --resume, reuse completed records even when code/configuration changed; opt-in because changes may affect comparability",
    )
    ap.add_argument(
        "--verbose-resume", action=argparse.BooleanOptionalAction, default=False,
        help="print one resume decision per job; default prints only a compact resume summary",
    )
    ap.add_argument(
        "--show-build-fingerprint", action=argparse.BooleanOptionalAction, default=False,
        help="show the internal build fingerprint in console output; it is always retained in result JSON for provenance",
    )
    ap.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    root_dir = Path(__file__).resolve().parents[1]
    build_fp = benchmark_build_fingerprint(root_dir, config_path)
    build_version = code_version(root_dir)
    cfg = yaml.safe_load(config_path.read_text())
    profile = resolve_profile(cfg, args.profile)
    if args.list:
        print(json.dumps({"suites": list_suites(), "models": profile.get("models"), "profiles": list(cfg["profiles"])}, indent=2))
        return 0

    try:
        device = resolve_device(args.device)
        workers = benchmark_workers(device, workers=args.workers, cpu_fraction=args.cpu_fraction)
    except (ValueError, RuntimeError) as exc:
        ap.error(str(exc))
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be >= 1")
    aggregate_every = int(args.aggregate_every)
    if aggregate_every < 0:
        raise ValueError("--aggregate-every must be >= 0")
    if aggregate_every == 0:
        aggregate_every = max(1, workers)

    models = _csv_list(args.models) or list(profile.get("models", []))
    try:
        _validate_optional_model_dependencies(models)
    except RuntimeError as exc:
        ap.error(str(exc))
    suites = _csv_list(args.suites)
    tasks = _select_tasks(profile, suites, _csv_list(args.tasks))
    seeds = [int(x) for x in (_csv_list(args.seeds) or profile.get("seeds", [0]))]
    width_values = profile.get("width_values")
    widths = [None] if not width_values else [int(x) for x in width_values]
    library_values = profile.get("library_values")
    libraries = [None] if not library_values else [str(x) for x in library_values]
    run_dir = Path(args.run_dir).resolve()
    runs_dir = run_dir / "runs"
    logs_dir = run_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "benchmark_config_snapshot.yaml").write_text(yaml.safe_dump({"profile": args.profile, "config": profile}, sort_keys=False))

    skipped_jobs = [
        {
            "task": task,
            "suite": TASKS[task].suite,
            "model": model,
            "seed": seed,
            "shared_capacity_width": width,
            "shared_symbolic_library": library,
            "reason": f"model {model!r} does not support task type {TASKS[task].task_type!r}",
        }
        for task in tasks for model in models for seed in seeds for width in widths for library in libraries
        if not model_supports_task(model, TASKS[task])
    ]
    _write_skipped_manifest(run_dir, skipped_jobs)

    jobs = [
        (task, model, seed, width, library)
        for task in tasks for model in models for seed in seeds for width in widths for library in libraries
        if model_supports_task(model, TASKS[task])
    ]
    print(f"[benchmark] profile={args.profile} jobs={len(jobs)} tasks={len(tasks)} models={models} seeds={seeds} widths={widths} libraries={libraries}")
    if args.show_build_fingerprint:
        print(f"[benchmark] code_version={build_version} build={build_fp}")
    else:
        print(f"[benchmark] code_version={build_version}")
    print(f"[benchmark] device={device} workers={workers} cpu_fraction={args.cpu_fraction:g} cpu_threads_per_job={args.cpu_threads_per_job}")
    if device == "mps":
        print("[benchmark] MPS CPU fallback enabled for unsupported PyTorch operators")
    if skipped_jobs:
        print(f"[benchmark] incompatible jobs not scheduled: {len(skipped_jobs)}; details: {run_dir / 'skipped_incompatible_jobs.csv'}")
    print(f"[benchmark] partial tables: {run_dir / 'summary.md'}")
    print(f"[benchmark] partial PDF figures: {run_dir / 'figures'}")
    timeout = int(profile.get("timeout_seconds", 3600))

    child_env = child_process_env(device, cpu_threads_per_job=args.cpu_threads_per_job)
    runnable: list[tuple[int, tuple[str, str, int, int | None, str | None], Path, Path, list[str], dict]] = []
    completed_from_resume = 0
    resumed_same_build = 0
    resumed_changed_build = 0
    rerun_changed_build = 0
    for idx, (task, model, seed, shared_width, shared_library) in enumerate(jobs, 1):
        result_path = runs_dir / _job_name(task, model, seed, shared_width, shared_library)
        if args.resume and result_path.exists():
            try:
                old = json.loads(result_path.read_text())
                if old.get("status") == "completed" and old.get("benchmark_build_fingerprint") == build_fp:
                    resumed_same_build += 1
                    completed_from_resume += 1
                    if args.verbose_resume:
                        print(f"[{idx}/{len(jobs)}] reuse completed {task} / {model} / seed={seed}")
                    continue
                if old.get("status") == "completed" and args.reuse_completed:
                    resumed_changed_build += 1
                    completed_from_resume += 1
                    if args.verbose_resume:
                        print(f"[{idx}/{len(jobs)}] reuse completed {task} / {model} / seed={seed} (code/configuration changed)")
                    continue
                if old.get("status") == "completed":
                    rerun_changed_build += 1
                    if args.verbose_resume:
                        print(f"[{idx}/{len(jobs)}] rerun completed {task} / {model} / seed={seed} (code/configuration changed)")
            except Exception:
                pass
        log_path = logs_dir / result_path.with_suffix(".log").name
        cmd = [
            sys.executable, "-m", "benchmarks.run_one",
            "--config", str(config_path), "--profile", args.profile,
            "--task", task, "--model", model, "--seed", str(seed),
            "--output", str(result_path), "--device", device,
        ]
        if device == "cpu":
            cmd += ["--cpu-threads", str(args.cpu_threads_per_job)]
        if args.edge_policy:
            cmd += ["--edge-policy", args.edge_policy]
        if args.gmp_refinement_policy:
            cmd += ["--gmp-refinement-policy", args.gmp_refinement_policy]
        if shared_width is not None:
            cmd += ["--shared-width", str(shared_width)]
        if shared_library is not None:
            cmd += ["--shared-library", str(shared_library)]
        failure_payload = {
            "profile": args.profile, "task": task, "suite": TASKS[task].suite,
            "task_type": TASKS[task].task_type, "model": model, "seed": seed,
            "shared_capacity_width": shared_width, "shared_symbolic_library": shared_library,
            "code_version": build_version, "benchmark_build_fingerprint": build_fp,
            "device": device,
        }
        runnable.append((idx, (task, model, seed, shared_width, shared_library), result_path, log_path, cmd, failure_payload))

    if args.resume and (completed_from_resume or rerun_changed_build):
        parts = []
        if resumed_same_build:
            parts.append(f"reused {resumed_same_build} completed")
        if resumed_changed_build:
            parts.append(f"reused {resumed_changed_build} completed despite code/configuration changes")
        if rerun_changed_build:
            parts.append(f"rerunning {rerun_changed_build} completed because code/configuration changed")
        print("[benchmark] resume: " + "; ".join(parts))

    if not runnable:
        aggregate(run_dir)
        print(f"[benchmark] finished. Results: {run_dir}")
        return 0

    print(f"[benchmark] runnable={len(runnable)} resumed={completed_from_resume} aggregate_every={aggregate_every}")
    failures: list[int] = []
    finished_since_aggregate = 0
    next_pos = 0
    stop_submitting = False

    def submit_one(executor: ThreadPoolExecutor, futures: dict) -> bool:
        nonlocal next_pos
        if stop_submitting or next_pos >= len(runnable):
            return False
        item = runnable[next_pos]
        next_pos += 1
        idx, job, result_path, log_path, cmd, failure_payload = item
        task, model, seed, shared_width, shared_library = job
        print(f"[{idx}/{len(jobs)}] run {task} / {model} / seed={seed} / width={shared_width} / library={shared_library}")
        fut = executor.submit(
            _run_subprocess_job,
            cmd=cmd, env=child_env, log_path=log_path, result_path=result_path,
            timeout=timeout, failure_payload=failure_payload,
        )
        futures[fut] = item
        return True

    progress_started = time.monotonic()
    heartbeat_seconds = 30.0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rulekan-job") as executor:
        futures: dict = {}
        for _ in range(min(workers, len(runnable))):
            submit_one(executor, futures)
        while futures:
            done, _ = wait(tuple(futures), timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                elapsed = time.monotonic() - progress_started
                queued = max(0, len(runnable) - next_pos)
                print(
                    f"[benchmark] still running: active={len(futures)} queued={queued} "
                    f"elapsed={elapsed:.0f}s; per-job logs: {run_dir / 'logs'}",
                    flush=True,
                )
                continue
            for fut in done:
                item = futures.pop(fut)
                idx, job, result_path, log_path, cmd, failure_payload = item
                task, model, seed, shared_width, shared_library = job
                try:
                    code = int(fut.result())
                except Exception as exc:
                    code = 125
                    failure_payload = dict(failure_payload)
                    failure_payload.update({"status": "failed", "error": f"worker: {type(exc).__name__}: {exc}"})
                    result_path.write_text(json.dumps(failure_payload, indent=2))
                finished_since_aggregate += 1
                print(f"[{idx}/{len(jobs)}] exit={code} {task} / {model} / seed={seed}")
                if code not in (0, 2):
                    failures.append(code)
                    if not args.keep_going:
                        stop_submitting = True
                elif code == 2 and not args.keep_going:
                    failures.append(code)
                    stop_submitting = True
                if not stop_submitting:
                    submit_one(executor, futures)
                if finished_since_aggregate >= aggregate_every:
                    aggregate(run_dir)
                    finished_since_aggregate = 0
                    print("[benchmark] refreshed tables/PDFs")

    aggregate(run_dir)
    print(f"[benchmark] finished. Results: {run_dir}")
    if failures and not args.keep_going:
        return failures[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
