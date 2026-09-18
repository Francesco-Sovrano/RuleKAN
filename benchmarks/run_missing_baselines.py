"""Safely append newly added external baselines to an existing benchmark result set.

The source result directory is never modified.  It is copied to a new destination,
only the explicitly selected model IDs are scheduled, and every pre-existing run
JSON is hash-checked after execution.  Completed rows are reused even if the code
fingerprint changed so adding a baseline cannot silently rerun prior experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_hashes(root: Path) -> dict[str, str]:
    runs = root / "runs"
    if not runs.is_dir():
        raise FileNotFoundError(f"missing benchmark runs directory: {runs}")
    return {str(p.relative_to(root)): _sha256(p) for p in sorted(runs.glob("*.json"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-results", required=True, help="existing result directory; never modified")
    ap.add_argument("--dest-results", required=True, help="new append-only working copy")
    ap.add_argument("--config", default="benchmarks/configs/default.yaml")
    ap.add_argument("--profile", default="research")
    ap.add_argument("--models", default="srkan", help="comma-separated newly added model IDs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cpu-threads-per-job", type=int, default=1)
    ap.add_argument("--reuse-destination", action="store_true", help="resume an already-created destination copy")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source = Path(args.source_results).resolve()
    dest = Path(args.dest_results).resolve()
    if source == dest:
        ap.error("source and destination must be different; source results are intentionally immutable")
    if not source.is_dir():
        ap.error(f"source result directory does not exist: {source}")

    source_hashes = _run_hashes(source)
    if dest.exists():
        if not args.reuse_destination:
            ap.error(f"destination already exists: {dest}; pass --reuse-destination to resume it")
    else:
        shutil.copytree(source, dest)

    # The destination must start as an exact copy of every source run record.
    initial_dest_hashes = _run_hashes(dest)
    mismatched_initial = [rel for rel, digest in source_hashes.items() if initial_dest_hashes.get(rel) != digest]
    if mismatched_initial:
        raise RuntimeError(f"destination is not an exact copy of source runs: {mismatched_initial[:5]}")

    before_files = set(initial_dest_hashes)
    cmd = [
        sys.executable, "-m", "benchmarks.run_benchmark",
        "--config", str((repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)),
        "--profile", args.profile,
        "--run-dir", str(dest),
        "--models", args.models,
        "--device", args.device,
        "--workers", str(args.workers),
        "--cpu-threads-per-job", str(args.cpu_threads_per_job),
        "--resume", "--reuse-completed",
    ]
    started = time.time()
    print(
        f"[safe-append] launching benchmark: models={args.models} "
        f"workers={args.workers} destination={dest}",
        flush=True,
    )

    # Stream the benchmark output live while retaining an exact copy for the
    # append manifest.  ``-u`` is important here: run_benchmark is otherwise
    # connected to a pipe rather than a TTY, so Python may block-buffer its
    # progress prints and make a healthy benchmark appear hung.
    stream_cmd = [cmd[0], "-u", *cmd[1:]]
    proc = subprocess.Popen(
        stream_cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
        return_code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    benchmark_output = "".join(output_lines)

    after_hashes = _run_hashes(dest)
    changed_old = [rel for rel, digest in source_hashes.items() if after_hashes.get(rel) != digest]
    restored = []
    if changed_old:
        # Defensive rollback: source is immutable, so any accidentally modified
        # pre-existing run record can be restored byte-for-byte.
        for rel in changed_old:
            src = source / rel
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(rel)
        after_hashes = _run_hashes(dest)
        still_changed = [rel for rel, digest in source_hashes.items() if after_hashes.get(rel) != digest]
        if still_changed:
            raise RuntimeError(f"failed to restore pre-existing run records: {still_changed[:5]}")

    new_runs = sorted(set(after_hashes) - before_files)
    manifest = {
        "source_results": str(source),
        "destination_results": str(dest),
        "profile": args.profile,
        "models_requested": [x.strip() for x in args.models.split(",") if x.strip()],
        "source_run_records": len(source_hashes),
        "preexisting_run_records_unchanged": len(source_hashes),
        "preexisting_records_restored_defensively": restored,
        "new_run_records": new_runs,
        "new_run_record_count": len(new_runs),
        "benchmark_return_code": return_code,
        "started_unix": started,
        "finished_unix": time.time(),
        "benchmark_output": benchmark_output,
    }
    (dest / "missing_baseline_append_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[safe-append] source run records preserved: {len(source_hashes)}")
    print(f"[safe-append] new run records: {len(new_runs)}")
    print(f"[safe-append] manifest: {dest / 'missing_baseline_append_manifest.json'}")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
