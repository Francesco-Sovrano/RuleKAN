from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def _load_records(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "runs").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except Exception as exc:
            rows.append({"status": "unreadable", "error": f"{type(exc).__name__}: {exc}", "_path": str(path)})
            continue
        row["_path"] = str(path)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain completed/failed/running/skipped RuleKAN benchmark records")
    ap.add_argument("run_dir")
    ap.add_argument("--show-completed", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    rows = _load_records(run_dir)
    statuses = Counter(str(r.get("status", "unknown")).lower() for r in rows)
    print(f"run_dir: {run_dir}")
    print(f"total run records: {len(rows)}")
    for status in ("completed", "failed", "skipped", "running", "unknown", "unreadable"):
        if statuses.get(status):
            print(f"{status}: {statuses[status]}")

    skip_path = run_dir / "skipped_incompatible_jobs.json"
    if skip_path.exists():
        try:
            skipped = json.loads(skip_path.read_text())
            print(f"incompatible jobs not scheduled: {len(skipped)}")
        except Exception as exc:
            print(f"incompatible manifest unreadable: {type(exc).__name__}: {exc}")

    now = time.time()
    shown = 0
    for row in rows:
        status = str(row.get("status", "unknown")).lower()
        if status == "completed" and not args.show_completed:
            continue
        shown += 1
        task = row.get("task", "?")
        model = row.get("model", "?")
        seed = row.get("seed", "?")
        print(f"\n[{status}] {task} / {model} / seed={seed}")
        if row.get("error"):
            print(f"  error: {row['error']}")
        if status == "running" and row.get("started_unix"):
            age = max(0.0, now - float(row["started_unix"]))
            print(f"  running-record age: {age:.0f}s; if no subprocess is active, the run was interrupted before terminal status was written")
        if row.get("elapsed_seconds") is not None:
            print(f"  elapsed_seconds: {row['elapsed_seconds']}")
        print(f"  record: {row.get('_path')}")
        log = run_dir / "logs" / (Path(str(row.get("_path"))).stem + ".log")
        if log.exists():
            print(f"  log: {log}")
    if shown == 0:
        print("\nNo incomplete run records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
