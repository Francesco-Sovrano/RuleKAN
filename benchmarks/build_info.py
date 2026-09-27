from __future__ import annotations

import hashlib
from pathlib import Path


def benchmark_build_fingerprint(root: Path, config_path: Path | None = None) -> str:
    """Hash benchmark-relevant source/config so resume never reuses stale runs."""
    root = Path(root).resolve()
    h = hashlib.sha256()
    candidates = []
    for rel in ("rulekan", "benchmarks", "external/Pub_Symbolic_KANs"):
        base = root / rel
        if base.exists():
            candidates.extend(sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts))
            marker = base / "UPSTREAM_COMMIT.txt"
            if marker.exists():
                candidates.append(marker)
    for rel in ("VERSION", "requirements.txt", "benchmarks/configs/default.yaml"):
        p = root / rel
        if p.exists():
            candidates.append(p)
    if config_path is not None:
        cp = Path(config_path).resolve()
        if cp.exists() and cp not in candidates:
            candidates.append(cp)
    for p in sorted(set(candidates), key=lambda q: str(q)):
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def code_version(root: Path) -> str:
    p = Path(root).resolve() / "VERSION"
    return p.read_text().strip() if p.exists() else "unknown"
