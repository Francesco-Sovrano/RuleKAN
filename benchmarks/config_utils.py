from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Set


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge profile dictionaries without mutating either input."""
    out: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve_profile(config: Mapping[str, Any], profile_name: str, _stack: Set[str] | None = None) -> Dict[str, Any]:
    """Resolve a benchmark profile, including optional ``extends`` inheritance.

    Child values override parent values. Nested mappings such as ``model_config``
    and ``shared_settings`` are merged recursively, while lists (notably
    ``models``/``tasks``/``suites``) replace the parent list.
    """
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"unknown benchmark profile {profile_name!r}")
    stack = set() if _stack is None else set(_stack)
    if profile_name in stack:
        chain = " -> ".join([*sorted(stack), profile_name])
        raise ValueError(f"cyclic benchmark profile inheritance: {chain}")
    stack.add(profile_name)

    profile = deepcopy(dict(profiles[profile_name]))
    parent_name = profile.get("extends")
    if not parent_name:
        profile.pop("extends", None)
        return profile
    parent = resolve_profile(config, str(parent_name), stack)
    return _deep_merge(parent, profile)
