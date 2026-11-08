"""Deterministic A/B assignment.

Hashing ``experiment:user_id`` gives every user a stable bucket without any stored state:
the same user always sees the same variant, and a new experiment name reshuffles everyone.
"""

from __future__ import annotations

import hashlib


def assign_variant(user_id: str, experiment: str, treatment_share: float = 0.5) -> str:
    if not 0.0 <= treatment_share <= 1.0:
        raise ValueError("treatment_share must be between 0 and 1")
    digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "treatment" if bucket < treatment_share else "control"
