from __future__ import annotations

from typing import Any


def expected_model_calls(instance: Any, adapter: Any) -> int:
    """Estimate calls to model.generate for one benchmark instance.

    The estimate is exact for the bundled HF baseline tests in protocol v0.2.7.
    Causal tests that require unsupported architecture primitives return zero.
    External adapters may perform additional internal work that is intentionally not
    counted as language-model generation calls.
    """
    code = str(instance.test).upper()
    probes = len(getattr(instance, "probes", []) or [])
    meta = getattr(instance, "meta", {}) or {}

    if code in {"T1", "T2", "T3", "B1", "B2", "B3", "E2", "E3", "A1", "A2", "A3"}:
        return probes
    if code == "E1":
        return 12 * max(2, int(getattr(adapter, "e1_samples", 5))) + 1
    if code == "M1":
        return len(meta.get("items", []))
    if code == "M2":
        return len(meta.get("items", []))
    if code == "M3":
        return 2 * len(meta.get("items", []))
    if code == "C1":
        return 3
    if code == "C2":
        return probes
    if code == "C3":
        return int(meta.get("k", 0)) + 2
    if code == "P1":
        return 3
    if code == "P2":
        return 2
    if code == "P3":
        return 3

    required = {
        "R1": "silent_steps",
        "R2": "adaptive_compute",
        "D1": "idle_dynamics",
        "D2": "idle_dynamics",
        "D3": "idle_dynamics",
        "L1": "persistent_latent_state",
        "L2": "persistent_latent_state",
        "L3": "dual_reset",
        "U1": "state_write",
        "U2": "checkpoint_rollback",
    }
    if code in required and not adapter.supports(required[code]):
        return 0

    fixed = {
        "R1": 3,
        "R2": 4,
        "R3": 1,
        "D1": 1,
        "D2": 1,
        "D3": 1,
        "L1": 1,
        "L2": 2,
        "L3": 2,
        "U1": 1,
        "U2": 1,
        "U3": 0,
        "I1": 0,
        "I2": 0,
        "I3": 0,
        "G1": 0,
        "G2": 0,
        "G3": 0,
        "H1": 0,
        "H2": 0,
        "H3": 0,
    }
    return int(fixed.get(code, probes))
