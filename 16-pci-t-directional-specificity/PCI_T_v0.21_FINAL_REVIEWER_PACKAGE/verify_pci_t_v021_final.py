#!/usr/bin/env python3
"""Standalone verifier for the PCI_T v0.21 final reviewer package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_results"

EXPECTED_SHA256 = {
    "R01_DIRECTIONAL_SPECIFICITY_RESULT_v0.21-X4.json": "c1e165a664c4e79ff08f883b26574e22d318ef51691cccfa969e530142f6407b",
    "R01_CAL3_LATE_LAYER_GATE_RESULT_v0.21-X7.json": "2ced898841193c455856a7fcf0866b563b1098508086a05dad968f07367a95af",
    "R01_SCORE2_FINAL_RESULT_v0.21-X8.json": "c711e2603b3785f27d99e640e93fbf71879db84bb41ee33908fd1eac81816baa",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name: str) -> dict:
    path = RAW / name
    actual = sha256(path)
    expected = EXPECTED_SHA256[name]
    assert actual == expected, f"SHA256 mismatch for {name}: {actual} != {expected}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_x4(x4: dict) -> None:
    assert x4["gates"]["passed"] is True
    assert x4["gates"]["failed"] == []
    assert x4["decision"] == "DIRECTIONAL_SPECIFICITY_SUPPORTED_FOR_QWEN_DEV_ONLY"
    adversarial = x4["gates"]["adversarial_median_enrichment"]
    structured = x4["gates"]["structured_median_enrichment"]
    assert max(adversarial.values()) <= 1.20
    assert min(structured.values()) >= 1.50
    assert x4["gates"]["separation_ratio"] > 7.0


def verify_x7(x7: dict) -> None:
    a = x7["analysis"]
    assert a["decision"] == "CAL3_LATE_LAYER_GATE_PASS_SCORE2_AUTHORIZED"
    assert a["gate_passed"] is True
    assert a["gate_failed"] == []
    assert a["deepest_dominance_count"] >= 5
    assert a["layer_system_medians"]["27"] > 1.20
    assert a["fraction_l27_medians"]["0.20"] > 1.20
    assert a["fraction_l27_medians"]["0.40"] > 1.20
    assert a["late_vs_lower_system_ratio"] > 1.20
    assert x7["future_data_firewall"]["score2_parsed"] is False


def verify_x8(x8: dict) -> tuple[float, float]:
    a = x8["analysis"]
    assert a["decision"] == "SCORE2_FINAL_PROSPECTIVE_MEASUREMENT_COMPLETE"
    assert a["selection_gate"] is None
    assert x8["score2_selection_gate"] is None
    assert x8["data_opening"]["authorized_by_x7"] is True
    assert x8["data_opening"]["opened_only_after_x8_contract_freeze"] is True
    assert x8["frozen_design"]["primary_layer"] == 27
    assert x8["frozen_design"]["directions_per_site"] == 16
    assert x8["frozen_design"]["orthogonal_controls_per_site"] == 15

    records = x8["records"]
    assert len(records) == 12
    grouped: dict[str, list[dict]] = {}
    for r in records:
        assert r["source_layer"] == 27
        assert r["source_win_rate"] == 1.0
        grouped.setdefault(r["episode_id"], []).append(r)
    assert len(grouped) == 6

    episode_primary = []
    episode_secondary = []
    for episode_id in sorted(grouped):
        rs = sorted(grouped[episode_id], key=lambda r: r["source_token_fraction"])
        assert [round(r["source_token_fraction"], 2) for r in rs] == [0.20, 0.40]
        episode_primary.append(median([r["source_enrichment"] for r in rs]))
        episode_secondary.append(median([r["source_score"] for r in rs]))

    primary = median(episode_primary)
    secondary = median(episode_secondary)
    assert abs(primary - a["primary_scalar"]) < 1e-12
    assert abs(secondary - a["secondary_scalar"]) < 1e-12
    return primary, secondary


def main() -> None:
    x4 = load("R01_DIRECTIONAL_SPECIFICITY_RESULT_v0.21-X4.json")
    x7 = load("R01_CAL3_LATE_LAYER_GATE_RESULT_v0.21-X7.json")
    x8 = load("R01_SCORE2_FINAL_RESULT_v0.21-X8.json")

    verify_x4(x4)
    verify_x7(x7)
    primary, secondary = verify_x8(x8)

    print("[PASS] X4 synthetic adversarial directional-specificity gates")
    print("[PASS] X7 prospective CAL3 late-layer replication gate")
    print("[PASS] X8 final SCORE2 provenance and frozen design")
    print(f"[FINAL] PCI_T_DS_L27={primary:.12f}")
    print(f"[SECONDARY] PCI_T_ST_L27_SOURCE={secondary:.12f}")
    print("[INTERPRETATION] Primary value is a directional enrichment ratio, not a before/after change from the historical v0.20 score.")


if __name__ == "__main__":
    main()
