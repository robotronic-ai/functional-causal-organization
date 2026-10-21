#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lambda_op_descriptor import ADDRESSES, OPS, calibrate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--construction", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source = Path(args.construction)
    panel = json.loads(source.read_text())
    if panel.get("stage") != "construction":
        raise ValueError("input is not a construction panel")
    if panel.get("development_limit_used"):
        raise ValueError("construction panel used a development limit")
    if panel.get("n_skipped") != 0:
        raise ValueError("construction panel contains technical skips")
    if any("behavior" in cell for row in panel["rows"] for cell in row["cells"]):
        raise ValueError("construction panel contains behavioral data")

    thresholds = calibrate(panel["rows"])
    result = {
        "schema": "co-qwen3-8b-lambda-op-v42.7-descriptor-freeze-v2",
        "descriptor": "LAMBDA_OP_OPERATION_ADDRESS_EFFECT",
        "operation_algebra": list(OPS),
        "addresses": list(ADDRESSES),
        "definition": (
            "Preserve the complete operation-by-address grid. For every non-ID cell, compare the target-blind "
            "internal corridor features with ID at the same address. Classify the cell as QUIET, ACTIVE_UP, "
            "ACTIVE_DOWN or ACTIVE_FLAT using construction-only KEEP-vs-ID 95th-percentile thresholds."
        ),
        "internal_features": ["primary_k_rms", "primary_v_rms", "reference_k_rms", "reference_v_rms", "corridor_ratio"],
        "threshold_reference": "KEEP versus ID at the same address",
        "threshold_quantile": 0.95,
        "thresholds": thresholds,
        "construction_n": len(panel["rows"]),
        "construction_panel_sha256": sha256(source),
        "behavior_used_for_freeze": False,
        "guardrail": "The descriptor and thresholds are frozen before the behavioral holdout is executed.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[LAMBDA-OP DESCRIPTOR FROZEN]")
    print("[DESCRIPTOR]", result["descriptor"])
    print("[CONSTRUCTION N]", result["construction_n"])
    print("[SHA256]", sha256(out))


if __name__ == "__main__":
    main()
