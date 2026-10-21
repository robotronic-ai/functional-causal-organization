#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
FEATURES = ("primary_k_rms", "primary_v_rms", "reference_k_rms", "reference_v_rms")
EPS = 1e-12


def quantile(values, q: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        raise ValueError("empty calibration values")
    pos = (len(xs) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def cell_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for cell in row["cells"]:
        key = (str(cell["operation"]), str(cell["address"]))
        if key in out:
            raise ValueError(f"duplicate cell {key} in unit={row.get('unit')}")
        out[key] = cell
    expected = {(op, address) for op in OPS for address in ADDRESSES}
    if set(out) != expected:
        missing = sorted(expected - set(out))
        extra = sorted(set(out) - expected)
        raise ValueError(f"invalid operation-address grid unit={row.get('unit')} missing={missing} extra={extra}")
    return out


def _log_feature(cell: dict[str, Any], name: str) -> float:
    value = float(cell["internal"][name])
    return math.log(max(value, EPS))


def effect(cell: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    deltas = [_log_feature(cell, name) - _log_feature(baseline, name) for name in FEATURES]
    norm = math.sqrt(sum(x * x for x in deltas))
    cr = math.log(max(float(cell["internal"]["corridor_ratio"]), EPS))
    br = math.log(max(float(baseline["internal"]["corridor_ratio"]), EPS))
    return {
        "effect_norm": float(norm),
        "corridor_log_delta": float(cr - br),
    }


def calibrate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep_norm = {address: [] for address in ADDRESSES}
    keep_corridor = {address: [] for address in ADDRESSES}
    for row in rows:
        cells = cell_map(row)
        for address in ADDRESSES:
            e = effect(cells[("KEEP", address)], cells[("ID", address)])
            keep_norm[address].append(e["effect_norm"])
            keep_corridor[address].append(abs(e["corridor_log_delta"]))

    thresholds = {}
    for address in ADDRESSES:
        thresholds[address] = {
            "effect_norm_q95_keep_vs_id": max(quantile(keep_norm[address], 0.95), 1e-8),
            "corridor_abs_q95_keep_vs_id": max(quantile(keep_corridor[address], 0.95), 1e-8),
        }
    return thresholds


def encode_row(row: dict[str, Any], thresholds: dict[str, Any]) -> list[list[str]]:
    cells = cell_map(row)
    descriptor = []
    for op in OPS:
        for address in ADDRESSES:
            if op == "ID":
                state = "BASELINE"
            else:
                e = effect(cells[(op, address)], cells[("ID", address)])
                t = thresholds[address]
                active = e["effect_norm"] > float(t["effect_norm_q95_keep_vs_id"])
                delta = e["corridor_log_delta"]
                dead = float(t["corridor_abs_q95_keep_vs_id"])
                if not active and abs(delta) <= dead:
                    state = "QUIET"
                elif delta > dead:
                    state = "ACTIVE_UP"
                elif delta < -dead:
                    state = "ACTIVE_DOWN"
                else:
                    state = "ACTIVE_FLAT"
            descriptor.append([op, address, state])
    return descriptor
