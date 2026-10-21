#!/usr/bin/env python3
from __future__ import annotations

import bisect
import math
from typing import Any

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
NON_ID_OPS = ("KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
BASE_FEATURES = (
    "primary_k_rms",
    "primary_v_rms",
    "reference_k_rms",
    "reference_v_rms",
    "corridor_ratio",
)
EPS = 1e-12


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
        raise ValueError(f"invalid grid unit={row.get('unit')} missing={missing} extra={extra}")
    return out


def _log(value: float) -> float:
    return math.log(max(float(value), EPS))


def feature_values(row: dict[str, Any]) -> dict[str, float]:
    cells = cell_map(row)
    out: dict[str, float] = {}

    for address in ADDRESSES:
        baseline = cells[("ID", address)]["internal"]
        for feat in BASE_FEATURES:
            out[f"ID:{address}:LOG:{feat}"] = _log(baseline[feat])

    for op in NON_ID_OPS:
        for address in ADDRESSES:
            cur = cells[(op, address)]["internal"]
            base = cells[("ID", address)]["internal"]
            deltas = []
            for feat in BASE_FEATURES:
                d = _log(cur[feat]) - _log(base[feat])
                out[f"{op}:{address}:DLOG:{feat}"] = d
                if feat != "corridor_ratio":
                    deltas.append(d)
            out[f"{op}:{address}:EFFECT_NORM"] = math.sqrt(sum(x * x for x in deltas))
    return out


def all_feature_names() -> list[str]:
    names = []
    for op in NON_ID_OPS:
        for address in ADDRESSES:
            names.append(f"{op}:{address}:EFFECT_NORM")
            for feat in BASE_FEATURES:
                names.append(f"{op}:{address}:DLOG:{feat}")
    for address in ADDRESSES:
        for feat in BASE_FEATURES:
            names.append(f"ID:{address}:LOG:{feat}")
    return names


def quantile(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        raise ValueError("empty values")
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def quantile_cutpoints(values: list[float], bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    raw = [quantile(values, i / bins) for i in range(1, bins)]
    cuts: list[float] = []
    for value in raw:
        if not cuts or value > cuts[-1] + 1e-15:
            cuts.append(float(value))
    return cuts


def encode_values(values: dict[str, float], feature_specs: list[dict[str, Any]]) -> tuple[int, ...]:
    encoded = []
    for spec in feature_specs:
        name = str(spec["name"])
        cuts = [float(x) for x in spec["cutpoints"]]
        encoded.append(int(bisect.bisect_right(cuts, float(values[name]))))
    return tuple(encoded)


def encode_row(row: dict[str, Any], feature_specs: list[dict[str, Any]]) -> tuple[int, ...]:
    return encode_values(feature_values(row), feature_specs)
