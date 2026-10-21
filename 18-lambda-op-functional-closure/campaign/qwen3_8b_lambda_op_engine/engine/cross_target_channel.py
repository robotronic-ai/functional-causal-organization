#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
INTERVENTION = "ZERO_SUFFIX"
HORIZONS = (1, 2)
METRICS = (
    "final_hidden_rel_rms",
    "primary_block_rel_rms",
    "post_primary_block_rel_rms",
    "reference_block_rel_rms",
)


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if int(data.get("n_skipped", 0)) != 0:
            raise ValueError(f"panel has technical skips: {path}")
        panel_rows = list(data.get("rows", []))
        if not panel_rows:
            raise ValueError(f"panel has no rows: {path}")
        tag = path.parent.name
        for row in panel_rows:
            copy = dict(row)
            copy["_panel_tag"] = tag
            rows.append(copy)
    return rows


def cell_map(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cells = {(str(c["operation"]), str(c["source_address"])): c for c in row["cells"]}
    expected = {(op, source) for op in OPS for source in ADDRESSES}
    if set(cells) != expected:
        raise ValueError(f"invalid operation-source grid in unit={row.get('unit')}")
    return cells


def target_vector(row: dict[str, Any]) -> tuple[str, ...]:
    cells = cell_map(row)
    return tuple(str(cells[(op, source)]["behavior"]["outcome"]) for op in OPS for source in ADDRESSES)


def response_value(
    row: dict[str, Any],
    op: str,
    source: str,
    target: str,
    horizon: int,
    metric: str,
) -> float:
    cell = cell_map(row)[(op, source)]
    records = cell["cross_target_kernel"][target][INTERVENTION]
    by_h = {int(r["horizon"]): r for r in records}
    if horizon not in by_h:
        raise ValueError(f"missing horizon {horizon}")
    return float(by_h[horizon][metric])


def _argmax(values: list[float]) -> int:
    return int(max(range(len(values)), key=lambda i: (values[i], -i)))


def _rank3(values: list[float]) -> int:
    order = tuple(sorted(range(3), key=lambda i: (-values[i], i)))
    perms = (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    )
    return perms.index(order)


def _diag_margin(values: list[float], source_index: int, rel_tol: float = 0.08) -> int:
    diag = float(values[source_index])
    off = max(float(values[i]) for i in range(3) if i != source_index)
    scale = max(abs(diag), abs(off), 1e-12)
    if abs(diag - off) <= rel_tol * scale:
        return 0
    return 1 if diag > off else 2


def add_forward_map(out: list[int], row: dict[str, Any], ops: tuple[str, ...], metrics: tuple[str, ...], horizons: tuple[int, ...], rank: bool = False) -> None:
    for op in ops:
        for horizon in horizons:
            for metric in metrics:
                for source in ADDRESSES:
                    vals = [response_value(row, op, source, target, horizon, metric) for target in ADDRESSES]
                    out.append(_rank3(vals) if rank else _argmax(vals))


def add_reverse_map(out: list[int], row: dict[str, Any], ops: tuple[str, ...], metrics: tuple[str, ...], horizons: tuple[int, ...]) -> None:
    for op in ops:
        for horizon in horizons:
            for metric in metrics:
                for target in ADDRESSES:
                    vals = [response_value(row, op, source, target, horizon, metric) for source in ADDRESSES]
                    out.append(_argmax(vals))


def add_diag_relation(out: list[int], row: dict[str, Any], ops: tuple[str, ...], metrics: tuple[str, ...], horizons: tuple[int, ...]) -> None:
    for op in ops:
        for horizon in horizons:
            for metric in metrics:
                for si, source in enumerate(ADDRESSES):
                    vals = [response_value(row, op, source, target, horizon, metric) for target in ADDRESSES]
                    out.append(_diag_margin(vals, si))


def descriptor(row: dict[str, Any], family: str) -> tuple[int, ...]:
    out: list[int] = []
    SUB = ("SUBSTITUTE",)
    FINAL = ("final_hidden_rel_rms",)
    FLOW = METRICS
    H2 = (2,)
    H12 = HORIZONS

    if family == "SUB_FINAL_H2_FORWARD":
        add_forward_map(out, row, SUB, FINAL, H2)
    elif family == "SUB_FINAL_H12_FORWARD":
        add_forward_map(out, row, SUB, FINAL, H12)
    elif family == "SUB_FINAL_H2_FORWARD_RANK":
        add_forward_map(out, row, SUB, FINAL, H2, rank=True)
    elif family == "SUB_FLOW_H2_FORWARD":
        add_forward_map(out, row, SUB, FLOW, H2)
    elif family == "SUB_FLOW_H12_FORWARD":
        add_forward_map(out, row, SUB, FLOW, H12)
    elif family == "SUB_FINAL_H2_BIDIR":
        add_forward_map(out, row, SUB, FINAL, H2)
        add_reverse_map(out, row, SUB, FINAL, H2)
    elif family == "SUB_FLOW_H2_BIDIR":
        add_forward_map(out, row, SUB, FLOW, H2)
        add_reverse_map(out, row, SUB, FLOW, H2)
    elif family == "SUB_FLOW_H2_TOPOLOGY":
        add_forward_map(out, row, SUB, FLOW, H2)
        add_reverse_map(out, row, SUB, FLOW, H2)
        add_diag_relation(out, row, SUB, FLOW, H2)
    elif family == "ALL_FINAL_H2_FORWARD":
        add_forward_map(out, row, OPS, FINAL, H2)
    elif family == "ALL_FLOW_H2_FORWARD":
        add_forward_map(out, row, OPS, FLOW, H2)
    elif family == "ALL_FINAL_H2_BIDIR":
        add_forward_map(out, row, OPS, FINAL, H2)
        add_reverse_map(out, row, OPS, FINAL, H2)
    else:
        raise ValueError(family)
    return tuple(out)


FAMILIES = (
    "SUB_FINAL_H2_FORWARD",
    "SUB_FINAL_H12_FORWARD",
    "SUB_FINAL_H2_FORWARD_RANK",
    "SUB_FLOW_H2_FORWARD",
    "SUB_FLOW_H12_FORWARD",
    "SUB_FINAL_H2_BIDIR",
    "SUB_FLOW_H2_BIDIR",
    "SUB_FLOW_H2_TOPOLOGY",
    "ALL_FINAL_H2_FORWARD",
    "ALL_FLOW_H2_FORWARD",
    "ALL_FINAL_H2_BIDIR",
)


def collision_metrics(descs: list[tuple[int, ...]], targets: list[str]) -> dict[str, int | float]:
    fibers: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for d, t in zip(descs, targets):
        fibers[d].append(t)
    collision_fibers = 0
    collision_pairs = 0
    nontrivial_fibers = 0
    systems_nontrivial = 0
    for tgts in fibers.values():
        if len(tgts) > 1:
            nontrivial_fibers += 1
            systems_nontrivial += len(tgts)
        counts: dict[str, int] = defaultdict(int)
        for t in tgts:
            counts[t] += 1
        if len(counts) > 1:
            collision_fibers += 1
            total = len(tgts) * (len(tgts) - 1) // 2
            same = sum(n * (n - 1) // 2 for n in counts.values())
            collision_pairs += total - same
    n = len(descs)
    return {
        "n_systems": n,
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": nontrivial_fibers,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "nontrivial_coverage": systems_nontrivial / n if n else 0.0,
        "n_collision_fibers": collision_fibers,
        "n_collision_pairs": collision_pairs,
    }


def collision_examples(rows: list[dict[str, Any]], descs: list[tuple[int, ...]], targets: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, d in enumerate(descs):
        groups[d].append(i)
    out: list[dict[str, Any]] = []
    for d, idxs in groups.items():
        target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i in idxs:
            target_groups[targets[i]].append({"panel": rows[i].get("_panel_tag"), "unit": int(rows[i]["unit"])})
        if len(target_groups) <= 1:
            continue
        out.append({
            "descriptor": list(d),
            "members": [
                {"target": json.loads(t), "systems": systems}
                for t, systems in sorted(target_groups.items())
            ],
        })
    return out
