#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
INTERVENTIONS = ("ZERO_SUFFIX", "HEAD_ROTATE")
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
    cells = {(str(c["operation"]), str(c["address"])): c for c in row["cells"]}
    expected = {(op, a) for op in OPS for a in ADDRESSES}
    if set(cells) != expected:
        raise ValueError(f"invalid cell grid in unit={row.get('unit')}")
    return cells


def target_vector(row: dict[str, Any]) -> tuple[str, ...]:
    cells = cell_map(row)
    return tuple(str(cells[(op, a)]["behavior"]["outcome"]) for op in OPS for a in ADDRESSES)


def metric_value(row: dict[str, Any], op: str, address: str, intervention: str, horizon: int, metric: str) -> float:
    cell = cell_map(row)[(op, address)]
    records = cell["causal_kernel"][intervention]
    by_h = {int(r["horizon"]): r for r in records}
    if horizon not in by_h:
        raise ValueError(f"missing horizon {horizon}")
    return float(by_h[horizon][metric])


def _rank_code(values: list[float]) -> int:
    # Stable six-way ordering of H1/H2/H3 from largest to smallest.
    order = tuple(sorted(range(3), key=lambda i: (-values[i], i)))
    perms = (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    )
    return perms.index(order)


def _argmax_code(values: list[float]) -> int:
    return int(max(range(3), key=lambda i: (values[i], -i)))


def _winner_code(a: float, b: float, rel_tol: float = 0.05) -> int:
    scale = max(abs(a), abs(b), 1e-12)
    if abs(a - b) <= rel_tol * scale:
        return 0
    return 1 if a > b else 2


def _trend_code(h1: float, h2: float, rel_tol: float = 0.05) -> int:
    scale = max(abs(h1), abs(h2), 1e-12)
    if abs(h2 - h1) <= rel_tol * scale:
        return 0
    return 1 if h2 > h1 else 2


def descriptor(row: dict[str, Any], family: str) -> tuple[int, ...]:
    out: list[int] = []

    def add_address_relation(ops: tuple[str, ...], metrics: tuple[str, ...], mode: str) -> None:
        for op in ops:
            for intervention in INTERVENTIONS:
                for horizon in HORIZONS:
                    for metric in metrics:
                        vals = [metric_value(row, op, a, intervention, horizon, metric) for a in ADDRESSES]
                        out.append(_argmax_code(vals) if mode == "argmax" else _rank_code(vals))

    def add_intervention_relation(ops: tuple[str, ...], metrics: tuple[str, ...]) -> None:
        for op in ops:
            for address in ADDRESSES:
                for horizon in HORIZONS:
                    for metric in metrics:
                        z = metric_value(row, op, address, "ZERO_SUFFIX", horizon, metric)
                        h = metric_value(row, op, address, "HEAD_ROTATE", horizon, metric)
                        out.append(_winner_code(z, h))

    def add_temporal_relation(ops: tuple[str, ...], metrics: tuple[str, ...]) -> None:
        for op in ops:
            for address in ADDRESSES:
                for intervention in INTERVENTIONS:
                    for metric in metrics:
                        h1 = metric_value(row, op, address, intervention, 1, metric)
                        h2 = metric_value(row, op, address, intervention, 2, metric)
                        out.append(_trend_code(h1, h2))

    FINAL = ("final_hidden_rel_rms",)
    PRIMARY = ("primary_block_rel_rms",)
    FLOW = ("primary_block_rel_rms", "post_primary_block_rel_rms", "reference_block_rel_rms")
    ALL = METRICS
    SUB = ("SUBSTITUTE",)

    if family == "SUB_FINAL_ARGMAX":
        add_address_relation(SUB, FINAL, "argmax")
    elif family == "SUB_FINAL_RANK":
        add_address_relation(SUB, FINAL, "rank")
    elif family == "SUB_PRIMARY_ARGMAX":
        add_address_relation(SUB, PRIMARY, "argmax")
    elif family == "SUB_PRIMARY_RANK":
        add_address_relation(SUB, PRIMARY, "rank")
    elif family == "SUB_FLOW_ARGMAX":
        add_address_relation(SUB, FLOW, "argmax")
    elif family == "SUB_ALL_ARGMAX":
        add_address_relation(SUB, ALL, "argmax")
    elif family == "SUB_RELATIONAL":
        add_address_relation(SUB, FLOW, "argmax")
        add_intervention_relation(SUB, FLOW)
        add_temporal_relation(SUB, FLOW)
    elif family == "ALL_FINAL_ARGMAX":
        add_address_relation(OPS, FINAL, "argmax")
    elif family == "ALL_PRIMARY_ARGMAX":
        add_address_relation(OPS, PRIMARY, "argmax")
    elif family == "ALL_FLOW_ARGMAX":
        add_address_relation(OPS, FLOW, "argmax")
    elif family == "ALL_RELATIONAL":
        add_address_relation(OPS, FLOW, "argmax")
        add_intervention_relation(OPS, FLOW)
        add_temporal_relation(OPS, FLOW)
    else:
        raise ValueError(family)
    return tuple(out)


FAMILIES = (
    "SUB_FINAL_ARGMAX",
    "SUB_FINAL_RANK",
    "SUB_PRIMARY_ARGMAX",
    "SUB_PRIMARY_RANK",
    "SUB_FLOW_ARGMAX",
    "SUB_ALL_ARGMAX",
    "SUB_RELATIONAL",
    "ALL_FINAL_ARGMAX",
    "ALL_PRIMARY_ARGMAX",
    "ALL_FLOW_ARGMAX",
    "ALL_RELATIONAL",
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
        tg: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i in idxs:
            tg[targets[i]].append({"panel": rows[i].get("_panel_tag"), "unit": int(rows[i]["unit"])})
        if len(tg) <= 1:
            continue
        out.append({
            "descriptor": list(d),
            "members": [
                {"target": json.loads(t), "systems": systems}
                for t, systems in sorted(tg.items())
            ],
        })
    return out
