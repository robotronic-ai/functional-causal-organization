#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
P_SIZE = 3
DELTA_Q = (0.5, 0.3, 0.2)
Q_INTERVENTIONS = ("CONTROL", "ZERO_LOCAL", "HEAD_ROTATE_LOCAL")
AUX_CONTEXTS = (
    "AUXILIARY CONTEXT ALPHA. Preserve the current episode state. Do not answer yet.",
    "AUXILIARY CONTEXT BETA. Preserve the current episode state. Do not answer yet.",
)
RETURN_CONTEXTS = (
    "RETURN CONTEXT GAMMA. Revisit the current episode state after this marker. Do not answer yet.",
    "RETURN CONTEXT DELTA. Revisit the current episode state after this marker. Do not answer yet.",
)
RHO_STEP = 1.0 / 16.0


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def wrap_value(value: int) -> int:
    return 30 + ((int(value) - 30) % 60)


def numeric_support(original: int, donor: int) -> list[int]:
    """Three deterministic, target-blind numeric states inside the campaign support."""
    candidates = [
        wrap_value(original),
        wrap_value(donor),
        wrap_value(original + 17),
        wrap_value(original + 31),
        wrap_value(donor + 23),
        wrap_value(original + 43),
    ]
    out: list[int] = []
    for value in candidates:
        if value not in out:
            out.append(value)
        if len(out) == P_SIZE:
            break
    if len(out) != P_SIZE:
        raise RuntimeError(f"could not build {P_SIZE}-state support from original={original}, donor={donor}")
    return out


def response_labels() -> tuple[str, ...]:
    return tuple([f"V{i}" for i in range(P_SIZE)] + ["NONE"])


def entropy_bits(weights: Iterable[float]) -> float:
    p = np.asarray(list(weights), dtype=np.float64)
    if p.size == 0:
        return 0.0
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    nz = p[p > 0.0]
    return float(-np.sum(nz * np.log2(nz)))


def mutual_information_bits(weights: Iterable[float], conditionals: Iterable[Iterable[float]]) -> float:
    q = np.asarray(list(weights), dtype=np.float64)
    ch = np.asarray(list(conditionals), dtype=np.float64)
    if ch.ndim != 2 or ch.shape[0] != q.size:
        raise ValueError("channel shape mismatch")
    q = q / q.sum()
    row_sums = ch.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("channel row has non-positive mass")
    ch = ch / row_sums
    py = np.sum(q[:, None] * ch, axis=0)
    mi = 0.0
    for i in range(q.size):
        for j in range(ch.shape[1]):
            p = float(ch[i, j])
            if p > 0.0 and py[j] > 0.0:
                mi += float(q[i]) * p * math.log2(p / float(py[j]))
    return float(max(0.0, mi))


def tagged_role_mixture(role_distributions: dict[str, list[float]], roles: Iterable[str]) -> list[float]:
    roles = tuple(roles)
    if not roles:
        raise ValueError("roles cannot be empty")
    out: list[float] = []
    scale = 1.0 / len(roles)
    for role in roles:
        vec = np.asarray(role_distributions[role], dtype=np.float64)
        vec = vec / vec.sum()
        out.extend((scale * vec).tolist())
    return out


def ordered_top2(probabilities: Iterable[float], labels: Iterable[str] | None = None) -> list[str]:
    p = list(float(x) for x in probabilities)
    names = list(labels if labels is not None else response_labels())
    if len(p) != len(names):
        raise ValueError("probability / label mismatch")
    order = sorted(range(len(p)), key=lambda i: (-p[i], names[i]))
    return [names[i] for i in order[:2]]


def canonical_partition(signatures: list[Any]) -> list[int]:
    mapping: dict[str, int] = {}
    out: list[int] = []
    for sig in signatures:
        key = stable(sig)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return out


def reentry_metrics(
    q_names: list[str],
    source_signatures: list[Any],
    return_conditionals: dict[str, list[list[float]]],
) -> dict[str, Any]:
    if len(q_names) != len(source_signatures):
        raise ValueError("q/signature mismatch")
    class_ids = canonical_partition(source_signatures)
    n_q = len(q_names)
    if n_q == 0:
        raise ValueError("empty intervention battery")
    class_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(class_ids):
        class_members[int(c)].append(i)
    class_order = sorted(class_members)
    p_c = [len(class_members[c]) / n_q for c in class_order]
    k_eff = entropy_bits(p_c)
    per_context = []
    for context_name, by_q in return_conditionals.items():
        if len(by_q) != n_q:
            raise ValueError("return channel q count mismatch")
        cond_c = []
        for c in class_order:
            arr = np.asarray([by_q[i] for i in class_members[c]], dtype=np.float64)
            cond_c.append(np.mean(arr, axis=0).tolist())
        mi = mutual_information_bits(p_c, cond_c)
        per_context.append({
            "context": context_name,
            "mutual_information_bits": float(mi),
            "conditionals_by_class": cond_c,
        })
    b_reentry = float(np.mean([x["mutual_information_bits"] for x in per_context])) if per_context else 0.0
    r = None if k_eff <= 1e-15 else float(b_reentry / k_eff)

    # Duplication invariance: duplicate every intervention without changing mass.
    dup_sigs = [sig for sig in source_signatures for _ in (0, 1)]
    dup_classes = canonical_partition(dup_sigs)
    dup_members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(dup_classes):
        dup_members[int(c)].append(i)
    dup_pc = [len(dup_members[c]) / len(dup_classes) for c in sorted(dup_members)]
    dup_k = entropy_bits(dup_pc)
    dup_mis = []
    for context_name, by_q in return_conditionals.items():
        duplicated = [row for row in by_q for _ in (0, 1)]
        cond_c = []
        for c in sorted(dup_members):
            arr = np.asarray([duplicated[i] for i in dup_members[c]], dtype=np.float64)
            cond_c.append(np.mean(arr, axis=0).tolist())
        dup_mis.append(mutual_information_bits(dup_pc, cond_c))
    dup_b = float(np.mean(dup_mis)) if dup_mis else 0.0
    dup_r = None if dup_k <= 1e-15 else float(dup_b / dup_k)
    invariant = abs(dup_k - k_eff) <= 1e-12 and abs(dup_b - b_reentry) <= 1e-12
    if r is not None and dup_r is not None:
        invariant = invariant and abs(dup_r - r) <= 1e-12
    return {
        "q_names": list(q_names),
        "source_signatures": source_signatures,
        "source_class_ids": class_ids,
        "n_source_classes": len(class_members),
        "class_probabilities": p_c,
        "K_eff_bits": float(k_eff),
        "return_contexts": per_context,
        "B_reentry_bits": float(b_reentry),
        "R": r,
        "label_duplication_invariance": bool(invariant),
        "duplicate_K_eff_bits": float(dup_k),
        "duplicate_B_reentry_bits": float(dup_b),
        "duplicate_R": dup_r,
    }


def q16(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    x = min(1.0, max(0.0, float(value)))
    return str(int(round(x / RHO_STEP)))


def gamma_key(row: dict[str, Any]) -> str:
    parts = []
    by = {(x["operation"], x["cut"]): x for x in row["Gamma"]["cuts"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, f"{source}|" + "+".join(a for a in ADDRESSES if a != source))]
            parts.extend([
                q16(rec["forward_normalized"]),
                q16(rec["backward_normalized"]),
                q16(rec["cut_score_arithmetic_mean"]),
                q16(rec["directional_balance_beta"]),
            ])
    return "|".join(parts)


def delta_key(row: dict[str, Any]) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["Delta"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, source)]
            parts.extend([q16(rec["kappa_q"]), q16(rec["delta_q"]), q16(rec["Delta_cap"])])
    return "|".join(parts)


def r_key(row: dict[str, Any]) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["R"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, source)]
            parts.extend([
                str(int(rec["n_source_classes"])),
                q16(None if rec["K_eff_bits"] is None else rec["K_eff_bits"] / math.log2(len(Q_INTERVENTIONS))),
                q16(rec["R"]),
            ])
            parts.append(stable(rec["source_class_ids"]))
    return "|".join(parts)


def lambda_key(row: dict[str, Any]) -> str:
    parts = []
    by = {(x["operation"], x["source_address"]): x for x in row["Lambda"]["cells"]}
    for op in OPS:
        for source in ADDRESSES:
            rec = by[(op, source)]
            for q_name in Q_INTERVENTIONS:
                parts.append(stable(rec["joint_signature_by_intervention"][q_name]))
    return "|".join(parts)


def component_keys(row: dict[str, Any]) -> dict[str, str]:
    return {
        "Gamma": gamma_key(row),
        "Delta": delta_key(row),
        "R": r_key(row),
        "Lambda": lambda_key(row),
    }


def family_components() -> dict[str, tuple[str, ...]]:
    return {
        "GAMMA": ("Gamma",),
        "DELTA": ("Delta",),
        "R": ("R",),
        "LAMBDA": ("Lambda",),
        "GAMMA+DELTA": ("Gamma", "Delta"),
        "GAMMA+R": ("Gamma", "R"),
        "GAMMA+LAMBDA": ("Gamma", "Lambda"),
        "DELTA+R": ("Delta", "R"),
        "DELTA+LAMBDA": ("Delta", "Lambda"),
        "R+LAMBDA": ("R", "Lambda"),
        "FULL_CO_OP": ("Gamma", "Delta", "R", "Lambda"),
    }


def family_label(keys: dict[str, str], components: Iterable[str]) -> str:
    return stable([[name, keys[name]] for name in components])
