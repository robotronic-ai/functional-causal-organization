#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
LOCAL_INTERVENTIONS = ("ZERO_LOCAL", "HEAD_ROTATE_LOCAL")
HORIZONS = (1, 2, 3, 4)
METRICS = (
    "final_hidden_rel_rms",
    "primary_block_rel_rms",
    "post_primary_block_rel_rms",
    "reference_block_rel_rms",
)


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _prefix_char_lengths(tok, token_ids: list[int]) -> list[int]:
    out = [0]
    for i in range(1, len(token_ids) + 1):
        out.append(len(tok.decode(token_ids[:i], skip_special_tokens=False)))
    return out


def _char_to_token(prefix_lengths: list[int], char_pos: int, *, right: bool = False) -> int:
    if char_pos <= 0:
        return 0
    for i in range(1, len(prefix_lengths)):
        if prefix_lengths[i] > char_pos or (right and prefix_lengths[i] >= char_pos):
            return i - 1 if not right else i
    return len(prefix_lengths) - 1


def control_clause_spans(tok, suffix_ids: list[int]) -> dict[str, tuple[int, int]]:
    """Locate H1/H2/H3 control-map clauses in the rendered instruction suffix.

    The operation instruction is unchanged. Spans are inferred from the decoded
    CONTROL_MAP text and are relative to the control suffix, not the full cache.
    """
    text = tok.decode(suffix_ids, skip_special_tokens=False)
    positions: dict[str, int] = {}
    for address in ADDRESSES:
        needle = f"{address}:"
        p = text.find(needle)
        if p < 0:
            raise RuntimeError(f"cannot locate {needle!r} in control suffix: {text!r}")
        positions[address] = p
    # The targeted address is rendered first by operation_instruction(), so the
    # CONTROL_MAP order is source-dependent (for example H2,H1,H3).  Clause
    # localization must therefore follow textual position rather than assume
    # canonical H1,H2,H3 ordering.
    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    last_pos = ordered[-1][1]
    map_end = text.find(". Apply each control", last_pos)
    if map_end < 0:
        map_end = text.find(". Apply", last_pos)
    if map_end < 0:
        raise RuntimeError("cannot locate end of CONTROL_MAP clause")
    bounds = {}
    for idx, (address, start_char) in enumerate(ordered):
        end_char = ordered[idx + 1][1] if idx + 1 < len(ordered) else map_end
        bounds[address] = (start_char, end_char)
    lens = _prefix_char_lengths(tok, suffix_ids)
    spans: dict[str, tuple[int, int]] = {}
    for address, (a, b) in bounds.items():
        start = _char_to_token(lens, a, right=False)
        end = _char_to_token(lens, b, right=True)
        if end <= start:
            raise RuntimeError(f"empty token span for {address}: {(start, end)}")
        decoded_span = tok.decode(suffix_ids[start:end], skip_special_tokens=False)
        if address not in decoded_span:
            raise RuntimeError(f"localized token span lost address marker {address}: {decoded_span!r}")
        spans[address] = (int(start), int(end))
    return spans


def bounded_response(x: float) -> float:
    x = max(0.0, float(x))
    return x / (1.0 + x)


def routing_tensor(row: dict[str, Any], metric: str = "final_hidden_rel_rms") -> np.ndarray:
    """Return [operation, source, intervention, target, horizon]."""
    if metric not in METRICS:
        raise ValueError(metric)
    out = np.zeros((len(OPS), len(ADDRESSES), len(LOCAL_INTERVENTIONS), len(ADDRESSES), len(HORIZONS)), dtype=np.float64)
    cells = {(c["operation"], c["source_address"]): c for c in row["cells"]}
    for oi, op in enumerate(OPS):
        for si, source in enumerate(ADDRESSES):
            rec = cells[(op, source)]["localized_routing"]
            for ii, intervention in enumerate(LOCAL_INTERVENTIONS):
                for ti, target in enumerate(ADDRESSES):
                    vals = rec[intervention][target]
                    by_h = {int(x["horizon"]): float(x[metric]) for x in vals}
                    for hi, h in enumerate(HORIZONS):
                        out[oi, si, ii, ti, hi] = bounded_response(by_h[h])
    return out


def gamma_profile(rt: np.ndarray) -> dict[str, Any]:
    # Horizon-1 response averaged over the two experimental interventions.
    M = rt[:, :, :, :, 0].mean(axis=2)  # op, source, target
    cuts = ((0,), (1,), (2,))
    per_op: list[dict[str, Any]] = []
    for oi, op in enumerate(OPS):
        cut_rows = []
        for A in cuts:
            B = tuple(i for i in range(3) if i not in A)
            forward = float(np.mean([M[oi, s, t] for s in A for t in B]))
            backward = float(np.mean([M[oi, s, t] for s in B for t in A]))
            cut_rows.append({
                "cut": f"{'+'.join(ADDRESSES[i] for i in A)}|{'+'.join(ADDRESSES[i] for i in B)}",
                "forward_role_normalized": forward,
                "backward_role_normalized": backward,
                "bidirectional_min": min(forward, backward),
                "bidirectional_geometric_mean": math.sqrt(max(0.0, forward * backward)),
            })
        per_op.append({
            "operation": op,
            "cuts": cut_rows,
            "gamma_min_bidirectional": min(x["bidirectional_min"] for x in cut_rows),
            "gamma_min_geometric": min(x["bidirectional_geometric_mean"] for x in cut_rows),
        })
    return {
        "estimator": "GAMMA_QWEN_ROLE_CUT_RESPONSE_V1",
        "status": "CANDIDATE_OPERATIONAL_ESTIMATOR",
        "definition": "minimum bidirectional source-target causal response over declared role cuts, using role-normalized means",
        "per_operation": per_op,
    }


def _effective_rank01(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    if s.size == 0 or float(s.sum()) <= 1e-12:
        return 0.0
    p = s / s.sum()
    ent = -float(np.sum(p * np.log(np.maximum(p, 1e-15))))
    er = math.exp(ent)
    return float((er - 1.0) / max(1.0, min(X.shape) - 1.0))


def delta_profile(rt: np.ndarray) -> dict[str, Any]:
    per_op = []
    for oi, op in enumerate(OPS):
        # Each source x intervention is one admissible perturbation pattern;
        # its response vector spans targets x horizons.
        V = rt[oi].reshape(len(ADDRESSES) * len(LOCAL_INTERVENTIONS), -1)
        delta_cap = _effective_rank01(V)
        h1 = rt[oi, :, :, :, 0].mean(axis=1)  # source,target
        total = float(h1.sum())
        off = float(sum(h1[s, t] for s in range(3) for t in range(3) if s != t))
        pi_role = off / total if total > 1e-12 else 0.0
        role_coverage = 0.0
        active = 0
        for t in range(3):
            vals = [h1[s, t] for s in range(3) if s != t]
            if vals and max(vals) > 0.05 * max(1e-12, float(h1.max())):
                active += 1
        role_coverage = active / 3.0
        per_op.append({
            "operation": op,
            "delta_cap_effective_rank": delta_cap,
            "pi_role_offdiagonal_mass": pi_role,
            "role_coverage_5pct": role_coverage,
            "delta_product": delta_cap * pi_role,
            "delta_min": min(delta_cap, pi_role),
            "delta_normalized_mean": 0.5 * (delta_cap + pi_role),
        })
    return {
        "estimator": "DELTA_QWEN_INTERVENTION_RICHNESS_X_ROLE_PROPAGATION_V1",
        "status": "CANDIDATE_OPERATIONAL_ESTIMATOR",
        "primary_composition": "delta_product = delta_cap_effective_rank * pi_role_offdiagonal_mass",
        "alternatives_reported_not_promoted": ["delta_min", "delta_normalized_mean"],
        "per_operation": per_op,
    }


def r_profile(rt: np.ndarray) -> dict[str, Any]:
    per_op = []
    for oi, op in enumerate(OPS):
        mh = rt[oi].mean(axis=1)  # source,target,horizon; intervention averaged
        h1 = mh[:, :, 0]
        row_sums = h1.sum(axis=1, keepdims=True)
        T = np.divide(h1, np.maximum(row_sums, 1e-12))
        cycle = []
        Tk = np.eye(3, dtype=np.float64)
        for k in HORIZONS:
            Tk = Tk @ T
            cycle.append(float(np.mean(np.diag(Tk))))
        d0 = float(np.mean(np.diag(h1)))
        persistence = []
        for hi, k in enumerate(HORIZONS):
            dk = float(np.mean(np.diag(mh[:, :, hi])))
            persistence.append(dk / d0 if d0 > 1e-12 else 0.0)
        per_op.append({
            "operation": op,
            "r_cycle_candidate": cycle,
            "persistence_control_profile": persistence,
            "transition_matrix_h1": T.tolist(),
        })
    return {
        "estimator": "R_QWEN_ROLE_CYCLE_PROFILE_V1",
        "status": "CANDIDATE_ALGEBRAIC_REENTRY_ESTIMATOR",
        "warning": "cycle return is derived from the localized role-routing transition matrix; persistence is reported separately and neither is asserted to equal the manuscript R(k) without construct validation",
        "horizons": list(HORIZONS),
        "per_operation": per_op,
    }


def gamma_vector(profile: dict[str, Any]) -> np.ndarray:
    vals = []
    for op in profile["per_operation"]:
        for cut in op["cuts"]:
            vals.extend([cut["forward_role_normalized"], cut["backward_role_normalized"]])
    return np.asarray(vals, dtype=np.float64)


def delta_vector(profile: dict[str, Any], composition: str = "product") -> np.ndarray:
    vals = []
    for op in profile["per_operation"]:
        vals.extend([op["delta_cap_effective_rank"], op["pi_role_offdiagonal_mass"], op["role_coverage_5pct"]])
        if composition == "product":
            vals.append(op["delta_product"])
        elif composition == "min":
            vals.append(op["delta_min"])
        elif composition == "mean":
            vals.append(op["delta_normalized_mean"])
        else:
            raise ValueError(composition)
    return np.asarray(vals, dtype=np.float64)


def r_vector(profile: dict[str, Any], include_persistence: bool = True) -> np.ndarray:
    vals = []
    for op in profile["per_operation"]:
        vals.extend(op["r_cycle_candidate"])
        if include_persistence:
            vals.extend(op["persistence_control_profile"])
    return np.asarray(vals, dtype=np.float64)


def source_content_feature_vector(source_row: dict[str, Any]) -> np.ndarray:
    # Preserve the existing Content-Typed 12/13 Lambda proxy exactly.
    cells = {(c["operation"], c["source_address"]): c for c in source_row["cells"]}
    vals = []
    for op in ("REVOKE", "SUBSTITUTE"):
        for address in ADDRESSES:
            effects = cells[(op, address)]["content_effect_kernel"]
            for intervention in ("ZERO_SUFFIX", "HEAD_ROTATE"):
                rec = effects[intervention]
                vals.append(float(rec["delta_margin_donor_vs_original"]))
                vals.append(float(rec["delta_margin_none_vs_original"]))
    return np.asarray(vals, dtype=np.float64)


def target_vector(source_row: dict[str, Any]) -> tuple[str, ...]:
    cells = {(c["operation"], c["source_address"]): c for c in source_row["cells"]}
    return tuple(str(cells[(op, a)]["behavior"]["outcome"]) for op in OPS for a in ADDRESSES)


def _collision_metrics(labels: list[str], targets: list[str]) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(labels):
        groups[x].append(i)
    pairs = 0
    fibers = 0
    nontrivial = 0
    systems_nontrivial = 0
    for idxs in groups.values():
        if len(idxs) > 1:
            nontrivial += 1
            systems_nontrivial += len(idxs)
        counts: dict[str, int] = defaultdict(int)
        for i in idxs:
            counts[targets[i]] += 1
        n = len(idxs)
        cp = n * (n - 1) // 2 - sum(v * (v - 1) // 2 for v in counts.values())
        if cp:
            fibers += 1
            pairs += cp
    n = len(labels)
    return {
        "n_fibers": len(groups),
        "n_collision_fibers": fibers,
        "n_collision_pairs": pairs,
        "n_nontrivial_fibers": nontrivial,
        "systems_in_nontrivial_fibers": systems_nontrivial,
        "nontrivial_coverage": systems_nontrivial / n if n else 0.0,
    }



def _entropy_nats(labels: list[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for x in labels:
        counts[str(x)] += 1
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c)


def _mutual_information_nats(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        raise ValueError("partition length mismatch")
    n = len(a)
    if n == 0:
        return 0.0
    ca: dict[str, int] = defaultdict(int)
    cb: dict[str, int] = defaultdict(int)
    cab: dict[tuple[str, str], int] = defaultdict(int)
    for x, y in zip(a, b):
        x = str(x); y = str(y)
        ca[x] += 1; cb[y] += 1; cab[(x, y)] += 1
    out = 0.0
    for (x, y), nij in cab.items():
        out += (nij / n) * math.log((n * nij) / (ca[x] * cb[y]))
    return float(out)


def _expected_mi_nats(a: list[str], b: list[str]) -> float:
    """Exact expected MI under the fixed-marginal permutation model."""
    if len(a) != len(b):
        raise ValueError("partition length mismatch")
    n = len(a)
    if n <= 1:
        return 0.0
    ca: dict[str, int] = defaultdict(int)
    cb: dict[str, int] = defaultdict(int)
    for x in a:
        ca[str(x)] += 1
    for y in b:
        cb[str(y)] += 1
    emi = 0.0
    for ai in ca.values():
        for bj in cb.values():
            lo = max(1, ai + bj - n)
            hi = min(ai, bj)
            denom = math.comb(n, bj)
            for nij in range(lo, hi + 1):
                prob = (math.comb(ai, nij) * math.comb(n - ai, bj - nij)) / denom
                emi += prob * (nij / n) * math.log((n * nij) / (ai * bj))
    return float(emi)


def adjusted_mutual_information(a: list[str], b: list[str]) -> float:
    """AMI with arithmetic-mean entropy normalization (Vinh et al. convention)."""
    if len(a) != len(b):
        raise ValueError("partition length mismatch")
    if not a:
        return 0.0
    mi = _mutual_information_nats(a, b)
    emi = _expected_mi_nats(a, b)
    ha = _entropy_nats(a)
    hb = _entropy_nats(b)
    denom = 0.5 * (ha + hb) - emi
    if abs(denom) <= 1e-15:
        return 1.0 if abs(mi - emi) <= 1e-15 and ha == hb else 0.0
    return float((mi - emi) / denom)


def partition_quality(labels: list[str], targets: list[str], *, permutations: int = 500, seed: int = 15015) -> dict[str, Any]:
    """Target-facing evaluation only; never used to define roles or raw component coordinates."""
    if len(labels) != len(targets):
        raise ValueError("partition length mismatch")
    n = len(labels)
    ht = _entropy_nats(targets)
    hc = _entropy_nats(labels)
    mi = _mutual_information_nats(labels, targets)
    hcond = max(0.0, ht - mi)
    suff_loss = (hcond / ht) if ht > 1e-15 else 0.0
    ami = adjusted_mutual_information(labels, targets)
    rng = np.random.default_rng(seed)
    target_arr = np.asarray(targets, dtype=object)
    perm_ami = []
    for _ in range(int(permutations)):
        perm = target_arr[rng.permutation(n)].tolist()
        perm_ami.append(adjusted_mutual_information(labels, perm))
    perm95 = float(np.quantile(np.asarray(perm_ami, dtype=np.float64), 0.95)) if perm_ami else 0.0
    p_ge = (1 + sum(float(x) >= ami - 1e-15 for x in perm_ami)) / (len(perm_ami) + 1)
    by_coord = []
    try:
        decoded = [json.loads(t) for t in targets]
        if decoded and all(isinstance(x, list) for x in decoded):
            width = min(len(x) for x in decoded)
            for j in range(width):
                coord = [str(x[j]) for x in decoded]
                by_coord.append({"target_index": j, "ami": adjusted_mutual_information(labels, coord)})
    except Exception:
        by_coord = []
    n_fibers = len(set(labels))
    return {
        "adjusted_mutual_information": float(ami),
        "ami_permutation_95": perm95,
        "ami_permutation_p_ge": float(p_ge),
        "ami_exceeds_permutation_95": bool(ami > perm95),
        "sufficiency_loss_H_target_given_descriptor_over_H_target": float(suff_loss),
        "target_entropy_bits": float(ht / math.log(2.0)) if ht else 0.0,
        "descriptor_partition_entropy_bits": float(hc / math.log(2.0)) if hc else 0.0,
        "descriptor_effective_fibers": float(math.exp(hc)) if hc else 1.0,
        "descriptor_fiber_fraction": float(n_fibers / n) if n else 0.0,
        "ami_by_target_coordinate": by_coord,
        "permutations": int(permutations),
        "permutation_seed": int(seed),
    }


def _quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    qs = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    raw = [float(np.quantile(values, q)) for q in qs]
    out = []
    for x in raw:
        if not out or x > out[-1] + 1e-12:
            out.append(x)
    return out


def _labels_for_specs(Z: np.ndarray, specs: tuple[tuple[int, int], ...], edge_map: dict[tuple[int, int], list[float]]) -> list[str]:
    out = []
    for row in Z:
        parts = []
        for j, bins in specs:
            edges = edge_map[(j, bins)]
            parts.append(str(int(np.searchsorted(np.asarray(edges), row[j], side="right"))))
        out.append("|".join(parts) if parts else "ALL")
    return out


def compact_search(X: np.ndarray, targets: list[str], max_features: int = 5) -> dict[str, Any]:
    """Small exploratory target-informed beam search; edges remain target-blind."""
    if X.ndim != 2 or X.shape[0] != len(targets):
        raise ValueError("shape mismatch")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    bins_grid = (2, 3, 4)
    edge_map = {(j, b): _quantile_edges(Z[:, j], b) for j in range(Z.shape[1]) for b in bins_grid}
    beam: list[tuple[tuple[int, int], ...]] = [tuple()]
    best = None
    seen = set(beam)
    for depth in range(1, max_features + 1):
        cand = []
        for specs in beam:
            used = {j for j, _ in specs}
            for j in range(Z.shape[1]):
                if j in used:
                    continue
                for b in bins_grid:
                    nxt = tuple(sorted(specs + ((j, b),)))
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    labels = _labels_for_specs(Z, nxt, edge_map)
                    m = _collision_metrics(labels, targets)
                    # Among functionally admissible partitions, prefer the coarsest
                    # partition before representation detail.  This explicitly
                    # penalizes quasi-injective solutions.
                    rank = (m["n_collision_pairs"], m["n_fibers"], len(nxt), -m["nontrivial_coverage"])
                    rec = {"specs": nxt, "metrics": m, "rank": rank}
                    cand.append(rec)
                    if m["n_collision_pairs"] == 0 and m["nontrivial_coverage"] >= 0.20:
                        if best is None or rank < best["rank"]:
                            best = rec
        cand.sort(key=lambda x: x["rank"])
        beam = [x["specs"] for x in cand[:64]]
        if best is not None:
            break
    if best is None:
        labels = [stable([round(float(v), 8) for v in row]) for row in Z]
        m = _collision_metrics(labels, targets)
        return {
            "status": "NO_COMPACT_CLOSED_QUOTIENT",
            "metrics": m,
            "partition_quality": partition_quality(labels, targets),
            "partition_labels": labels,
            "note": "Fallback quality is reported for the full rounded continuous partition and is diagnostic only.",
        }
    specs_json = []
    for j, b in best["specs"]:
        specs_json.append({"feature_index": int(j), "bins": int(b), "edges": edge_map[(j, b)]})
    labels = _labels_for_specs(Z, best["specs"], edge_map)
    return {
        "status": "COMPACT_CLOSED_QUOTIENT_FOUND",
        "selected_feature_count": len(specs_json),
        "selected_features": specs_json,
        "metrics": best["metrics"],
        "partition_quality": partition_quality(labels, targets),
        "partition_labels": labels,
        "standardizer": {"mu": mu.tolist(), "sd": sd.tolist()},
    }
