#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import qwen_v427_runtime as rt
import runtime as base
from co_base_operator_profile import control_clause_spans
from run_co_base_operator_profile import apply_local_intervention
from formula_measurement17 import (
    ADDRESSES,
    AUX_CONTEXTS,
    DELTA_Q,
    OPS,
    P_SIZE,
    Q_INTERVENTIONS,
    RETURN_CONTEXTS,
    entropy_bits,
    mutual_information_bits,
    numeric_support,
    ordered_top2,
    reentry_metrics,
    response_labels,
    tagged_role_mixture,
)

PRIMARY_LAYERS = (21, 22, 23)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _softmax_log_scores(scores: list[float]) -> list[float]:
    x = np.asarray(scores, dtype=np.float64)
    x = x - float(np.max(x))
    p = np.exp(x)
    p /= p.sum()
    return p.tolist()


def _probe_distribution(
    model,
    tok,
    cache,
    history_messages: list[dict[str, str]],
    address: str,
    support: list[int],
    device: str,
) -> dict[str, Any]:
    probe = base.SELECT_PROBE_TEMPLATE.format(handle=address)
    _, probe_suffix = base.split_append(
        tok,
        history_messages,
        [{"role": "user", "content": probe}],
        final_generation_prompt=True,
    )
    state = base.advance_cache(model, cache, probe_suffix, base.cache_seq_length(cache), device)
    full_prompt = history_messages + [{"role": "user", "content": probe}]
    texts = [str(int(v)) for v in support] + ["NONE"]
    scores: list[float] = []
    token_counts: list[int] = []
    for text in texts:
        tids = base.answer_token_ids(tok, full_prompt, text)
        token_counts.append(len(tids))
        if len(tids) == 1:
            lp = torch.log_softmax(state["logits"].float().reshape(-1), dim=-1)
            scores.append(float(lp[int(tids[0])].item()))
        else:
            scores.append(float(base.sequence_logprob(model, state, tids, device)))
    probs = _softmax_log_scores(scores)
    labels = response_labels()
    return {
        "labels": list(labels),
        "support_values": [int(v) for v in support],
        "logprob": {labels[i]: float(scores[i]) for i in range(len(labels))},
        "probability": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "probability_vector": probs,
        "ordered_top2": ordered_top2(probs, labels),
        "token_counts": {labels[i]: int(token_counts[i]) for i in range(len(labels))},
    }


def _build_control(model, tok, bindings: dict[str, int], donors: dict[str, int], operation: str, source: str, device: str):
    presentation = rt.presentation(bindings)
    prefix_cache, prefix_ids = rt.build_prefix(model, tok, bindings, device)
    instruction = rt.operation_instruction(operation, source, donors[source])
    control_cache, suffix = rt.control_cache(model, tok, prefix_cache, prefix_ids, presentation, instruction, device)
    history = [
        {"role": "user", "content": presentation},
        {"role": "user", "content": instruction},
    ]
    del prefix_cache
    return control_cache, prefix_ids, suffix, history


def _append_context(model, tok, cache, history: list[dict[str, str]], text: str, device: str):
    _, suffix = base.split_append(tok, history, [{"role": "user", "content": text}], final_generation_prompt=False)
    state = base.advance_cache(model, cache, suffix, base.cache_seq_length(cache), device)
    return state["cache"], history + [{"role": "user", "content": text}]


def _role_distributions(model, tok, cache, history, supports: dict[str, list[int]], roles, device):
    return {r: _probe_distribution(model, tok, cache, history, r, supports[r], device) for r in roles}


def _pre_span_max_diff(reference_cache, perturbed_cache, end: int) -> float:
    mx = 0.0
    for li in PRIMARY_LAYERS:
        rk, rv = base.layer_kv(reference_cache, li)
        pk, pv = base.layer_kv(perturbed_cache, li)
        for a, b in ((rk[..., :end, :], pk[..., :end, :]), (rv[..., :end, :], pv[..., :end, :])):
            if a.numel():
                mx = max(mx, float(torch.max(torch.abs(a.float() - b.float())).item()))
    return mx


def _gamma_delta_for_cell(model, tok, base_bindings, donors, supports, operation, source, device):
    other = [a for a in ADDRESSES if a != source]
    forward_contexts = []
    delta_channel = None
    native_control_for_reuse = None
    native_meta = None

    # q_B has two fixed, target-blind contexts: native and support-index 2 on the opposite side.
    for context_idx in (0, 1):
        conditionals = []
        for p_idx in range(P_SIZE):
            bindings = dict(base_bindings)
            bindings[source] = supports[source][p_idx]
            if context_idx == 1:
                for role in other:
                    bindings[role] = supports[role][2]
            cache, prefix_ids, suffix, history = _build_control(model, tok, bindings, donors, operation, source, device)
            roles_to_probe = ADDRESSES if context_idx == 0 else tuple(other)
            dists = _role_distributions(model, tok, cache, history, supports, roles_to_probe, device)
            target_mix = tagged_role_mixture({r: dists[r]["probability_vector"] for r in other}, other)
            conditionals.append(target_mix)
            if context_idx == 0:
                all_mix = tagged_role_mixture({r: dists[r]["probability_vector"] for r in ADDRESSES}, ADDRESSES)
                if delta_channel is None:
                    delta_channel = []
                delta_channel.append(all_mix)
            del cache
        mi = mutual_information_bits([1.0 / P_SIZE] * P_SIZE, conditionals)
        forward_contexts.append({
            "context_index": context_idx,
            "I_source_to_other_bits": float(mi),
            "channel": conditionals,
        })

    forward_j = float(np.mean([x["I_source_to_other_bits"] for x in forward_contexts]))

    backward_contexts = []
    for source_context_idx in (0, 1):
        conditionals = []
        for p_idx in range(P_SIZE):
            bindings = dict(base_bindings)
            bindings[source] = supports[source][0 if source_context_idx == 0 else 2]
            for role in other:
                bindings[role] = supports[role][p_idx]
            cache, prefix_ids, suffix, history = _build_control(model, tok, bindings, donors, operation, source, device)
            dist = _probe_distribution(model, tok, cache, history, source, supports[source], device)
            conditionals.append(dist["probability_vector"])
            del cache
        mi = mutual_information_bits([1.0 / P_SIZE] * P_SIZE, conditionals)
        backward_contexts.append({
            "context_index": source_context_idx,
            "I_other_to_source_bits": float(mi),
            "channel": conditionals,
        })

    backward_j = float(np.mean([x["I_other_to_source_bits"] for x in backward_contexts]))
    norm = math.log2(P_SIZE)
    x = min(1.0, forward_j / norm) if norm > 0 else 0.0
    y = min(1.0, backward_j / norm) if norm > 0 else 0.0
    cut_score = 0.5 * (x + y)
    beta = 0.0 if x + y <= 1e-15 else float(2.0 * min(x, y) / (x + y))

    if delta_channel is None:
        raise RuntimeError("missing Delta channel")
    h_p = entropy_bits(DELTA_Q)
    i_py = mutual_information_bits(DELTA_Q, delta_channel)
    kappa = h_p / math.log2(P_SIZE)
    delta_q = 0.0 if h_p <= 1e-15 else i_py / h_p
    delta_cap = i_py / math.log2(P_SIZE)

    gamma = {
        "operation": operation,
        "cut": f"{source}|" + "+".join(other),
        "source_role": source,
        "other_roles": other,
        "forward_J_bits": forward_j,
        "backward_J_bits": backward_j,
        "normalization_bits": norm,
        "forward_normalized": float(x),
        "backward_normalized": float(y),
        "cut_score_arithmetic_mean": float(cut_score),
        "directional_balance_beta": float(beta),
        "forward_contexts": forward_contexts,
        "backward_contexts": backward_contexts,
    }
    delta = {
        "operation": operation,
        "source_address": source,
        "P_star_size": P_SIZE,
        "q_P": list(DELTA_Q),
        "H_P_bits": float(h_p),
        "I_P_Y_bits": float(i_py),
        "kappa_q": float(kappa),
        "delta_q": float(delta_q),
        "Delta_cap": float(delta_cap),
        "readout": "tagged mixture over H1/H2/H3 finite role-value distributions",
        "channel": delta_channel,
    }
    return gamma, delta


def _r_lambda_for_cell(model, tok, bindings, donors, supports, operation, source, device):
    control_cache, prefix_ids, suffix, history = _build_control(model, tok, bindings, donors, operation, source, device)
    spans = control_clause_spans(tok, suffix)
    rel_start, rel_end = spans[source]
    abs_start = len(prefix_ids) + rel_start
    abs_end = len(prefix_ids) + rel_end

    source_signatures = []
    return_by_context: dict[str, list[list[float]]] = {f"RET{i}": [] for i in range(len(RETURN_CONTEXTS))}
    lambda_by_q = {}
    temporal_controls = {}

    for q_name in Q_INTERVENTIONS:
        if q_name == "CONTROL":
            qcache = base.clone_cache(control_cache)
        else:
            qcache = apply_local_intervention(control_cache, abs_start, abs_end, q_name)
        pre_diff = _pre_span_max_diff(control_cache, qcache, abs_start)
        temporal_controls[q_name] = {
            "pre_source_span_max_abs_cache_difference": float(pre_diff),
            "pass": bool(pre_diff <= 1e-12),
        }

        aux_sig = []
        for aux_text in AUX_CONTEXTS:
            acache, ahistory = _append_context(model, tok, qcache, history, aux_text, device)
            dist = _probe_distribution(model, tok, acache, ahistory, source, supports[source], device)
            aux_sig.append(dist["ordered_top2"])
            del acache
        source_signatures.append(aux_sig)

        for i, ret_text in enumerate(RETURN_CONTEXTS):
            rcache, rhistory = _append_context(model, tok, qcache, history, ret_text, device)
            dist = _probe_distribution(model, tok, rcache, rhistory, source, supports[source], device)
            return_by_context[f"RET{i}"].append(dist["probability_vector"])
            del rcache

        role_dists = _role_distributions(model, tok, qcache, history, supports, ADDRESSES, device)
        lambda_by_q[q_name] = {
            "joint_signature": [role_dists[a]["ordered_top2"] for a in ADDRESSES],
            "marginal_probability_vectors": {a: role_dists[a]["probability_vector"] for a in ADDRESSES},
        }
        del qcache

    r = reentry_metrics(list(Q_INTERVENTIONS), source_signatures, return_by_context)
    r.update({
        "operation": operation,
        "source_address": source,
        "auxiliary_context_count": len(AUX_CONTEXTS),
        "return_context_count": len(RETURN_CONTEXTS),
        "negative_temporal_storage_control": temporal_controls,
        "negative_temporal_storage_control_pass": all(x["pass"] for x in temporal_controls.values()),
        "control_note": "The negative control verifies that a localized cache intervention does not alter cached positions causally prior to the source clause. It is an adapted storage-level causal control, not the exact logit-position control used in the Pythia protocol.",
    })
    lam = {
        "operation": operation,
        "source_address": source,
        "interventions": list(Q_INTERVENTIONS),
        "joint_signature_by_intervention": {q: lambda_by_q[q]["joint_signature"] for q in Q_INTERVENTIONS},
        "marginal_probability_vectors_by_intervention": {q: lambda_by_q[q]["marginal_probability_vectors"] for q in Q_INTERVENTIONS},
        "joint_signature_definition": "ordered top-2 content labels for H1/H2/H3 retained jointly; no marginal pooling in the primary signature",
    }
    del control_cache
    return r, lam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    schedule_path = Path(args.schedule)
    schedule = json.loads(schedule_path.read_text())
    model, tok = rt.load_model(Path(args.model_dir), args.device, args.dtype)
    rows = []
    skips = []
    started = time.perf_counter()

    for i, episode in enumerate(schedule, 1):
        try:
            bindings = {str(k): int(v) for k, v in episode["bindings"].items()}
            donors = {str(k): int(v) for k, v in episode["donor_values"].items()}
            supports = {a: numeric_support(bindings[a], donors[a]) for a in ADDRESSES}
            gamma_cuts = []
            delta_cells = []
            r_cells = []
            lambda_cells = []
            for op in OPS:
                for source in ADDRESSES:
                    g, d = _gamma_delta_for_cell(model, tok, bindings, donors, supports, op, source, args.device)
                    r, lam = _r_lambda_for_cell(model, tok, bindings, donors, supports, op, source, args.device)
                    gamma_cuts.append(g)
                    delta_cells.append(d)
                    r_cells.append(r)
                    lambda_cells.append(lam)
            gamma_by_op = []
            for op in OPS:
                op_cuts = [x for x in gamma_cuts if x["operation"] == op]
                minimum = min(float(x["cut_score_arithmetic_mean"]) for x in op_cuts)
                argmin = [x["cut"] for x in op_cuts if abs(float(x["cut_score_arithmetic_mean"]) - minimum) <= 1e-15]
                gamma_by_op.append({"operation": op, "Gamma_min": minimum, "argmin_cuts": argmin})
            rows.append({
                "unit": int(episode["unit"]),
                "source_unit": int(episode["source_unit"]),
                "bindings": bindings,
                "donor_values": donors,
                "finite_value_supports": supports,
                "Gamma": {
                    "status": "MEASURED_ON_DECLARED_FINITE_BATTERY",
                    "definition": "directional interventional-information cut profile with arithmetic-mean directional aggregator",
                    "cuts": gamma_cuts,
                    "per_operation_minimum": gamma_by_op,
                },
                "Delta": {
                    "status": "MEASURED_ON_DECLARED_FINITE_BATTERY",
                    "definition": "primary profile (kappa_q, delta_q), Delta_cap=kappa_q*delta_q=I(P;Y)/log2(|P*|)",
                    "cells": delta_cells,
                },
                "R": {
                    "status": "ADAPTED_QWEN_REENTRY_MEASUREMENT",
                    "definition": "source quotient from auxiliary channel independent of return contexts; R=I(C;Y_ret|U)/H(C|U)",
                    "cells": r_cells,
                },
                "Lambda": {
                    "status": "FINITE_JOINT_TYPED_SIGNATURE",
                    "definition": "joint H1/H2/H3 typed response signature under declared internal interventions",
                    "cells": lambda_cells,
                },
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            skips.append({"unit": int(episode.get("unit", -1)), "source_unit": int(episode.get("source_unit", -1)), "error": repr(exc)})
        elapsed = (time.perf_counter() - started) / 60.0
        print(f"[formula17] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m")

    result = {
        "schema": "co-qwen3-8b-formula-measurement-17-panel-v1",
        "stage": "development_formula_measurement",
        "schedule_file": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "dtype": args.dtype,
        "operation_algebra": list(OPS),
        "generalization_level": "OP",
        "generalization_level_changed": False,
        "P_star_size": P_SIZE,
        "Delta_q": list(DELTA_Q),
        "q_interventions": list(Q_INTERVENTIONS),
        "auxiliary_contexts": list(AUX_CONTEXTS),
        "return_contexts": list(RETURN_CONTEXTS),
        "measurement_uses_behavioral_target": False,
        "system_operations_unchanged": True,
        "experimental_interventions_are_not_system_operations": True,
        "n_kept": len(rows),
        "n_skipped": len(skips),
        "rows": rows,
        "skips": skips,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO FORMULA MEASUREMENT 17 PANEL COMPLETE]", out)
    print("[SHA256]", sha256(out))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
