#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

import runtime as base

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
ADDRESSES = ("H1", "H2", "H3")
PRIMARY_LAYERS = (21, 22, 23)
REFERENCE_LAYERS = (30, 31, 32)


def load_model(model_dir: Path, device: str = "cuda", dtype: str = "bfloat16"):
    return base.load_model(model_dir, device, dtype)


def presentation(bindings: dict[str, int]) -> str:
    return base.select_presentation(bindings)


def operation_instruction(operation: str, address: str, donor_value: int) -> str:
    if operation not in OPS:
        raise ValueError(operation)
    if address not in ADDRESSES:
        raise ValueError(address)
    if operation == "ID":
        action = "ID"
    elif operation == "KEEP":
        action = "KEEP"
    elif operation == "REVOKE":
        action = "REVOKE"
    else:
        action = f"SUBSTITUTE={int(donor_value)}"
    others = [h for h in ADDRESSES if h != address]
    control = [f"{address}:{action}"] + [f"{h}:KEEP" for h in others]
    return (
        "CONTROL_MAP=" + ",".join(control) + ". "
        "Apply each control only to its named handle. "
        "ID means no transformation, KEEP means preserve, REVOKE means remove the active binding, "
        "and SUBSTITUTE means replace the active value with the stated donor value."
    )


def build_prefix(model, tok, bindings: dict[str, int], device: str):
    return base.build_native_prefix(model, tok, bindings, device)


def control_cache(model, tok, prefix_cache, prefix_ids: list[int], presentation_text: str, instruction: str, device: str):
    _, suffix = base.split_append(
        tok,
        [{"role": "user", "content": presentation_text}],
        [{"role": "user", "content": instruction}],
        final_generation_prompt=False,
    )
    c = base.clone_cache(prefix_cache)
    x = torch.tensor([suffix], dtype=torch.long, device=device)
    total = len(prefix_ids) + len(suffix)
    with torch.inference_mode():
        out = model(
            input_ids=x,
            attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
            past_key_values=c,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    return out.past_key_values, suffix


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.float()
    return {
        "rms": float(torch.sqrt(torch.mean(y * y)).item()),
        "abs_mean": float(torch.mean(torch.abs(y)).item()),
        "mean": float(torch.mean(y).item()),
    }


def kv_suffix_features(cache, prefix_len: int) -> dict[str, Any]:
    def per_layer(indices):
        rows = []
        for li in indices:
            k, v = base.layer_kv(cache, int(li))
            if prefix_len < 0 or prefix_len >= k.shape[-2]:
                raise RuntimeError(
                    f"invalid prefix_len={prefix_len} for layer={li} sequence={k.shape[-2]}"
                )
            ks = k[..., prefix_len:, :]
            vs = v[..., prefix_len:, :]
            rows.append({
                "layer": int(li),
                "suffix_tokens": int(ks.shape[-2]),
                "k": _tensor_stats(ks),
                "v": _tensor_stats(vs),
            })
        return rows

    primary = per_layer(PRIMARY_LAYERS)
    reference = per_layer(REFERENCE_LAYERS)

    def avg(rows, side: str, key: str):
        return sum(float(r[side][key]) for r in rows) / len(rows)

    p_k = avg(primary, "k", "rms")
    p_v = avg(primary, "v", "rms")
    r_k = avg(reference, "k", "rms")
    r_v = avg(reference, "v", "rms")
    eps = 1e-12
    corridor_ratio = 0.5 * (p_k / max(r_k, eps) + p_v / max(r_v, eps))
    return {
        "primary_layers": primary,
        "reference_layers": reference,
        "primary_k_rms": p_k,
        "primary_v_rms": p_v,
        "reference_k_rms": r_k,
        "reference_v_rms": r_v,
        "corridor_ratio": float(corridor_ratio),
        "suffix_tokens": int(primary[0]["suffix_tokens"]),
    }


def _probe_prompt(tok, presentation_text: str, instruction: str, address: str):
    probe = base.SELECT_PROBE_TEMPLATE.format(handle=address)
    prefix_msgs = [
        {"role": "user", "content": presentation_text},
        {"role": "user", "content": instruction},
    ]
    _, probe_suffix = base.split_append(
        tok,
        prefix_msgs,
        [{"role": "user", "content": probe}],
        final_generation_prompt=True,
    )
    return probe, prefix_msgs, probe_suffix


def probe_candidate_scores(
    model,
    tok,
    control_cache_,
    presentation_text: str,
    instruction: str,
    address: str,
    original_value: int,
    donor_value: int,
    device: str,
):
    probe, prefix_msgs, probe_suffix = _probe_prompt(tok, presentation_text, instruction, address)
    state = base.advance_cache(
        model,
        control_cache_,
        probe_suffix,
        base.cache_seq_length(control_cache_),
        device,
    )
    full_prompt = prefix_msgs + [{"role": "user", "content": probe}]
    candidates = {
        "ORIGINAL": str(int(original_value)),
        "NONE": "NONE",
        "DONOR": str(int(donor_value)),
    }
    scores = {}
    token_counts = {}
    for label, response in candidates.items():
        tids = base.answer_token_ids(tok, full_prompt, response)
        token_counts[label] = len(tids)
        scores[label] = base.sequence_logprob(model, state, tids, device)
    winner = max(scores, key=scores.get)
    return {
        "logprob": {k: float(v) for k, v in scores.items()},
        "token_counts": token_counts,
        "outcome": winner,
    }


def expected_outcome(operation: str) -> str:
    return {
        "ID": "ORIGINAL",
        "KEEP": "ORIGINAL",
        "REVOKE": "NONE",
        "SUBSTITUTE": "DONOR",
    }[operation]
