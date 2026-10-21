#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


POSITIVE_TERMS = {
    "persistent": (
        "memory",
        "recurrent",
        "state",
        "stream",
        "kv_cache",
        "paged",
        "sparse",
    ),
    "causal": (
        "attn",
        "attention",
        "fusion",
        "gate",
        "mhc",
        "router",
        "routing",
        "latent",
        "virtual",
    ),
}

HARD_EXCLUDE_TERMS = (
    "rope",
    "rotary",
    "norm",
    "embedding",
    "embed_tokens",
    "lm_head",
    "dropout",
)

HARD_EXCLUDE_CLASSES = (
    "ropecaching",
    "deepnorm",
    "rmsnorm",
    "layernorm",
    "embedding",
)


def shape_size(shape: list[int] | None) -> int:
    if not shape:
        return 0
    size = 1
    for value in shape:
        size *= int(value)
    return int(size)


def first_tensor_shape(record: dict[str, Any] | None, key: str) -> list[int] | None:
    if not record:
        return None
    for call in record.get("calls", []):
        summary = call.get(key)
        if isinstance(summary, dict):
            shape = summary.get("shape")
            if shape:
                return [int(value) for value in shape]
    return None


def is_hard_excluded(name: str, class_name: str | None) -> tuple[bool, str | None]:
    label = f"{name} {class_name or ''}".lower()
    class_lower = (class_name or "").lower()

    for term in HARD_EXCLUDE_TERMS:
        if term in label:
            return True, f"excluded_name_term:{term}"

    for term in HARD_EXCLUDE_CLASSES:
        if term in class_lower:
            return True, f"excluded_class:{term}"

    return False, None


def architecture_score(
    name: str,
    class_name: str | None,
    s_record: dict[str, Any] | None,
    f_record: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    label = f"{name} {class_name or ''}".lower()
    score = 0.0
    reasons: list[str] = []

    for term in POSITIVE_TERMS["persistent"]:
        if term in label:
            score += 5.0
            reasons.append(f"persistent:{term}")

    for term in POSITIVE_TERMS["causal"]:
        if term in label:
            score += 2.0
            reasons.append(f"causal:{term}")

    s_shape = first_tensor_shape(s_record, "output")
    f_shape = first_tensor_shape(f_record, "output")
    output_shape = s_shape or f_shape

    active_both = s_record is not None and f_record is not None
    if active_both:
        score += 1.0
        reasons.append("active_in_both_flows")

    if output_shape is None:
        score -= 4.0
        reasons.append("no_tensor_output")
    else:
        # Prefer actual batch/sequence/feature activations such as [B,T,D].
        if len(output_shape) == 3:
            score += 3.0
            reasons.append("rank3_activation")
        elif len(output_shape) == 4:
            score += 2.5
            reasons.append("rank4_activation")
        elif len(output_shape) == 2:
            score += 0.5
            reasons.append("rank2_activation")

        if output_shape[-1] >= 256:
            score += 1.0
            reasons.append("large_feature_axis")

        size = shape_size(output_shape)
        if size > 0:
            score += min(1.5, math.log10(size + 1.0) / 4.0)

    # Prefer modules that are not just leafless wrappers with no direct state.
    return score, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select causal-state candidate intervention sites from a complete "
            "Quantum Gamma inspection report."
        )
    )
    parser.add_argument(
        "--inspection",
        default="quantum_gamma_inspection_full.json",
    )
    parser.add_argument(
        "--output",
        default="quantum_gamma_site_selection_v2.json",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=60,
    )
    args = parser.parse_args()

    report = json.loads(Path(args.inspection).read_text())

    flow_probes = {
        item["flow"]: item
        for item in report.get("flow_probes", [])
    }

    s_events = flow_probes.get("s_attn", {}).get("hook_events", {})
    f_events = flow_probes.get("fused", {}).get("hook_events", {})

    metadata = {
        item["name"]: item
        for item in report.get("candidate_modules", [])
    }

    active_names = sorted(set(s_events) | set(f_events))

    selected = []
    excluded = []

    for name in active_names:
        meta = metadata.get(name, {})
        class_name = meta.get("class")

        excluded_flag, exclusion_reason = is_hard_excluded(name, class_name)
        if excluded_flag:
            excluded.append(
                {
                    "name": name,
                    "class": class_name,
                    "reason": exclusion_reason,
                }
            )
            continue

        score, reasons = architecture_score(
            name,
            class_name,
            s_events.get(name),
            f_events.get(name),
        )

        s_shape = first_tensor_shape(s_events.get(name), "output")
        f_shape = first_tensor_shape(f_events.get(name), "output")

        selected.append(
            {
                "name": name,
                "class": class_name,
                "architecture_score": float(score),
                "reasons": reasons,
                "active_in_s_attn": name in s_events,
                "active_in_fused": name in f_events,
                "s_attn_output_shape": s_shape,
                "fused_output_shape": f_shape,
                "direct_parameter_count": int(
                    meta.get("direct_parameter_count", 0)
                ),
                "direct_buffer_count": int(
                    meta.get("direct_buffer_count", 0)
                ),
            }
        )

    selected.sort(
        key=lambda item: (
            -item["architecture_score"],
            item["name"],
        )
    )
    selected = selected[: args.top]

    result = {
        "selection_version": 2,
        "selection_rule": (
            "Architecture-only selection after explicit exclusion of positional "
            "encoding caches, normalization layers, embeddings, and LM head. "
            "No behavioral target or Gamma value is used."
        ),
        "model_class": report.get("model_class"),
        "kv_cache_class": report.get("kv_cache_class"),
        "active_module_count": len(active_names),
        "hard_excluded_count": len(excluded),
        "selected_count": len(selected),
        "selected_sites": selected,
        "excluded_sites": excluded,
        "important_note": (
            "A high ranking is not a quotient certificate. Candidate sites must "
            "still pass matched-intervention transport, causal sufficiency, "
            "irreducibility, and independent holdout validation."
        ),
    }

    Path(args.output).write_text(json.dumps(result, indent=2))

    print("=== QUANTUM GAMMA SITE SHORTLIST V2 ===")
    print(f"Active modules: {len(active_names)}")
    print(f"Hard excluded: {len(excluded)}")
    print(f"Selected candidates: {len(selected)}")
    print()

    for rank, item in enumerate(selected[:30], start=1):
        reasons = ",".join(item["reasons"][:4])
        print(
            f"{rank:2d}. score={item['architecture_score']:.2f} "
            f"{item['name']} [{item['class']}] "
            f"s_attn={item['s_attn_output_shape']} "
            f"fused={item['fused_output_shape']} "
            f"reasons={reasons}"
        )

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
