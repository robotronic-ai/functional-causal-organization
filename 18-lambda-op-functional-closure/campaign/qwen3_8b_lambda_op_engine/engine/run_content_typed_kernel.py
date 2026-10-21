#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

import qwen_v427_runtime as rt
from co_u_causal_kernel import PRIMARY_LAYERS, apply_self_intervention
from content_typed_kernel import ADDRESSES, CONTENT_OPS, INTERVENTIONS, OPS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def centered_effect(base_lp: dict[str, float], pert_lp: dict[str, float]) -> dict[str, object]:
    delta = {k: float(pert_lp[k]) - float(base_lp[k]) for k in ("ORIGINAL", "NONE", "DONOR")}
    return {
        "delta_logprob": delta,
        "delta_margin_donor_vs_original": float(delta["DONOR"] - delta["ORIGINAL"]),
        "delta_margin_none_vs_original": float(delta["NONE"] - delta["ORIGINAL"]),
    }


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
            presentation = rt.presentation(bindings)
            prefix_cache, prefix_ids = rt.build_prefix(model, tok, bindings, args.device)
            cells = []

            for operation in OPS:
                for address in ADDRESSES:
                    instruction = rt.operation_instruction(operation, address, donors[address])
                    control_cache, suffix = rt.control_cache(
                        model, tok, prefix_cache, prefix_ids, presentation, instruction, args.device
                    )
                    control_start = len(prefix_ids)
                    behavior = rt.probe_candidate_scores(
                        model,
                        tok,
                        control_cache,
                        presentation,
                        instruction,
                        address,
                        bindings[address],
                        donors[address],
                        args.device,
                    )

                    content_effect = None
                    if operation in CONTENT_OPS:
                        content_effect = {}
                        for intervention in INTERVENTIONS:
                            perturbed_cache = apply_self_intervention(
                                control_cache,
                                start=control_start,
                                kind=intervention,
                                layers=PRIMARY_LAYERS,
                            )
                            perturbed = rt.probe_candidate_scores(
                                model,
                                tok,
                                perturbed_cache,
                                presentation,
                                instruction,
                                address,
                                bindings[address],
                                donors[address],
                                args.device,
                            )
                            content_effect[intervention] = centered_effect(behavior["logprob"], perturbed["logprob"])
                            del perturbed_cache

                    cells.append({
                        "operation": operation,
                        "source_address": address,
                        "instruction_suffix_tokens": len(suffix),
                        "behavior": {
                            "outcome": behavior["outcome"],
                            "logprob": behavior["logprob"],
                            "token_counts": behavior["token_counts"],
                        },
                        "content_effect_kernel": content_effect,
                    })
                    del control_cache

            rows.append({
                "unit": int(episode["unit"]),
                "bindings": bindings,
                "donor_values": donors,
                "cells": cells,
            })
            del prefix_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            skips.append({"unit": int(episode.get("unit", -1)), "error": repr(exc)})

        elapsed = (time.perf_counter() - started) / 60.0
        print(f"[content-kernel] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m")

    result = {
        "schema": "co-qwen3-8b-content-typed-causal-effect-panel-v1",
        "stage": "exploratory_content_typed_kernel",
        "schedule_file": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "executed_schedule_size": len(schedule),
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "dtype": args.dtype,
        "operation_algebra": list(OPS),
        "content_effect_operations": list(CONTENT_OPS),
        "addresses": list(ADDRESSES),
        "experimental_interventions": list(INTERVENTIONS),
        "intervention_layers": list(PRIMARY_LAYERS),
        "content_effect_definition": "candidate-sequence log-probability change under an internal intervention, centered against ORIGINAL; baseline winner is excluded from descriptor construction",
        "functional_target": "ordered 12-cell behavior vector: operation x address",
        "descriptor_uses_baseline_outcome": False,
        "descriptor_uses_baseline_absolute_logprob": False,
        "behavior_used_only_as_closure_target_and_exploratory_feature_selection": True,
        "n_kept": len(rows),
        "n_skipped": len(skips),
        "rows": rows,
        "skips": skips,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO-U CONTENT-TYPED KERNEL COMPLETE]", out)
    print("[SHA256]", sha256(out))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
