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
from co_u_causal_kernel import PRIMARY_LAYERS, apply_self_intervention, compare_probe_trajectory, probe_token_ids, run_probe_trajectory
from cross_target_channel import ADDRESSES, HORIZONS, INTERVENTION, OPS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    probe_ids = {address: probe_token_ids(tok, address) for address in ADDRESSES}

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
                for source in ADDRESSES:
                    instruction = rt.operation_instruction(operation, source, donors[source])
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
                        source,
                        bindings[source],
                        donors[source],
                        args.device,
                    )

                    perturbed_cache = apply_self_intervention(
                        control_cache, start=control_start, kind=INTERVENTION, layers=PRIMARY_LAYERS
                    )
                    cross = {}
                    for target in ADDRESSES:
                        baseline_traj = run_probe_trajectory(
                            model, control_cache, probe_ids[target], args.device, horizons=max(HORIZONS)
                        )
                        perturbed_traj = run_probe_trajectory(
                            model, perturbed_cache, probe_ids[target], args.device, horizons=max(HORIZONS)
                        )
                        cross[target] = {
                            INTERVENTION: compare_probe_trajectory(baseline_traj, perturbed_traj)
                        }
                        del baseline_traj, perturbed_traj

                    cells.append({
                        "operation": operation,
                        "source_address": source,
                        "instruction_suffix_tokens": len(suffix),
                        "behavior": {
                            "outcome": behavior["outcome"],
                            "logprob": behavior["logprob"],
                            "token_counts": behavior["token_counts"],
                        },
                        "cross_target_kernel": cross,
                    })
                    del control_cache, perturbed_cache

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
        print(f"[cross-channel] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m")

    result = {
        "schema": "co-qwen3-8b-cross-target-active-channel-panel-v1",
        "stage": "exploratory_cross_target_channel",
        "schedule_file": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "executed_schedule_size": len(schedule),
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "dtype": args.dtype,
        "operation_algebra": list(OPS),
        "source_addresses": list(ADDRESSES),
        "target_addresses": list(ADDRESSES),
        "experimental_intervention": INTERVENTION,
        "intervention_layers": list(PRIMARY_LAYERS),
        "horizons": list(HORIZONS),
        "functional_target": "diagonal 12-cell behavior vector: operation x source address",
        "descriptor_target_blind": True,
        "behavior_used_only_as_closure_target": True,
        "n_kept": len(rows),
        "n_skipped": len(skips),
        "rows": rows,
        "skips": skips,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[CO-U CROSS-TARGET CHANNEL COMPLETE]", out)
    print("[SHA256]", sha256(out))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
