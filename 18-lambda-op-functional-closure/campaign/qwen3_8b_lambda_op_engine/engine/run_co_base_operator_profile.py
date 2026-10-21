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
import runtime as base
from co_base_operator_profile import ADDRESSES, HORIZONS, LOCAL_INTERVENTIONS, OPS, control_clause_spans
from co_u_causal_kernel import compare_probe_trajectory, probe_token_ids, run_probe_trajectory

PRIMARY_LAYERS = (21, 22, 23)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def apply_local_intervention(cache, start: int, end: int, kind: str):
    c = base.clone_cache(cache)
    if end <= start:
        raise ValueError((start, end))
    for li in PRIMARY_LAYERS:
        k, v = base.layer_kv(c, int(li))
        if start < 0 or end > k.shape[-2]:
            raise RuntimeError(f"invalid localized span {(start, end)} layer={li} seq={k.shape[-2]}")
        ks = k[..., start:end, :].clone()
        vs = v[..., start:end, :].clone()
        if kind == "ZERO_LOCAL":
            k[..., start:end, :].zero_()
            v[..., start:end, :].zero_()
        elif kind == "HEAD_ROTATE_LOCAL":
            if k.ndim != 4 or v.ndim != 4 or k.shape[1] < 2:
                raise RuntimeError(f"unexpected KV shape {tuple(k.shape)}")
            perm = torch.roll(torch.arange(k.shape[1], device=k.device), shifts=1)
            k[..., start:end, :].copy_(ks[:, perm, :, :])
            v[..., start:end, :].copy_(vs[:, perm, :, :])
        else:
            raise ValueError(kind)
    return c


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
            for op in OPS:
                for source in ADDRESSES:
                    instruction = rt.operation_instruction(op, source, donors[source])
                    control_cache, suffix = rt.control_cache(model, tok, prefix_cache, prefix_ids, presentation, instruction, args.device)
                    spans = control_clause_spans(tok, suffix)
                    rel_start, rel_end = spans[source]
                    abs_start = len(prefix_ids) + rel_start
                    abs_end = len(prefix_ids) + rel_end
                    reference_by_target = {}
                    for target in ADDRESSES:
                        tids = probe_token_ids(tok, target)
                        reference_by_target[target] = run_probe_trajectory(
                            model, control_cache, tids, args.device, horizons=max(HORIZONS)
                        )
                    localized = {}
                    for intervention in LOCAL_INTERVENTIONS:
                        perturbed_cache = apply_local_intervention(control_cache, abs_start, abs_end, intervention)
                        by_target = {}
                        for target in ADDRESSES:
                            tids = probe_token_ids(tok, target)
                            pert_traj = run_probe_trajectory(model, perturbed_cache, tids, args.device, horizons=max(HORIZONS))
                            by_target[target] = compare_probe_trajectory(reference_by_target[target], pert_traj)
                            del pert_traj
                        localized[intervention] = by_target
                        del perturbed_cache
                    del reference_by_target
                    cells.append({
                        "operation": op,
                        "source_address": source,
                        "control_suffix_tokens": len(suffix),
                        "localized_clause_span_relative": [rel_start, rel_end],
                        "localized_clause_span_absolute": [abs_start, abs_end],
                        "localized_routing": localized,
                    })
                    del control_cache
            rows.append({
                "unit": int(episode["unit"]),
                "source_unit": int(episode["source_unit"]),
                "bindings": bindings,
                "donor_values": donors,
                "cells": cells,
            })
            del prefix_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            skips.append({"unit": int(episode.get("unit", -1)), "source_unit": int(episode.get("source_unit", -1)), "error": repr(exc)})
        elapsed = (time.perf_counter() - started) / 60.0
        print(f"[co-base-op] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m")

    result = {
        "schema": "co-qwen3-8b-base-operator-routing-panel-v1",
        "stage": "exploratory_base_operator_component_audit",
        "schedule_file": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "executed_schedule_size": len(schedule),
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "dtype": args.dtype,
        "operation_algebra": list(OPS),
        "addresses": list(ADDRESSES),
        "experimental_interventions": list(LOCAL_INTERVENTIONS),
        "intervention_layers": list(PRIMARY_LAYERS),
        "horizons": list(HORIZONS),
        "localization": "token-span-localized intervention on the named CONTROL_MAP clause",
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
    print("[CO BASE-OP ROUTING PANEL COMPLETE]", out)
    print("[SHA256]", sha256(out))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
