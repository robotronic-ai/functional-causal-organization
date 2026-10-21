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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
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
            pres = rt.presentation(bindings)
            prefix_cache, prefix_ids = rt.build_prefix(model, tok, bindings, args.device)
            cells = []
            for operation in rt.OPS:
                for address in rt.ADDRESSES:
                    instruction = rt.operation_instruction(operation, address, donors[address])
                    cache, suffix = rt.control_cache(model, tok, prefix_cache, prefix_ids, pres, instruction, args.device)
                    internal = rt.kv_suffix_features(cache, len(prefix_ids))
                    behavior = rt.probe_candidate_scores(
                        model, tok, cache, pres, instruction, address,
                        bindings[address], donors[address], args.device,
                    )
                    behavior["expected_outcome"] = rt.expected_outcome(operation)
                    behavior["correct"] = behavior["outcome"] == behavior["expected_outcome"]
                    cells.append({
                        "operation": operation,
                        "address": address,
                        "instruction_suffix_tokens": len(suffix),
                        "internal": internal,
                        "behavior": behavior,
                    })
                    del cache
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
        print(f"[validation] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} elapsed={elapsed:.1f}m")

    result = {
        "schema": "co-qwen3-8b-lambda-op-candidate-validation-panel-v1",
        "stage": "validation",
        "schedule_file": schedule_path.name,
        "schedule_sha256": sha256(schedule_path),
        "executed_schedule_size": len(schedule),
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "dtype": args.dtype,
        "operation_algebra": list(rt.OPS),
        "primary_layers": list(rt.PRIMARY_LAYERS),
        "reference_layers": list(rt.REFERENCE_LAYERS),
        "n_kept": len(rows),
        "n_skipped": len(skips),
        "rows": rows,
        "skips": skips,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[VALIDATION COMPLETE]", out)
    print("[SHA256]", sha256(out))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
