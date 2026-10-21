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
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["construction", "holdout"], required=True)
    ap.add_argument("--model-dir", default="./Qwen3-8B")
    ap.add_argument("--output-dir", default="./CO_v427_qwen_results_2026-09-26")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=0, help="Development smoke-test only. Zero means the full frozen schedule.")
    args = ap.parse_args()

    manifest_path = root / "data" / "v427_schedule_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    info = manifest["schedules"][args.stage]
    schedule_path = root / "data" / info["file"]
    if sha256(schedule_path) != info["sha256"]:
        raise RuntimeError(f"Frozen {args.stage} schedule SHA256 mismatch")
    schedule = json.loads(schedule_path.read_text())
    if args.limit:
        schedule = schedule[: args.limit]

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
                    cache, suffix = rt.control_cache(
                        model, tok, prefix_cache, prefix_ids, pres, instruction, args.device
                    )
                    internal = rt.kv_suffix_features(cache, len(prefix_ids))
                    cell = {
                        "operation": operation,
                        "address": address,
                        "instruction_suffix_tokens": len(suffix),
                        "internal": internal,
                    }
                    if args.stage == "holdout":
                        behavior = rt.probe_candidate_scores(
                            model,
                            tok,
                            cache,
                            pres,
                            instruction,
                            address,
                            bindings[address],
                            donors[address],
                            args.device,
                        )
                        behavior["expected_outcome"] = rt.expected_outcome(operation)
                        behavior["correct"] = behavior["outcome"] == behavior["expected_outcome"]
                        cell["behavior"] = behavior
                    cells.append(cell)
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
        print(
            f"[{args.stage}] {i}/{len(schedule)} kept={len(rows)} skipped={len(skips)} "
            f"elapsed={elapsed:.1f}m"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "co-v42.7-qwen-operator-panel-v1",
        "stage": args.stage,
        "schedule_file": info["file"],
        "schedule_sha256": info["sha256"],
        "full_frozen_schedule_size": int(info["n"]),
        "executed_schedule_size": len(schedule),
        "development_limit_used": bool(args.limit),
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
        "guardrail": "Construction contains no behavioral scores. Holdout behavior must not be opened before descriptor freezing.",
    }
    out_path = out_dir / f"V427_{args.stage.upper()}_PANEL.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[COMPLETE]", out_path)
    print("[SHA256]", sha256(out_path))
    if skips:
        raise SystemExit(f"Technical skips occurred: {len(skips)}")


if __name__ == "__main__":
    main()
