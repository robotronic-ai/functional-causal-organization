from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fca_bench
from fca_bench import REGISTRY
from fca_bench.adapters import adapter_from_config, load_config
from fca_bench.core import Result
from fca_bench.progress import expected_model_calls
from fca_bench.run import aggregate, render

EXPECTED_PROTOCOL = "0.2.7"
EXPECTED_RUNNER = "FCA027_GPU_R6"
DEFAULT_CONFIG = str(ROOT / "config.qwen3_8b_gpu8.json")
ALLOWED_TESTS = ["T1", "T2", "T3", "B1", "C3", "R1", "R3", "D1", "L2", "U3", "I3", "E1"]
STRUCTURAL_ABSENT_ON_BASELINE = {"R1", "D1", "L2", "U3", "I3"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _result_from_dict(obj: dict) -> Result:
    return Result(
        instance_id=obj["instance_id"],
        dim=obj["dim"],
        test=obj["test"],
        score=float(obj["score"]),
        raw_score=float(obj.get("raw_score", obj["score"])),
        valid=bool(obj.get("valid", True)),
        status=str(obj.get("status", "executed")),
        details=dict(obj.get("details", {})),
    )


def _parse_tests(value: str) -> list[str]:
    tests = [x.strip().upper() for x in value.split(",") if x.strip()]
    if not tests:
        raise ValueError("--tests must contain at least one test code.")
    unknown = [x for x in tests if x not in ALLOWED_TESTS]
    if unknown:
        raise ValueError(
            "This locked runner only accepts the baseline test set. Unknown/not allowed: "
            + ", ".join(unknown)
        )
    if len(set(tests)) != len(tests):
        raise ValueError("--tests contains duplicate test codes.")
    return tests


def _validate_runner(config: dict, cwd: Path) -> None:
    if fca_bench.__version__ != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"Wrong FCA protocol imported: {fca_bench.__version__!r}; expected {EXPECTED_PROTOCOL!r}."
        )
    if getattr(fca_bench, "__runner_id__", None) != EXPECTED_RUNNER:
        raise RuntimeError(
            f"Wrong runner build imported: {getattr(fca_bench, '__runner_id__', None)!r}; "
            f"expected {EXPECTED_RUNNER!r}."
        )
    if config.get("required_runner_id") != EXPECTED_RUNNER:
        raise RuntimeError(
            f"Config required_runner_id must be {EXPECTED_RUNNER!r}; "
            f"got {config.get('required_runner_id')!r}."
        )
    if not bool(config.get("strict_gpu", False)):
        raise RuntimeError("This campaign requires strict_gpu=true.")
    if str(config.get("quantization", "")).lower() not in {"bnb8", "8bit", "int8"}:
        raise RuntimeError("This runner requires quantization='bnb8'.")
    if int(config.get("e1_samples", 0)) < 20:
        raise RuntimeError("Publication campaign requires e1_samples >= 20.")
    model_path = Path(str(config.get("model_path", "")))
    if not model_path.is_absolute():
        model_path = (cwd / model_path).resolve()
    if not model_path.exists():
        raise RuntimeError(f"Model directory does not exist: {model_path}")


def _preflight_only(config: dict) -> int:
    try:
        import torch
        import transformers
    except Exception as exc:
        print(f"PRECHECK FAILED: core dependency import failed: {exc}", file=sys.stderr)
        return 2
    try:
        import bitsandbytes
        bnb_version = getattr(bitsandbytes, "__version__", "unknown")
    except Exception as exc:
        print(
            "PRECHECK FAILED: bitsandbytes is required for the GPU-only 8-bit campaign.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print("PRECHECK FAILED: CUDA is not available.", file=sys.stderr)
        return 2
    gpu_index = int(config.get("gpu_index", 0))
    if gpu_index >= torch.cuda.device_count():
        print(
            f"PRECHECK FAILED: cuda:{gpu_index} requested but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible.",
            file=sys.stderr,
        )
        return 2
    props = torch.cuda.get_device_properties(gpu_index)
    print(f"Protocol version: {fca_bench.__version__}")
    print(f"Runner ID: {fca_bench.__runner_id__}")
    print(f"Runner file: {Path(__file__).resolve()}")
    print(f"Package file: {Path(fca_bench.__file__).resolve()}")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"bitsandbytes: {bnb_version}")
    print(f"GPU: cuda:{gpu_index} {props.name}")
    print(f"GPU VRAM: {props.total_memory / (1024**3):.2f} GiB")
    print("Mode: GPU-only, bitsandbytes 8-bit, CPU/disk offload forbidden")
    print("Preflight PASSED. Model weights were not loaded.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="FCA-Bench v0.2.7 Qwen3-8B GPU-only baseline runner R6."
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--tests",
        required=False,
        default=",".join(ALLOWED_TESTS),
        help="Comma-separated subset of the locked 12-test baseline.",
    )
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=False, default="results/FCA027_GPU_R6_results.json")
    ap.add_argument("--fresh", action="store_true", help="Archive an existing checkpoint and start again.")
    ap.add_argument("--preflight", action="store_true", help="Validate runtime and GPU without loading weights.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cwd = Path.cwd()
    config = load_config(args.config)
    _validate_runner(config, cwd)
    selected_tests = _parse_tests(args.tests)

    if args.n < 20:
        raise RuntimeError("This publication runner refuses n < 20.")
    if args.preflight:
        return _preflight_only(config)

    output = Path(args.output)
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.json")
    live_progress = output.with_suffix(output.suffix + ".progress.json")

    completed: dict[tuple[str, str], Result] = {}
    if checkpoint.exists() and not args.fresh:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != EXPECTED_PROTOCOL or payload.get("runner_id") != EXPECTED_RUNNER:
            raise RuntimeError("Checkpoint belongs to a different protocol/runner build. Use --fresh after archiving it.")
        if payload.get("selected_tests") != selected_tests:
            raise RuntimeError(
                "Checkpoint selected_tests do not match this command. Use a different --output or --fresh."
            )
        if int(payload.get("n_requested", -1)) != args.n or int(payload.get("seed", -1)) != args.seed:
            raise RuntimeError("Checkpoint n/seed do not match this command. Use a different --output or --fresh.")
        for obj in payload.get("instances", []):
            r = _result_from_dict(obj)
            completed[(r.test, r.instance_id)] = r
        print(f"[RESUME] loaded {len(completed)} completed instances from {checkpoint}", flush=True)
    elif checkpoint.exists() and args.fresh:
        archived = checkpoint.with_suffix(checkpoint.suffix + f".disabled-{int(time.time())}")
        checkpoint.replace(archived)
        print(f"[FRESH] previous checkpoint moved to {archived}", flush=True)

    # Estimate progress before model loading. This uses the known standard-transformer baseline capabilities.
    estimate_adapter = type(
        "EstimateAdapter",
        (),
        {
            "e1_samples": int(config["e1_samples"]),
            "supports": lambda self, _: False,
        },
    )()

    plan = []
    total_sequences_all = 0
    for code in selected_tests:
        for k in range(args.n):
            seed = args.seed + k
            inst = REGISTRY[code].generate(seed)
            calls = expected_model_calls(inst, estimate_adapter)
            if code in STRUCTURAL_ABSENT_ON_BASELINE:
                calls = 0
            plan.append((code, seed, inst, calls))
            total_sequences_all += calls

    remaining = [(c, s, i, calls) for c, s, i, calls in plan if (c, i.id) not in completed]
    remaining_sequences = sum(x[3] for x in remaining)
    total_instances = len(plan)

    print(
        f"FCA027 GPU R6: protocol={EXPECTED_PROTOCOL} runner={EXPECTED_RUNNER} "
        f"tests={','.join(selected_tests)} instances={total_instances} n={args.n}",
        flush=True,
    )
    print(
        f"Progress units: {total_sequences_all} generation sequences total; "
        f"{remaining_sequences} remain. E1 samples={config['e1_samples']} "
        f"E1 micro-batch={config.get('e1_batch_size', 1)}.",
        flush=True,
    )
    print(f"Checkpoint: {checkpoint}", flush=True)
    print(f"Live progress: {live_progress}", flush=True)

    adapter = adapter_from_config(config)
    adapter.configure_progress(total_calls=remaining_sequences, enabled=True)
    if hasattr(adapter, "set_progress_file"):
        adapter.set_progress_file(live_progress)

    results = list(completed.values())
    completed_count = len(completed)

    def _order_key(r: Result):
        return (selected_tests.index(r.test), int(r.instance_id.split("-")[-1]))

    def save_checkpoint() -> None:
        ordered = sorted(results, key=_order_key)
        _atomic_json(
            checkpoint,
            {
                "benchmark": "FCA-Bench",
                "benchmark_expansion": "Functional Causal Access Benchmark",
                "protocol_version": EXPECTED_PROTOCOL,
                "runner_id": EXPECTED_RUNNER,
                "campaign": "Qwen3-8B GPU-only bnb8 selected baseline tests",
                "system_id": getattr(adapter, "system_id", adapter.name),
                "runtime": adapter.runtime_metadata() if hasattr(adapter, "runtime_metadata") else {},
                "n_requested": args.n,
                "seed": args.seed,
                "selected_tests": selected_tests,
                "completed_instances": len(ordered),
                "total_instances": total_instances,
                "instances": [r.__dict__ for r in ordered],
            },
        )

    try:
        for code, seed, inst, _calls in remaining:
            completed_count += 1
            if hasattr(adapter, "set_progress_context"):
                adapter.set_progress_context(code, inst.id)
            print(
                f"[TEST {completed_count}/{total_instances}] START {code} seed={seed} instance={inst.id}",
                flush=True,
            )
            t0 = time.perf_counter()
            test = REGISTRY[code]
            try:
                records = test.run(inst, adapter)
                result = test.score(inst, records)
            except Exception as exc:
                if args.verbose:
                    traceback.print_exc()
                result = Result(
                    inst.id,
                    test.dim,
                    code,
                    0.0,
                    0.0,
                    valid=False,
                    status="invalid",
                    details={"error": type(exc).__name__, "message": str(exc)},
                )
            results.append(result)
            save_checkpoint()
            dt = time.perf_counter() - t0
            print(
                f"[TEST {completed_count}/{total_instances}] DONE  {code} seed={seed} "
                f"score={result.score:.3f} status={result.status} dt={dt:.2f}s checkpoint=SAVED",
                flush=True,
            )
    except KeyboardInterrupt:
        save_checkpoint()
        print("\n[INTERRUPTED] checkpoint saved. Rerun the same command to resume.", file=sys.stderr, flush=True)
        return 130

    ordered = sorted(results, key=_order_key)
    agg = aggregate(ordered)
    payload = {
        "benchmark": "FCA-Bench",
        "benchmark_expansion": "Functional Causal Access Benchmark",
        "protocol_version": EXPECTED_PROTOCOL,
        "runner_id": EXPECTED_RUNNER,
        "campaign": "Qwen3-8B GPU-only bnb8 selected baseline tests",
        "system_id": getattr(adapter, "system_id", adapter.name),
        "model": adapter.name,
        "runtime": adapter.runtime_metadata() if hasattr(adapter, "runtime_metadata") else {},
        "n_requested": args.n,
        "seed": args.seed,
        "selected_tests": selected_tests,
        **agg,
        "instances": [r.__dict__ for r in ordered],
    }
    _atomic_json(output, payload)
    save_checkpoint()
    print("\n" + render(agg, adapter.name), flush=True)
    print(f"\nFINAL JSON: {output}", flush=True)
    print(f"Checkpoint retained for provenance: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
