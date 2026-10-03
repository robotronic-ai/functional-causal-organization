from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from . import BEHAVIORAL_DIMENSIONS, CAUSAL_DIMENSIONS, DIMENSIONS, REGISTRY, __version__
from .adapters import adapter_from_config, load_config
from .core import Result
from .metrics import bootstrap_ci, mean
from .progress import expected_model_calls


def _seed_bank_hash(path: str | None) -> str | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_seed_bank(path: str | None) -> dict[str, list[int]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tests = payload.get("tests")
    if not isinstance(tests, dict):
        raise ValueError("Seed bank must contain an object named 'tests'.")
    out: dict[str, list[int]] = {}
    for code, values in tests.items():
        if code not in REGISTRY:
            continue
        if not isinstance(values, list) or not all(isinstance(x, int) for x in values):
            raise ValueError(f"Seed bank entry {code} must be a list of integers.")
        out[code] = values
    return out


def run_test(code, adapter, n, seed0=0, verbose=False, explicit_seeds=None, progress=True, instance_offset=0, instance_total=None):
    test = REGISTRY[code]
    out = []
    seeds = list(explicit_seeds) if explicit_seeds is not None else [seed0 + k for k in range(n)]
    for local_index, seed in enumerate(seeds, start=1):
        inst = test.generate(seed)
        if hasattr(adapter, "set_progress_context"):
            adapter.set_progress_context(code, inst.id)
        ordinal = instance_offset + local_index
        if progress:
            total_text = str(instance_total) if instance_total is not None else "?"
            print(f"[TEST {ordinal}/{total_text}] START {code} seed={seed} instance={inst.id}", flush=True)
        t0 = time.perf_counter()
        try:
            records = test.run(inst, adapter)
            result = test.score(inst, records)
        except Exception as exc:
            if verbose:
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
        out.append(result)
        if progress:
            dt = time.perf_counter() - t0
            print(f"[TEST {ordinal}/{total_text}] DONE  {code} seed={seed} score={result.score:.3f} status={result.status} dt={dt:.2f}s", flush=True)
    return out


# Tests whose aggregate is chance-corrected after averaging instances
# (instance scores are raw balanced accuracies; chance = 0.5).
CHANCE_CORRECTED_TESTS = {"T2"}


def _chance_corrected(x):
    return max(0.0, 2.0 * x - 1.0)


def aggregate(results):
    by_test = {}
    for r in results:
        by_test.setdefault(r.test, []).append(r)
    tests = {}
    for code, rs in sorted(by_test.items()):
        valid = [r for r in rs if r.valid]
        scores = [r.score for r in valid]
        executed = [r for r in valid if r.status == "executed"]
        structural = [r for r in valid if r.status in {"structural_absent", "structural_zero"}]
        lo, hi = bootstrap_ci(scores) if scores else (0, 0)
        mean_score = mean(scores)
        if code in CHANCE_CORRECTED_TESTS and scores:
            mean_score, lo, hi = _chance_corrected(mean_score), _chance_corrected(lo), _chance_corrected(hi)
        tests[code] = {
            "dim": rs[0].dim,
            "track": "causal" if REGISTRY[code].track == "mechanistic" else REGISTRY[code].track,
            "label": REGISTRY[code].label,
            "score": round(mean_score, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "n": len(rs),
            "n_valid": len(valid),
            "n_executed": len(executed),
            "n_structural_absent": len(structural),
            "measured_score": (round(mean_score, 4) if code in CHANCE_CORRECTED_TESTS else round(mean([r.score for r in executed]), 4)) if executed else None,
            "invalid_rate": round(1 - len(valid) / len(rs), 4) if rs else 0,
        }
    profile = {}
    for d in DIMENSIONS:
        vals = [v["score"] for v in tests.values() if v["dim"] == d]
        if vals:
            profile[d] = round(mean(vals), 4)
    behavioral_profile = {d: profile[d] for d in BEHAVIORAL_DIMENSIONS if d in profile}
    causal_profile = {d: profile[d] for d in CAUSAL_DIMENSIONS if d in profile}
    return {
        "tests": tests,
        "profile": profile,
        "behavioral_profile": behavioral_profile,
        "causal_profile": causal_profile,
        # Deprecated compatibility alias. New reports should use causal_profile.
        "mechanistic_profile": causal_profile,
    }


def render(agg, model_name):
    lines = [f"FCA-Bench — {model_name}", ""]
    for title, dims in [
        ("Behavioral access profile", BEHAVIORAL_DIMENSIONS),
        ("Causal/interventional access profile", CAUSAL_DIMENSIONS),
    ]:
        present = [d for d in dims if d in agg["profile"]]
        if not present:
            continue
        lines.append(title)
        lines.append("dimension  score   tests")
        for d in present:
            codes = [c for c, v in agg["tests"].items() if v["dim"] == d]
            parts = []
            for c in sorted(codes):
                item = agg["tests"][c]
                if item["n_structural_absent"] == item["n_valid"] and item["n_valid"] > 0:
                    parts.append(f"{c}=STRUCTURAL_ABSENT")
                elif item["n_structural_absent"]:
                    parts.append(f"{c}={item['score']:.2f}[absent={item['n_structural_absent']}/{item['n_valid']}]")
                else:
                    parts.append(f"{c}={item['score']:.2f}")
            detail = "  ".join(parts)
            lines.append(f"{d:>3} {dims[d]:<34} {agg['profile'][d]:.3f}   {detail}")
        lines.append("")
    lines.append("No official aggregate scalar is defined by FCA-Bench.")
    lines.append("STRUCTURAL_ABSENT denotes an unavailable architecture primitive, not a failed executed task.")
    invalid = {c: v["invalid_rate"] for c, v in agg["tests"].items() if v["invalid_rate"] > 0}
    if invalid:
        lines.append("Invalid instance rates: " + ", ".join(f"{c}={x:.0%}" for c, x in invalid.items()))
    return "\n".join(lines)


def select_codes(suite, tests, dims):
    if suite == "causal":
        suite = "mechanistic"
    if tests:
        codes = [x.strip().upper() for x in tests.split(",") if x.strip()]
    else:
        codes = sorted(REGISTRY)
        if suite != "all":
            codes = [c for c in codes if REGISTRY[c].track == suite]
    if dims:
        keep = {x.strip().upper() for x in dims.split(",") if x.strip()}
        codes = [c for c in codes if REGISTRY[c].dim in keep]
    unknown = [c for c in codes if c not in REGISTRY]
    if unknown:
        raise ValueError("Unknown tests: " + ", ".join(unknown))
    return codes


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run FCA-Bench on a local model or custom architecture adapter.")
    ap.add_argument("--config", help="JSON adapter configuration file")
    ap.add_argument("--suite", choices=["behavioral", "causal", "mechanistic", "all"], default="behavioral", help="Use causal for the causal/interventional track; mechanistic is retained as a compatibility alias.")
    ap.add_argument("--tests", default="", help="Comma-separated test codes")
    ap.add_argument("--dims", default="", help="Comma-separated dimension codes")
    ap.add_argument("--n", type=int, default=3, help="Instances per selected test when no seed bank is supplied")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-bank", default="", help="Private JSON seed bank generated by fca_bench.sealing")
    ap.add_argument("--output", default="results.json")
    ap.add_argument("--list-tests", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output; the final report is still printed")
    args = ap.parse_args(argv)

    if args.list_tests:
        for code in sorted(REGISTRY):
            t = REGISTRY[code]
            display_track = "causal" if t.track == "mechanistic" else t.track
            print(f"{code:>3}  {display_track:<11}  {t.dim}  {t.label}")
        return 0
    if not args.config:
        ap.error("--config is required unless --list-tests is used")

    config = load_config(args.config)
    adapter = adapter_from_config(config)
    codes = select_codes(args.suite, args.tests, args.dims)
    bank = load_seed_bank(args.seed_bank)

    execution_plan = []
    for code in codes:
        if bank is not None:
            if code not in bank:
                raise ValueError(f"Selected test {code} is missing from the supplied seed bank.")
            seeds = list(bank[code])
        else:
            seeds = [args.seed + k for k in range(args.n)]
        for seed in seeds:
            inst = REGISTRY[code].generate(seed)
            execution_plan.append((code, seed, expected_model_calls(inst, adapter)))

    total_instances = len(execution_plan)
    total_calls = sum(item[2] for item in execution_plan)
    progress = not args.quiet
    if hasattr(adapter, "configure_progress"):
        adapter.configure_progress(total_calls=total_calls, enabled=progress)
    if progress:
        print(
            f"Execution plan: {len(codes)} tests, {total_instances} instances, "
            f"~{total_calls} model generation calls. E1 samples={getattr(adapter, 'e1_samples', 'n/a')}",
            flush=True,
        )

    results = []
    offset = 0
    for code in codes:
        explicit = None
        if bank is not None:
            explicit = bank[code]
        count = len(explicit) if explicit is not None else args.n
        results.extend(
            run_test(
                code, adapter, args.n, args.seed, args.verbose,
                explicit_seeds=explicit, progress=progress,
                instance_offset=offset, instance_total=total_instances,
            )
        )
        offset += count

    agg = aggregate(results)
    print(render(agg, adapter.name))
    payload = {
        "benchmark": "FCA-Bench",
        "benchmark_expansion": "Functional Causal Access Benchmark",
        "protocol_version": __version__,
        "system_id": getattr(adapter, "system_id", adapter.name),
        "model": adapter.name,
        "suite": args.suite,
        "n_requested": args.n,
        "seed": None if bank is not None else args.seed,
        "seed_bank_sha256": _seed_bank_hash(args.seed_bank),
        "selected_tests": codes,
        **agg,
        "instances": [r.__dict__ for r in results],
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
