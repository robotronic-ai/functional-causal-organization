from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .core import Result
from .run import aggregate, render


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "FCA-Bench":
        raise ValueError(f"{path}: not an FCA-Bench result file.")
    if payload.get("protocol_version") != __version__:
        raise ValueError(
            f"{path}: protocol_version={payload.get('protocol_version')!r}, "
            f"expected {__version__!r}."
        )
    if not isinstance(payload.get("instances"), list):
        raise ValueError(f"{path}: missing instances list.")
    return payload


def _as_result(obj: dict) -> Result:
    allowed = {
        "instance_id", "dim", "test", "score", "raw_score",
        "valid", "status", "details"
    }
    data = {k: obj[k] for k in allowed if k in obj}
    return Result(**data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge FCA-Bench v0.2.7 result JSON files without changing scores."
    )
    ap.add_argument("inputs", nargs="+", help="Input FCA-Bench JSON result files")
    ap.add_argument("--output", required=True, help="Merged JSON output path")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.inputs]
    payloads = [_load(p) for p in paths]

    system_ids = {p.get("system_id") for p in payloads}
    models = {p.get("model") for p in payloads}
    if len(system_ids) != 1:
        raise ValueError(f"Input files have different system_id values: {sorted(system_ids)}")
    if len(models) != 1:
        raise ValueError(f"Input files have different model labels: {sorted(models)}")

    results = []
    selected_tests = []
    seen_instance_keys = set()

    for path, payload in zip(paths, payloads):
        for code in payload.get("selected_tests", []):
            if code not in selected_tests:
                selected_tests.append(code)
        for obj in payload["instances"]:
            key = (obj.get("test"), obj.get("instance_id"))
            if key in seen_instance_keys:
                raise ValueError(
                    f"Duplicate benchmark instance across inputs: test={key[0]} "
                    f"instance_id={key[1]}. Refusing to double-count."
                )
            seen_instance_keys.add(key)
            results.append(_as_result(obj))

    agg = aggregate(results)
    n_by_test = {code: data["n"] for code, data in agg["tests"].items()}

    out = {
        "benchmark": "FCA-Bench",
        "benchmark_expansion": "Functional Causal Access Benchmark",
        "protocol_version": __version__,
        "system_id": next(iter(system_ids)),
        "model": next(iter(models)),
        "suite": "merged-selected-tests",
        "selected_tests": selected_tests,
        "n_by_test": n_by_test,
        "source_files": [str(p) for p in paths],
        **agg,
        "instances": [r.__dict__ for r in results],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(render(agg, out["model"]))
    print(f"\nMerged JSON written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
