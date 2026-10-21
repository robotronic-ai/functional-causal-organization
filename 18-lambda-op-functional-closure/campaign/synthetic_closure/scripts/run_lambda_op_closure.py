#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:
    np = None

OPS = ("ID", "KEEP", "REVOKE", "SUBSTITUTE")
PRESENTATIONS = (0, 1, 2, 3)
LAMBDA_OP = {
    "ID": (0, 1, 2, 3),
    "KEEP": (0, 1, 2, 3),
    "REVOKE": (-1, 1, 2, 3),
    "SUBSTITUTE": (1, 1, 2, 3),
}
OPERATOR_TARGET = {
    "ID": ("ORIGINAL", "ORIGINAL", "ORIGINAL", "ORIGINAL"),
    "KEEP": ("ORIGINAL", "ORIGINAL", "ORIGINAL", "ORIGINAL"),
    "REVOKE": ("NONE", "ORIGINAL", "ORIGINAL", "ORIGINAL"),
    "SUBSTITUTE": ("DONOR", "ORIGINAL", "ORIGINAL", "ORIGINAL"),
}


def stable(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value) -> str:
    return hashlib.sha256((stable(value) + "\n").encode()).hexdigest()


def horizons(law: tuple[int, ...]):
    h1 = tuple(law[p] for p in PRESENTATIONS)
    h2 = tuple(law[x] for x in h1)
    h3 = tuple(law[x] for x in h2)
    return h1, h2, h3


def descriptor(law: tuple[int, ...]):
    h1, h2, _ = horizons(law)
    reached = tuple(sorted(set(h2)))
    return {
        "source_binding_h1": h1,
        "state_h2": h2,
        "r2_law": tuple((state, law[state]) for state in reached),
        "lambda_op": tuple((op, tuple(LAMBDA_OP[op])) for op in OPS),
    }


def target(law: tuple[int, ...]):
    _, _, h3 = horizons(law)
    return {
        "h3": h3,
        "operator_target": tuple((op, tuple(OPERATOR_TARGET[op])) for op in OPS),
    }


def analyze_rows(rows: list[dict]) -> dict:
    fibers = defaultdict(list)
    for row in rows:
        fibers[stable(row["descriptor"])].append(row)
    nontrivial = [group for group in fibers.values() if len(group) > 1]
    collisions = []
    for group in nontrivial:
        targets = {stable(row["target"]) for row in group}
        if len(targets) > 1:
            collisions.append({
                "system_ids": [row["system_id"] for row in group],
                "distinct_target_count": len(targets),
            })
    passed = not collisions
    return {
        "closure": "PASS" if passed else "COLLISION_PRESENT",
        "closure_type": ("NONTRIVIAL" if nontrivial else "TRIVIAL_INJECTIVE") if passed else "NONE",
        "n_systems": len(rows),
        "n_fibers": len(fibers),
        "n_nontrivial_fibers": len(nontrivial),
        "systems_in_nontrivial_fibers": sum(len(group) for group in nontrivial),
        "n_functional_collisions": len(collisions),
        "collision_examples": collisions[:8],
    }


def all_laws(n_bits: int) -> Iterable[tuple[int, ...]]:
    n = 1 << n_bits
    return itertools.product(range(n), repeat=n)


def run_2bit() -> dict:
    rows = []
    for i, law_ in enumerate(all_laws(2)):
        law = tuple(law_)
        rows.append({"system_id": f"L{i:03d}", "descriptor": descriptor(law), "target": target(law)})
    result = analyze_rows(rows)
    return {
        "schema": "co-lambda-op-closure-v1",
        "bits": 2,
        "coverage": "EXHAUSTIVE",
        "total_deterministic_maps": 256,
        **result,
    }


def _encode_3bit_chunk(codes):
    count = len(codes)
    laws = np.empty((count, 8), dtype=np.uint8)
    x = codes.copy()
    for state in range(8):
        laws[:, state] = x & 7
        x >>= 3
    h1 = laws[:, :4]
    h2 = np.take_along_axis(laws, h1, axis=1)
    h3 = np.take_along_axis(laws, h2, axis=1)

    desc = np.zeros(count, dtype=np.uint64)
    for j in range(4):
        desc |= h1[:, j].astype(np.uint64) << np.uint64(3 * j)
        desc |= h2[:, j].astype(np.uint64) << np.uint64(12 + 3 * j)
    for state in range(8):
        present = np.any(h2 == state, axis=1)
        code = np.where(present, laws[:, state].astype(np.uint64) + 1, 0)
        desc |= code << np.uint64(24 + 4 * state)

    tgt = np.zeros(count, dtype=np.uint16)
    for j in range(4):
        tgt |= h3[:, j].astype(np.uint16) << np.uint16(3 * j)

    h2_sorted = np.sort(h2, axis=1)
    r2_size = 1 + np.sum(h2_sorted[:, 1:] != h2_sorted[:, :-1], axis=1)
    return desc, tgt, r2_size.astype(np.uint8)


def run_3bit(work_dir: Path, chunk_size: int = 500_000) -> dict:
    if np is None:
        raise RuntimeError("NumPy is required for the exhaustive 3-bit execution")
    total = 8 ** 8
    desc_path = work_dir / "three_bit_desc.u64"
    target_path = work_dir / "three_bit_target.u16"
    desc_mm = np.memmap(desc_path, dtype=np.uint64, mode="w+", shape=(total,))
    tgt_mm = np.memmap(target_path, dtype=np.uint16, mode="w+", shape=(total,))
    r2_hist = Counter()
    started = time.perf_counter()

    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        codes = np.arange(start, stop, dtype=np.uint64)
        desc, tgt, r2_size = _encode_3bit_chunk(codes)
        desc_mm[start:stop] = desc
        tgt_mm[start:stop] = tgt
        vals, counts = np.unique(r2_size, return_counts=True)
        for value, count in zip(vals.tolist(), counts.tolist()):
            r2_hist[int(value)] += int(count)
    desc_mm.flush()
    tgt_mm.flush()

    desc = np.asarray(desc_mm)
    tgt = np.asarray(tgt_mm)
    order = np.argsort(desc, kind="stable")
    sorted_desc = desc[order]
    sorted_tgt = tgt[order]
    starts = np.r_[0, np.flatnonzero(sorted_desc[1:] != sorted_desc[:-1]) + 1]
    ends = np.r_[starts[1:], total]
    sizes = ends - starts
    nontrivial = sizes > 1

    target_min = np.minimum.reduceat(sorted_tgt, starts)
    target_max = np.maximum.reduceat(sorted_tgt, starts)
    collision_mask = target_min != target_max
    collisions = int(np.sum(collision_mask))
    passed = collisions == 0

    result = {
        "schema": "co-lambda-op-closure-v1",
        "bits": 3,
        "coverage": "EXHAUSTIVE",
        "total_deterministic_maps": total,
        "closure": "PASS" if passed else "COLLISION_PRESENT",
        "closure_type": "NONTRIVIAL" if passed and np.any(nontrivial) else ("TRIVIAL_INJECTIVE" if passed else "NONE"),
        "n_fibers": int(len(starts)),
        "n_nontrivial_fibers": int(np.sum(nontrivial)),
        "systems_in_nontrivial_fibers": int(np.sum(sizes[nontrivial])),
        "n_functional_collisions": collisions,
        "r2_reached_set_size_histogram": {str(k): v for k, v in sorted(r2_hist.items())},
        "elapsed_seconds": time.perf_counter() - started,
    }

    del sorted_desc, sorted_tgt, order, desc, tgt, desc_mm, tgt_mm
    for path in (desc_path, target_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return result


def random_law(rng: random.Random, n: int) -> tuple[int, ...]:
    return tuple(rng.randrange(n) for _ in range(n))


def make_closed_pair(rng: random.Random, n_bits: int):
    n = 1 << n_bits
    while True:
        law = list(random_law(rng, n))
        h1, h2, _ = horizons(tuple(law))
        relevant = set(PRESENTATIONS) | set(h1) | set(h2)
        nuisance = [state for state in range(n) if state not in relevant]
        if not nuisance:
            continue
        state = rng.choice(nuisance)
        choices = [value for value in range(n) if value != law[state]]
        law2 = law.copy()
        law2[state] = rng.choice(choices)
        a, b = tuple(law), tuple(law2)
        if stable(descriptor(a)) == stable(descriptor(b)) and stable(target(a)) == stable(target(b)):
            return a, b


def run_transport(n_bits: int, n_fibers: int = 512) -> dict:
    rng = random.Random(2026092600 + n_bits)
    rows = []
    for i in range(n_fibers):
        a, b = make_closed_pair(rng, n_bits)
        for suffix, law in (("A", a), ("B", b)):
            rows.append({
                "system_id": f"F{i:04d}_{suffix}",
                "descriptor": descriptor(law),
                "target": target(law),
            })
    result = analyze_rows(rows)
    return {
        "schema": "co-lambda-op-closure-v1",
        "bits": n_bits,
        "coverage": "DECLARED_TRANSPORT_ENVELOPE",
        "generated_fibers": n_fibers,
        **result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)

    results = {"2_bit": run_2bit()}
    (out / "2_bit_result.json").write_text(json.dumps(results["2_bit"], indent=2, sort_keys=True) + "\n")
    results["3_bit"] = run_3bit(work)
    (out / "3_bit_result.json").write_text(json.dumps(results["3_bit"], indent=2, sort_keys=True) + "\n")
    for bits in (4, 6, 8):
        results[f"{bits}_bit"] = run_transport(bits)
        (out / f"{bits}_bit_result.json").write_text(json.dumps(results[f"{bits}_bit"], indent=2, sort_keys=True) + "\n")

    summary = {
        "schema": "co-lambda-op-closure-suite-v1",
        "operation_algebra": list(OPS),
        "presentations": list(PRESENTATIONS),
        "results": results,
    }
    summary["sha256"] = sha256_json(summary)
    (out / "LAMBDA_OP_CLOSURE_2_3_4_6_8.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"closure": row["closure"], "closure_type": row["closure_type"]} for name, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
