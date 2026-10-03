from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from . import REGISTRY, __version__


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_obj(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def select_codes(suite: str, tests: str) -> list[str]:
    if suite == "causal":
        suite = "mechanistic"
    if tests:
        codes = [x.strip().upper() for x in tests.split(",") if x.strip()]
    else:
        codes = sorted(REGISTRY)
        if suite != "all":
            codes = [c for c in codes if REGISTRY[c].track == suite]
    unknown = [c for c in codes if c not in REGISTRY]
    if unknown:
        raise ValueError("Unknown tests: " + ", ".join(unknown))
    return codes


def create_bank(codes: list[str], n: int) -> tuple[dict, dict]:
    if n < 1:
        raise ValueError("n must be at least 1")
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    bank_id = secrets.token_hex(12)
    tests = {code: [secrets.randbits(63) for _ in range(n)] for code in codes}
    private_payload = {
        "benchmark": "FCA-Bench",
        "protocol_version": __version__,
        "bank_id": bank_id,
        "created_utc": created,
        "tests": tests,
    }
    commitments = {code: [sha256_obj({"code": code, "seed": seed}) for seed in seeds] for code, seeds in tests.items()}
    public_payload = {
        "benchmark": "FCA-Bench",
        "protocol_version": __version__,
        "bank_id": bank_id,
        "created_utc": created,
        "test_counts": {code: len(seeds) for code, seeds in tests.items()},
        "seed_commitments": commitments,
        "private_bank_sha256": sha256_obj(private_payload),
        "note": "Publish this manifest before evaluation. Keep the seed bank private until the run is frozen.",
    }
    return private_payload, public_payload


def verify(private_payload: dict, public_payload: dict) -> tuple[bool, list[str]]:
    errors = []
    if private_payload.get("bank_id") != public_payload.get("bank_id"):
        errors.append("bank_id mismatch")
    if sha256_obj(private_payload) != public_payload.get("private_bank_sha256"):
        errors.append("private bank SHA-256 mismatch")
    for code, seeds in private_payload.get("tests", {}).items():
        expected = public_payload.get("seed_commitments", {}).get(code, [])
        actual = [sha256_obj({"code": code, "seed": seed}) for seed in seeds]
        if actual != expected:
            errors.append(f"seed commitments mismatch for {code}")
    return (not errors), errors


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create or verify a sealed FCA-Bench seed bank.")
    sub = ap.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create")
    create.add_argument("--suite", choices=["behavioral", "causal", "mechanistic", "all"], default="all", help="Use causal for the causal/interventional track; mechanistic is retained as a compatibility alias.")
    create.add_argument("--tests", default="")
    create.add_argument("--n", type=int, default=20)
    create.add_argument("--private", default="sealed_seed_bank.private.json")
    create.add_argument("--public", default="sealed_seed_bank.public.json")

    check = sub.add_parser("verify")
    check.add_argument("--private", required=True)
    check.add_argument("--public", required=True)

    args = ap.parse_args(argv)
    if args.command == "create":
        codes = select_codes(args.suite, args.tests)
        private_payload, public_payload = create_bank(codes, args.n)
        Path(args.private).write_text(json.dumps(private_payload, indent=2), encoding="utf-8")
        Path(args.public).write_text(json.dumps(public_payload, indent=2), encoding="utf-8")
        print(f"Created sealed bank {public_payload['bank_id']} for {len(codes)} tests x {args.n} instances.")
        print(f"Private: {args.private}")
        print(f"Public commitment: {args.public}")
        return 0

    private_payload = json.loads(Path(args.private).read_text(encoding="utf-8"))
    public_payload = json.loads(Path(args.public).read_text(encoding="utf-8"))
    ok, errors = verify(private_payload, public_payload)
    if not ok:
        for error in errors:
            print("ERROR:", error)
        return 1
    print("Seed bank verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
