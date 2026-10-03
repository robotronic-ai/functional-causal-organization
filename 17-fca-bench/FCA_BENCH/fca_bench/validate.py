from __future__ import annotations

import sys
from collections import Counter

from . import BEHAVIORAL_DIMENSIONS, CAUSAL_DIMENSIONS, REGISTRY
from .tests_time import _normalized_numeric


def main() -> int:
    expected = set(BEHAVIORAL_DIMENSIONS) | set(CAUSAL_DIMENSIONS)
    counts = Counter(test.dim for test in REGISTRY.values())
    problems = []
    if len(REGISTRY) != 42:
        problems.append(f"Expected 42 registered tests, found {len(REGISTRY)}.")
    for dim in sorted(expected):
        if counts[dim] != 3:
            problems.append(f"Dimension {dim} should contain 3 tests, found {counts[dim]}.")
    unknown = sorted(set(counts) - expected)
    if unknown:
        problems.append("Unknown dimensions: " + ", ".join(unknown))
    numeric_cases = [
        (16, 16.0, True),
        ("16", 16.0, False),
        ("16 minutes", 16.0, False),
        ("about 16 minutes", 16.0, False),
        ("16 or 17", None, False),
        (None, None, False),
    ]
    for value, expected, expected_native in numeric_cases:
        got, native = _normalized_numeric(value)
        if got != expected or native != expected_native:
            problems.append(
                f"T2 numeric normalization failed for {value!r}: "
                f"got {(got, native)!r}, expected {(expected, expected_native)!r}."
            )

    if problems:
        print("FCA-Bench package validation FAILED")
        for problem in problems:
            print("- " + problem)
        return 1
    print("FCA-Bench package validation PASSED")
    print("Registered tests: 42")
    print("Behavioral dimensions: 7 x 3 tests")
    print("Causal/interventional dimensions: 7 x 3 tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
