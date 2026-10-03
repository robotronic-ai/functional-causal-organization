from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def pearson(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if den == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / den


def ranks(values):
    pairs = sorted(enumerate(values), key=lambda p: p[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[pairs[k][0]] = rank
        i = j
    return out


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(ranks(xs), ranks(ys))


def zscores(values):
    if len(values) < 2:
        return [0.0 for _ in values]
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [(x - m) / sd for x in values]


def build_general_capability(models, metadata):
    bench_names = sorted({b for sid in models for b in metadata[sid].get("general_benchmarks", {})})
    per_model = {sid: [] for sid in models}
    for bench in bench_names:
        present = [(sid, metadata[sid].get("general_benchmarks", {}).get(bench)) for sid in models]
        present = [(sid, val) for sid, val in present if isinstance(val, (int, float))]
        if len(present) < 2:
            continue
        zs = zscores([float(val) for _, val in present])
        for (sid, _), z in zip(present, zs):
            per_model[sid].append(z)
    return {sid: mean(vals) if vals else None for sid, vals in per_model.items()}


def rounded(x):
    return None if x is None else round(float(x), 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyze whether FCA profiles track model scale or general capability.")
    ap.add_argument("--results", nargs="+", required=True, help="FCA result JSON files from different systems")
    ap.add_argument("--metadata", required=True, help="JSON metadata keyed by system_id")
    ap.add_argument("--output", default="specificity_analysis.json")
    args = ap.parse_args(argv)

    metadata_payload = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    metadata = metadata_payload.get("systems", metadata_payload)
    results = []
    for path in args.results:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sid = payload.get("system_id") or payload.get("model")
        if sid not in metadata:
            raise ValueError(f"No metadata entry for system_id {sid!r}")
        results.append((sid, payload))

    systems = [sid for sid, _ in results]
    gc = build_general_capability(systems, metadata)
    dims = sorted({d for _, payload in results for d in payload.get("profile", {})})
    correlations = {}
    for dim in dims:
        rows = []
        for sid, payload in results:
            score = payload.get("profile", {}).get(dim)
            params = metadata[sid].get("parameters")
            if isinstance(score, (int, float)):
                rows.append((sid, float(score), params, gc.get(sid)))
        p_rows = [r for r in rows if isinstance(r[2], (int, float)) and r[2] > 0]
        g_rows = [r for r in rows if isinstance(r[3], (int, float))]
        log_params = [math.log10(float(r[2])) for r in p_rows]
        p_scores = [r[1] for r in p_rows]
        g_vals = [float(r[3]) for r in g_rows]
        g_scores = [r[1] for r in g_rows]
        correlations[dim] = {
            "n_parameter": len(p_rows),
            "pearson_log_parameters": rounded(pearson(log_params, p_scores)),
            "spearman_log_parameters": rounded(spearman(log_params, p_scores)),
            "n_general_capability": len(g_rows),
            "pearson_general_capability": rounded(pearson(g_vals, g_scores)),
            "spearman_general_capability": rounded(spearman(g_vals, g_scores)),
        }

    groups = {}
    for sid, payload in results:
        group = metadata[sid].get("matched_group")
        condition = metadata[sid].get("condition")
        if group and condition:
            groups.setdefault(group, {})[condition] = (sid, payload)
    matched = {}
    for group, conditions in groups.items():
        if len(conditions) < 2:
            continue
        names = sorted(conditions)
        pairs = []
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                lp = conditions[left][1].get("profile", {})
                rp = conditions[right][1].get("profile", {})
                shared = sorted(set(lp) & set(rp))
                pairs.append({
                    "left": left,
                    "right": right,
                    "delta_right_minus_left": {d: round(rp[d] - lp[d], 4) for d in shared},
                })
        matched[group] = pairs

    output = {
        "analysis": "FCA specificity controls",
        "systems": systems,
        "general_capability_z_composite": {sid: rounded(gc.get(sid)) for sid in systems},
        "dimension_correlations": correlations,
        "matched_group_contrasts": matched,
        "interpretation_rule": (
            "Correlation with scale or general capability is a confound diagnostic, not a proof of invalidity. "
            "Matched same-base-model interventions are the stronger specificity test."
        ),
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
