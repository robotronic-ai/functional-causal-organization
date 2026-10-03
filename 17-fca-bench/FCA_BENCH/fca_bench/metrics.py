from __future__ import annotations

import math
import random
from collections import Counter
from typing import Hashable, Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def geomean(values: Sequence[float]) -> float:
    if not values or any(x <= 0 for x in values):
        return 0.0
    return math.exp(sum(math.log(x) for x in values) / len(values))


def exact(gold, pred) -> float:
    return float(gold == pred)


def normalized_text(value) -> str:
    return " ".join(str(value).strip().upper().split())


def set_f1(gold: Iterable[Hashable], pred: Iterable[Hashable]) -> float:
    g, p = set(gold), set(pred)
    if not g and not p:
        return 1.0
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Iterable[str] | None = None) -> float:
    labs = sorted(set(labels) if labels is not None else set(gold) | set(pred))
    scores = []
    for label in labs:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def brier_skill(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    if not outcomes:
        return 0.0
    base = mean(outcomes)
    bs = mean([(p - y) ** 2 for p, y in zip(probs, outcomes)])
    ref = mean([(base - y) ** 2 for y in outcomes])
    if ref <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - bs / ref))


def ece(probs: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    if not outcomes:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            total += len(idx) / len(outcomes) * abs(mean([probs[i] for i in idx]) - mean([outcomes[i] for i in idx]))
    return total


def _inv_norm(p: float) -> float:
    p = min(max(p, 1e-8), 1 - 1e-8)
    a1, a2, a3, a4, a5, a6 = -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924
    b1, b2, b3, b4, b5 = -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857
    c1, c2, c3, c4, c5, c6 = -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878
    d1, d2, d3, d4 = 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742
    plow, phigh = 0.02425, 0.97575
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    q = p - 0.5
    r = q*q
    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)


def signal_detection(hits: int, misses: int, false_alarms: int, correct_rejections: int) -> tuple[float, float]:
    hit_rate = (hits + 0.5) / (hits + misses + 1.0)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1.0)
    zh, zf = _inv_norm(hit_rate), _inv_norm(fa_rate)
    return zh - zf, -0.5 * (zh + zf)


def dprime_to_unit(dprime: float, cap: float = 3.0) -> float:
    return max(0.0, min(1.0, dprime / cap))


def jsd_from_samples(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    if not a or not b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    na, nb = len(a), len(b)
    pa = {k: ca[k] / na for k in keys}
    pb = {k: cb[k] / nb for k in keys}
    m = {k: 0.5 * (pa[k] + pb[k]) for k in keys}

    def kl(p, q):
        return sum(v * math.log2(v / q[k]) for k, v in p.items() if v > 0)

    return max(0.0, min(1.0, 0.5 * kl(pa, m) + 0.5 * kl(pb, m)))



def jsd_permutation(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    support: Sequence[Hashable] | None = None,
    reps: int = 256,
    seed: int = 17,
) -> tuple[float, dict[str, float]]:
    """Permutation-bias-corrected empirical Jensen-Shannon divergence.

    Small samples on a wide categorical support have a strongly positive plug-in
    JSD bias even when both arms are sampled from the same distribution. We
    estimate that finite-sample null bias by repeatedly permuting the pooled arm
    labels, subtract it from the observed JSD, and report diagnostics.

    The caller should pass a closed support whenever the probe defines one.
    """
    if not a or not b:
        return 0.0, {"observed": 0.0, "null_mean": 0.0, "corrected": 0.0, "p_value": 1.0}

    support_values = list(dict.fromkeys(support or list(a) + list(b)))

    def _jsd_closed(x: Sequence[Hashable], y: Sequence[Hashable]) -> float:
        cx, cy = Counter(x), Counter(y)
        nx, ny = len(x), len(y)
        pa = {k: cx.get(k, 0) / nx for k in support_values}
        pb = {k: cy.get(k, 0) / ny for k in support_values}
        m = {k: 0.5 * (pa[k] + pb[k]) for k in support_values}

        def _kl(p, q):
            return sum(v * math.log2(v / q[k]) for k, v in p.items() if v > 0 and q[k] > 0)

        return max(0.0, min(1.0, 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)))

    observed = _jsd_closed(a, b)
    pooled = list(a) + list(b)
    n_a = len(a)
    rnd = random.Random(seed)
    null = []
    for _ in range(max(32, int(reps))):
        perm = pooled[:] 
        rnd.shuffle(perm)
        null.append(_jsd_closed(perm[:n_a], perm[n_a:]))
    null_mean = mean(null)
    corrected = max(0.0, min(1.0, observed - null_mean))
    p_value = (1.0 + sum(x >= observed for x in null)) / (len(null) + 1.0)
    return corrected, {
        "observed": observed,
        "null_mean": null_mean,
        "corrected": corrected,
        "p_value": p_value,
    }

def bootstrap_ci(values: Sequence[float], seed: int = 17, reps: int = 1000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    r = random.Random(seed)
    n = len(values)
    samples = sorted(mean([values[r.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return samples[int(0.025 * reps)], samples[min(reps - 1, int(0.975 * reps))]
