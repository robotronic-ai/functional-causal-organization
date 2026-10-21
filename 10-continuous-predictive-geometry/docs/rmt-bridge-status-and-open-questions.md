# Applying this to a real model (RMT bridge), status, and open questions

Source material: `RMT_BRIDGE.md`, `STATUS_AND_HANDOFF.md`, `OPEN_QUESTIONS.md`.

## How this theory would be applied to a frozen model representation

For a history/state representation `h` (e.g. a hidden state inside a trained model),
estimate future distributions of frozen downstream observables under frozen
interventions, using a discrepancy such as total variation for discrete categorical
outcomes, bounded-Lipschitz distance, Wasserstein distance, or a characteristic-kernel
MMD — any of these must be normalized or otherwise bounded before the universal
discounted-tail certificate (`docs/theorem-and-proof-sketch.md`, Section 2) can be
applied. The empirical discounted predictive distance is

```
dhat_{gamma,H}(h, h') = (1 - gamma) * sum_{m=1}^H gamma^(m-1) * dhat_m(h, h'),
```

and the workflow reports three separate quantities: the finite-horizon estimate, its
statistical uncertainty, and the deterministic temporal-tail allowance `gamma^H`.

**Suggested experimental workflow** (not yet executed against a real model in this
bundle): (1) freeze readouts and interventions; (2) build predictive features on
calibration data only; (3) choose dimension/cover radius without using holdout
outcomes; (4) evaluate future-law error on held-out horizons; (5) evaluate context/
intervention transfer; (6) only after predictive transport is established, evaluate
causal-use interventions.

**Why this is preferable to raw latent geometry.** Euclidean distance between hidden
states is coordinate-dependent. Predictive distance is defined by consequences under
declared future experiments and is therefore aligned with the functional claim this
project is making.

**Relation to causal retargeting.** A predictive state may be highly accurate while the
encoded variable is not causally used by the native computation. Conversely,
causal-use evidence (such as this project's RMT causal-retargeting result, reported
elsewhere in the archive) should not be interpreted as proof that a compact predictive
realization exists. The two layers — predictive capacity and realized causal use — are
complementary, not substitutable.

## What is closed in this protocol

- A continuous approximate predictive pseudometric is defined from discounted
  finite-horizon future-law discrepancies, and zero distance preserves exact predictive
  equivalence.
- A universal temporal truncation certificate is proved:
  `d_gamma <= d_{gamma,H} + gamma^H`.
- Automatic epsilon quotients are rejected because threshold relations need not be
  transitive (proved by explicit counterexample).
- Compact reachable systems with continuous finite-horizon predictive discrepancies
  admit finite predictive epsilon-covers.
- A representative abstraction gives a one-state future-law guarantee, and multi-step
  abstract rollout is separated from one-state approximation via an explicit
  Lipschitz-plus-defect error recurrence.
- Infinite-path TV is shown to be unsuitable as a generic approximate geometry, via the
  Bernoulli product-law singularity example.

## Exact / numerical verification performed

The included Bernoulli verifier checks: monotonic finite-horizon TV; discounted
truncation intervals; approximate triangle inequalities on a rational parameter grid;
convergence of finite-horizon TV toward one for distinct Bernoulli parameters; and
continuity-like behavior of the discounted metric on nearby parameters. The
nontransitivity verifier gives an explicit predictive-distance threshold
counterexample. Both are exact, deterministic, closed-form computations — no model
inference, sampling, or statistical estimation is involved.

## Explicit next target (not yet done)

> Apply the protocol to a controlled continuous representation with frozen readouts and
> interventions, using a strict held-out transport axis.

The strongest useful next result named in the source material would combine, in a
single finite-sample theorem: statistical future-law error, deterministic discounted
tail, abstraction error, and dynamic rollout defect. None of this has been attempted in
this bundle.

## Scientific typing

Approximate predictive geometry quantifies predictive capacity / behavioral
similarity. It does not by itself establish realized causal use, and — like the rest of
this project's predictive-state theory — it is not evidence for or against
phenomenal consciousness.

## Open questions carried forward

1. **Sharp continuous tail bounds without discounting** — structural assumptions under
   which finite-horizon path-law error controls undiscounted infinite-path error.
2. **Predictive contraction** — sufficient conditions for a controlled RNN or
   stochastic kernel to be contractive in a declared predictive metric.
3. **Finite-sample metric estimation** — simultaneous confidence bounds for
   multi-horizon TV, Wasserstein, or MMD discrepancies under dependent trajectory
   samples.
4. **Approximate Hankel-to-law guarantees** — relating weighted low-rank Hankel
   approximation to a direct predictive law metric.
5. **Predictive covering rates** — bounding `N_gamma(epsilon)` from smoothness,
   intrinsic dimension, and controlled sensitivity assumptions.
6. **Adaptive intervention design** — selecting interventions that most efficiently
   shrink predictive uncertainty while preserving strict holdouts.
7. **RMT integration** — testing whether a frozen predictive representation transports
   downstream future-law distributions across held-out contexts/interventions before
   comparing it with realized causal-use evidence.
8. **Organization relation** — characterizing the set of predictive geometries
   compatible with a fixed organizational profile, instead of forcing a direct scalar
   map from organization to access.
