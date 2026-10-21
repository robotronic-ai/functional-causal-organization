# Experimental protocol and held-out transport certification

Source material: `PROTOCOL.md` and `HELD_OUT_TRANSPORT.md`. This document describes how
the discounted predictive geometry is meant to be *applied* to a real controlled
system — it is a protocol specification, not a record of an application that has
already happened (see `docs/rmt-bridge-status-and-open-questions.md`).

## Six-phase protocol

**Phase 1 — Declare the predictive experiment.** Freeze the reachable state/history
definition, observation/readout variables, intervention/control class, future-law
discrepancy, discount `gamma`, calibration horizon, and holdout indices, before any
candidate is built.

**Phase 2 — Construct the approximate predictive geometry.** Estimate `d_1, ..., d_H`
and the discounted partial metric `d_{gamma,H}`. Report the certified tail allowance
`gamma^H`. Construct either an epsilon-cover, a bounded-diameter partition, or a metric
embedding with certified distortion.

**Phase 3 — Separate complexity notions.** Report predictive covering number /
empirical cover size, effective linear predictive rank, and future-law reconstruction
error **separately** — never collapsed into a single complexity scalar without a
separate theorem justifying the collapse (see
`docs/complexity-notions-and-guardrails.md`).

**Phase 4 — Held-out transport.** Evaluate at least one strict holdout: a later
horizon, an unseen control, an unseen context, or an unseen downstream observation
composition. A representation that only fits calibration indices is not accepted as a
transport result.

**Phase 5 — Dynamic rollout.** If an abstract transition model is learned, estimate
the initial cover/quantization error, the one-step transition defect, the predictive
Lipschitz/contraction coefficient, and the actual held-out rollout error. Compare the
observed rollout error against the theoretical recurrence bound
(`docs/theorem-and-proof-sketch.md`, Section 7) when applicable.

**Phase 6 — Causal-use layer.** Only after predictive transport is established should
causal-use interventions be interpreted. Predictive capacity and realized causal use
must remain distinct claims.

## Held-out transport certification (detail on Phase 4)

1. **Frozen declarations before model selection**: source/history construction,
   readout/observation variables, intervention/control class, `gamma`, calibration
   horizon `H_cal`, holdout horizons, holdout contexts/interventions, predictive
   discrepancy `D_m`, and approximation tolerance.
2. **Calibration object**: estimate finite-horizon predictive discrepancies or a
   predictive representation using only calibration indices. Held-out future outcomes
   must not be used to choose rank, clustering, embedding dimension, predictive
   representatives, or regularization strength.
3. **Temporal tail certificate**: for any pair of predictive representatives,
   `d_gamma <= d_{gamma,H_cal} + gamma^H_cal`. If the finite-horizon terms are
   estimated statistically, replace them by simultaneous upper confidence bounds.
4. **Held-out transport test**: a candidate predictive representation passes transport
   only if it predicts held-out future laws under at least one strict new index —
   longer horizon, unseen context, unseen intervention, or unseen consumer/readout
   composition.
5. **Dynamic model certification**: if a learned abstract transition is rolled forward,
   report the initial predictive quantization error, the one-step abstract transition
   defect, the estimated predictive Lipschitz/contraction coefficient, and the observed
   held-out rollout error, separately. One-state cover accuracy must not be treated as
   a multi-step rollout theorem on its own.
6. **Realized causal use**: even a perfect predictive representation establishes
   predictive capacity, not necessarily causal use by the original system. Causal
   retargeting or intervention tests remain a separate scientific layer (this project's
   RMT causal-retargeting results, referenced in
   `docs/rmt-bridge-status-and-open-questions.md`, are a separate line of evidence).

## Status of this protocol in the present bundle

No real controlled system, learned predictive representation, or held-out transport
test has been run against this protocol in this bundle. What has been done is the
exact mathematical verification described in `docs/theorem-and-proof-sketch.md` and
`docs/why-discounting-is-needed.md`, on a closed-form toy family. The protocol above is
frozen and ready to be applied to a real target; see
`docs/rmt-bridge-status-and-open-questions.md` for the recommended next step.
