# Specification structure and layers — v0.28

A-dagger is the functional layer of the project. It is deliberately behavioral and architecture-neutral: the observable interventional kernel and declared deployment boundary determine the measurement, not an assumed internal mechanism.

The sealed v0.28 specification separates:

- the versioned `A†-SPEC`, which fixes clauses, score typing, calibration/inference rules, status semantics and interpretation prohibitions;
- an `INSTANCE`, which fixes the concrete deployment, cues, controls, data roles, resampling unit, score functions and quality flags for a run;
- optional bridge-layer hypotheses, which are outside the behavioral measurement itself.

The core ladder is nested:

- `A†0 = {DELAYED-USE, SUBST}`
- `A†1 = A†0 + {INTERFERENCE}`
- `A†2 = A†1 + {POSTCUE-MULTI-USE, CONTROL}`

Four certificates remain independent of that ladder:

- `Cert_T = {TEMPORAL-AUTONOMY, ELAPSED-TIME-CONTROL}`
- `Cert_E = {HARD-ERASURE, SELECTIVE-ERASURE}`
- `Cert_B = {SOURCE-BINDING}`
- `Cert_A = {CAUSAL-OWNERSHIP}`

v0.28 uses clause verdicts `PASS`, `FAIL`, `NA_interface`, and `INVALID`. Quality flags are reported separately and do not become pseudo-verdicts. Object propagation is `INVALID > FAIL > BLOCKED(NA_interface) > PASS`.

The specification explicitly prohibits interpreting the profile as a consciousness score or as evidence about phenomenology.
