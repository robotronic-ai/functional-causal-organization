# 04 — Γ / Δ characterization

Clean edition of the campaign answering manuscript §7.1–§7.2: is the arithmetic-mean
aggregator used for the bidirectional causal-integration score Γ axiomatically forced, does
the differentiation score Δ decompose into independent, separately-characterizable factors,
and are Γ and Δ functionally independent of one another?

## Reproduction discipline

This campaign follows the same discipline as every other campaign in this repository (see
the root `README.md` for the full statement): a protocol is frozen and hashed before any
candidate is coded; a generator produces fixtures and a private oracle records the expected
verdict for each one; a candidate is written against the frozen fixtures; and where the
result claims a **candidate-fixed holdout**, a second, independent fixture set is generated
*after* the candidate's hash is frozen and the same unmodified candidate is re-run against
it. Verdicts are reported as `V1/POINT` (a unique value), `V3/PARTIAL` (several values remain
compatible, reported as a set) or `V4/NA` (the intervention algebra abstains) — no result here
silently turns a partial finding into a single number.

## Main results

### Γ — not uniquely forced by the minimal axioms

Under the minimal axioms (domain/codomain typing, symmetry, causal-recoding invariance,
coordinatewise monotonicity, diagonal calibration `F(t,t)=t`, and boundary values), the
arithmetic mean, geometric mean and min all qualify — the current formula is not singled out.
Verdict: **`GAMMA: AX-NONUNIQUE + SEMANTIC-BRANCH(G-WEAK)`**. A follow-up audit shows that
adding one further, explicitly substantive axiom — marginal independence of the two
directional contributions — forces the arithmetic mean uniquely, and that the geometric mean
and min both violate it (explicit finite-grid counter-witnesses). See
`docs/gamma-axiom-characterization.md`.

### Δ — exact decomposition into coverage × discriminability

\[
\Delta_q = \Delta_{\rm cap} = \kappa_q \cdot \delta_q,
\qquad
\kappa_q = \frac{H(P)}{\log_2|P^*|},
\qquad
\delta_q = \frac{I(P;Y)}{H(P)}.
\]

This is an exact analytic identity, confirmed on finite rational witnesses. Verdict:
**`DELTA actuelle = D-CAP, pas pure D-DEC`** (the current Δ formula realizes the
support-normalized D-CAP semantics, not the pure decodability-normalized D-DEC semantics).
Two restricted-but-exact axiomatic characterizations pin down D-CAP and D-DEC separately
within their respective function classes. See `docs/delta-decomposition.md`.

### Γ and Δ are mutually independent

Two exact finite witnesses (IND-1, IND-2) show that neither score is a universal function of
the other on the witness domain. Verdict: **`GAMMA/DELTA: INDEPENDENCE-WITNESS`**. See
`docs/independence-witnesses.md`.

### GD-SEM prospective campaign + candidate-fixed holdout: both 32/32

The frozen semantic choice (Γ stays the arithmetic mean under the `G-WEAK` additive
interpretation, with the directional balance β reported separately; Δ's primary profile
becomes `(κ_q, δ_q)`, with `Δ_cap = κ_q·δ_q` retained as a historical scalar synthesis) was
tested prospectively on 32 fixtures (16 families × 2 relabelings) — **32/32**, verdict
**`GD-V4 FINITE NON-REFUTATION`** — and then re-run unchanged against a second,
candidate-fixed holdout of 32 fixtures (8 families × 4 relabelings) generated after the
candidate's SHA-256 was frozen — **32/32**, verdict **`GD-CF-HOLDOUT-PASS`**. The holdout
includes a three-minimizing-cut witness: three distinct bipartitions achieve the identical
system-level Γ = 1/2 while their directional balances take three different values,
β ∈ {0, 0.5, 1} — the reason β is reported as a separate profile rather than folded into Γ.
See `docs/gd-sem-prospective-validation.md`.

### Final status: CLOSE-CONDITIONAL

The corrected, final closure note (`POINT3_GAMMA_DELTA_CLOSURE_v0_2.md` — an earlier `v0.1`
closure note in this same work stream was left at `PENDING INDEPENDENT REPRODUCTION` and is
superseded by this one) confirms independent reproduction of the manifest and hashes, the
analytic Γ/Δ audits, the 32/32 prospective run and the 32/32 candidate-fixed holdout, and
declares the work stream **`CLOSE-CONDITIONAL`**. See `docs/closure-status.md` for the full
closure text and the exact manuscript patch it authorizes.

## How to verify

```bash
# Analytic / finite-grid audits (Gamma axioms, Delta decomposition, independence witnesses)
python3 scripts/gamma_delta_axiom_audit_v0_1.py
python3 scripts/gamma_delta_characterization_audit_v0_2.py

# GD-SEM prospective campaign (regenerates the frozen fixtures/oracles, then scores the candidate)
python3 scripts/gamma_delta_sem_generator_v0_1.py
python3 scripts/gamma_delta_sem_harness_v0_1.py scripts/gamma_delta_sem_candidate_v0_1.py
# Expected: EXACT_PASSES: 32/32 ; GD-SEM VERDICT: GD-V4 FINITE NON-REFUTATION

# Candidate-fixed holdout (independent fixture set, same unmodified candidate)
python3 scripts/gamma_delta_sem_holdout_cf_generator_v0_1.py
python3 scripts/gamma_delta_sem_holdout_cf_harness_v0_1.py scripts/gamma_delta_sem_candidate_v0_1.py
# Expected: EXACT_PASSES: 32/32 ; GD-SEM VERDICT: GD-CF-HOLDOUT-PASS
```

The candidate's SHA-256 (`7681fe8c17c3bfd9e94d0e4b566e14254f718821dbea3ba2dae66334c21f26c6`)
is identical between the prospective run and the holdout run — this is what makes the holdout
a "candidate-fixed" reproducibility test rather than a second attempt at fitting the data.
All scripts use only the Python standard library (`fractions`, `math`, `json`, `importlib`).

See the root `README.md` for the full protocol → manuscript-section table.
