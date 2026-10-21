# Gamma-Cert target transformer — certified-interval protocol, NOT yet executed

Clean English edition of the target-transformer Γ certification package, folder
`07-gamma-scaling-and-certification/gamma-cert-target-transformer/`. **This is a frozen protocol
and a working, self-tested implementation — it has not been run against the real target model.**
See `docs/execution-status.md` for the full, plainly-stated status.

## Reproduction discipline

This package follows the same discipline as the rest of the repository: the protocol
(`docs/certification-protocol.md`) is preregistered — quotient gate, four independent
statistical streams, error budget, and the `V1 ≤ 0.10 width` / `V3 > 0.10 width` / `NA
quotient-fails` precision policy — before any real trial is collected. The certificate never
substitutes a raw activation-space score for a failed quotient gate, and it never substitutes
another model's result for this target's. Verdicts are `V1`/`V3`/`NA`, exactly as elsewhere in
this repository; nothing here silently reports a number where the protocol requires `NA` or an
interval.

## Main result: NOT executed on the real target model

**`status: blocked_on_target_model_artifacts`, `target_experiment_executed: false`.** The
certificate implementation is ready and self-tested (synthetic system, exact Γ `0.2`, certified
interval `[0.093, 0.252]`, width `0.159` — itself only `V3`, since it exceeds the `0.10`
precision band). No real interval exists yet for the manuscript's recurrent-memory transformer;
the checkpoint, tokenizer, intervention hook, and quotient-discovery procedure are all missing
from this runtime. See `docs/execution-status.md` for the full breakdown.

## Layout

```
README.md                          this file
docs/
  certification-protocol.md        estimand, quotient gate, statistical streams, bounds, V1/V3/NA policy
  execution-status.md              the plain, prominent NOT-executed status and self-test result
scripts/
  gamma_certificate_from_trials.py architecture-independent certificate (lower + upper bound, MILP-free)
  adapter_contract.py              strict model-specific interface the certifier requires
  selftest_tiny_system.py          end-to-end software self-test on a synthetic system
results/
  RUN_STATUS.json                  exact execution status and missing target inputs (verbatim)
  selftest_results.json            exact self-test certificate output (verbatim)
  experiment_config.json           frozen statistical settings (verbatim)
  TARGET_INPUT_MANIFEST.json       exact list of required model-adapter inputs (verbatim)
```

## Relation to the rest of this folder

- `gamma-quantum-campaign/` shipped a byte-identical copy of this package as its own
  `baseline_certifier/`; it is not duplicated a third time. That campaign's real, executed
  `Γ_dir` measurements are on a *different* model (Quantum) under a *different*, not
  statistically certified normalization, and do not satisfy this package's target-run
  obligation.
- `gamma-cert-causal-quotient/` is a dedicated stress test of the quotient gate this protocol
  requires (§"Non-negotiable quotient gate" in `docs/certification-protocol.md`), showing exactly
  when a causally sufficient macrostate quotient can and cannot be trusted.
- `gamma-cert-v0.41/` develops a different, additive/scalable family of certified Γ lower and
  upper bounds (fractional packing, spectral Cheeger bounds) aimed at systems too large for this
  package's decoder-graph-only construction to certify tightly.

## How to verify

```bash
# Self-test only; no target model required.
python3 scripts/selftest_tiny_system.py
# Expected: interval [0.09285393537647317, 0.252291157775851], status V3, selftest_pass true.
```

The certificate cannot be run against the real target because the model-specific adapter inputs
listed in `results/TARGET_INPUT_MANIFEST.json` are not available in this runtime (see
`docs/execution-status.md`).
