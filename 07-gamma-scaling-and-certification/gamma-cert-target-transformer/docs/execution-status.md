# Execution status: NOT executed on the real target model

**Status: `blocked_on_target_model_artifacts`. `target_experiment_executed: false`.** This is
stated plainly and prominently because the certificate machinery itself is implemented, tested,
and passing — it would be easy to mistake the software self-test below for a real result on the
manuscript's recurrent-memory transformer. It is not. No target-model Γ interval exists for this
package. (Verbatim source: `results/RUN_STATUS.json`.)

## What is implemented and passing

- `certificate_implementation_ready: true`
- `certificate_selftest_completed: true`, `certificate_selftest_passed: true`

## What is missing (why the real run has not happened)

| Field | Value |
|---|---|
| `target_model_artifacts_present_in_conversation` | `false` |
| `github_repository_reachable_from_runtime` | `false` |
| `transformers_package_available` | `false` |
| `huggingface_checkpoint_cache_present` | `false` |

Missing inputs, verbatim: frozen post-memory-training checkpoint or local model loader;
tokenizer and context generator; recurrent-memory intervention hook; causal minimal-state
quotient discovery or frozen quotient map.

**Scientific rule enforced here:** *"Do not substitute another architecture and label it as the
target experiment."* No other model's numbers stand in for this one. (In particular: the real,
executed `Γ_dir` measurements on the "Quantum" model in the sibling folder
`gamma-quantum-campaign/` are a genuinely different model and a genuinely different,
not-statistically-certified normalization — they do not satisfy this obligation and are not a
substitute for it.)

## The self-test — a software check, not a scientific result

`scripts/selftest_tiny_system.py` runs the certificate end-to-end on a small synthetic system
with a known exact Γ, to check the software (decoder graph, Fano bound, spectral lower bound,
cross-entropy upper bound, MILP-free packing) is wired correctly:

| Quantity | Value |
|---|---:|
| exact Γ (synthetic system) | `0.2` |
| certified lower bound `L_Γ` | `0.09285393537647317` |
| certified upper bound `U_Γ` | `0.252291157775851` |
| interval width `U_Γ - L_Γ` | `0.1594372223993778` |
| precision threshold | `0.10` |

The interval correctly contains the exact value (`selftest_interval_covers_exact: true`,
`selftest_pass: true`), confirming the certificate logic is sound. But note the width
(`0.159`) **exceeds** the `0.10` precision band, so even this synthetic self-test only reaches
status **`V3`**, not `V1` — recorded verbatim as `"status": "V3"` in
`results/selftest_results.json`. This is expected for a tiny 10-role synthetic system with a
modest trial budget; it is not evidence about the eventual real-model interval width, and it is
not itself the target result.

## Bottom line

There is currently no Γ certificate — of any status, `V1`, `V3`, or `NA` — for the manuscript's
recurrent-memory transformer. What exists is: a working, self-tested certificate implementation,
a frozen protocol, and an exact list of the model-specific artifacts still required to run it.
