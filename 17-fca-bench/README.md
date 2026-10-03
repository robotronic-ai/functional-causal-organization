# 17 — FCA-Bench: Functional Causal Access Benchmark

FCA-Bench evaluates whether information, goals, state variables and control signals in a system are actually accessible and governable. *Causal* means access demonstrated by intervention: information is accessible when it can be shown to influence, update, suppress, route, preserve or release other parts of the system. The benchmark is independent of A† and of C_O. It is not an intelligence score or a consciousness score, and it never infers a mechanism from a model's verbal description of itself.

It supports manuscript **§17.9** (and the FCA-Bench row of Table 1, §2).

## Design

- **Behavioral access profile** `(T, B, E, M, C, A, P)` — temporal grounding, internal/external boundary, context governance and erasure, metacognitive monitoring, cognitive control, causal agency, functional self-prediction. Observable through the ordinary model interface.
- **Causal/interventional access profile** `(R, D, L, U, I, G, H)` — latent recurrence, autonomous temporal dynamics, persistent latent state, mutable internal state, endogenous interrupt, global workspace, higher-order attention control. Requires an architecture primitive or an exposed intervention.
- Three tests per dimension, 42 in total. The two profiles are **never merged** into one score, so that strong general competence cannot compensate for a missing mechanism.
- Three result statuses: `executed` (measured score), `structural_absent` (the required primitive does not exist in the architecture — not a failed attempt), `invalid` (excluded).
- Specificity controls (behavior-without-mechanism, same-base-model pair, compute-matched twin, sham intervention, scale/capability diagnostics) and a sealed seed-bank workflow are part of the protocol.

## Qwen3-8B standard baseline

Qwen3-8B as released, thinking mode off, weights quantized to 8 bits (bitsandbytes) and executed entirely on GPU. Twenty instances per test, seed 0; 95 % confidence intervals in brackets. Protocol v0.2.7. Per-instance records: `results/FCA027_GPU_R6_12TESTS_n20_FINAL.json`; summary: `results/qwen3_8b_standard_baseline.json`.

**Behavioral access profile**

| Test | Capability | Score |
|---|---|---|
| T1 | Event timeline reconstruction | 0.89 [0.86, 0.93] |
| T2 | Pause-aware temporal accounting (active intervals, suspended deadline) | 0.26 [0.11, 0.41] |
| T3 | Temporal self-location | 0.85 [0.75, 0.93] |
| B1 | Provenance attribution from episode history | 0.87 [0.80, 0.93] |
| E1 | Counterfactual hard erasure (four matched arms, 20 samples per arm; 19/20 valid) | 0.72 [0.67, 0.76] |
| C3 | Interrupt and resume | 0.46 [0.41, 0.51] |

Dimension summaries: T = 0.67 · B = 0.87 · E = 0.72 · C = 0.46.

T2 is the chance-corrected balanced accuracy of interval membership, `max(0, 2·mean(BA) − 1)` (raw mean BA = 0.63, chance = 0.5): balanced accuracy is 0.79 for the intervals that count as active time and 0.48 for the intervals that consume a suspended deadline. Direct numeric answers are recorded as an arithmetic diagnostic and excluded from the score (protocol section 15).

**Causal/interventional access profile**

| Test | Capability | Result |
|---|---|---|
| R1 | Zero-token deepening | STRUCTURAL_ABSENT |
| R3 | Hidden-state iteration stress test | 0.40 [0.20, 0.60] |
| D1 | Silent deadline crossing | STRUCTURAL_ABSENT |
| L2 | Context-equivalent twins | STRUCTURAL_ABSENT |
| U3 | Semantic garbage collection | STRUCTURAL_ABSENT |
| I3 | Independent stream clocks | STRUCTURAL_ABSENT |

**Reading.** Qwen3-8B orders events, locates itself in time and attributes the provenance of statements reliably; it identifies the intervals that count as active time, erases content mostly effectively and resumes an interrupted task about half the time. The causal profile describes the architecture itself: five of the six tests require an operation a standard Transformer does not have — computing without emitting a token, evolving during silence, keeping a state independent of the visible context, rewriting its own state, running two streams on separate clocks — which matches the preregistered expectation for the `standard_transformer` control (`FCA_BENCH/CONTROL_EXPECTATIONS.json`).

## Why this baseline matters

It is the reference for the controlled comparison the manuscript privileges: the same base model with a latent loop or a dual stream enabled, same weights, seeds and decoding, next to a compute-matched twin and a sham intervention. The expected difference is not only "the new model answers better" but a transition from `STRUCTURAL_ABSENT` to a measured value — a new class of causal operations becoming available to the system at comparable behavioral performance.

## Contents

- `FCA_BENCH/` — protocol v0.2.7 (42 tests, specificity controls, sealing workflow, architecture-hook contract), the `fca_bench` package, the Qwen3-8B GPU runner (`run_qwen3_8b_gpu.py`, `config.qwen3_8b_gpu8.json`) and `verify_results.py`.
- `results/` — per-instance results and summary.

## Reproduce

Recompute every reported aggregate from the per-instance records (no model needed):

```bash
cd FCA_BENCH
python verify_results.py ../results/FCA027_GPU_R6_12TESTS_n20_FINAL.json
python -m fca_bench.run --list-tests
```

Run the baseline on a CUDA GPU with a local Qwen3-8B directory: see `FCA_BENCH/README.md`. `FCA_BENCH/MANIFEST.json` lists the SHA-256 of every file of the package.
