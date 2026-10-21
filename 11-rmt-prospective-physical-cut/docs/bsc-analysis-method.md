# Analysis specification and bypass audit

Source material: `ANALYSIS_SPEC.md` and `BYPASS_AUDIT.md`. Implemented in
`scripts/analyze_rmt_cut.py`.

## Analysis specification

**Input.** A JSON response table satisfying the adapter contract
(`docs/adapter-contract-and-execution-status.md`).

**Bypass score.** For each `(context, horizon, forced_t)`, compute the total variation
between response laws under `source_c=0` and `source_c=1`. Report
`beta = max TV` across all of these. The theorem test is qualified only if the
no-bypass gate passes.

**Exact BSC reweighting.** For every noise level `p`, source content `c`, and forced
message `t`: `P(T=t|C=c) = 1-p` if `t=c`, else `p`. Because the extractor only ever
needs to force `T=0` and `T=1` once per context, every `p` on the grid is then computed
analytically by this exact reweighting; no Monte Carlo sampling is used.

**Distortion.** For each response law, correctness is the probability assigned to
`BITc`; `EMPTY` and `OTHER` are errors. Average over balanced `c`, the frozen
confirmatory (or qualification) contexts, and the exact BSC branch weights.

**Information terms.** `I_p = 1 - h2(p)`; `R_p = 1 - h2(D_p)` if `D_p <= 0.5` else `0`;
report the gap `G_p = I_p - R_p` (theorem requires `G_p >= 0`) and the distortion-
frontier diagnostic `F_p = D_p - p` (theorem requires `F_p >= 0`).

**Status labels** (the analyzer's complete, fixed vocabulary):

- `QUALIFIED_PASS` — access gate, bypass gate, and all theorem inequalities pass.
- `ACCESS_FAIL` — noiseless receiver distortion exceeds `0.10`.
- `BYPASS_FAIL` — fixed-`T` source invariance fails.
- `THEOREM_VIOLATION` — all qualification assumptions pass but a quantitative
  inequality fails.
- `NOT_EXECUTED_REAL_MODEL` — no real response table was supplied. This is the actual
  status of this campaign as a whole; see `docs/adapter-contract-and-execution-status.md`.

## Extractor no-bypass requirements (B1-B7)

**B1 — content-neutral final query.** The final query string must be identical
for `C=0` and `C=1` when `(T,Y,H)` are held fixed; the literal source content must
never appear in the query.

**B2 — forced-T decoupling.** The extractor must be able to run `source_c=0,
forced_t=t` and `source_c=1, forced_t=t` with every receiver-visible variable
otherwise identical. If the local model API cannot implement this decoupling, the
campaign cannot certify a no-bypass cut at all.

**B3 — Role-B clamp.** Role B must be set to a frozen collateral value
independent of `C`.

**B4 — cache hygiene.** Any standard prompt KV cache and any custom cache
location not part of the declared Role-A cut must be reset, clamped, or demonstrated
independent of `C`.

**B5 — no source label in consumed metadata.** Source IDs may be written to
analysis logs but must never enter model inference.

**B6 — fixed-T response invariance (the primary no-bypass gate).**
`beta = max_{y,t} TV(P(O|C=0,T=t,Y=y), P(O|C=1,T=t,Y=y))`. Required: `beta = 0` for
deterministic one-hot output, or `beta <= 0.01` for normalized logits.

**B7 — zero-information channel.** At `p=0.5`, `T` is independent of `C` by
construction; any above-chance source recovery at this noise level is an independent
bypass alarm, redundant with the falsification control in the preregistration.

## Coverage of B1-B7 in this bundle

Only the three synthetic controls (`results/synthetic_ideal.json`,
`synthetic_degraded.json`, `synthetic_bypass.json`) have been run through the analyzer.
B1-B5 are extractor-implementation requirements that cannot be verified from
response-table data alone — they must be enforced by the real extraction pipeline
itself, which does not exist in this bundle. Only B6 (fixed-`T` response
invariance) and B7 (zero-information-channel, folded into the theorem test at
`p=0.5`) are actually computed by `analyze_rmt_cut.py`, and only on synthetic data so
far. The synthetic bypass control confirms the B6 computation catches an obvious
violation (`max_fixed_T_source_TV = 1.0`), and the synthetic ideal/degraded controls
confirm it does not falsely trigger on a legitimate no-bypass receiver
(`max_fixed_T_source_TV = 0.0` in both cases). See `results/SYNTHETIC_VERIFICATION.json`.
