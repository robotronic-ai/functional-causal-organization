# Adapter contract and real-execution status

Source material: `V086_ADAPTER_CONTRACT.md` and `REAL_EXECUTION_STATUS.md`.

## ACCESS-KERNEL v0.86 adapter contract

**Existing v0.86 facts used.** The existing (separate) ACCESS-KERNEL campaign defines
typed two-role memory states; Role-A and Role-B identity queries; `write_target`,
`write_select`, `erase_target`, `erase_select`, and delay operators; response
categories `BIT0`, `BIT1`, `EMPTY`, `OTHER`; smoke horizons `1,2`; full horizons
`1,2,3,4,5`; and cache reset before each prompt.

**Required new capability.** The physical-cut extractor must expose two independent
variables: `source_c` (the intended source label, used only for analysis and BSC
construction) and `forced_t` (the actual bit written into Role A). The model must
receive `forced_t` but must not receive `source_c` except indirectly through
`forced_t`.

**Required response table.** One JSON or NPZ row per (frozen context index, horizon,
`source_c in {0,1}`, `forced_t in {0,1}`), with fields: `context_index`, `horizon`,
`source_c`, `forced_t`, `response_prob_BIT0`, `response_prob_BIT1`,
`response_prob_EMPTY`, `response_prob_OTHER`, `query_text_hash`, `role_b_state_id`,
`cache_reset_confirmed`.

**Primary model call count.** Using four confirmatory contexts and primary horizon
`H=2`: `4 contexts x 2 C x 2 T = 16` confirmatory model calls. The qualification split
requires another 16 calls. The primary campaign therefore needs **32 calls** after the
adapter exists. Secondary horizons `1,3,4,5` increase this to 160 total calls across
all eight contexts and five horizons.

**Preferred output probabilities.** Forced-choice normalized logits over the four
frozen response categories are preferred; if unavailable, the one-hot greedy v0.86
output is acceptable for the first exact causal test.

## Real execution status

**Status: `NOT_EXECUTED_REAL_MODEL`.**

**Reason.** The supplied ACCESS-KERNEL v0.77–v0.86 archive contains protocol
documentation and states that the real Quantum extraction is intended for the user's
local WSL project environment. The runtime this bundle was produced in does not expose
the documented local project path, checkpoint, v0.86 extractor source, or a generated
response bundle. A real Quantum/RMT numerical result therefore cannot be produced from
the supplied archive alone.

**What has been executed here.** The model-independent physical-cut analysis
(`scripts/analyze_rmt_cut.py`) has been implemented and exactly verified on three
synthetic controls (`scripts/generate_synthetic_controls.py`,
`results/SYNTHETIC_VERIFICATION.json`):

1. an ideal no-bypass receiver that outputs `T` and saturates `D=p` at every noise
   level;
2. a degraded no-bypass receiver that ignores `T` and obeys `D>=p` strictly (in fact
   `D=0.5` at every `p`, since it always predicts `BIT0` regardless of input);
3. a direct-content bypass receiver that violates fixed-`T` invariance and produces
   `D<p`.

These synthetic tests validate that the analysis correctly **distinguishes** a
legitimate cut from a bypassed cut and from an uninformative receiver — this is a
test of the analyzer's own correctness, not a measurement of any real model.

**Exact next action in the local Quantum environment.** Implement the adapter
contract above, generate the response table, and run:

```bash
python3 scripts/analyze_rmt_cut.py real_response_table.json --split confirmatory --horizon 2
```

No change to the preregistration (`docs/preregistration.md`) is allowed after viewing
that output.
