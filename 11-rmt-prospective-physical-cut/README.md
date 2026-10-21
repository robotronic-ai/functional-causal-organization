# 11 — RMT prospective physical-cut BSC campaign

Clean English edition of the RMT Prospective Physical-Cut BSC campaign: a preregistered
protocol to test, on a real trained model, whether a receiver's access to a piece of
content is bounded by the information-theoretic capacity of a *controlled* binary
symmetric channel (BSC) deliberately inserted at a declared write interface.

## Important execution status — read this first

**This campaign's real-model result is `NOT_EXECUTED_REAL_MODEL`.** The source bundle is
a complete preregistration, cut definition, bypass-qualification protocol, and
model-independent analyzer, together with three synthetic controls that validate the
analysis engine itself — but it does **not** contain the target model's checkpoint, the
local extraction environment, or any generated real-model response table. No real
pass/fail verdict on an actual trained model is claimed anywhere in this bundle, and
none should be inferred from the synthetic-control results below. This status is stated
explicitly in the source material's own `REAL_EXECUTION_STATUS.md`
(`docs/adapter-contract-and-execution-status.md` in this archive) and in `protocol.json`
(`real_model_status: "NOT_EXECUTED_REAL_MODEL"`), and is carried through unchanged here.

## What question this protocol answers (once run on a real model)

Take a controlled binary content variable `C in {0,1}` and force the receiver to see
only a noised version `T = C XOR N`, `N ~ Bernoulli(p)`, written at a designated
"Role-A" interface. For balanced `C` independent of receiver-side context `Y`,
`I(C; T | Y) = 1 - h2(p)` (binary channel capacity). Under a certified no-bypass cut —
i.e. every path from content to the receiver's output provably passes through `T` — the
general rate-distortion inequality `I(C; T|Y) >= R_{Z|Y}(D)` collapses, for balanced
binary targets and Hamming distortion, to a clean exact frontier:

```
D >= p.
```

Any receiver that reconstructs `C` with distortion `D < p` under a certified no-bypass
cut has access to content information that did not cross through `T` — a direct
contradiction of the frozen causal model. See `docs/theorem-and-cut-definition.md`.

## Main results

1. **The physical-cut inequality reduces exactly to `D >= p`** for a balanced binary
   content variable with Hamming distortion. Proved in `docs/theorem-and-cut-definition.md`.
2. **A complete preregistration is frozen**: model target, roles, content variable,
   receiver-context definition, the cut channel itself, the noise grid
   `p in {0, 0.1, ..., 0.5}`, the primary horizon `H=2`, the qualification/confirmatory
   context split, response categories, two qualification gates (noiseless access, and
   no-bypass), the primary theorem test, and a zero-capacity falsification control at
   `p=0.5`. See `docs/preregistration.md`.
3. **The model-independent analysis engine is implemented and exactly verified on three
   synthetic controls** (no model involved — these are hand-constructed response
   tables): an ideal no-bypass receiver that outputs `T` exactly and saturates
   `D = p` at every noise level (`status QUALIFIED_PASS`); a degraded no-bypass receiver
   that ignores `T` and always predicts `BIT0`, failing the noiseless-access gate
   (`status ACCESS_FAIL`, as expected — this is a correct rejection, not a bug); and a
   direct-content bypass receiver that reads `C` directly, producing `D < p` and a
   fixed-`T` source-dependence of `TV = 1.0` (`status BYPASS_FAIL`, correctly caught by
   the bypass gate). Overall synthetic-engine verification: **PASS**. See
   `docs/bsc-analysis-method.md` and `results/SYNTHETIC_VERIFICATION.json`.
4. **What remains before a real result can be reported**: the model-side adapter
   contract (exposing `source_c` and `forced_t` as independently controllable
   variables, and writing the required 32-call confirmatory+qualification response
   table) has not been implemented against any real model in this bundle. See
   `docs/adapter-contract-and-execution-status.md`.

## Reproduction discipline

This campaign follows the same discipline as the rest of the repository (see the root
`README.md` for the full statement):

1. A **protocol** is written and frozen (hashed conceptually, by being committed as a
   `_FROZEN`-equivalent file — here, `PREREGISTRATION.md` and `protocol.json`) *before*
   any candidate code is written or run. The preregistration is explicit that no
   checkpoint may be selected, and no analysis choice may be made, after viewing
   physical-cut results.
2. A **generator + private oracle** pattern is realized here as
   `generate_synthetic_controls.py` (the generator, producing the three hand-specified
   synthetic response tables as fixtures) and `analyze_rmt_cut.py` (the
   model-independent oracle/analyzer, which does not know in advance which of the three
   controls it is scoring).
3. Where this campaign would claim **"candidate-fixed holdout"**: the preregistration's
   qualification/confirmatory context split (indices `{0,2,4,6}` vs. `{1,3,5,7}`) is
   exactly this pattern — qualification gates must pass on one context set before the
   theorem test is read on the disjoint confirmatory set — but this has only been
   exercised on synthetic data in this bundle, never against a real held-out model run.
4. Verdicts are never silently converted into a single point value when the protocol
   doesn't license one: the analyzer emits a fixed vocabulary of statuses
   (`QUALIFIED_PASS`, `ACCESS_FAIL`, `BYPASS_FAIL`, `THEOREM_VIOLATION`,
   `NOT_EXECUTED_REAL_MODEL`) and a missing real response table yields
   `NOT_EXECUTED_REAL_MODEL`, never a fabricated or silently-omitted `D`/`R` value. This
   is the actual, current status of this campaign, stated plainly rather than
   converted into a claimed point result.

## How to verify / rerun

```bash
python3 scripts/generate_synthetic_controls.py     # writes synthetic_{ideal,degraded,bypass}.json
python3 scripts/analyze_rmt_cut.py results/synthetic_ideal.json --split confirmatory --horizon 2
python3 scripts/verify_campaign.py                 # regenerates controls and checks all three expected statuses
```

To produce a real result, the local Quantum/RMT extractor must implement the adapter
contract in `docs/adapter-contract-and-execution-status.md`, generate a real response
table in the schema it specifies, and then run:

```bash
python3 scripts/analyze_rmt_cut.py real_response_table.json --split confirmatory --horizon 2
```

No change to the preregistration is permitted after viewing that output.

## Contents

```
README.md
docs/
  theorem-and-cut-definition.md          — physical-cut inequality, D>=p reduction, and
                                            the frozen source/cut/receiver/context roles
  preregistration.md                     — frozen model, roles, content variable, noise
                                            grid, horizon, context split, gates, tests
  bsc-analysis-method.md                 — exact BSC reweighting, distortion/information
                                            computation, and the seven-point bypass audit
  adapter-contract-and-execution-status.md — required extractor capability/response
                                            schema, and the explicit NOT_EXECUTED_REAL_MODEL
                                            status with what has and has not been run
scripts/
  generate_synthetic_controls.py         — builds the three synthetic response tables
  analyze_rmt_cut.py                     — model-independent BSC physical-cut analyzer
  verify_campaign.py                     — regenerates controls, runs the analyzer on
                                            all three, checks expected statuses
results/
  protocol.json                          — frozen machine-readable protocol parameters
  synthetic_ideal.json                   — synthetic no-bypass receiver, saturates D=p
  synthetic_degraded.json                — synthetic no-bypass receiver, fails access gate
  synthetic_bypass.json                  — synthetic direct-content-access receiver
  SYNTHETIC_VERIFICATION.json            — full analyzer output on all three controls
  STATUS.json                            — machine-readable summary of verdicts and the
                                            explicit NOT_EXECUTED_REAL_MODEL status
```
