# 15 — Causal Construct Validity Protocol

## What this folder is

Folder `14` shows that axis-completeness (localization) does not, by
itself, certify that the compact declared profile `C_O = (Γ, Δ, R(k), Λ)`
is sufficient to predict functional access. This folder is the empirical
and methodological response to that gap: it contains (a) an executed
adversarial repair chain that found and fixed two concrete
information-loss modes in the compact profile, (b) an independent
necessity argument for why `Γ`, `Δ`, `R(k)` are the right targets to test
at all, (c) the safeguards needed to make a causal-intervention test of the
profile-to-access relation actually identify that relation rather than a
confound, and (d) the frozen (never yet executed) protocol design for
running that test against real architecture pairs.

Everything under `scripts/` and `results/` in this folder is copied
as-is from the underlying campaigns' final canonical harnesses and frozen
JSON results — logic and data untouched, comments translated to English
where the source was in French.

## Contents

- `docs/organizational-closure-repair-chain.md` — the executed v0.3→v0.6.1
  chain: typed-binding insufficiency → typed-marginal repair →
  joint-synergy insufficiency → joint-typed-channel repair → static
  sufficiency theorem. This is where the two counterexamples referenced in
  folder `14`'s Compression Limit Theorem document come from.
- `docs/functional-access-necessity-bridge.md` — an independently defined
  access construct `A(e) = [K_A]_semantic`, with three proved necessity
  theorems (Breadth ⟹ Δ>0, bounded-depth Temporal ⟹ R(k)>0, Unity ⟹ Γ>0)
  and their conjunction `A_BTU`, explicitly labeled a necessity theorem,
  not a sufficiency theorem.
- `docs/intervention-identification-safeguards.md` — why a bare
  before/after intervention test is not enough (rival-pathway confound),
  and the same-profile-invariance / convergent-intervention / exclusion-
  sentinel safeguards needed, validated only on synthetic adjudicator
  cases.
- `docs/fci3-protocol-specification.md` — the full frozen FCI-3
  preregistration template (intervention families, hypotheses,
  manipulation checks, negative controls, failure/success criteria).
  **Never executed against real architecture pairs.**
- `scripts/` — the final canonical harnesses (`*.py`) together with the
  frozen protocol, theory, manifest, and fixture/pair files each harness
  reads at run time (hash-verified against a pre-registration manifest
  before scoring). They are kept side by side, unmodified, so a reviewer
  can run `python3 scripts/<harness>.py` directly from this folder and
  reproduce the recorded verdict. `results/` holds the same result JSON
  files as a stable, script-independent copy of record.
- `results/STATUS.json` — machine-readable status/verdict index, including
  which harnesses were actually re-executed for this archive edition and
  which could not be (see below).

## Reading this folder honestly

Most of this folder (the repair chain, the necessity bridge, the
identification-adjudicator validation) is **executed on frozen synthetic
fixtures**, and the verdicts are exact conditional theorems or adversarial
witnesses, not simulations tuned toward a preferred answer. The one
substantial exception is `fci3-protocol-specification.md`: it is a
carefully frozen design, but it has not yet been run against any real
architecture pair, and no claim in this folder should be read as if it had.

The `IG_S_DEEP_METROLOGY` workstream referenced by some of the source
material is deliberately **excluded** from this archive edition: its own
status file states that the referenced artifacts are not currently
materialized and that no instrument version is authorized yet.

Four of the nine executable items in this folder (v0.3 witness, v0.4 repair,
the FCI3 adjudicator validation, and the AF-identification-limits witness)
were re-run from the files in this archive alone and reproduced their
recorded verdicts exactly. Two items (the v0.5 joint-synergy witness and
the v0.6 repair, plus the v0.6.1 holdout that depends on it) could **not**
be re-executed here: their harnesses verify a source-hash manifest against
a `*_SOURCE_REFS_*.json` file that was not present in the material supplied
to this archive edition. Their frozen result files are included as the
record of the original run, clearly marked as not independently
re-verified in `results/STATUS.json`.

## Notation mapping from the source campaigns

The source campaigns predate this manuscript's fourth organizational
component Λ and used `C_F = [(Γ, Δ_prop, R(k))]`. Throughout this folder,
`Δ_prop` corresponds to this manuscript's `Δ`. None of the repair chain or
protocol content below used or required Λ; the mapping is a notational
courtesy for readers of the main manuscript, not a claim that these
results were re-run under the four-component `C_O`.

## Manuscript consequence

Together with folder `14`, this folder supports the following manuscript
posture: do not claim `C_O ⟹ A*` as an established identification theorem.
Do claim (1) that a compact organizational profile can be adversarially
tested and repaired against independently frozen access consequences, with
an exact sufficiency result for the static case; (2) that the three
targeted components have an independent necessity justification via BTU;
and (3) that a fully specified, falsifiable protocol exists for testing
causal construct validity empirically, which is exactly what §17.6/17.7 of
the manuscript (H_C†,I test, PCI_T covariate) and this folder's FCI-3
specification together aim at.
