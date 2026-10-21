# Preregistration

Source material: `PREREGISTRATION.md`, machine-readable mirror in `results/protocol.json`.
Every parameter below is frozen before any real-model response table is read.

## Frozen model target

The same Quantum/RMT checkpoint and loader selected by the project's ACCESS-KERNEL
v0.86 line. No different checkpoint may be selected after viewing physical-cut
results.

## Frozen roles

- **Source role**: `Role A` write interface.
- **Collateral role**: `Role B`, held fixed independently of source content.
- **Receiver**: the installed final response readout for an identity query of `Role A`.

## Frozen content variable

`C in {0,1}` with equal weight on `BIT0` and `BIT1` within every frozen receiver-side
context, so `H(C|Y) = 1` bit by construction.

## Frozen receiver-side context `Y`

Every variable available to the receiver independently of source content: the frozen
context/background item, the fixed Role-B collateral value, the query template, the
delay-marker template, and any other fixed non-content prompt material. The
confirmatory design must include both source contents for every value of `Y`.

## Frozen cut

The only source-content message allowed to cross the cut is `T = C XOR N`,
`N ~ Bernoulli(p)`. The value actually written to Role A is `T`, never `C`.

## Frozen noise grid

`p in {0, 0.10, 0.20, 0.30, 0.40, 0.50}`. Model calls do not need to be repeated for
every `p`: the extractor evaluates both forced cut values `T=0` and `T=1` once, and
every BSC noise level is then evaluated analytically by reweighting
(`docs/analysis-and-bypass-audit.md`).

## Frozen horizon

Primary confirmatory horizon: **`H = 2`**, already present in both the smoke and full
ACCESS-KERNEL v0.86 batteries. Secondary descriptive horizons, if the full extractor is
available: `H in {1,3,4,5}`. No secondary horizon can substitute for the primary
result.

## Frozen context split

Using the existing ordered list of eight full-mode v0.86 contexts: qualification
contexts `{0, 2, 4, 6}`; confirmatory contexts `{1, 3, 5, 7}`. The split is index-based
and frozen before the physical-cut result is read — qualification gates are checked on
one half, the theorem test is read only on the disjoint other half.

## Frozen response categories

`BIT0`, `BIT1`, `EMPTY`, `OTHER` (the existing v0.86 categories). For Hamming distortion
relative to source `C`, only the matching `BITC` response counts as correct; `EMPTY`
and `OTHER` count as errors. Forced-choice normalized logits are preferred when
available; deterministic one-hot responses are also accepted.

## Primary qualification gates

**Gate Q1 — access exists when the cut is noiseless.** At `p=0`, `H=2`: `D_0 <= 0.10`.
If this fails, the physical-cut theorem is uninformative because the declared receiver
does not realize the target access task at all.

**Gate Q2 — no bypass.** For every confirmatory context `y`, cut message `t`, and the
primary horizon, compare runs with different source labels `c` while holding `t` fixed.
The final response law must be invariant to `c`:
`P(O|C=0,T=t,Y=y) = P(O|C=1,T=t,Y=y)`. For one-hot v0.86 responses the required maximum
TV defect is exactly `0`; for normalized logits the preregistered tolerance is
`max TV <= 0.01`. If this gate fails, the physical-cut inequality is not tested at all,
because `T` is not shown to be the complete content-carrying transcript.

## Primary theorem test

For every preregistered `p`, compute confirmatory distortion `D_p` exactly by weighting
the two forced cut-message branches. Compute `I_p = 1 - h2(p)` and `R_p = R(D_p)`.
Primary theorem criterion: `I_p - R_p >= -1e-9` for every `p` on the grid. Equivalent
distortion-frontier criterion: `D_p >= p`.

## Zero-capacity falsification control

At `p = 0.5`, `I(C;T|Y) = 0`. Under a qualified no-bypass design, the receiver cannot
reconstruct balanced `C` with distortion below `0.5`, so `D_{0.5} >= 0.5` is required.
Any `D_{0.5} < 0.5` is direct evidence of a bypass, of receiver-side side information
omitted from `Y`, or of an invalid extraction implementation — it is a built-in
falsifier, not a tunable target.
