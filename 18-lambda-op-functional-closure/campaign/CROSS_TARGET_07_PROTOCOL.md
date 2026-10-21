# Cross-Target Active Channel 07

## Purpose

The local active-kernel and relational-summary branches did not yield a compact closed quotient. This screen therefore measures a source-conditioned cross-target response matrix rather than an address-local response.

The functional closure target is unchanged: the ordered 12-cell behavioral vector over

`{ID, KEEP, REVOKE, SUBSTITUTE} x {H1, H2, H3}`.

Only the internal descriptor changes.

## Source-conditioned internal channel

For each system operation `u` and source address `s`, the operation is applied to `s`. The resulting control state is probed at every target address `t` in `{H1,H2,H3}`.

For each `(u,s,t)` cell, the target-blind active response is measured under the experimental intervention `ZERO_SUFFIX` applied to layers 21-23. Relative RMS responses are recorded at horizons 1 and 2 for:

- final hidden state;
- primary block 21-23;
- post-primary block 24-26;
- reference block 30-32.

This yields a source-conditioned `3 x 3` response matrix for each operation, horizon, and metric.

The experimental intervention is global to the control suffix inside a source-conditioned control state. The screen therefore estimates a source-conditioned cross-target response channel; it does not claim a perfectly source-localized physical cut.

## Target independence

Behavior is not used to construct any descriptor feature. Behavioral outcomes are used only after descriptors are computed to evaluate exact fibers and select among the declared exploratory families.

## Exploratory descriptor families

The declared families encode categorical routing topology rather than fitted amplitude bins. They include forward source-to-target argmax maps, complete target rankings, reverse target-to-source maps, diagonal-versus-off-diagonal relations, and variants restricted to `SUBSTITUTE` or extended across all four operations.

No k-means or behavior-fitted threshold is used in this screen.

## Schedule

The exploratory panel contains 24 fresh systems. It is deliberately smaller than the validation panel because each source operation is probed at three internal target addresses.

## Promotion rule

A family is a useful exploratory candidate only if both conditions hold:

1. zero functional collision pairs;
2. at least 20% of systems belong to nontrivial fibers.

A collision-free nearly injective family is not promoted.

## Fresh validation

If the screen returns `USEFUL_EXPLORATORY_CANDIDATE`, Validation 07 freezes the selected family and tests it on 48 fresh systems without family changes or fitted parameters.
