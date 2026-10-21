# Joint Typed Trajectory Refinement 08

## Purpose

Cross-Target Active Channel 07 showed that source-to-target routing topology is highly compressive but not closed. Refinement 08 therefore keeps the same functional target and the same measured cross-target panel, but replaces categorical routing summaries with a quantitative joint typed trajectory representation.

## Frozen feature construction

The feature construction is target-blind. For every operation, source address, target address, and internal metric measured under `ZERO_SUFFIX`, the representation retains:

1. the mean log response level across horizons 1 and 2; and
2. the ordered log change from horizon 1 to horizon 2.

The internal metrics are:

- `final_hidden_rel_rms`
- `primary_block_rel_rms`
- `post_primary_block_rel_rms`
- `reference_block_rel_rms`

Address identity and operation identity remain explicit. No behavioral outcome is used to construct or fit any feature or codebook.

## Exploratory codebook families

Deterministic standardized k-means is fitted without behavioral targets for the following representation families:

- `FULL`: one code for the complete operation/source/target trajectory tensor;
- `BY_OPERATION`: one code per operation;
- `SUBSTITUTE_BY_SOURCE`: one code for each `SUBSTITUTE` source address;
- `ALL_BY_OP_SOURCE`: one code for every operation-source slice.

For each family, `K` is searched over `{2,3,4,6,8,12}`. Behavioral targets are used only to choose among these already-fitted exploratory candidates.

## Useful-candidate rule

A candidate is useful only if both conditions hold on the exploratory panel:

- zero collision pairs; and
- nontrivial coverage at least 20%.

A useful candidate is frozen exactly as fitted and must be tested on a fresh panel without refitting normalization or codebooks.

## Functional target

The functional target remains the ordered 12-cell categorical behavior vector over `ID`, `KEEP`, `REVOKE`, and `SUBSTITUTE` crossed with `H1`, `H2`, and `H3`.
