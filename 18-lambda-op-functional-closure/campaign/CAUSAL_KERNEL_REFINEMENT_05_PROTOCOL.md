# Active Causal-Kernel Hybrid Refinement 05

## Scope

This phase remains exploratory and target-informed at candidate selection time. The underlying active causal-kernel measurements are target-blind.

The source panel contains 32 Qwen3-8B systems measured under:

- system operations: `ID`, `KEEP`, `REVOKE`, `SUBSTITUTE`;
- addresses: `H1`, `H2`, `H3`;
- experimental interventions: `ZERO_SUFFIX`, `HEAD_ROTATE`;
- intervention layers: 21-23;
- horizons: 1 and 2.

## Motivation

The preregistered active-kernel screen did not produce a useful candidate under its original family grid. Its best zero-collision candidate covered only 6.25% of systems through nontrivial fibers.

A near-frontier candidate, `K3_ALL_PER_OPERATION_K6`, reached 21.875% nontrivial coverage and had only one collision pair. That collision differed only in the address-specific `SUBSTITUTE` behavior.

The refinement therefore preserves a coarse operation-indexed active-kernel code and adds address-resolved `SUBSTITUTE` codes.

## Search space

The base descriptor is:

- one active-kernel vector quantizer for each of `ID`, `KEEP`, `REVOKE`, and `SUBSTITUTE`;
- common base codebook size chosen from `K = {2,3,4,5,6,8,10,12}`.

The optional refinements add zero, one, or two address-resolved `SUBSTITUTE` codebooks over distinct addresses in `H1/H2/H3`, with the same K grid.

All codebooks are fitted without behavioral targets. Behavioral targets are used only to select among the already fitted descriptor families.

## Exploratory selection rule

A candidate is considered useful only when both conditions hold:

1. zero functional collision pairs;
2. nontrivial fiber coverage at least 20%.

Within useful candidates, selection prefers fewer collision pairs, higher nontrivial coverage, lower total codebook size, lower maximum K, and fewer address refinements.

## Selected candidate

The deterministic search on the 32-system active-kernel panel selected:

`HYBRID_K3K2__SUB_H1_K4__SUB_H3_K12`

Structure:

- coarse operation-indexed base: K=2 for `ID`, `KEEP`, `REVOKE`, `SUBSTITUTE`;
- address-resolved `SUBSTITUTE:H1`: K=4;
- address-resolved `SUBSTITUTE:H3`: K=12.

Exploratory metrics:

- systems: 32;
- fibers: 27;
- nontrivial fibers: 3;
- systems in nontrivial fibers: 8;
- nontrivial coverage: 25.0%;
- collision fibers: 0;
- collision pairs: 0.

## Epistemic status

This candidate is target-informed because behavioral targets were used to select it from a large exploratory family. The codebooks themselves were target-blind.

The exploratory result is not a validation result. The candidate must be frozen and applied without refitting to a fresh panel.

## Fresh Validation 05

Validation 05 uses 64 fresh systems excluded from all prior Qwen panels used by this campaign.

The frozen candidate is copied before the validation panel is executed. No codebook centers, standardization parameters, K values, address refinements, or selection thresholds may be changed after opening the fresh panel.

Promotion requires:

- zero collision pairs; and
- nontrivial coverage at least 20%.
