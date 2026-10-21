# Content-Typed Pooled Refinement 13 Protocol

## Purpose

Search quickly for a new exploratory Lambda proxy candidate after Fresh Validation 12 refuted the two-feature content-typed quotient.

Construction 12 and Fresh Validation 12 are now both opened exploratory data. They are pooled into a 96-system training set. Any candidate selected here requires a new independent Fresh Validation 13.

## Fixed system algebra

System operations remain exactly:

- `ID`
- `KEEP`
- `REVOKE`
- `SUBSTITUTE`

Experimental interventions remain distinct from system operations:

- `ZERO_SUFFIX`
- `HEAD_ROTATE`

## Measured descriptor source

The measured rich proxy remains the 24D typed content-effect kernel from Phase 12:

- operations measured: `REVOKE`, `SUBSTITUTE`
- addresses: `H1`, `H2`, `H3`
- interventions: `ZERO_SUFFIX`, `HEAD_ROTATE`
- signed effects: `DONOR_MINUS_ORIGINAL`, `NONE_MINUS_ORIGINAL`

Baseline outcome and baseline absolute log-probabilities remain excluded from descriptor construction.

## Opened-data search

Inputs:

- 32-system Content-Typed 12 construction panel
- 64-system refuted Fresh Validation 12 panel

Total: 96 opened systems.

Standardization and quantile edges are fit without behavioral targets. Feature selection is explicitly exploratory and target-informed.

Families searched:

- `SUBSTITUTE_ONLY`
- `SUBSTITUTE_DONOR`
- `SUBSTITUTE_NONE`
- `REVOKE_SUBSTITUTE`

Search method:

- beam width: 48
- maximum selected features: 8
- bins per selected feature: 2, 3, 4, 5, 6, or 8

A useful exploratory candidate must satisfy on the pooled 96 opened systems:

- zero collision pairs
- nontrivial coverage at least 0.20
- within-fiber distortion at most 0.25

Candidate ranking additionally prefers higher minimum nontrivial coverage across the two opened source panels.

## Fresh Validation 13

If a candidate is found, all learned quantities are frozen before a new schedule is created:

- standardizer
- selected feature indices
- quantile edges
- bin counts

Fresh Validation 13 uses 64 new systems and excludes exact binding/donor tuples from all opened panels to date, including Fresh Validation 12.

Promotion requires the same fixed rule:

- zero collision pairs
- nontrivial coverage at least 0.20
- within-fiber distortion at most 0.25

## Scope

This is an exploratory refinement of the typed content-effect Lambda proxy. It is not a scalar validation of complete `C_O=(Gamma,Delta,R,Lambda)`.
