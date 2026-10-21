# Exploratory Lambda-op Candidate Search

## Purpose

Find a compact Qwen3-8B Lambda-op descriptor that separates the functional collisions observed under `LAMBDA_OP_OPERATION_ADDRESS_EFFECT`, while retaining nontrivial fibers whenever possible.

## Epistemic status

This phase is target-informed and exploratory. Behavioral targets are allowed during candidate selection. Any selected candidate requires a new frozen validation panel before it can support a closure claim.

## Feature pool

For every non-ID operation (`KEEP`, `REVOKE`, `SUBSTITUTE`) and every address (`H1`, `H2`, `H3`), the search exposes:

- log change in primary K RMS relative to ID at the same address;
- log change in primary V RMS relative to ID at the same address;
- log change in reference K RMS relative to ID at the same address;
- log change in reference V RMS relative to ID at the same address;
- log change in corridor ratio relative to ID at the same address;
- an effect norm over the four K/V RMS log changes.

The pool also contains absolute log internal features for the ID state at each address.

## Search

Each scalar is discretized by empirical quantile cutpoints. Candidate bin counts are `2, 3, 4, 5, 6, 8, 12, 16`.

For each bin count, greedy forward selection repeatedly adds the feature that most reduces target-discordant pairs inside exact descriptor fibers. Search stops when no further improvement is available, all collision pairs are removed, or 24 features have been selected.

Candidates are ranked by:

1. zero target-discordant pairs;
2. nontrivial closure rather than injective closure;
3. fewer selected features;
4. fewer total fibers;
5. more systems retained in nontrivial fibers.

## Fresh validation

The chosen candidate, including all selected feature names and cutpoints, is copied unchanged into a new validation directory. A fresh 128-system schedule is generated using seed `202609270801`. The candidate cannot be altered using the fresh validation panel.
