# T341 — Frozen graded functional closure protocol (proposed V45.1 §17.7bis)

**Status:** Methodology approved for implementation and preregistration, **not** a closure certificate on Qwen3-8B. The original V45.1 §17.7 remains unaltered. This document supersedes the draft clarifications of T340 when they differ.

## A. Fixed objects, no retrospective selection

Before accessing the assessment targets, freeze the finite universe U of comparable system realizations, causal individual/boundary, fixed model architecture and weights when studying within-Qwen variation, environment and intervention protocol, independently measured quantitative or typed organizational profile Phi_CO, target panels, target distance rules d_j and target tolerances epsilon_j, organizational cell map Q_eta, a **complete ordered grid of eta values**, uncertainty procedure, and the negative-control panel. The grid is reported **in its entirety**, not reduced to favorable eta values. Treat arms/interventions and worlds as realizations only when each has its own comparably measured paired organizational and external target outcomes. A donor configuration is not automatically a distinct complete system.

Pre-register epsilon_j based on repeatability/independent calibration, **never** on success or failure of the closure test. Keep PCI_T directional specificity separate from raw complexity. Keep the FCA behavioral and causal vectors separate and report each dimension individually. A structurally absent test is a valid typed categorical outcome different from measured zero; an invalid test is neither zero nor a valid categorical class.

## B. Primary closure criterion: direct pairs, not transitive chains

For each eta, target j and organizational cell c, define the eligible set

    U_eta,c = {s in U : Q_eta(Phi_CO(s)) == c and target j is valid/comparable}.

Its target diameter is

    D_eta,c,j = max_{s,t in U_eta,c} d_j(Y_j(s), Y_j(t)),

with an empty/singleton cell contributing no comparable pairs, not evidence of informative closure. The strict collision condition is `d_j(Y_j(s),Y_j(t)) > epsilon_j` for any within-cell pair. The formal implication holds for the declared universe exactly when every target cell diameter is <= epsilon_j. Retain counterexample IDs and the full pair and cell counts. **Never** use connected components of the pairwise `d<=epsilon` graph as a replacement: `0, 0.9*epsilon, 1.8*epsilon` is a required counterexample.

For noisy targets, construct simultaneous uncertainty coverage for **the maximum distance over all eligible within-cell pairs, all cells, all tested targets, AND every eta in the frozen grid**, including any shared multiple-comparison family used for the headline claim. This means joint/familywise coverage, not isolated pointwise intervals. When an exact deterministic target is available, exact comparisons suffice. A valid lower bound above epsilon proves a collision; if a simultaneously valid upper bound on every cell's diameter is <= epsilon closure passes; otherwise `OPEN_UNCERTAIN`. If any required target values are invalid/missing such that the declared assessment cannot be evaluated, use `OPEN_INVALID_OR_MISSING_TARGET` for that target/scope, not an invented numeric value.

## C. Non-triviality assessed **per target and per eta**

Define the target's full-universe diameter for target j:

    D_U,j = max_{s,t in U, comparable} d_j(Y_j(s), Y_j(t)).

A target is **resolved at its measurement resolution** only if `D_U,j > epsilon_j` (with appropriate uncertainty support if noisy). This is separate from target variation at finer, unvalidated resolution. For categorical typed structural absence, predeclare a categorical mismatch rule, e.g. distance 1 between absent and measured and epsilon<1, rather than interpreting absence as zero.

For an **informative** closure label, require all of the following, without treating them as extra conditions for the *logical* implication: (i) the direct-pair closure test passes, (ii) the target is resolved, (iii) at least two organizational cells are occupied, and (iv) at least one cell has >=2 independent realizations and at least one assessable within-cell pair. If a target is constant or unresolved (`D_U,j<=epsilon_j`), if all profiles are unique, or if comparable repeated pairs are absent, report `FORMAL_CLOSURE_ONLY_NON_INFORMATIVE` with specific reasons. If confidence bounds cannot decide whether the target is resolved, retain the appropriate uncertainty flag; do not certify informativeness.

## D. Secondary measures (NEVER replace the primary criterion)

1. **Information fraction:** for a separately preregistered categorical target partition `Z_j`, report `I(C_eta;Z_j)/H(Z_j)` only when `H(Z_j)>0`; otherwise undefined, not 0 or 1. Avoid graph-connectivity-induced chaining. Publish sample sizes, class frequencies, occupied and repeated cell counts, and same-cell pair counts. A plug-in value of 1 can be entirely due to memorization.
2. **Held-out/leave-one-realization-out prediction:** predict target class using *other* realizations with the identical organization-cell label, only on test realizations whose cell is repeated in the training portion. Report coverage, unseen-cell abstentions and class-prediction performance. If no evaluable repeats exist, report undefined. Any discretization/calibration must be independent of held-out targets.
3. **Permutation reference:** for each target and eta, compute the observed fraction of within-cell pairs with `d_j<=epsilon_j`, and the reference distribution after permuting target outcomes among comparable realizations **while keeping the organizational-cell membership and occupancy sizes fixed**. Preserve complete typed target outcomes. Predeclare a seed, permutation count or exact enumeration, and any legitimate exchangeability strata; do not permute across incomparable environments or measurement regimes. Publish the reference distribution and randomization uncertainty. This is descriptive discrimination, not a logical closure prerequisite.
4. **Adversarial rich-but-disorganized controls:** a separately preregistered instrument-discrimination panel (PCI_T designated direction versus norm-matched orthogonal directions and FCA target-specific comparisons), not an additional condition on the logical closure of U. Keep A-dagger as a supplemental coarse external diagnostic.

## E. Per-target, per-eta outcome logic

- `GRADED_CLOSURE_REFUTED_WITH_PAIR`: at least one reliably demonstrated within-cell target difference >epsilon; disclose concrete pair(s).
- `OPEN_INVALID_OR_MISSING_TARGET`: the target for this declared evaluation cannot be assessed due to invalid/missing data (a valid structural absence is NOT invalid).
- `OPEN_UNCERTAIN`: uncertainty cannot decide whether relevant max-difference inequalities hold, or informativeness cannot be supported under the preregistered measurement rule. Report which question is unresolved.
- `FORMAL_CLOSURE_ONLY_NON_INFORMATIVE`: the implication passes but the target is unresolved at epsilon, or there are no repeated assessable profile cells, or the organizational profile has fewer than two occupied cells; publish explicit reason(s).
- `GRADED_CLOSURE_INFORMATIVE_CERTIFIED`: all within-cell pairs pass with appropriate simultaneous coverage AND non-triviality/repeated-cell conditions hold.

For each eta and j, publish every occupied cell, number of realizations, number of eligible pairs, maximal observed within-cell distance, uncertainty interval if relevant, full-universe target diameter, target-resolution verdict, count and IDs of violating pairs, the information diagnostics (when defined), permutation baseline and holdout coverage. Never merge target-level decisions into one compensated scalar.

## F. Strict minimality and existing statuses

The requirements in B establish the finite-universe closure implication. The conditions in C distinguish **informative** closure from logically vacuous closure; D adds diagnostics only. Do not demand component-by-component necessity, universal generalization, full causal mediation, a full-model bisimulation, or recertification of pre-existing nonzero results merely to run this closure test.

Prior recorded nonzero certificates on Qwen3-8B L24 remain `Gamma`, `Delta`, `R`, and `Lambda0` = `NONZERO_CERTIFIED` under their declared contracts. There is currently no evaluated paired Qwen organization/independent-target table; Qwen graded closure remains `OPEN_NOT_ESTABLISHED`. **Do not encode `Gamma(u25)=0` as an established measurement:** the finite 11-role Gamma certificate is on u26; earlier positive measurements on an eight-role sub-individual and different role selections do not establish zero Gamma for u25 under the same contract.

## G. Required synthetic acceptance tests

1. `epsilon=1`, one cell with targets `0,0.9,1.8`: refuted because diameter 1.8 >1 despite two near-neighbor edges.
2. Unique organization cell per realization, nonconstant targets: formal implication true but `FORMAL_CLOSURE_ONLY_NON_INFORMATIVE`; plug-in I/H=1 does not change the status.
3. Constant target (or noisy target whose full-universe variation is below epsilon), with repeated organization cells: formal implication can hold but target is non-informative.
4. Repeated organizational cell with injected target separation >epsilon: refuted with explicit pair.
5. Epsilon so large that every target pair is within it, despite differing numeric scores: target is unresolved and closure is non-informative.
6. Valid typed structural absence compared to measured zero: distinct categorical statuses, not collapsed; invalid target remains invalid.
7. Complete preregistered eta grid: all eta results appear; a single favorable eta cannot be cherry-picked.
8. Noisy multiple-cell/multiple-target/multiple-eta data: pointwise upper bounds may pass while simultaneous upper bound does not; do not certify in that case.
9. Permutation preserves organization-cell sizes; if there is no eligible within-cell pair, report the pair-compliance statistic undefined.
10. Held-out prediction never scores unique/unseen organizational cells as successful generalization.
