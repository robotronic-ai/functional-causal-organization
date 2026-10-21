# Protocol FCI-3 INTERVENTION IDENTIFICATION v0.2 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen before synthetic fixtures.

## A. Required intervention families

For each primary \(C_F\) component under test, preregister at least two
mechanistically distinct intervention families.

Examples:

### R

- latent-state recurrence;
- recurrent controller with matched state capacity.

### Delta_prop

- multi-role broadcast;
- distributed typed routing with matched output bandwidth.

### Gamma

- explicit bidirectional module coupling;
- shared causal bottleneck with matched local module competence.

The exact empirical implementations must be frozen before model results.

## B. Pairwise matching contract

For every intervention pair used in a same-profile test, require equivalence
margins for:

- \(I_G\);
- \(S\);
- compute;
- parameter count;
- external bandwidth;
- passive memory capacity;
- context length;
- optimization steps;
- base task performance;
- Functional Orientation.

Additional rivals may be added before data collection.

## C. Profile-equivalence rule

For profile component vector \(\mathbf c\), define frozen tolerances:

\[
\boldsymbol\epsilon_C.
\]

Two systems are `PROFILE-EQUIVALENT` iff every simultaneous confidence interval
for:

\[
c_j^{(1)}-c_j^{(2)}
\]

lies inside the corresponding equivalence band.

Otherwise return:

- `PROFILE-NOT-EQUIVALENT`, or
- `PROFILE-UNRESOLVED`.

## D. AF-equivalence rule

Similarly freeze:

\[
\boldsymbol\epsilon_A.
\]

Two systems are `AF-EQUIVALENT` iff every primary AF difference is inside the
corresponding equivalence band.

## E. Core sufficiency test

For pairs that are:

- `PROFILE-EQUIVALENT`;
- rival-equivalent;

test AF equivalence.

Outcomes:

### E1 — profile survives

\[
C_F\text{-equivalent}
\land
AF\text{-equivalent}.
\]

Status:

`FIBER-CONSISTENT`.

### E2 — profile fails

\[
C_F\text{-equivalent}
\land
AF\text{-not-equivalent}.
\]

Status:

\[
\boxed{\texttt{FIBER-VIOLATION}}.
\]

A replicated `FIBER-VIOLATION` is direct evidence that the compact profile is
insufficient in the tested domain.

## F. Convergent-instrument test

For each component-specific hypothesis, estimate effect ratios from at least
two valid intervention families.

Freeze an admissible heterogeneity margin:

\[
\epsilon_\beta.
\]

If the simultaneous confidence interval for:

\[
\beta_{j1}-\beta_{j2}
\]

lies outside:

\[
[-\epsilon_\beta,\epsilon_\beta],
\]

return:

`INSTRUMENT-DISAGREEMENT`.

Otherwise:

`INSTRUMENT-CONSISTENT`.

## G. Direct-effect sentinel

Every intervention family must include at least one outcome predicted to be
insensitive to the target \(C_F\) component but sensitive to plausible direct
architectural side-effects.

A sentinel change without the preregistered target AF pattern returns:

`EXCLUSION-UNSUPPORTED`.

This does not prove exclusion when absent; it is a falsification aid.

## H. Synthetic protocol-validation fixtures

The frozen harness must classify:

1. valid convergent instruments;
2. a direct-effect contaminated instrument;
3. a rival-variable mismatch;
4. a same-profile AF fiber violation;
5. an unresolved confidence-interval case.

These fixtures validate the adjudication logic only. They do not validate
\(C_F\) as a construct.
