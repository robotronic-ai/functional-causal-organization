# Protocol CF++ JOINT TYPED PROPAGATION REPAIR v0.6 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen before repair candidate implementation.

## Purpose

Replace the insufficient list of typed marginal response channels with the
typed joint intervention-response channel while preserving the scalar
\(\Delta_{\rm prop}\).

## Candidate form

\[
\mathcal D_q^{++}
=
(
\Delta_{\rm prop},
[K^{\rm joint}_{q,\mathcal R}]_{\rm adm}
)
\]

and:

\[
C_F^{++}
=
[(\Gamma,\mathcal D_q^{++},R(k))].
\]

## Frozen obligations

### Previous typed-binding witness

T0. The v0.3 aligned and cross-bound systems must have different joint-channel
equivalence classes.

T1. Their \(C_F^{++}\) profiles must differ.

T2. Their scalar \(\Delta_{\rm prop}\) values remain unchanged.

### v0.5 synergy witness

S0. `CONTENT-SYNERGY` and `NUISANCE-SYNERGY` must have different joint-channel
equivalence classes.

S1. Their \(C_F^{++}\) profiles must differ.

S2. Their scalar \(\Delta_{\rm prop}\) values remain equal to \(1/2\).

### Invariances

I0. Coherent bijective recoding of output labels within every role preserves
the joint-channel equivalence class.

I1. Type-preserving renaming of role identifiers preserves the equivalence
class.

I2. Cross-type role permutations are not free unless the type contract is
transported.

I3. Representation-level duplication already collapsed by the causal role
quotient must not alter the joint channel.

I4. A one-role homogeneous fixture may retain a structural joint channel but
its scalar gate remains \(\Delta_{\rm prop}=0\).

## No sufficiency overclaim

A pass establishes:

1. exact repair of the v0.3 and v0.5 witnesses;
2. static-access sufficiency conditional on the theorem's declared scope.

It does not establish full sufficiency for delayed/update/temporal access.
