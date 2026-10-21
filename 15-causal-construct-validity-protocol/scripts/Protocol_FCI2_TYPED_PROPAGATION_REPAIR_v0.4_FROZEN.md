# Protocol FCI-2 Typed Propagation Repair v0.4 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen after the v0.3 witness and before repair implementation.

## 1. Purpose

Repair the v0.3 typed-binding insufficiency without adding a new primitive
family and without changing the independent access-consequence battery.

## 2. Keep the existing scalar summary

Retain:

\[
\Delta_{\rm prop}
=
\Delta_{\rm cap}\Pi_{\rm role,q}.
\]

It remains the scalar measure of how much perturbation information is
discriminable and how broadly it is distributed across certified roles.

It is not sufficient to describe **which typed content distinctions reach
which typed roles**.

## 3. Add a typed propagation signature

For each certified target role \(r_j\), retain its intervention-response
channel:

\[
K_{j,q}:
p
\mapsto
P_q(Y^{(j)}\mid do(P=p)).
\]

Define the typed propagation signature:

\[
\boxed{
\mathcal B_q
=
[
(r_j,K_{j,q})_{j=1}^{m}
]
}
\]

under admissible source/output recodings and type-preserving role transports.

Cross-type role permutations are not admissible.

The repaired differentiation object is:

\[
\boxed{
\mathcal D_q
=
(\Delta_{\rm prop},\mathcal B_q).
}
\]

The proposed repaired organizational profile is:

\[
\boxed{
C_F^{+}
=
[(\Gamma,\mathcal D_q,R(k))].
}
\]

No new scalar factor is introduced.

## 4. Frozen obligations

R0. The v0.3 aligned and cross-bound systems must have different
\(\mathcal B_q\).

R1. They must therefore have different \(C_F^{+}\) profiles.

R2. Their scalar \(\Delta_{\rm prop}\) values must remain equal to 1.

R3. Coherent within-role answer-label recoding must preserve
\(\mathcal B_q\)'s equivalence class.

R4. A type-preserving renaming of role identifiers must preserve the
equivalence class.

R5. A cross-type role swap must not be treated as an admissible transport.

R6. Pure duplication inside a certified role followed by the declared causal
role quotient must leave \(\mathcal B_q\) unchanged.

R7. For a one-role homogeneous system, the scalar gate
\(\Delta_{\rm prop}=0\) remains zero; retaining a structural channel signature
must not turn the scalar differentiation factor positive.

## 5. Interpretation

A pass establishes only that the typed propagation signature repairs the
specific v0.3 information-loss witness and respects the frozen invariances.

It does not establish universal sufficiency of \(C_F^{+}\) for functional
access.

Further adversarial attacks remain required.
