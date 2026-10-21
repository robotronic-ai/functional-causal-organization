# Protocol FCI2 ACCESS-ROLE BINDING v0.3 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen before witness generation.

## Purpose

Attack sufficiency of the compressed functional-consciousness profile:

\[
C_F
=
[(\Gamma,\Delta_{\mathrm{prop}},R(k))]
\]

without using decoy roles or outcome-dependent context selection.

## Frozen content

The perturbation/content source is:

\[
P=(X_A,X_B)
\]

with:

\[
X_A,X_B
\overset{\mathrm{iid}}{\sim}
Bernoulli(1/2).
\]

Thus:

\[
H(P)=2\text{ bits}.
\]

## Frozen typed access roles

Exactly two roles exist in the access context:

- `TASK-A`;
- `TASK-B`.

Both are certified access-relevant before witness generation.

Their semantics are distinct:

- `TASK-A` must receive/use \(X_A\);
- `TASK-B` must receive/use \(X_B\).

Cross-type role permutations are forbidden.

Within-role answer-label recodings remain admissible.

There are no decoy roles.

## Frozen recurrent causal core

Both systems use the same two-bit recurrent kernel:

\[
Z_{1,t+1}
=
Z_{1,t}\oplus Z_{2,t},
\]

\[
Z_{2,t+1}
=
Z_{1,t}.
\]

Under the uniform intervention distribution this kernel is bijective.

For the only nontrivial bipartition:

\[
J(Z_1\to Z_2)=1,
\qquad
J(Z_2\to Z_1)=1.
\]

Therefore the current \(\Gamma\) candidate gives:

\[
\Gamma=1.
\]

Because the recurrent map is bijective and the full active state returns to the
same two typed state roles:

\[
R(k)=1
\]

for every positive integer \(k\).

## Witness-generation constraint

After this protocol is frozen, generate exactly two systems.

### Aligned system

\[
Y_{\mathrm{TASK-A}}=X_A,
\qquad
Y_{\mathrm{TASK-B}}=X_B.
\]

### Cross-bound system

\[
Y_{\mathrm{TASK-A}}=X_B,
\qquad
Y_{\mathrm{TASK-B}}=X_A.
\]

No other structural difference is allowed.

## Frozen C_F obligations

For both systems:

- B0: \(I(P;Y)=2\) bits;
- B1: \(\Delta_{\mathrm{cap}}=1\);
- B2: role Shapley information contributions are exactly one bit each;
- B3: role weights are \((1/2,1/2)\);
- B4: \(N_{\mathrm{eff}}=2\);
- B5: \(\Pi_{\mathrm{role},q}=1\);
- B6: \(\Delta_{\mathrm{prop}}=1\);
- B7: \(\Gamma=1\);
- B8: \(R(1)=R(2)=R(4)=1\);
- B9: the compressed \(C_F\) profiles are equal.

## Frozen independent-access obligation

`AF-FLEX` has two typed transformations:

- `TASK-A`: predict/use \(X_A\);
- `TASK-B`: predict/use \(X_B\).

The access scorer is fixed independently of \(C_F\).

For the aligned system, each role has the correct bit and achieves oracle
access.

For the cross-bound system, each role only receives the other independent bit.
The Bayes-optimal prediction of its required bit is therefore the content-blind
uniform predictor.

Required:

\[
a_{\mathrm{flex}}(\mathrm{aligned})=1,
\]

\[
a_{\mathrm{flex}}(\mathrm{crossbound})=0.
\]

## Anti-circularity obligations

- C0: role certification uses only the preregistered typed interface contract;
- C1: no role is added or removed after observing AF or C_F;
- C2: no mutual-information, Shapley, AF, or success value is used to define
  role membership;
- C3: cross-type role swaps are not admissible recodings;
- C4: a coherent within-role answer-label recoding preserves the verdict.

## Verdict

If B0–B9 and C0–C4 pass and `AF-FLEX` differs as frozen:

\[
\boxed{
\texttt{FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING}.
}
\]

This means only:

> The compressed coordinates
> \((\Gamma,\Delta_{\mathrm{prop}},R(k))\) do not retain enough typed causal
> organization to determine this independently defined functional-access
> consequence.

It does not establish phenomenal consciousness and does not create a new
primitive family.
