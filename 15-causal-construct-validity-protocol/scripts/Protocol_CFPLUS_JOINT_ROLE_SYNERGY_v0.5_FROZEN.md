# Protocol CF+ JOINT-ROLE SYNERGY ATTACK v0.5 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen before witness generation.

## 1. Source

\[
P=(X,S)
\]

with:

\[
X,S\overset{iid}{\sim}Bernoulli(1/2).
\]

`X` is the access-relevant content.

`S` is an access-neutral nuisance/source-status bit.

The two bits are independent.

## 2. Certified access roles

Exactly two roles exist:

- `ACCESS-L`;
- `ACCESS-R`.

Both roles are certified access-relevant before witness generation.

No role is added or removed after observing \(C_F^+\) or AF.

Role identifiers may be renamed; cross-type role permutations are not
admissible unless the type contract is transported coherently.

## 3. Shared recurrent core

Both systems use the same recurrent two-bit core:

\[
Z_{1,t+1}=Z_{1,t}\oplus Z_{2,t},
\qquad
Z_{2,t+1}=Z_{1,t}.
\]

As in the prior exact fixture:

\[
\Gamma=1
\]

and:

\[
R(1)=R(2)=R(4)=1.
\]

## 4. Witness generation rule

Introduce one fresh internal random bit:

\[
U\sim Bernoulli(1/2)
\]

independent of \(X,S\).

### CONTENT-SYNERGY

\[
Y_L=U,
\qquad
Y_R=U\oplus X.
\]

Therefore:

\[
X=Y_L\oplus Y_R.
\]

### NUISANCE-SYNERGY

\[
Y_L=U,
\qquad
Y_R=U\oplus S.
\]

Therefore:

\[
S=Y_L\oplus Y_R.
\]

No other difference is allowed.

## 5. Frozen C_F+ obligations

For both systems:

- J0: each typed marginal channel
  \(P\mapsto P(Y_L|P)\) and \(P\mapsto P(Y_R|P)\)
  is uniform and independent of \(P\);
- J1: \(I(P;Y_L)=I(P;Y_R)=0\);
- J2: \(I(P;Y_L,Y_R)=1\) bit;
- J3: role Shapley contributions are exactly \(1/2\) bit each;
- J4: role weights are \((1/2,1/2)\);
- J5: \(N_{\rm eff}=2\);
- J6: \(\Pi_{\rm role,q}=1\);
- J7: \(\Delta_{\rm cap}=1/2\);
- J8: \(\Delta_{\rm prop}=1/2\);
- J9: the v0.4 typed marginal propagation signatures
  \(\mathcal B_q\) are equal;
- J10: \(\Gamma\) is equal;
- J11: \(R(1),R(2),R(4)\) are equal;
- J12: the complete current \(C_F^+\) profiles are equal.

## 6. Frozen independent access consequence

The AF task requires access to the target content \(X\).

The task may read both certified access roles jointly.

The scoring rule is the already frozen AF log-score normalization.

### CONTENT-SYNERGY

Because:

\[
X=Y_L\oplus Y_R,
\]

the system returns the oracle distribution for \(X\).

Required:

\[
a_{\rm flex}=1.
\]

### NUISANCE-SYNERGY

Because:

\[
(Y_L,Y_R)
\]

contains only \(S\) in its cross-role relation and \(S\perp X\), the
Bayes-optimal predictor of \(X\) is the content-blind uniform predictor.

Required:

\[
a_{\rm flex}=0.
\]

## 7. Anti-circularity

- A0: access-role certification is frozen before witness generation;
- A1: the access target \(X\) is frozen before witness generation;
- A2: `S` is declared access-neutral before witness generation;
- A3: no AF score is used in \(\Gamma,\Delta,R,\mathcal B_q\);
- A4: no \(C_F^+\) quantity is used in AF scoring;
- A5: coherent output-label recoding preserves the verdict.

## 8. Verdict

If J0–J12 and A0–A5 pass:

\[
\boxed{
\texttt{CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY}.
}
\]

Interpretation:

> The v0.4 typed marginal propagation repair still discards
> access-relevant information carried only by joint dependence across certified
> roles.
