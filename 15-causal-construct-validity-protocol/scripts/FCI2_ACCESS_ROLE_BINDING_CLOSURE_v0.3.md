# FCI-2 Access-Role Binding Attack — Closure v0.3

**Date:** 2026-09-14

## Verdict

\[
\boxed{
\texttt{FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING}
}
\]

All frozen obligations passed.

## Exact witness

Two systems use the same recurrent causal core:

\[
Z_{1,t+1}=Z_{1,t}\oplus Z_{2,t},
\qquad
Z_{2,t+1}=Z_{1,t}.
\]

They therefore have the same:

\[
\Gamma=1
\]

and:

\[
R(1)=R(2)=R(4)=1.
\]

The perturbation/content source is:

\[
P=(X_A,X_B),
\qquad
X_A,X_B\overset{iid}{\sim}Bernoulli(1/2).
\]

Both systems have exactly two preregistered access roles:

- `TASK-A`, which semantically requires \(X_A\);
- `TASK-B`, which semantically requires \(X_B\).

No decoy roles exist.

### Aligned system

\[
Y_{\rm TASK-A}=X_A,
\qquad
Y_{\rm TASK-B}=X_B.
\]

### Cross-bound system

\[
Y_{\rm TASK-A}=X_B,
\qquad
Y_{\rm TASK-B}=X_A.
\]

For both systems:

\[
I(P;Y)=2,
\]

\[
\Delta_{\rm cap}=1,
\]

\[
(\phi_{\rm TASK-A},\phi_{\rm TASK-B})=(1,1),
\]

\[
(w_{\rm TASK-A},w_{\rm TASK-B})=(1/2,1/2),
\]

\[
N_{\rm eff}=2,
\qquad
\Pi_{\rm role,q}=1,
\qquad
\Delta_{\rm prop}=1.
\]

Thus the current compressed profiles are exactly equal:

\[
\boxed{
C_F^{\rm compressed}(A)
=
C_F^{\rm compressed}(B)
=
[(1,1,R(k)=1)].
}
\]

The independently frozen access consequence differs:

\[
a_{\rm flex}(A)=1,
\qquad
a_{\rm flex}(B)=0.
\]

The verdict survives coherent within-role answer-label recoding.

## Interpretation

The result does **not** refute the causal family \(\mathcal C\).

It refutes the sufficiency of the current compression:

\[
\boxed{
\mathcal C
\longrightarrow
(\Gamma,\Delta_{\rm prop},R(k))
}
\]

for typed functional-access consequences.

The missing information is not the amount of propagation and not the effective
number of reached roles.  It is the **binding between typed content
distinctions and typed access roles**.

Therefore:

\[
\boxed{
C_F(e_1)=C_F(e_2)
\not\Rightarrow
\mathcal A_F(e_1)=\mathcal A_F(e_2)
}
\]

for the current numerical profile.

## Scientific consequence

The bridge:

\[
A=H(C_F)
\]

cannot currently be claimed even as a finite non-refuted profile-sufficiency
statement.

The stronger and still viable statement is:

\[
A=H_C(\mathcal C)
\]

conditional on four-family localization assumptions.

A repair must retain enough typed propagation structure to distinguish the
witness without introducing a new primitive family.

No repair is adopted in this closure file.
