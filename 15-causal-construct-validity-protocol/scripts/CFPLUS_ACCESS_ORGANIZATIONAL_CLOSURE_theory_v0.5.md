# CF+ ACCESS ORGANIZATIONAL CLOSURE v0.5

**Date:** 2026-09-14  
**Status:** attack theory frozen before witness generation.

## 1. Current candidate

After the v0.3 typed-binding counterexample and v0.4 repair, the current
organizational profile candidate is:

\[
C_F^+
=
[(\Gamma,\mathcal D_q,R(k))]
\]

with:

\[
\mathcal D_q
=
(\Delta_{\rm prop},\mathcal B_q)
\]

and:

\[
\mathcal B_q
=
[(r_j,K_{j,q})_{j=1}^{m}],
\qquad
K_{j,q}:p\mapsto P_q(Y^{(j)}\mid do(P=p)).
\]

Thus \(C_F^+\) retains the typed marginal intervention-response channel of
every certified role.

## 2. Remaining possible information loss

The collection of typed marginal channels does not, in general, determine the
joint channel:

\[
p
\mapsto
P_q(Y^{(1)},\ldots,Y^{(m)}\mid do(P=p)).
\]

Therefore two systems may have:

- the same \(\Gamma\);
- the same \(R(k)\);
- the same \(\Delta_{\rm prop}\);
- the same typed marginal role channels \(\mathcal B_q\);

while differing in information available only through **joint dependence
between roles**.

This is a distinct attack from the v0.3 typed-binding witness.

## 3. Target question

Does there exist a pair:

\[
C_F^+(e_1)=C_F^+(e_2)
\]

while an independently defined functional-access consequence differs because
the target content is encoded only in a cross-role relation?

If yes:

\[
\boxed{
C_F^+
\text{ is not sufficient for functional access.}
}
\]

The failure would remain inside the organizational family \(\mathcal C\); it
would not establish a new primitive family.

## 4. Frozen attack principle

Use two preregistered access roles and two independent source bits:

\[
P=(X,S),
\qquad
X,S\overset{iid}{\sim}Bernoulli(1/2).
\]

Each individual access role is marginally uniform and therefore carries zero
information about either source bit.

The two-role **joint dependence** carries exactly one bit.

In system `CONTENT-SYNERGY`, the joint relation carries the target content
\(X\).

In system `NUISANCE-SYNERGY`, the joint relation carries the independent
nuisance bit \(S\).

Because both systems carry one joint bit and have identical role marginals,
the current \(C_F^+\) compression is predicted to identify them.

The independent AF task is frozen to require flexible use of \(X\), not \(S\).

## 5. No post-hoc repair

If the frozen witness passes, retain the verdict.  Any repair must be frozen in
a separate subsequent phase.
