# CF–Access Organizational Closure — Static Phase Summary v0.6.1

**Date:** 2026-09-14

## Result 1 — typed marginal repair is insufficient

The v0.4 profile:

\[
C_F^+
=
[(\Gamma,(\Delta_{\rm prop},\mathcal B_q),R(k))]
\]

with one typed marginal response channel per certified role was attacked
prospectively.

Two systems were constructed with identical:

\[
\Gamma=1,\qquad
R(1)=R(2)=R(4)=1,
\]

\[
\Delta_{\rm cap}=1/2,\qquad
\Pi_{\rm role,q}=1,\qquad
\Delta_{\rm prop}=1/2,
\]

and identical typed marginal channels.

In `CONTENT-SYNERGY`, the target content \(X\) exists only in the cross-role
relation:

\[
X=Y_L\oplus Y_R.
\]

In `NUISANCE-SYNERGY`, the same amount of joint information carries an
independent nuisance bit \(S\):

\[
S=Y_L\oplus Y_R.
\]

The independent AF-FLEX consequence is:

\[
1
\]

versus:

\[
0.
\]

Verdict:

\[
\boxed{
\texttt{CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY}
}
\]

Therefore typed role marginals are not sufficient.

## Result 2 — joint typed response channel

The repair retains the full typed joint intervention-response channel:

\[
K^{\rm joint}_{q,\mathcal R}
:
p\mapsto
P_q(Y_{\mathcal R}\mid do(P=p)).
\]

The differentiation object becomes:

\[
\boxed{
\mathcal D_q^{++}
=
(
\Delta_{\rm prop},
[K^{\rm joint}_{q,\mathcal R}]_{\rm adm}
)
}
\]

and:

\[
\boxed{
C_F^{++}
=
[
(
\Gamma,
\mathcal D_q^{++},
R(k)
)
].
}
\]

This repairs both:

- the v0.3 typed content-to-role witness;
- the v0.5 cross-role synergy witness.

## Result 3 — static-access sufficiency theorem

At fixed source distribution, target semantics, loss, and admissible
decision-rule class, equal typed joint response channels imply equal expected
loss for every fixed static decision rule, hence equal optimal achievable loss.

Thus the typed joint channel is sufficient for every static access consequence
that reads only the certified joint response.

This is an exact conditional theorem, not a finite non-refutation.

## Control adjudication

The initial v0.6 harness entered `REVIEW` because one cross-type swap control
used a channel exactly symmetric under that swap.

The candidate was not modified.

A candidate-fixed asymmetric holdout was frozen and passed:

\[
\boxed{
\texttt{CFPLUSPLUS-CROSS-TYPE-HOLDOUT-PASS}
}
\]

The candidate is therefore retained.

## Current boundary

The static part of the access problem is now characterized much more strongly.

Still open are consequences that depend on the **causal process across time**:

- AF-DELAY;
- AF-UPDATE;
- temporal order;
- path-specific gating;
- transition-specific content binding.

The next object to test is therefore not another static scalar.  It is a typed
trajectory/intervention-process extension of the causal response kernel.
