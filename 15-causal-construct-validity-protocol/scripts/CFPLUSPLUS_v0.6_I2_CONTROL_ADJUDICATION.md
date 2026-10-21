# CF++ v0.6 — I2 Control Adjudication

**Date:** 2026-09-14

## Initial result

The v0.6 harness returned:

\[
\texttt{CFPLUSPLUS-JOINT-TYPED-PROPAGATION-REVIEW}
\]

because obligation `I2_CROSS_TYPE_SWAP_NOT_FREE` failed.

All other frozen obligations passed, including repair of both prior
counterexamples.

## Diagnosis

The I2 fixture used the v0.5 `CONTENT-SYNERGY` channel:

\[
Y_L=U,
\qquad
Y_R=U\oplus X,
\qquad
U\sim Bernoulli(1/2).
\]

Swapping the two output coordinates gives:

\[
(Y_R,Y_L)
=
(U\oplus X,U).
\]

For every fixed \(X\), this has exactly the same joint distribution as
\((U,U\oplus X)\).

Therefore the frozen I2 fixture is **distributionally symmetric under the
forbidden swap**.

No statistic computed from the joint intervention-response distribution can
distinguish that swap on this fixture.

The failed check is thus non-discriminating: it does not show that the
candidate treats cross-type permutations as admissible.

## Required correction

Do not modify the v0.6 candidate.

Run a candidate-fixed holdout using an asymmetric typed channel for which a
cross-type output swap changes the joint intervention-response distribution.

Until that holdout passes, retain v0.6 status as `REVIEW`.
