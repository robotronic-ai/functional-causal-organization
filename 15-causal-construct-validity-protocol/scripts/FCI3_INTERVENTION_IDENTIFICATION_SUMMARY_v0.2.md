# FCI-3 Intervention Identification — Summary v0.2

**Date:** 2026-09-14

## Main result

Architecture intervention is not enough for construct validation.

The pattern:

\[
Z_{\rm arch}\to C_F
\]

and:

\[
Z_{\rm arch}\to \mathcal A_F
\]

does not establish:

\[
C_F\to \mathcal A_F
\]

because the intervention may act through rival architectural pathways.

The stronger identification target is a profile-response relation under matched
rivals:

\[
m(\mathbf c,\mathbf r)
=
E[\mathbf a\mid do(\mathbf c),\mathbf r].
\]

## Strongest empirical sufficiency stress test

The primary test should be same-profile invariance:

\[
\mathbf c(e_1)\simeq\mathbf c(e_2),
\qquad
\mathbf r(e_1)\simeq\mathbf r(e_2)
\Rightarrow
\mathbf a(e_1)\simeq\mathbf a(e_2).
\]

A replicated violation gives:

\[
\boxed{\texttt{FIBER-VIOLATION}}
\]

and is direct evidence that the compact profile is incomplete in the tested
domain.

## Convergent interventions

At least two mechanistically distinct interventions should target the same
\(C_F\) component.

If they produce equivalent profile changes but incompatible AF effects, return:

\[
\boxed{\texttt{INSTRUMENT-DISAGREEMENT}}.
\]

This distinguishes profile incompleteness from a single implementation-specific
effect.

## Exclusion

An architectural manipulation is not assumed to satisfy exclusion merely
because compute, memory, and other known rivals are matched.

Direct-effect sentinels are falsification aids.

A sentinel failure returns:

\[
\boxed{\texttt{EXCLUSION-UNSUPPORTED}}.
\]

## Synthetic adjudicator validation

Five frozen cases were used:

- valid convergent intervention;
- direct-effect contamination;
- rival mismatch;
- same-profile AF fiber violation;
- unresolved profile-equivalence interval.

All were classified as preregistered.

Verdict:

\[
\boxed{\texttt{FCI3-IDENTIFICATION-ADJUDICATOR-PASS}}.
\]

This validates the protocol logic only. It is not evidence for \(C_F\)
construct validity.

## Consequence for the manuscript

The empirical bridge should not be stated merely as:

\[
do(\text{architecture})\to\Delta C_F\to\Delta A.
\]

It should state that construct validity requires:

1. successful manipulation of the causal profile;
2. rival-variable equivalence;
3. same-profile outcome invariance across distinct implementations;
4. convergent component-response effects;
5. prospectively frozen failure criteria.

This converts the \(C_F\)-to-access relation into a genuinely falsifiable
causal identification program.
