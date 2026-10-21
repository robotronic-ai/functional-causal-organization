# Protocol CF–AF NECESSARY-CONDITION ATTACK v0.1 — FROZEN

**Date:** 2026-09-14  
**Status:** frozen before witness construction.

## Upstream frozen scorer

Use the existing AF candidate unchanged.

The scorer hash is recorded in the manifest.

## Obligations

### N0 — AF scorer independence

The candidate AF scorer must remain byte-identical to the previously frozen
candidate.

### N1 — oracle AF profile for acyclic witness

The acyclic compiled witness must score:

\[
a_{\rm flex}
=
a_{\rm delay}
=
a_{\rm update}
=
a_{\rm status}
=
1.
\]

### N2 — zero recurrence

The acyclic witness must have no path returning to the same declared
functional role within the horizon:

\[
R(k)=0
\]

for every tested positive \(k\).

### N3 — zero Gamma with oracle AF

The decomposed-parallel witness must preserve the oracle AF profile while
having an admissible partition with zero cross-cut causal information:

\[
\Gamma=0.
\]

### N4 — zero Delta_prop with AF-FLEX oracle

The single-role multiplexed witness must have:

\[
a_{\rm flex}=1,
\]

positive discriminability, one certified causal role, and therefore:

\[
\Pi_{\rm role,q}=0,
\qquad
\Delta_{\rm prop}=0.
\]

### N5 — answer-label recoding

A coherent answer-label bijection must preserve all AF scores.

## Verdict

If N0–N5 pass:

\[
\boxed{
\texttt{AF-NOT-STRUCTURALLY-IDENTIFYING}
}
\]

and the three necessary-condition implications for the AF indicator profile
are rejected on the frozen finite fixtures.
