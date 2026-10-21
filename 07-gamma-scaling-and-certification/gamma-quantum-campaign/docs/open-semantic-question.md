# The remaining open question: bottleneck saturation vs. mean utilization

The candidate normalization (`directional-normalization.md`) and the previous manuscript
normalization answer **different questions** on unequal-capacity cuts, and this difference is a
semantic choice, not a numerical error in either formula.

- The previous form asks approximately: *how much bidirectional causal information crosses the
  cut relative to the capacity of the smaller side?* (bottleneck saturation)
- The candidate form asks: *what fraction of each side's intervention entropy is transmitted
  across the cut, averaged across directions?* (mean utilization of both sides' causal degrees
  of freedom)

## The XOR-5 illustration

A one-bit causal signal crossing from a four-bit source side is *complete* relative to a
one-bit receiving bottleneck (previous normalization: saturates at 1), but it is only
*one quarter* of the four-bit source's entropy (candidate normalization: `0.25`). Neither
reading is a mathematical error; they are different constructs.

## If the construct should measure balanced utilization of both sides

The candidate normalization is preferable because it is: bounded without future-state
cardinality assumptions; stable under deeper state-space closure (§closure invariance); compatible
with non-uniform intervention distributions; invariant under coherent physical replication when
`H_q` is computed correctly (§replication invariance); and explicitly directional.

## If the construct should measure saturation of the smaller causal bottleneck

The previous normalization should be retained, but its denominator must be redefined from a
stable causal macro alphabet rather than raw K/V state cardinality — the closure-sensitivity
result (v11) shows the raw-cardinality version is not fit for purpose regardless of which
semantics is chosen.

## Current recommendation (v15, status: `current`)

Do **not** edit the manuscript definition of Γ yet. The next formal step is to state both
candidate constructs side by side and test them against a preregistered set of axioms for causal
integration; the decisive axiom is the treatment of unequal-capacity cuts. In the meantime, the
empirical Quantum result should continue to report the raw directional quantities `J_q`
alongside any normalized Γ value, so the mechanistic conclusion (layer 35's token-mediated
re-entry, the entry-block weak cut, the horizon-dependent return signal) remains independent of
the eventual normalization choice.

The recommended descriptive name for the candidate, regardless of which semantics is eventually
adopted for the manuscript's primary Γ, is **source-normalized interventional information
efficiency**.

## Literature anchors for the information-theoretic ingredients (not the causal construction)

- MIT OpenCourseWare 6.441, *Information Theory*, Ch. 2 — `I(X;Y) ≤ min{H(X),H(Y)}`.
- Stanford EE376A lecture notes — mutual information identities and conditional information.
- *Behavior Research Methods*, "Reliability of a probabilistic knowledge structure" —
  uncertainty coefficient `I(X;Y)/H(Y)` and its `[0,1]` range.
- *Data Mining and Knowledge Discovery*, "TCMI: a non-parametric mutual-dependence estimator" —
  normalized mutual information as a fraction of information.

These sources support the mutual-information inequality the boundedness proof relies on; they do
not themselves propose the directional, interventional construction used here.
