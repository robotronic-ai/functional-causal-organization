# Core Theorems and Proofs

## T1. Fiber Factorization Theorem

Let `F : C -> Y` and `L : C -> Z`.

\[
\exists G:\operatorname{Im}F\to Z\;\text{such that}\;L=G\circ F
\]

if and only if

\[
F(x)=F(x')\Rightarrow L(x)=L(x')
\]

for all `x,x'`.

### Proof

Necessity is immediate from substitution. For sufficiency, define `G(y)=L(x)` for any representative `x` with `F(x)=y`. Fiber constancy makes this well-defined. `G` is unique on `Im F`. QED.

---

## T2. Refinement Monotonicity Theorem

Write `F <= F'` when there exists `q` such that `F=q o F'`, so `F'` is finer.

Then

\[
\mathcal U(F')\subseteq\mathcal U(F)
\]

and, for any fixed metric,

\[
\Omega(F')\le\Omega(F).
\]

### Proof

`F'(x)=F'(x')` implies `F(x)=F(x')`. Every unresolved pair under `F'` is therefore also unresolved under `F`. QED.

---

## T3. Minimal Admissible Sufficiency

Let `A` be a finite descriptor library. If at least one descriptor family is sufficient, then at least one inclusion-minimal sufficient family exists.

### Proof

The set of sufficient subsets of a finite set is finite and non-empty. Choose one of minimal cardinality; no strict subset can be sufficient. QED.

The same statement holds in any well-founded admissible poset.

---

## T4. Symmetry Obstruction Theorem

Suppose a transformation `h` satisfies

\[
F(hx)=F(x)
\]

for all `x`, but for some `x`

\[
L(hx)\neq L(x).
\]

Then no factorization `L=G o F` exists.

### Proof

If it existed, `L(hx)=G(F(hx))=G(F(x))=L(x)`, contradiction. QED.

A necessary condition is therefore

\[
\operatorname{Stab}(F)\subseteq\operatorname{Stab}(L).
\]

---

## T5. Causal-Groupoid Factorization Theorem

Fix a causal context and let `G_c` be a groupoid of target-blind typed causal conjugacies. Assume:

1. `F(x)=F(x')` exactly when `x` and `x'` lie in the same groupoid orbit.
2. `L` is invariant under every groupoid morphism.

Then `L=G o F` on the declared envelope.

### Proof

Equal `F` values imply the two points lie in one orbit. A finite composition of groupoid morphisms connects them, and invariance along each morphism gives equal target values. Apply T1. QED.

---

## T6. Closed-Quotient Transport Theorem

For deterministic intervention-indexed transitions `T_i`, let `q` satisfy

\[
q\circ T_i=\bar T_{\bar i}\circ q
\]

for every declared generator intervention.

Then every finite composition of declared transitions also descends through `q`.

### Proof

Apply the commuting identity repeatedly. For two steps,

\[
qT_jT_i=\bar T_{\bar j}qT_i=\bar T_{\bar j}\bar T_{\bar i}q.
\]

Induction gives arbitrary finite compositions. QED.

The stochastic version follows because pushforwards compose.

---

## T7. Certified-Quotient Composition Theorem

Suppose

\[
S\xrightarrow{q}Q\xrightarrow{h}Z
\]

and both stages are closed:

\[
qT_i=\bar T_{\bar i}q,
\qquad
h\bar T_{\bar i}=\widetilde T_{\tilde i}h.
\]

Then

\[
(hq)T_i=\widetilde T_{\tilde i}(hq).
\]

### Proof

Substitute the first commuting equation into the second:

\[
(hq)T_i=h(qT_i)=h\bar T_{\bar i}q=\widetilde T_{\tilde i}hq.
\]

QED.

### Important non-converse

A composite quotient can be closed even when an intermediate quotient is not. Intermediate closure is what grants modular reuse of the intermediate causal object.
