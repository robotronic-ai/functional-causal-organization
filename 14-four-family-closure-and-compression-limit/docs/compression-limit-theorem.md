# Compression Limit Theorem for C_O-to-A* Sufficiency

**Status:** proved (direct proof by contradiction). Self-contained; does not
depend on any executed harness.

## 1. Why this theorem is here

The manuscript's four-axis closure result (`four-family-axis-completeness.md`)
argues, by elimination over a declared ontology, that any residual variation
in functional access A* must be located inside active causal organization
𝒞 — not in knowledge (I_G), embodiment/situatedness (S), or functional
orientation (O). That is a **localization** claim.

It is tempting to read a localization result as if it already proved that the
declared four-component profile

```
C_O = [(Γ, Δ, R(k), Λ)]
```

*captures* that variation — i.e. that C_O is sufficient to predict A*. It does
not. The two claims are logically independent, and this theorem is the reason
why.

## 2. Statement

Let `p : 𝒞 → Z` be any organizational profile (a map from the full causal
organization of a system to some compact summary space `Z`, e.g. `Z` = the
range of `(Γ, Δ, R(k), Λ)`).

Suppose `p` is **non-injective**: there exist two organizations `𝒞₁ ≠ 𝒞₂`
with `p(𝒞₁) = p(𝒞₂)`.

**Proposition.** If the admissible target class contains a functional
property `A*` that distinguishes `𝒞₁` and `𝒞₂` (i.e. `A*(𝒞₁) ≠ A*(𝒞₂)`),
then no function `H` satisfies `A* = H ∘ p`.

## 3. Proof

Assume, for contradiction, that `A* = H ∘ p` for some function `H`.

Since `p(𝒞₁) = p(𝒞₂)`, applying `H` to both sides gives
`H(p(𝒞₁)) = H(p(𝒞₂))`, i.e. `A*(𝒞₁) = A*(𝒞₂)`.

This contradicts the hypothesis that `A*` distinguishes `𝒞₁` and `𝒞₂`.
Therefore no such `H` exists. **QED.**

This is a two-line argument; its force comes entirely from exhibiting a real
`p` that is non-injective and a real `A*` that is sensitive to what `p`
discards — which is exactly what the organizational-closure repair chain in
folder `15` does (see `docs/organizational-closure-repair-chain.md` there for
the concrete witnesses).

## 4. Corollary for this manuscript

Four-family closure of

```
Ψ(e) = (𝒮, 𝒢, 𝒞, 𝒪)
```

can localize residual functional-access variation inside 𝒞. It **cannot, by
itself**, prove that a non-injective compression `C_O = p(𝒞)` is sufficient
for functional access. A separate restriction on the target construct A*, a
representation theorem, or direct empirical evidence is required in addition.

Concretely: closure answers *where* the variation must live; it does not
answer *whether the four declared scalars of C_O see all of it*.

## 5. Why the counterexamples in folder 15 matter

The typed-binding witness (`FCI2-PROFILE-INSUFFICIENT-TYPED-BINDING`) and the
joint-synergy witness (`CFPLUS-PROFILE-INSUFFICIENT-JOINT-SYNERGY`) are not
coding accidents. They are concrete instances of the abstract proposition
above: two organizations identical on the compact profile
`[(Γ, Δ, R(k))]`, distinguished by an independently frozen access consequence
(`a_flex`). Each successive repair (adding a typed marginal channel, then a
joint typed channel) is a change of `p` to a strictly more informative map,
which is the only way the theorem allows the collapse to be undone.

## 6. Limit of the repair strategy

One can always retain more of the interventional structure of 𝒞 inside the
profile. In the limit, a complete typed interventional causal model becomes
sufficient for every functional consequence defined on that model — but at
that point the "compact profile" has converged to 𝒞 itself, and the exercise
has stopped being a compression.

The scientific target is therefore **not**:

> keep enriching C_O until it is trivially sufficient for every possible
> functional consequence.

It is:

> identify a compact, target-independent causal profile that prospectively
> predicts a preregistered family of access consequences better than rival
> variables.

That is construct validation (folder `15`), not taxonomic closure (this
folder).

## 7. One-paragraph summary for the manuscript

> Closure of the four-family ontology localizes any residual variation in
> functional access to active causal organization 𝒞, but a compression
> limit theorem shows this localization cannot by itself certify that the
> declared four-component profile C_O = (Γ, Δ, R(k), Λ) is sufficient to
> predict that variation: sufficiency of a non-injective summary can be
> refuted by a single distinguishing witness, and must otherwise be argued
> or tested separately (see the organizational-closure repair chain and the
> FCI-3 causal construct validity protocol).
