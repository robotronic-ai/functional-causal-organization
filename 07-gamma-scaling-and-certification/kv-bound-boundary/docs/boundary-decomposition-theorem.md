# KV-BOUND v0.3: the temporal boundary and a bound-with-bypass theorem

Translated and restated from the source note `temporal_boundary_note.md` (French). The
mathematics and theorem statement are unchanged.

## 1. The notation problem the self-test revealed

The notation `P_{t+1}` is ambiguous if it denotes both the component simply carried over from the
past *and* the new history after `Y_t` is appended. These two objects must be kept separate.

Define **`C_t`**, the **carried persistence**: the component of the future macrostate that stays
invariant under every admissible intervention on the source `Z*_t`. In a standard autoregressive
transformer:

- `C_t = H_t`, the prefix already acquired before the current emission;
- `Y_t` does **not** belong to `C_t`, because it can change under `do(Z*_t)`;
- the new history is `(C_t, Y_t)`.

This definition forbids accidentally conditioning on a descendant of the source and thereby
artificially erasing the very channel being measured.

## 2. Estimand

\[
B_q = I_q^{do}(Z_t^*;N_{t+1}\mid C_t,U_t).
\]

Define the **conditional bypass**:

\[
C_{\mathrm{bypass},q} = I_q^{do}(Z_t^*;N_{t+1}\mid C_t,U_t,Y_t),
\]

which measures exactly what remains of the source-to-next-state transport once the symbolic
(token) channel is held fixed.

## 3. Decomposition theorem

For any finite interventional law for which these quantities are defined,

\[
\boxed{B_q \le H_q(Y_t\mid C_t,U_t) + C_{\mathrm{bypass},q}.}
\]

**Proof.** `I(Z;N∣C,U) ≤ I(Z;Y,N∣C,U)`; by the chain rule,
`I(Z;Y,N∣C,U) = I(Z;Y∣C,U) + I(Z;N∣Y,C,U)`; and `I(Z;Y∣C,U) ≤ H(Y∣C,U)`. Combining:
`B_q ≤ H(Y_t∣C_t,U_t) + C_bypass,q`. No hypothesis that "the token is the only channel" is used
anywhere in this inequality.

## 4. The 15–17 bit corollary — now experimentally falsifiable

If a test establishes `C_bypass,q = 0`, then `B_q ≤ H_q(Y_t∣C_t,U_t) ≤ log₂|V|`. The "15–17 bit"
bound therefore becomes an **experimentally falsifiable corollary**, not an assumed property of
the abstraction: it holds if and only if the conditional bypass is measured to be zero on the
real system under study.

## 5. Consequence for the temporal boundary

The self-test (`selftest-results.md`) distinguishes four intervention modes:

- **`post_cycle`** — intervention *after* the persistent writes; bypass is zero.
- **`prewrite`** — intervention *before* a K/V write; bypass is positive.
- **`raw_kv`** — direct surgery on a persistent memory; bypass is positive.
- **`positive_latent`** — an explicit latent side channel; bypass is positive.

Consequently, if `Z*_t` includes pre-write variables, the strict `log₂|V|` bound does **not**
hold in general. To claim the bound at the `state` grain, the source must be measured as a
**cycle-boundary state**: after the persistent writes of the cycle have closed. Any effective
capacity `K_eff` used in a denominator must be estimated on that *same* source and the *same*
battery `q` — a `K_eff` measured on a pre-write intermediate activation cannot be combined with a
numerator bounded by `log₂|V|` without separately demonstrating its transport to the terminal,
quotiented state.

## 6. Experimental criterion for the real model — NOT YET EXECUTED

The source note's own closing instruction is to apply this next, on the real model — this has
**not been done** in this bundle; it is a specified next step, not a completed result. It
requires measuring separately:

- `C_bypass^terminal` for a terminal, post-write source, and
- `C_bypass^prewrite(ℓ)` for several internal layers `ℓ`.

The "17 bits" claim is confirmed for the terminal grain if the first quantity is zero at a
preregistered threshold, *even if* the second quantities are positive. If the first quantity is
positive, the claim is directly falsified for that grain. This falsification protocol is fully
specified but has not been run against any real transformer in this bundle — see
`selftest-results.md` for exactly what *has* been executed (a synthetic toy-attention self-test
of the theorem itself, not this real-model criterion).
