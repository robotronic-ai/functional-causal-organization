# Theorem and frozen cut definition

Source material: `THEOREM.md` and `CUT_DEFINITION.md`.

## Assumptions

1. `C` is balanced binary conditional on every receiver context `Y`.
2. `T = C XOR N` with `N ~ Bernoulli(p)` independent of `(C, Y)`.
3. The receiver output is a function only of `(T, Y)` after the cut.
4. Target `Z = C`.
5. Distortion is binary Hamming loss, with response categories `EMPTY` and `OTHER`
   counted as errors.

## The quantitative theorem

For a balanced binary symmetric channel, `I(C; T | Y) = 1 - h2(p)`, where `h2` is the
binary entropy function. Because `Z = C` is balanced conditional on `Y`, the binary
rate-distortion function is

```
R_{Z|Y}(D) = 1 - h2(D)   for D <= 1/2
           = 0            for D >= 1/2.
```

The general physical-cut inequality (source information bound is at least the
task rate-distortion requirement) gives `1 - h2(p) >= R(D)`. For `0 <= p <= 1/2` and any
informative reconstruction with `D <= 1/2`, binary entropy is increasing on `[0, 1/2]`,
so the full information-theoretic test reduces to an **exact distortion frontier**:

```
D >= p.
```

- **Equality case.** If the receiver outputs `T` perfectly, `D = p` and the
  rate-distortion inequality is saturated — the receiver is exactly as good as the
  cut allows, no better and no worse.
- **Strictly degraded receiver.** Any receiver that loses information downstream of
  `T` has `D > p` for at least some noise levels.
- **Bypass receiver.** A receiver with direct access to `C` (violating assumption 3)
  can obtain `D < p`, which violates the frozen-cut inequality precisely because the
  causal assumption behind it is false. This is why the bypass audit
  (`docs/analysis-and-bypass-audit.md`) is a **qualification gate**, not an optional
  diagnostic: a `D < p` reading is uninterpretable as evidence about the theorem unless
  bypass has independently been ruled out.

## Frozen cut definition

**Source side `L`.** Contains the externally selected typed content label
`C in {BIT0, BIT1}`, used only to generate the controlled cut message. It must not
appear in the final content-neutral query or any other receiver-visible channel.

**Cut transcript `T`.** The binary value physically/architecturally written to the
designated Role-A memory interface after prospective corruption: `T = C XOR N`. For the
extractor, `T` is forced directly. `C` and `T` must be independently configurable, so
that fixed-`T`, varying-`C` bypass tests are possible (this is exactly what
`docs/analysis-and-bypass-audit.md` exercises).

**Receiver side `R`.** The post-write model computation, the frozen identity query of
Role A, and the final installed categorical response readout. The receiver target is
`Z = C`.

**Receiver-side context `Y`.** All receiver-side information available independently of
`C` — the campaign deliberately *conditions on* `Y` rather than treating context as
noise, since `Y` includes the frozen background item, the fixed Role-B collateral
value, the query template, the delay-marker template, and any other fixed non-content
prompt material.

**Architectural status.** This is an **architectural RMT cut** at the typed Role-A
write interface. It should not be described as a spatially physical neural cut unless
the local implementation establishes a corresponding physical module boundary — the
"physical" in the campaign name refers to the physical-cut information-theoretic
inequality, not to a claim about physical/spatial locality inside the model.
