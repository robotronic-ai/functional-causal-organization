#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRP candidate interface v0.1 — frozen pre-candidate surface.

No candidate implementation is included in this file.
The interface is intentionally P/N-free.

Scientific object returned by a future candidate:
    K_tilde(T_{t+1} | T_t, U_t)

represented as an exact rational table on the source/context repertoires
declared by the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import (
    FrozenSet, Mapping, Protocol, Sequence, Tuple, runtime_checkable
)

Var = int
Val = int
Assignment = Tuple[Tuple[Var, Val], ...]
ValueTuple = Tuple[Val, ...]
Distribution = Mapping[ValueTuple, Fraction]


@dataclass(frozen=True)
class WeightedIntervention:
    """
    One member of q.

    `do` is an immutable sorted tuple of (opaque variable id, opaque value).
    `weight` is exact and the weights over the battery must sum to 1.
    """
    do: Assignment
    weight: Fraction


@dataclass(frozen=True)
class EmulationRow:
    """
    One exact row of K_tilde.

    `source` follows ctx.source_state() order.
    `context` follows ctx.context_state() order.
    `target_dist` follows ctx.target_state() order.
    """
    source: ValueTuple
    context: ValueTuple
    target_dist: Distribution


@dataclass(frozen=True)
class EmulationKernel:
    """
    Complete emulated one-transition channel.

    The three orders MUST exactly equal those returned by the context on the
    same anonymized call.  `rows` must contain exactly one normalized row for
    every source/context assignment required by the declared repertoires.
    """
    source_order: Tuple[Var, ...]
    context_order: Tuple[Var, ...]
    target_order: Tuple[Var, ...]
    rows: Tuple[EmulationRow, ...]


@runtime_checkable
class IRPContext(Protocol):
    """
    O1-only context exposed to a future IRP candidate.

    IDs and value codes are opaque and may be bijectively rerandomized before
    every call.  No method exposes P/N labels, constructor labels such as
    "active"/"persistent", oracle decompositions, or hidden fixture ids.
    """

    def variables(self) -> Sequence[Var]: ...
    def alphabet(self, v: Var) -> Sequence[Val]: ...
    def parents(self, v: Var) -> FrozenSet[Var]: ...
    def roles(self) -> Mapping[Var, int]: ...

    def source_state(self) -> Tuple[Var, ...]:
        """Variables constituting T_t / the perturbed source macro-state."""
        ...

    def target_state(self) -> Tuple[Var, ...]:
        """Untyped variables constituting T_{t+1}.  This replaces pn_split()."""
        ...

    def context_state(self) -> Tuple[Var, ...]:
        """Declared conditioning/context variables U_t (possibly empty)."""
        ...

    def temporal_transport(self) -> Mapping[ValueTuple, ValueTuple]:
        """
        Certified same-role transport from target-state values at t+1 to
        source-state values for the next transition.

        Keys follow target_state() order; values follow source_state() order.
        For the v0.3 IRP campaign this transport MUST be a bijection between
        the complete target and source repertoires.

        This is structural O1/rho information, not a P/N label.
        """
        ...

    def battery(self) -> Sequence[WeightedIntervention]:
        """
        Full weighted q at the quotient grain.
        Unlike Protocole F v1.0, weights are explicitly exposed.
        """
        ...

    def kernel(
        self,
        do: Assignment,
        targets: Tuple[Var, ...],
    ) -> Distribution:
        """Exact ordinary interventional kernel P(targets | do)."""
        ...


@runtime_checkable
class IRPCandidate(Protocol):
    name: str
    version: str

    def __call__(self, ctx: IRPContext) -> EmulationKernel:
        """
        Return K_tilde(T_{t+1} | T_t,U_t).

        The candidate is stateless between calls and may use only `ctx`.
        """
        ...


def validate_emulation_kernel(ctx: IRPContext, out: EmulationKernel) -> None:
    """
    Interface-level validation only.

    This does NOT judge scientific correctness.  It checks typing, exact
    normalization, alphabets, uniqueness, and coverage of every Cartesian
    source/context assignment implied by the exposed alphabets.
    """
    if out.source_order != tuple(ctx.source_state()):
        raise ValueError("source_order mismatch")
    if out.context_order != tuple(ctx.context_state()):
        raise ValueError("context_order mismatch")
    if out.target_order != tuple(ctx.target_state()):
        raise ValueError("target_order mismatch")

    src_alph = [tuple(ctx.alphabet(v)) for v in out.source_order]
    ctx_alph = [tuple(ctx.alphabet(v)) for v in out.context_order]
    tgt_alph = [set(ctx.alphabet(v)) for v in out.target_order]

    def product(xs):
        acc = [()]
        for vals in xs:
            acc = [a + (x,) for a in acc for x in vals]
        return acc

    required = set(product(src_alph))
    required_ctx = set(product(ctx_alph)) if ctx_alph else {()}
    required_keys = {(s, u) for s in required for u in required_ctx}

    seen = set()
    for row in out.rows:
        key = (tuple(row.source), tuple(row.context))
        if key in seen:
            raise ValueError(f"duplicate row {key}")
        seen.add(key)

        if key not in required_keys:
            raise ValueError(f"row outside declared repertoire: {key}")

        total = sum(row.target_dist.values(), Fraction(0))
        if total != 1:
            raise ValueError(f"row {key} not normalized exactly: {total}")

        for y, p in row.target_dist.items():
            if p < 0:
                raise ValueError("negative probability")
            if len(y) != len(out.target_order):
                raise ValueError("wrong target tuple arity")
            for i, val in enumerate(y):
                if val not in tgt_alph[i]:
                    raise ValueError(f"target value {val} outside alphabet")

    if seen != required_keys:
        missing = sorted(required_keys - seen)
        extra = sorted(seen - required_keys)
        raise ValueError(f"incomplete row coverage; missing={missing}, extra={extra}")


def validate_temporal_transport(ctx: IRPContext) -> None:
    """
    Validate the exact temporal value transport required for multi-step
    same-role reasoning in the frozen IRP campaign.
    """
    src_alph = [tuple(ctx.alphabet(v)) for v in ctx.source_state()]
    tgt_alph = [tuple(ctx.alphabet(v)) for v in ctx.target_state()]

    def product(xs):
        acc = [()]
        for vals in xs:
            acc = [a + (x,) for a in acc for x in vals]
        return set(acc)

    src = product(src_alph)
    tgt = product(tgt_alph)
    tr = dict(ctx.temporal_transport())

    if set(tr) != tgt:
        raise ValueError(
            f"temporal transport target coverage mismatch: "
            f"missing={sorted(tgt-set(tr))}, extra={sorted(set(tr)-tgt)}"
        )
    vals = list(tr.values())
    if any(v not in src for v in vals):
        raise ValueError("temporal transport maps outside source repertoire")
    if len(set(vals)) != len(vals):
        raise ValueError("temporal transport is not injective")
    if set(vals) != src:
        raise ValueError("temporal transport is not surjective")


__all__ = [
    "Var", "Val", "Assignment", "ValueTuple", "Distribution",
    "WeightedIntervention", "EmulationRow", "EmulationKernel",
    "IRPContext", "IRPCandidate", "validate_emulation_kernel", "validate_temporal_transport",
]
