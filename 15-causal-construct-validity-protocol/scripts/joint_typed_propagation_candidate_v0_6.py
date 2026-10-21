#!/usr/bin/env python3
"""Joint typed propagation candidate v0.6."""

from __future__ import annotations

import itertools


def _bits_from_key(key: str) -> tuple[int, ...]:
    if key == "":
        return tuple()
    return tuple(int(x) for x in key.split(","))


def _key_from_bits(bits) -> str:
    return ",".join(str(int(x)) for x in bits)


def _flip_bits(bits, mask):
    return tuple((b ^ f) for b, f in zip(bits, mask))


def canonical_joint_channel(source_types, role_types, channel):
    """
    Canonical typed joint channel under independent binary label complements
    within each declared source type and each declared role type.

    source_types and role_types are ordered typed coordinates. Cross-type
    coordinate permutations are not free.

    channel maps source-state keys like "0,1" to distributions over joint
    role-response keys like "1,0".
    """
    source_types = tuple(source_types)
    role_types = tuple(role_types)

    if len(set(source_types)) != len(source_types):
        raise ValueError("source types must be unique in this candidate")
    if len(set(role_types)) != len(role_types):
        raise ValueError("role types must be unique after causal quotient")

    src_masks = list(itertools.product([0,1], repeat=len(source_types)))
    out_masks = list(itertools.product([0,1], repeat=len(role_types)))

    reps = []
    for sm in src_masks:
        for om in out_masks:
            rows = []
            for src_key, dist in channel.items():
                src = _bits_from_key(src_key)
                if len(src) != len(source_types):
                    raise ValueError("source arity mismatch")
                canon_src = _flip_bits(src, sm)

                out_items = []
                for y_key, p in dist.items():
                    y = _bits_from_key(y_key)
                    if len(y) != len(role_types):
                        raise ValueError("response arity mismatch")
                    canon_y = _flip_bits(y, om)
                    out_items.append((_key_from_bits(canon_y), round(float(p), 15)))
                out_items.sort()
                rows.append((_key_from_bits(canon_src), tuple(out_items)))
            rows.sort()
            reps.append((source_types, role_types, tuple(rows)))

    return min(reps)


def repaired_delta_object(delta_prop, source_types, role_types, channel):
    return {
        "Delta_prop": float(delta_prop),
        "joint_typed_channel": canonical_joint_channel(
            source_types, role_types, channel
        ),
    }


def repaired_cf_profile(
    gamma,
    delta_prop,
    source_types,
    role_types,
    channel,
    r_profile,
):
    return {
        "Gamma": float(gamma),
        "D_q": repaired_delta_object(
            delta_prop, source_types, role_types, channel
        ),
        "R": tuple(sorted((int(k), float(v)) for k, v in r_profile.items())),
    }
