#!/usr/bin/env python3
"""Typed propagation signature candidate v0.4."""

from __future__ import annotations

from collections import defaultdict
import json


def _canon_distribution(d):
    """Canonicalize a binary output distribution under label complement."""
    # Input keys are "0","1". Consider identity and complement output recodings.
    p0 = float(d["0"])
    p1 = float(d["1"])
    identity = (round(p0, 15), round(p1, 15))
    complement = (round(p1, 15), round(p0, 15))
    return min(identity, complement)


def _channel_repr(channel):
    """
    Channel is a mapping from typed source state key to binary output distribution.
    Source state keys are already typed and are not permuted across source types.
    """
    rows = []
    # A coherent output recoding must be the SAME permutation across all rows,
    # so canonicalize the whole channel, not each row independently.
    keys = sorted(channel.keys())
    identity = []
    complement = []
    for key in keys:
        d = channel[key]
        identity.append((key, round(float(d["0"]),15), round(float(d["1"]),15)))
        complement.append((key, round(float(d["1"]),15), round(float(d["0"]),15)))
    return min(tuple(identity), tuple(complement))


def typed_binding_signature(role_channels):
    """
    role_channels: list of
      {"role_type": str, "role_id": str, "channel": {source_key: {"0":..,"1":..}}}

    Multiple entries of the same role_type are treated as representation-level
    duplicates only if their canonical channels are identical. They then
    collapse to one certified role channel.

    Returns a type-keyed canonical tuple. Role identifiers do not enter it.
    """
    by_type = defaultdict(list)
    for item in role_channels:
        by_type[item["role_type"]].append(_channel_repr(item["channel"]))

    out = []
    for role_type in sorted(by_type):
        reps = by_type[role_type]
        unique = sorted(set(reps))
        if len(unique) != 1:
            raise ValueError(
                f"same role_type {role_type!r} has causally non-equivalent copies"
            )
        out.append((role_type, unique[0]))
    return tuple(out)


def repaired_delta_object(delta_prop, role_channels):
    return {
        "Delta_prop": float(delta_prop),
        "typed_binding_signature": typed_binding_signature(role_channels),
    }


def repaired_cf_profile(gamma, delta_prop, role_channels, r_profile):
    return {
        "Gamma": float(gamma),
        "D_q": repaired_delta_object(delta_prop, role_channels),
        "R": tuple(sorted((int(k), float(v)) for k, v in r_profile.items())),
    }
