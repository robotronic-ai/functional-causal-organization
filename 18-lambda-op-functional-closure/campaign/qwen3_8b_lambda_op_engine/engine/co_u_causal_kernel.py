#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import torch

import runtime as base

ADDRESSES = tuple(base.SELECT_HANDLES)
INTERVENTIONS = ("ZERO_SUFFIX", "HEAD_ROTATE")
PRIMARY_LAYERS = (21, 22, 23)
HORIZONS = (1, 2)


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    num = torch.sqrt(torch.mean((af - bf) ** 2))
    den = torch.sqrt(torch.mean(af ** 2)) + 1e-8
    return float((num / den).item())


def apply_self_intervention(cache, start: int, kind: str, layers: Iterable[int] = PRIMARY_LAYERS):
    c = base.clone_cache(cache)
    for li in layers:
        k, v = base.layer_kv(c, int(li))
        if start < 0 or start >= k.shape[-2]:
            raise RuntimeError(f"invalid control suffix start={start} for layer={li} shape={tuple(k.shape)}")
        ks = k[..., start:, :].clone()
        vs = v[..., start:, :].clone()
        if kind == "ZERO_SUFFIX":
            k[..., start:, :].zero_()
            v[..., start:, :].zero_()
        elif kind == "HEAD_ROTATE":
            if k.ndim != 4 or v.ndim != 4 or k.shape[1] < 2:
                raise RuntimeError(f"unexpected KV shape for head rotation: {tuple(k.shape)}")
            perm = torch.roll(torch.arange(k.shape[1], device=k.device), shifts=1)
            k[..., start:, :].copy_(ks[:, perm, :, :])
            v[..., start:, :].copy_(vs[:, perm, :, :])
        else:
            raise ValueError(kind)
    return c


def probe_token_ids(tok, address: str) -> list[int]:
    text = f" {address} ... ... ..."
    ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
    if len(ids) < max(HORIZONS):
        raise RuntimeError(f"probe tokenization too short for {address}: {ids}")
    return ids


def run_probe_trajectory(model, cache, token_ids: list[int], device: str, horizons: int = 2):
    c = base.clone_cache(cache)
    records = []
    for h in range(1, horizons + 1):
        tid = int(token_ids[(h - 1) % len(token_ids)])
        x = torch.tensor([[tid]], dtype=torch.long, device=device)
        plen = base.cache_seq_length(c)
        mask = torch.ones((1, plen + 1), dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(
                input_ids=x,
                attention_mask=mask,
                past_key_values=c,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        hs = out.hidden_states
        if hs is None or len(hs) < 37:
            raise RuntimeError(f"expected hidden states for 36 decoder layers, got {None if hs is None else len(hs)}")
        records.append({
            "horizon": h,
            "token_id": tid,
            "layer_states": [z[:, -1, :].detach() for z in hs[1:37]],
            "final_state": hs[-1][:, -1, :].detach(),
        })
        c = out.past_key_values
    return records


def _block_mean(reference, perturbed, indices: Iterable[int]) -> float:
    vals = [rel_rms(reference[li], perturbed[li]) for li in indices]
    return float(sum(vals) / len(vals))


def compare_probe_trajectory(reference, perturbed):
    out = []
    for r, p in zip(reference, perturbed):
        out.append({
            "horizon": int(r["horizon"]),
            "final_hidden_rel_rms": rel_rms(r["final_state"], p["final_state"]),
            "primary_block_rel_rms": _block_mean(r["layer_states"], p["layer_states"], (21, 22, 23)),
            "post_primary_block_rel_rms": _block_mean(r["layer_states"], p["layer_states"], (24, 25, 26)),
            "reference_block_rel_rms": _block_mean(r["layer_states"], p["layer_states"], (30, 31, 32)),
        })
    return out
