#!/usr/bin/env python3
"""Layer-local execution for Gamma on HF Qwen3 models: run ONLY decoder layer l+1 for the active position.

Why: on the user's RTX 5060 Ti a full Qwen3-8B forward costs ~1.2 s; a Gamma transition only needs layer l+1,
i.e. ~1/36 of the network.  The residual leaving layer l at the active position is captured once from a normal
forward; interventions on parts of layer l are additive residual-space vectors (the residual stream is the sum of
the component writes), so the local run is:

    x_l (+ v_A + v_B)  --decoder layer l+1 (with the real prefix KV cache)-->  component writes of layer l+1

Measured state B' = writes of the B-part components of layer l+1 at the active position:
    head h write  = W_o[:, h] @ o_h          (captured from the o_proj input)
    MLP block j   = W_down[:, blk_j] @ a_blk (captured from the down_proj input)
These writes are invariant objects under the exact recodings used for the resolution (VO_DIAG, UPDOWN_SCALE,
QKNORM_RECIP), up to floating-point rounding.

Exact-gauge replicas: weights are recoded in place, the prefix/suffix cache and x_l are REBUILT under the recoded
weights (lesson of T110B/T111), then restored from a CPU backup and verified by digest.

Correctness gate: `verify_local_equals_full` compares the local layer-(l+1) output with the one produced inside a
normal full forward of the same token; it must be bit-identical (or the tool refuses to run).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
import time

import numpy as np
import torch

torch.set_grad_enabled(False)


# ----------------------------------------------------------------------------- model access helpers
def decoder(model):
    return model.model if hasattr(model, "model") else model


def n_layers(model):
    return len(decoder(model).layers)


def geom(model):
    cfg = model.config
    nq, nkv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // nq))
    return nq, nkv, hd, nq // nkv


def crop_cache(cache, n):
    if hasattr(cache, "crop"):
        extra = int(cache.get_seq_length()) - int(n) if hasattr(cache, "get_seq_length") else None
        try:
            cache.crop(-extra if extra else n)          # newer transformers: negative = number of tokens to drop
        except Exception:
            cache.crop(n)
        return cache
    raise RuntimeError("cache has no crop(); update transformers or use clone_cache per run")


# ----------------------------------------------------------------------------- base pass: cache + x_l
def prepare_active_position(model, base, control_cache, suffix, layer, device):
    """Run suffix[:-1] on a clone of the control cache, then the last token with a hook capturing the output of
    decoder layer `layer` at the active position.  Returns (cache_before_last_token, last_token_id, x_l, full_next)."""
    dec = decoder(model)
    c = base.clone_cache(control_cache)
    n0 = int(base.cache_seq_length(c))
    if len(suffix) > 1:
        ids = torch.tensor([suffix[:-1]], device=device)
        model(input_ids=ids, attention_mask=torch.ones((1, n0 + ids.shape[1]), dtype=torch.long, device=device),
              past_key_values=c, use_cache=True)
    n1 = int(base.cache_seq_length(c))
    keep = base.clone_cache(c)                                   # cache with every position before the active one
    cap = {}
    h1 = dec.layers[layer].register_forward_hook(lambda m, a, o: cap.__setitem__("x_l", (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach().clone()))
    h2 = dec.layers[layer + 1].register_forward_hook(lambda m, a, o: cap.__setitem__("x_l1", (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach().clone()))
    try:
        ids = torch.tensor([[suffix[-1]]], device=device)
        model(input_ids=ids, attention_mask=torch.ones((1, n1 + 1), dtype=torch.long, device=device), past_key_values=c, use_cache=True)
    finally:
        h1.remove(); h2.remove()
    return dict(cache=keep, n=n1, x_l=cap["x_l"], x_l1_full=cap["x_l1"])


# ----------------------------------------------------------------------------- local run of layer l+1
class LocalLayer:
    """Context manager: temporarily make the decoder contain only layer l+1 (it keeps its own layer_idx)."""

    def __init__(self, model, layer):
        self.dec = decoder(model); self.layer = layer

    def __enter__(self):
        self.saved = self.dec.layers
        self.dec.layers = torch.nn.ModuleList([self.saved[self.layer + 1]])
        return self

    def __exit__(self, *a):
        self.dec.layers = self.saved


def local_run(model, prep, x_in, layer, device, want_writes=True):
    """x_in: (B, 1, d) residual entering layer l+1.  Runs the single layer for each batch row (batch 1 each, the
    prefix cache is cropped back after every run).  Returns layer output (B, d) and component writes."""
    nq, nkv, hd, group = geom(model)
    target = prep["layer_module"]
    cap = {}
    ho = target.self_attn.o_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("o_in", a[0][:, -1, :].detach()))
    hd_ = target.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("d_in", a[0][:, -1, :].detach()))
    hout = target.register_forward_hook(lambda m, a, o: cap.__setitem__("out", (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()))
    outs, heads, mlps = [], [], []
    Wo = target.self_attn.o_proj.weight; Wd = target.mlp.down_proj.weight
    n = prep["n"]
    try:
        with LocalLayer(model, layer):
            for b in range(x_in.shape[0]):
                cache = prep["cache"]
                decoder(model)(inputs_embeds=x_in[b:b + 1].to(Wo.dtype), attention_mask=torch.ones((1, n + 1), dtype=torch.long, device=device),
                               position_ids=torch.tensor([[n]], device=device), past_key_values=cache, use_cache=True)
                crop_cache(cache, n)
                outs.append(cap["out"][0].float().cpu())
                if want_writes:
                    o = cap["o_in"][0]; a = cap["d_in"][0]
                    heads.append(torch.stack([(Wo[:, h * hd:(h + 1) * hd] @ o[h * hd:(h + 1) * hd]).float().cpu() for h in range(nq)]))
                    mlps.append(a.float().cpu())
    finally:
        ho.remove(); hd_.remove(); hout.remove()
    res = dict(out=torch.stack(outs))
    if want_writes:
        res["head_writes"] = torch.stack(heads)              # (B, nq, d)
        res["mlp_act"] = torch.stack(mlps)                   # (B, dff)  -> block writes computed on demand
        res["Wd"] = Wd
    return res


def block_writes(res, blocks, n_blocks):
    Wd = res["Wd"]; dff = Wd.shape[1]; bs = dff // n_blocks; out = []
    for j in blocks:
        sl = slice(j * bs, (j + 1) * bs)
        out.append((res["mlp_act"][:, sl].to(Wd.device, Wd.dtype) @ Wd[:, sl].T).float().cpu())
    return out


def verify_local_equals_full(model, prep, layer, device):
    prep["layer_module"] = decoder(model).layers[layer + 1]
    r = local_run(model, prep, prep["x_l"], layer, device, want_writes=False)
    full = prep["x_l1_full"][0, 0].float().cpu()
    return bool(torch.equal(r["out"][0], full)), float((r["out"][0] - full).abs().max())


# ----------------------------------------------------------------------------- units, directions, measurement
def default_units(model, n_blocks=4, groups_per_unit=4):
    """Units of a layer: KV-group bundles of heads and MLP neuron blocks."""
    nq, nkv, hd, group = geom(model)
    units = []
    for g0 in range(0, nkv, groups_per_unit):
        heads = [h for g in range(g0, min(nkv, g0 + groups_per_unit)) for h in range(g * group, (g + 1) * group)]
        units.append(("heads", tuple(heads)))
    per = max(1, n_blocks // 2)
    for j0 in range(0, n_blocks, per):
        units.append(("mlp", tuple(range(j0, min(n_blocks, j0 + per)))))
    return units


def part_directions(model, layer, part, k, n_blocks, seed=0):
    """k orthonormal residual directions mixing, with fixed random signs, the top write directions of every
    component of the part (built from the UNGAUGED weights, reused for every replica)."""
    L = decoder(model).layers[layer]; nq, nkv, hd, group = geom(model)
    Wo = L.self_attn.o_proj.weight.double().cpu(); Wd = L.mlp.down_proj.weight.double().cpu(); bs = Wd.shape[1] // n_blocks
    comps = []
    for kind, idx in part:
        if kind == "heads":
            comps += [Wo[:, h * hd:(h + 1) * hd] for h in idx]
        else:
            comps += [Wd[:, j * bs:(j + 1) * bs] for j in idx]
    Us = [torch.linalg.svd(W, full_matrices=False)[0][:, :k] for W in comps]
    g = torch.Generator().manual_seed(seed + 97 * layer)
    dirs = []
    for i in range(k):
        v = torch.zeros(Wo.shape[0], dtype=torch.float64)
        for U in Us:
            v += (1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0) * U[:, i % U.shape[1]]
        dirs.append(v)
    return torch.linalg.qr(torch.stack(dirs, 1))[0].T.float()


def measure_B(res, part, n_blocks):
    vecs = []
    for kind, idx in part:
        if kind == "heads":
            vecs += [res["head_writes"][:, h, :] for h in idx]
        else:
            vecs += block_writes(res, idx, n_blocks)
    return torch.cat(vecs, 1)


# ----------------------------------------------------------------------------- exact gauges with cache rebuild
def _gauge_params(model, layer, kind):
    dec = decoder(model)
    if kind == "QKNORM_RECIP":
        return {f"{i}.{n}": getattr(l.self_attn, n).weight for i, l in enumerate(dec.layers) for n in ("q_norm", "k_norm")}
    out = {}
    for li in (layer, layer + 1):
        l = dec.layers[li]
        if kind == "UPDOWN_SCALE":
            out[f"{li}.up"] = l.mlp.up_proj.weight; out[f"{li}.down"] = l.mlp.down_proj.weight
        elif kind == "VO_DIAG":
            out[f"{li}.v"] = l.self_attn.v_proj.weight; out[f"{li}.o"] = l.self_attn.o_proj.weight
    return out


def _digest(params):
    h = hashlib.sha256()
    for n, p in sorted(params.items()):
        h.update(n.encode()); h.update(p.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def apply_gauge_(model, layer, kind, seed, log_std=0.3):
    dec = decoder(model); nq, nkv, hd, group = geom(model); g = torch.Generator().manual_seed(1009 * seed + len(kind))
    for li in ((layer, layer + 1) if kind != "QKNORM_RECIP" else range(len(dec.layers))):
        l = dec.layers[li]
        if kind == "UPDOWN_SCALE":
            up, down = l.mlp.up_proj.weight, l.mlp.down_proj.weight
            f = torch.exp(log_std * torch.randn(up.shape[0], generator=g, dtype=torch.float64))
            for a in range(0, up.shape[0], 2048):
                fa = f[a:a + 2048].to(up.device)
                up[a:a + 2048].copy_((up[a:a + 2048].double() * fa[:, None]).to(up.dtype))
                down[:, a:a + 2048].copy_((down[:, a:a + 2048].double() / fa[None, :]).to(down.dtype))
        elif kind == "VO_DIAG":
            v, o = l.self_attn.v_proj.weight, l.self_attn.o_proj.weight
            for gi in range(nkv):
                f = torch.exp(log_std * torch.randn(hd, generator=g, dtype=torch.float64)).to(v.device)
                sl = slice(gi * hd, (gi + 1) * hd)
                v[sl].copy_((v[sl].double() * f[:, None]).to(v.dtype))
                for qh in range(gi * group, (gi + 1) * group):
                    osl = slice(qh * hd, (qh + 1) * hd)
                    o[:, osl].copy_((o[:, osl].double() / f[None, :]).to(o.dtype))
        elif kind == "QKNORM_RECIP":
            c = math.exp(log_std * float(torch.randn(1, generator=g)))
            l.self_attn.q_norm.weight.copy_((l.self_attn.q_norm.weight.double() * c).to(l.self_attn.q_norm.weight.dtype))
            l.self_attn.k_norm.weight.copy_((l.self_attn.k_norm.weight.double() / c).to(l.self_attn.k_norm.weight.dtype))


class Gauged:
    """Context manager applying an exact gauge in place; restores from a CPU backup and verifies the digest."""

    def __init__(self, model, layer, kind, seed):
        self.m, self.l, self.k, self.s = model, layer, kind, seed

    def __enter__(self):
        self.params = _gauge_params(self.m, self.l, self.k); self.d0 = _digest(self.params)
        self.bk = {n: p.detach().cpu().clone() for n, p in self.params.items()}
        apply_gauge_(self.m, self.l, self.k, self.s); return self

    def __exit__(self, *a):
        for n, p in self.params.items():
            p.copy_(self.bk[n].to(p.device))
        if _digest(self.params) != self.d0:
            raise RuntimeError("exact-gauge restoration failed")


GAUGE_SET = [("UPDOWN_SCALE", s) for s in (0, 1, 2)] + [("VO_DIAG", s) for s in (0, 1, 2)] + [("QKNORM_RECIP", s) for s in (0, 1, 2)]


# ----------------------------------------------------------------------------- Gamma
def entropy(labels):
    _, c = np.unique(np.asarray(labels), return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def pairwise_rel(Y):
    Y = Y.double(); r = Y.pow(2).mean(1).sqrt()
    return ((Y[:, None] - Y[None]).pow(2).mean(-1).sqrt() / (0.5 * (r[:, None] + r[None]) + 1e-30)).numpy()


def complete_link_labels(D, tol):
    D = np.array(D, float); n = len(D); CD = D.copy(); np.fill_diagonal(CD, np.inf)
    mem = {i: [i] for i in range(n)}; act = list(range(n))
    while len(act) > 1:
        sub = CD[np.ix_(act, act)]; k = int(np.argmin(sub)); a, b = divmod(k, len(act))
        if not sub[a, b] <= tol: break
        ia, ib = act[min(a, b)], act[max(a, b)]; row = np.maximum(CD[ia], CD[ib])
        CD[ia, :] = row; CD[:, ia] = row; CD[ia, ia] = np.inf; CD[ib, :] = np.inf; CD[:, ib] = np.inf
        mem[ia] += mem.pop(ib); act.remove(ib)
    own = {i: r for r, m in mem.items() for i in m}; lut = {}
    return [lut.setdefault(own[i], len(lut)) for i in range(n)]


def gamma_local(model, base, build_control_cache, suffix, layer, device, units=None, n_blocks=4, k_side=2, dose=0.5,
                gauges=GAUGE_SET, band=2.0, family=None, log=print):
    """Tested-cut Gamma upper bounds U_tested(delta) and U_tested(band*delta) at the
    transition layer -> layer+1, active position, with exact-gauge replicas (cache rebuilt under each gauge).

    build_control_cache: zero-argument callable returning the prompt/control KV cache computed with the CURRENT
    weights (e.g. lambda: build_receiver(...)["control_cache"]).  It is called again under every gauge: a cache
    built before a recoding keeps un-recoded K/V and breaks exactness (T110B/T111 lesson)."""
    t0 = time.time()
    units = units or default_units(model, n_blocks)
    prep = prepare_active_position(model, base, build_control_cache(), suffix, layer, device)
    ok, err = verify_local_equals_full(model, prep, layer, device)
    if not ok:
        raise RuntimeError(f"local layer run is not bit-identical to the full forward (max abs diff {err:.3g})")
    scale = float(prep["x_l"].float().pow(2).mean().sqrt())
    if family is None:
        family = []
        for mask in range(1, 2 ** (len(units) - 1)):
            A = [u for i, u in enumerate(units) if mask >> i & 1]; B = [u for i, u in enumerate(units) if not mask >> i & 1]
            family.append((A, B))
    # One product battery q = q_A x q_B.  Each side uses the SAME marginal
    # whether it is the source or the conditioned side (T121 correction).
    C_SIDE = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=k_side)))
    # batteries (fixed residual vectors, from the ungauged weights)
    dirs = {}
    def vecs(part, k, C):
        key = (tuple(part), k)
        if key not in dirs:
            # unit-norm codewords scaled so that the per-coordinate RMS of the intervention is dose * RMS(x_l)
            dirs[key] = dose * scale * math.sqrt(prep["x_l"].shape[-1]) * (C @ part_directions(model, layer, part, k, n_blocks)) / math.sqrt(k)
        return dirs[key]

    def measure_all(prep_):
        prep_["layer_module"] = decoder(model).layers[layer + 1]
        out = {}
        for A, B in family:
            for src, dst in ((A, B), (B, A)):
                va = vecs(src, k_side, C_SIDE); vb = vecs(dst, k_side, C_SIDE)
                X = (prep_["x_l"][0, 0].float().cpu()[None, None, :] + va[None].transpose(0, 1)).repeat(vb.shape[0], 1, 1)
                X = X + vb.repeat_interleave(va.shape[0], 0)[:, None, :]
                res = local_run(model, prep_, X.to(device), layer, device)
                out[(str(src), str(dst))] = measure_B(res, dst, n_blocks).view(vb.shape[0], va.shape[0], -1)
        return out

    ref = measure_all(prep)
    reps = []
    for kind, s in gauges:
        with Gauged(model, layer, kind, s):
            prep_g = prepare_active_position(model, base, build_control_cache(), suffix, layer, device)   # cache REBUILT
            reps.append(measure_all(prep_g))
    # ONE resolution for the measurement site (component writes of layer l+1), frozen from ALL exact-gauge orbits
    # before any Gamma value is computed (T120 acceptance gate): the maximum orbit diameter over every cut/direction.
    site_delta = 0.0; per_key_delta = {}
    for key, Y0 in ref.items():
        S = torch.stack([Y0] + [r[key] for r in reps])
        dk = max(float(pairwise_rel(S[:, b, a, :]).max()) for b in range(S.shape[1]) for a in range(S.shape[2]))
        per_key_delta[key] = dk; site_delta = max(site_delta, dk)
    delta = math.nextafter(site_delta, math.inf)
    rows = []
    n = math.log2(C_SIDE.shape[0])
    for A, B in family:
        J = {}
        for src, dst in ((A, B), (B, A)):
            Y0 = ref[(str(src), str(dst))]
            hi = np.mean([entropy(complete_link_labels(pairwise_rel(Y0[b]), delta)) for b in range(Y0.shape[0])])
            lo = np.mean([entropy(complete_link_labels(pairwise_rel(Y0[b]), band * delta)) for b in range(Y0.shape[0])])
            J[(str(src), str(dst))] = (hi, lo)
        jab, jba = J[(str(A), str(B))], J[(str(B), str(A))]
        rows.append(dict(A=A, B=B, cut_value_at_delta=(jab[0] + jba[0]) / (2 * n), cut_value_at_band=(jab[1] + jba[1]) / (2 * n),
                         orbit_diameter_AB=per_key_delta[(str(A), str(B))], orbit_diameter_BA=per_key_delta[(str(B), str(A))]))
    up_hi = min(r["cut_value_at_delta"] for r in rows); up_lo = min(r["cut_value_at_band"] for r in rows)
    out = dict(layer=layer, units=units, units_status="DIAGNOSTIC physical packs, NOT certified quotient roles",
               n_tested_cuts=len(rows), site_delta=delta, band=band, k_side=k_side,
               q_contract="symmetric product battery: same uniform marginal on each side whether source or conditioned",
               upper_bound_tested_cuts_at_delta=up_hi, upper_bound_tested_cuts_at_band=up_lo,
               lower_certificate="NOT_COMPUTED (L_Gamma = NA; the T116 additive/spectral certificate is to be plugged in)",
               report="L_Gamma = NA; U_tested(delta) is an upper bound on the true partition minimum. U_tested(2delta) is a resolution-sensitivity companion upper bound, not a lower Gamma bound.",
               verify_local_bit_identical=ok, seconds=round(time.time() - t0, 1), partitions=rows)
    log(f"[gamma-local] layer {layer}->{layer+1}: site delta={delta:.4g}; upper bound over {len(rows)} tested cuts: "
        f"U(delta)={up_hi:.3f} (U({band:g}delta)={up_lo:.3f}); lower certificate NA ({out['seconds']}s)")
    return out
