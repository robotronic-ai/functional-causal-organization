#!/usr/bin/env python3
"""ROLEQ v3.2 - quotient-role certification for HF Qwen3 models, layer-local 2-layer windows.
Conforms to the frozen protocol TURN126 (PROTOCOL_FROZEN_MINI_VALIDATED), rule by rule:

  physical components -> Stage 0 refactorization -> Stage 1 activity -> Stage 2 role equivalence -> certified roles
  (Stage 3, cross-layer transport, is a separate relation: candidates of different layers are never fused.)

Candidates of layer a (active position): head packets and MLP neuron blocks (any kind; fusion is same-layer only).
Site: output of layer a+1 at the active position.  Only layers a, a+1 are executed (prefix cache reused) after a
bit-identity check against the full forward.

Declared battery (per exact-recoding replica, every prompt cache rebuilt under the recoding):
    calibration contexts t (k = 0) and REFAC_PERTURB counterfactual interventions k on the residual entering layer a
    (incoming-intervention battery), plus single-candidate mean ablations and cross-context substitutions at k = 0.
delta_site (ONE per site, frozen before any pairwise decision) = maximum exact-recoding orbit diameter (rel-RMS) of
    every site state of the declared battery: clean (t, k), ablation (c, t), substitution (c, t, u).
Exact recodings: identity + Q.GAUGE_SET (3 UPDOWN_SCALE, 3 VO_DIAG on layers a, a+1; 3 QKNORM_RECIP on all layers)
    = 10 replicas.  Consensus: a status holds only if every replica gives it; mixed verdicts are AMBIGUOUS.

Stage 0 (refactorization; geometry is a prefilter/report only, it never decides):
    per replica, fit one fixed factor s on the CENTRED writes over the whole battery (calibration + counterfactual),
    replace the write of y by its reconstruction  mean(w_y) + s (w_x - mean(w_x))  (residual-space correction added
    at the output of y's projection, so layer a's own MLP sees it), and symmetrically x from y;
    e = max over battery states and both directions of rel(site_replaced, site_clean).
    FUSION if e <= delta in all replicas; DISTINCT if e > 2 delta in all replicas; otherwise AMBIGUOUS.
    Fusion groups = connected components of FUSION edges that are also cliques; a non-clique component is NOT fused
    (members kept separate, reported as STAGE0_NON_TRANSITIVE).
Stage 1 (activity of each macro-candidate, mean ablation over the calibration contexts):
    ACTIVE_ROLE if effect > 2 delta in all replicas; INACTIVE if <= delta in all replicas; otherwise AMBIGUOUS.
Stage 2 (equivalence of ACTIVE macro-candidates):
    downstream signature distance d_down = max(ablation contrast, substitution contrast) (raw site states);
    incoming signature distance d_in between incoming-dependence profiles (relative write change under each
    counterfactual intervention), with tau_in = exact-recoding orbit diameter of those profiles (measured, frozen).
    FUSION if d_down <= delta and d_in <= tau_in; DISTINCT if d_down > 2 delta or d_in > 2 tau_in; else AMBIGUOUS;
    consensus over replicas; certified roles = cliques of Stage-2 FUSION pairs among ACTIVE macro-candidates.
AMBIGUOUS is never read as zero or as fusion.
"""
from __future__ import annotations

import itertools
import math
import time

import torch

import qwen_local_transition as Q

torch.set_grad_enabled(False)
VERSION = "ROLEQ_v3.2_T126_CONFORMANT"
REFAC_PERTURB = 8          # counterfactual upstream interventions per calibration context
REFAC_DOSE = 0.3           # relative RMS of those interventions on the residual entering layer a


# ----------------------------------------------------------------------------- window execution
def prepare_window(model, base, control_cache, suffix, a, device):
    dec = Q.decoder(model)
    c = base.clone_cache(control_cache); n0 = int(base.cache_seq_length(c))
    if len(suffix) > 1:
        ids = torch.tensor([suffix[:-1]], device=device)
        model(input_ids=ids, attention_mask=torch.ones((1, n0 + ids.shape[1]), dtype=torch.long, device=device), past_key_values=c, use_cache=True)
    n1 = int(base.cache_seq_length(c)); keep = base.clone_cache(c); cap = {}
    grab = lambda key: (lambda m, args, out: cap.__setitem__(key, (out[0] if isinstance(out, tuple) else out)[:, -1:, :].detach().clone()))
    hooks = [dec.layers[a + 1].register_forward_hook(grab("site"))]
    if a > 0:
        hooks.append(dec.layers[a - 1].register_forward_hook(grab("x_in")))
    else:
        hooks.append(dec.embed_tokens.register_forward_hook(lambda m, args, out: cap.__setitem__("x_in", out[:, -1:, :].detach().clone())))
    try:
        model(input_ids=torch.tensor([[suffix[-1]]], device=device), attention_mask=torch.ones((1, n1 + 1), dtype=torch.long, device=device),
              past_key_values=c, use_cache=True)
    finally:
        for h in hooks: h.remove()
    return dict(cache=keep, n=n1, x_in=cap["x_in"], site_full=cap["site"])


class Window:
    def __init__(self, model, a):
        self.dec = Q.decoder(model); self.a = a

    def __enter__(self):
        self.saved = self.dec.layers
        self.dec.layers = torch.nn.ModuleList([self.saved[self.a], self.saved[self.a + 1]]); return self

    def __exit__(self, *x):
        self.dec.layers = self.saved


def run_window(model, prep, a, device, edits=None, adds=None, capture=False, x_in=None):
    """Run layers a, a+1 for the active token.
    edits: (kind, slice, value) overwrite layer-a internal outputs ('o' = o_proj input, 'd' = down_proj input);
    adds : (kind, vector) residual-space corrections added to the OUTPUT of o_proj ('o') or down_proj ('d')."""
    dec = Q.decoder(model); La = dec.layers[a]; cap = {}

    def pre(kind):
        def f(m, args):
            x = args[0]
            cap[kind] = x[:, -1, :].detach().clone()
            if edits and any(k == kind for k, _, _ in edits):
                x = x.clone()
                for k, sl, val in edits:
                    if k == kind:
                        x[:, -1, sl] = val.to(x.dtype)
                return (x,) + tuple(args[1:])
            return None
        return f

    def post(kind):
        def f(m, args, out):
            vs = [v for k, v in (adds or []) if k == kind]
            if not vs:
                return None
            out = out.clone()
            out[:, -1, :] = (out[:, -1, :].float() + sum(v.float().to(out.device) for v in vs)).to(out.dtype)
            return out
        return f

    hs = [La.self_attn.o_proj.register_forward_pre_hook(pre("o")), La.mlp.down_proj.register_forward_pre_hook(pre("d")),
          La.self_attn.o_proj.register_forward_hook(post("o")), La.mlp.down_proj.register_forward_hook(post("d")),
          dec.layers[a + 1].register_forward_hook(lambda m, ar, out: cap.__setitem__("site", (out[0] if isinstance(out, tuple) else out)[:, -1, :].detach().clone()))]
    n = prep["n"]
    try:
        with Window(model, a):
            dec(inputs_embeds=prep["x_in"] if x_in is None else x_in, attention_mask=torch.ones((1, n + 1), dtype=torch.long, device=device),
                position_ids=torch.tensor([[n]], device=device), past_key_values=prep["cache"], use_cache=True)
        Q.crop_cache(prep["cache"], n)
    finally:
        for h in hs: h.remove()
    out = dict(site=cap["site"][0].float().cpu())
    if capture:
        out["o"] = cap["o"][0]; out["d"] = cap["d"][0]
    return out


# ----------------------------------------------------------------------------- candidates
def candidate_registry(model, packets=None, n_blocks=4):
    nq, nkv, hd, group = Q.geom(model)
    if packets is None:
        packets = [tuple(range(g * group, (g + 1) * group)) for g in range(nkv)]
    dff = Q.decoder(model).layers[0].mlp.down_proj.weight.shape[1]; bs = dff // n_blocks
    cands = [dict(name=f"heads{list(p)}", kind="o", slices=[slice(h * hd, (h + 1) * hd) for h in p]) for p in packets]
    cands += [dict(name=f"mlp{j}", kind="d", slices=[slice(j * bs, (j + 1) * bs)]) for j in range(n_blocks)]
    return cands


def _kinds(c):
    return c.get("kinds") or [c["kind"]] * len(c["slices"])


def cand_values(internal, c):
    return [internal[k][sl] for k, sl in zip(_kinds(c), c["slices"])]


def edits_for(c, values):
    return [(k, sl, v) for k, sl, v in zip(_kinds(c), c["slices"], values)]


def cand_write(model, a, c, internal):
    L = Q.decoder(model).layers[a]; tot = 0
    for k, sl in zip(_kinds(c), c["slices"]):
        W = L.self_attn.o_proj.weight if k == "o" else L.mlp.down_proj.weight
        tot = tot + (W[:, sl].float() @ internal[k][sl].float())
    return tot.cpu()


def rel(a, b):
    a = a.double(); b = b.double()
    return float((a - b).pow(2).mean().sqrt() / (0.5 * (a.pow(2).mean().sqrt() + b.pow(2).mean().sqrt()) + 1e-30))


def merge_cands(cands, groups):
    out = []
    for g in groups:
        mem = [next(c for c in cands if c["name"] == n) for n in g]
        if len(mem) == 1:
            out.append(mem[0]); continue
        out.append(dict(name="+".join(g), kind=mem[0]["kind"], slices=sum((m["slices"] for m in mem), []),
                        kinds=sum((_kinds(m) for m in mem), []), members=list(g)))
    return out


# ----------------------------------------------------------------------------- declared battery (one replica)
def battery(model, base, build_caches, suffixes, a, device, cands, seed=0, n_perturb=REFAC_PERTURB, dose=REFAC_DOSE):
    """Calibration contexts (k=0) + counterfactual upstream interventions (k>=1): clean sites and candidate writes;
    plus single-candidate mean ablations and substitutions on the calibration contexts."""
    g = torch.Generator().manual_seed(seed)
    preps = [prepare_window(model, base, cc, sfx, a, device) for cc, sfx in zip(build_caches(), suffixes)]
    entries, clean, writes = [], [], {c["name"]: [] for c in cands}
    internals = []
    for t, p in enumerate(preps):
        x0 = p["x_in"]; scale = float(x0.float().pow(2).mean().sqrt())
        for k in range(n_perturb + 1):
            x = x0 if k == 0 else (x0.float() + dose * scale * torch.randn(x0.shape, generator=g).to(x0.device)).to(x0.dtype)
            r = run_window(model, p, a, device, capture=True, x_in=x)
            entries.append((t, k, x)); clean.append(r["site"]); internals.append((r["o"], r["d"]))
            for c in cands:
                writes[c["name"]].append(cand_write(model, a, c, r))
    B = dict(preps=preps, entries=entries, clean=clean, writes={n: torch.stack(v) for n, v in writes.items()}, internals=internals)
    B.update(ablations(model, B, a, device, cands))
    return B


def ablations(model, B, a, device, cands):
    """Mean ablation and cross-context substitution of each candidate on the calibration contexts (k = 0)."""
    cal = [i for i, (t, k, x) in enumerate(B["entries"]) if k == 0]
    clean_int = {i: dict(o=B["internals"][i][0], d=B["internals"][i][1]) for i in cal}
    means = {c["name"]: [torch.stack([cand_values(clean_int[i], c)[j] for i in cal]).mean(0) for j in range(len(c["slices"]))] for c in cands}
    abl, sub = {}, {}
    for c in cands:
        abl[c["name"]] = [run_window(model, B["preps"][B["entries"][i][0]], a, device, edits_for(c, means[c["name"]]), x_in=B["entries"][i][2])["site"] for i in cal]
        sub[c["name"]] = {(i, j): run_window(model, B["preps"][B["entries"][i][0]], a, device, edits_for(c, cand_values(clean_int[j], c)),
                                             x_in=B["entries"][i][2])["site"] for i in cal for j in cal if i != j}
    return dict(cal=cal, abl=abl, sub=sub)


def site_states(B):
    s = list(B["clean"])
    for n in B["abl"]:
        s += B["abl"][n] + [B["sub"][n][k] for k in sorted(B["sub"][n])]
    return s


def consensus(vs, pos="FUSION", neg="DISTINCT"):
    return pos if all(v == pos for v in vs) else (neg if all(v == neg for v in vs) else "AMBIGUOUS")


def band(e, d):
    return "FUSION" if e <= d else ("DISTINCT" if e > 2 * d else "AMBIGUOUS")


# ----------------------------------------------------------------------------- stage 0
def stage0_pair(model, B, a, device, cx, cy):
    """Residual causal effect of replacing each write by its fixed-factor (centred) reconstruction from the other."""
    wx, wy = B["writes"][cx["name"]].double(), B["writes"][cy["name"]].double()
    mx, my = wx.mean(0, keepdim=True), wy.mean(0, keepdim=True); X, Y = wx - mx, wy - my
    nX, nY = float((X * X).sum()), float((Y * Y).sum())
    s_xy = float((X * Y).sum()) / nX if nX > 0 else 0.0     # a constant write reconstructs nothing but its mean
    s_yx = float((X * Y).sum()) / nY if nY > 0 else 0.0
    geo = max(rel(Y, s_xy * X) if nY > 0 else 0.0, rel(X, s_yx * Y) if nX > 0 else 0.0)
    e = 0.0
    for i, (t, k, x) in enumerate(B["entries"]):
        corr_y = (my[0] + s_xy * X[i]) - wy[i]          # y's write := reconstruction from x
        corr_x = (mx[0] + s_yx * Y[i]) - wx[i]          # x's write := reconstruction from y
        for c, corr in ((cy, corr_y), (cx, corr_x)):
            site = run_window(model, B["preps"][t], a, device, adds=[(c["kind"], corr)], x_in=x)["site"]
            e = max(e, rel(site, B["clean"][i]))
    return dict(e=e, geometric_residual=geo, s_xy=s_xy, s_yx=s_yx)


def components_cliques(names, edges):
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for u, v in edges:
        parent[find(v)] = find(u)
    comps = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    groups, nontransitive = [], []
    es = {frozenset(e) for e in edges}
    for comp in comps.values():
        if len(comp) > 1 and not all(frozenset((u, v)) in es for u, v in itertools.combinations(comp, 2)):
            nontransitive.append(comp); groups += [[n] for n in comp]
        else:
            groups.append(comp)
    return groups, nontransitive


def incoming_profile(B, name):
    w = B["writes"][name].double(); out = []
    for i, (t, k, x) in enumerate(B["entries"]):
        if k == 0:
            i0 = i
        else:
            out.append(float((w[i] - w[i0]).norm() / (w[i0].norm() + 1e-30)))
    return torch.tensor(out, dtype=torch.float64)


# ----------------------------------------------------------------------------- full certification
def certify_layer(model, base, build_caches, suffixes, a, device, cands=None, gauges=Q.GAUGE_SET, log=print, n_perturb=REFAC_PERTURB):
    t0 = time.time()
    cands = cands or candidate_registry(model)
    names = [c["name"] for c in cands]
    for cc, sfx in zip(build_caches(), suffixes):
        p = prepare_window(model, base, cc, sfx, a, device)
        if not torch.equal(run_window(model, p, a, device)["site"], p["site_full"][0, 0].float().cpu()):
            raise RuntimeError("2-layer local window is not bit-identical to the full forward")
    replicas = [("IDENTITY", 0)] + list(gauges)

    def per_replica(fn):
        out = []
        for kind, s in replicas:
            if kind == "IDENTITY":
                out.append(fn())
            else:
                with Q.Gauged(model, a, kind, s):
                    out.append(fn())
        return out

    # PASS 1, replica by replica (only one replica's prompt caches alive at a time): declared battery on the
    # PHYSICAL candidates + raw stage-0 residual effects.  Verdicts are banded only after delta is frozen.
    pairs = list(itertools.combinations(cands, 2)); st0 = {f"{x['name']}|{y['name']}": [] for x, y in pairs}; Bs = []

    def pass1():
        B = battery(model, base, build_caches, suffixes, a, device, cands, n_perturb=n_perturb)
        for x, y in pairs:
            st0[f"{x['name']}|{y['name']}"].append(stage0_pair(model, B, a, device, x, y))
        del B["preps"]; Bs.append(B)
    per_replica(pass1)
    S = [site_states(B) for B in Bs]
    diam = max(rel(S[r][i], S[q][i]) for i in range(len(S[0])) for r, q in itertools.combinations(range(len(S)), 2))
    delta = math.nextafter(diam, math.inf)
    log(f"[roleq v3.2] L{a}: delta_site={delta:.4g} over {len(S[0])} battery site states x {len(S)} replicas")
    stage0 = {}
    for key, rows in st0.items():
        vs = [band(rw["e"], delta) for rw in rows]
        stage0[key] = dict(status=consensus(vs), replica_statuses=vs, residual_effect_over_delta=[rw["e"] / delta for rw in rows],
                           geometric_residual_max=max(rw["geometric_residual"] for rw in rows),
                           factor_range=[min((rw["s_xy"] for rw in rows if rw["s_xy"] is not None), default=None),
                                         max((rw["s_xy"] for rw in rows if rw["s_xy"] is not None), default=None)])
    edges = [tuple(k.split("|")) for k, v in stage0.items() if v["status"] == "FUSION"]
    groups0, nontrans = components_cliques(names, edges)
    macros = merge_cands(cands, groups0)
    # PASS 2, replica by replica: caches rebuilt under the replica's recoding, macro ablations / substitutions
    it = iter(range(len(replicas)))

    def pass2():
        r = next(it); B = Bs[r]
        B["preps"] = [prepare_window(model, base, cc, sfx, a, device) for cc, sfx in zip(build_caches(), suffixes)]
        B.update(ablations(model, B, a, device, macros)); del B["preps"]
        for m in macros:
            if "members" in m:
                B["writes"][m["name"]] = sum(B["writes"][n] for n in m["members"])
    per_replica(pass2)
    cal = Bs[0]["cal"]
    act = {m["name"]: [max(rel(B["abl"][m["name"]][j], B["clean"][i]) for j, i in enumerate(cal)) for B in Bs] for m in macros}
    def act_status(v):
        return "ACTIVE_ROLE" if all(x > 2 * delta for x in v) else ("INACTIVE" if all(x <= delta for x in v) else "AMBIGUOUS")
    activity = {n: act_status(v) for n, v in act.items()}
    prof = {m["name"]: [incoming_profile(B, m["name"]) for B in Bs] for m in macros}
    tau_in = math.nextafter(max((rel(prof[n][0], prof[n][r]) for n in prof for r in range(1, len(Bs))), default=0.0), math.inf)
    active = [m["name"] for m in macros if activity[m["name"]] == "ACTIVE_ROLE"]
    stage2 = {}
    for x, y in itertools.combinations(active, 2):
        vs, dd, di = [], [], []
        for r, B in enumerate(Bs):
            d_abl = max(rel(B["abl"][x][j], B["abl"][y][j]) for j in range(len(cal)))
            d_sub = max(rel(B["sub"][x][k], B["sub"][y][k]) for k in B["sub"][x])
            d_down = max(d_abl, d_sub); d_in = rel(prof[x][r], prof[y][r])
            if d_down <= delta and d_in <= tau_in:
                v = "FUSION"
            elif d_down > 2 * delta or d_in > 2 * tau_in:
                v = "DISTINCT"
            else:
                v = "AMBIGUOUS"
            vs.append(v); dd.append(d_down / delta); di.append(d_in / tau_in if tau_in > 0 else math.inf)
        stage2[f"{x}|{y}"] = dict(status=consensus(vs), replica_statuses=vs, d_down_over_delta=[min(dd), max(dd)], d_in_over_tau_in=[min(di), max(di)])
    roles = []
    for n in active:
        for g in roles:
            if all(stage2.get(f"{m}|{n}", stage2.get(f"{n}|{m}"))["status"] == "FUSION" for m in g):
                g.append(n); break
        else:
            roles.append([n])
    amb = dict(stage0=[k for k, v in stage0.items() if v["status"] == "AMBIGUOUS"], stage0_non_transitive=nontrans,
               stage1=[n for n, v in activity.items() if v == "AMBIGUOUS"], stage2=[k for k, v in stage2.items() if v["status"] == "AMBIGUOUS"])
    # exclusion of everything partially identified (GPT Q1): a role is certified only if none of its macros has an
    # AMBIGUOUS stage-2 relation and none of its physical members has an AMBIGUOUS stage-0 relation
    amb_macro = {n for k in amb["stage2"] for n in k.split("|")}
    amb_phys = {n for k in amb["stage0"] for n in k.split("|")} | {n for comp in nontrans for n in comp}
    members_of = {m["name"]: m.get("members", [m["name"]]) for m in macros}
    def excluded(g):
        return any(n in amb_macro for n in g) or any(p in amb_phys for n in g for p in members_of[n])
    roles_all = roles; roles = [g for g in roles_all if not excluded(g)]
    role_map = {}
    for m in macros:
        g = next((g for g in roles_all if m["name"] in g), None)
        rid = f"L{a}.R{roles.index(g)}" if g is not None and g in roles else None
        st = "CERTIFIED_ROLE" if rid else ("AMBIGUOUS_EXCLUDED" if g is not None else activity[m["name"]])
        for n in members_of[m["name"]]:
            role_map[n] = dict(role=rid, macro=m["name"], status=st)
    out = dict(version=VERSION, layer=a, role_map=role_map, site=f"post-L{a+1}", replicas=len(replicas), delta_site=delta, tau_in=tau_in,
               stage0=stage0, stage0_groups=groups0, activity_status=activity, activity_over_delta={n: [min(v) / delta, max(v) / delta] for n, v in act.items()},
               stage2=stage2, certified_roles=roles, active_groups_before_exclusion=roles_all, inactive=[n for n, v in activity.items() if v == "INACTIVE"],
               partial_identification=amb, quotient_is_partially_identified=bool(any(amb.values())), seconds=round(time.time() - t0, 1))
    log(f"[roleq v3.2] L{a} -> post-L{a+1}: stage0 groups={[g for g in groups0 if len(g) > 1]}; certified roles={roles}; "
        f"inactive={out['inactive']}; ambiguous={ {k: len(v) for k, v in amb.items()} } ({out['seconds']}s)")
    return out
