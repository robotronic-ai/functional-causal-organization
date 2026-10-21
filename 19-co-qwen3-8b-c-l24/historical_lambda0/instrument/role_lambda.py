#!/usr/bin/env python3
"""L1 - Lambda_O: typed, partially identified content -> certified-role relation (diagnostic).  DIAGNOSTIC ONLY.

Inputs (all declared before measurement): certified roles of layer a (merged macro candidates from ROLEQ v3.2), a DECLARED
reference content, and a list of content interventions z_1..z_n (each a residual vector entering layer a, built per replica
from the replica's own prompt caches).  For every calibration context t (attention cache) the layer-a window is run with the
residual replaced by x_ref and by x_z; the observed object is the WRITE of each role (exact-recoding invariant).
    e(r, z) = max_t rel( write_r | x_z , write_r | x_ref )
ONE delta_w for the site (role writes) = max exact-recoding orbit diameter over every state, frozen before any verdict;
10 replicas, caches rebuilt under every recoding.  Per cell:
    RESOLVED_POSITIVE  if e > 2 delta_w in every replica;
    UNRESOLVED         otherwise (band_detail: BELOW_RESOLUTION if e <= delta_w in every replica, else AMBIGUOUS).
There is deliberately NO negative status: a finite battery never proves "content z does not go to role r".
Content classes: z, z' are resolved as distinct contents only if rel(x_z, x_z') > 2 delta_x in every replica (delta_x = orbit
diameter of the content vectors); otherwise the pair is reported CONTENT_PAIR_UNRESOLVED and must be merged or dropped.
Lambda_O (lower relation) = {(z, r) : RESOLVED_POSITIVE}; every superset consistent with the UNRESOLVED cells is a completion.
Two readings are reported, never mixed: the LABELED relation (contents named, roles named) and its ISOMORPHISM CLASS
(modulo joint relabelling of contents and roles).  A one-sided permutation changes the labeled relation; whether it changes the
class depends on whether labels are anchored across systems (open protocol point for GPT).
"""
from __future__ import annotations

import itertools, math, time
import torch

import qwen_local_transition as Q
import role_quotient as RQ
import role_transport as RT

torch.set_grad_enabled(False)
VERSION = "ROLELAMBDA_v1_PROPOSED"
REF = "__REF__"


def one_replica(model, base, build_caches, suffixes, a, device, roles, contents):
    preps = [RQ.prepare_window(model, base, cc, sfx, a, device) for cc, sfx in zip(build_caches(), suffixes)]
    xs = {REF: contents["ref"](preps)}; xs.update({c["name"]: c["fn"](preps) for c in contents["items"]})
    w = {}
    for t, p in enumerate(preps):
        for zn, x in xs.items():
            r = RQ.run_window(model, p, a, device, capture=True, x_in=x)
            w[(t, zn)] = {rn: RQ.cand_write(model, a, c, r).double() for rn, c in roles.items()}
    del preps
    return dict(w=w, xin={zn: x[0, 0].float().cpu().double() for zn, x in xs.items()})


def canonical_signature(M):
    """Isomorphism class of a 0/1 matrix (contents x roles) modulo joint relabelling: exact brute force over the smaller
    side when it has <= 8 elements, otherwise a 2-round WL hash (exact=False)."""
    import hashlib
    nz, nr = len(M), (len(M[0]) if M else 0)
    if nz == 0 or nr == 0:
        return f"empty-{nz}x{nr}", True
    if nr <= 8:
        best = min(tuple(sorted(tuple(row[p] for p in perm) for row in M)) for perm in itertools.permutations(range(nr)))
        return hashlib.sha256(repr((nz, nr, "C", best)).encode()).hexdigest()[:16], True
    if nz <= 8:
        cols = list(zip(*M))
        best = min(tuple(sorted(tuple(col[p] for p in perm) for col in cols)) for perm in itertools.permutations(range(nz)))
        return hashlib.sha256(repr((nz, nr, "R", best)).encode()).hexdigest()[:16], True
    rl = [sum(r) for r in M]; cl = [sum(c) for c in zip(*M)]
    for _ in range(2):
        rl2 = [hash((rl[i], tuple(sorted(cl[j] for j in range(nr) if M[i][j])))) for i in range(nz)]
        cl2 = [hash((cl[j], tuple(sorted(rl[i] for i in range(nz) if M[i][j])))) for j in range(nr)]
        rl, cl = rl2, cl2
    return hashlib.sha256(repr((nz, nr, "WL", sorted(rl), sorted(cl))).encode()).hexdigest()[:16], False


def finalize(states, roles, content_names, a, log=print):
    R = len(states); names = [REF] + list(content_names)
    vw = [{(t, zn, rn): v for (t, zn), d in s["w"].items() for rn, v in d.items()} for s in states]
    diam = max(RQ.rel(vw[r][k], vw[q][k]) for k in vw[0] for r, q in itertools.combinations(range(R), 2))
    delta_w = math.nextafter(diam, math.inf)
    xd = max(RQ.rel(states[r]["xin"][z], states[q]["xin"][z]) for z in names for r, q in itertools.combinations(range(R), 2))
    delta_x = math.nextafter(xd, math.inf); T = max(t for t, _ in states[0]["w"]) + 1
    cells, M = {}, []
    for z in content_names:
        row = []
        for rn in roles:
            es = [max(RQ.rel(s["w"][(t, z)][rn], s["w"][(t, REF)][rn]) for t in range(T)) for s in states]
            pos = all(e > 2 * delta_w for e in es)
            cells[f"{z}->{rn}"] = dict(status="RESOLVED_POSITIVE" if pos else "UNRESOLVED",
                                       band_detail=("POSITIVE" if pos else ("BELOW_RESOLUTION" if all(e <= delta_w for e in es) else "AMBIGUOUS")),
                                       effect_over_delta=[min(es) / delta_w, max(es) / delta_w])
            row.append(int(pos))
        M.append(row)
    pairs = {}
    for z1, z2 in itertools.combinations(names, 2):
        es = [RQ.rel(s["xin"][z1], s["xin"][z2]) for s in states]
        pairs[f"{z1}|{z2}"] = "RESOLVED_DISTINCT" if all(e > 2 * delta_x for e in es) else "CONTENT_PAIR_UNRESOLVED"
    unresolved_pairs = [k for k, v in pairs.items() if v != "RESOLVED_DISTINCT"]
    sig, exact = canonical_signature(M)
    lower = {z: [rn for rn, v in zip(roles, M[i]) if v] for i, z in enumerate(content_names)}
    n_unres = sum(c["status"] == "UNRESOLVED" for c in cells.values())
    out = dict(version=VERSION, layer=a, site=f"role writes at layer {a}", replicas=R, delta_w=delta_w, delta_x=delta_x, roles=list(roles),
               contents=list(content_names), reference=REF, cells=cells, lower_relation=lower,
               labeled_relation=sorted([z, rn] for z, rs in lower.items() for rn in rs), iso_class=sig, iso_class_exact=exact,
               content_pairs=pairs, content_pairs_unresolved=unresolved_pairs, n_cells=len(cells), n_unresolved_cells=n_unres,
               completions=f"every relation containing the {sum(map(sum, M))} certified cells (2^{n_unres} candidates)",
               note="no negative status exists: UNRESOLVED never means 'does not go there'")
    log(f"[rolelambda] L{a}: delta_w={delta_w:.4g} delta_x={delta_x:.3g}; lower relation={lower}; unresolved cells={n_unres}/{len(cells)}; unresolved content pairs={len(unresolved_pairs)}")
    return out


def lambda_layer(model, base, build_caches, suffixes, a, device, roles, contents, gauges=Q.GAUGE_SET, ckpt_dir=None, resume=False,
                 log=print, stop_after=None):
    """contents = dict(ref=fn(preps)->x_in tensor (1,1,d), items=[dict(name, fn, sig)]) ; roles = {role: merged candidate}."""
    t0 = time.time()
    for cc, sfx in zip(build_caches(), suffixes):
        p = RQ.prepare_window(model, base, cc, sfx, a, device)
        if not torch.equal(RQ.run_window(model, p, a, device)["site"], p["site_full"][0, 0].float().cpu()):
            raise RuntimeError("2-layer local window is not bit-identical to the full forward")
    signature = RT.sig_of(dict(a=a, roles={n: [str(s) for s in c["slices"]] + RT._k(c) for n, c in roles.items()}, items=[(c["name"], c.get("sig")) for c in contents["items"]],
                               ref=contents.get("ref_sig"), n_ctx=len(suffixes), suffixes=suffixes, gauges=[list(g) for g in gauges],
                               dtype=str(next(model.parameters()).dtype), model=RT.model_fingerprint(model, a)))
    states = RT.replica_loop(model, a, gauges, lambda: one_replica(model, base, build_caches, suffixes, a, device, roles, contents),
                             ckpt_dir, resume, signature, log, stop_after)
    out = finalize(states, roles, [c["name"] for c in contents["items"]], a, log)
    out["seconds"] = round(time.time() - t0, 1); out["signature"] = signature
    return out
