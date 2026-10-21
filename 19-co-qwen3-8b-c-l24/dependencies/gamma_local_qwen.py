#!/usr/bin/env python3
"""Diagnostic-only Gamma probe on Qwen3-8B with layer-local execution (layer l -> l+1, active position).

Computes NO R, B or P.  Reports: bit-identity of the local layer run against the full forward, the measured cost
of a full single-token forward versus a single-layer local run, the exact-gauge resolution of the measured state,
and two tested-cut upper bounds, U_tested(delta) and U_tested(2 delta), over the declared partition family.
No lower causal certificate is computed here: a positive minimum over tested cuts does not prove Gamma > 0.

Usage (user machine):
  $PY gamma_local_qwen.py --engine-dir $ENG --schedule $SCH --model-dir Qwen3-8B --timing-only --output V_GAMMA_LOCAL_TIMING.json
  $PY gamma_local_qwen.py --engine-dir $ENG --schedule $SCH --model-dir Qwen3-8B --output V_GAMMA_LOCAL_PROBE.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_local_transition as Q          # noqa: E402

torch.set_grad_enabled(False)


# --------------------------------------------------------------------- engine glue (same calls as v77 / T111B)
def _imports(engine_dir):
    p = str(Path(engine_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import runtime as base            # type: ignore
    import qwen_v427_runtime as rt    # type: ignore
    import run_direct_co_op_panel as eng  # type: ignore
    return base, rt, eng


def load_model(a, rt):
    return rt.load_model(Path(a.model_dir), a.device, a.dtype)


def schedule_cases(schedule, unit_ids, ops, addrs):
    out = []
    for ep0 in schedule:
        if int(ep0["unit"]) not in set(unit_ids):
            continue
        ep = dict(ep0)
        ep["bindings"] = {k: int(v) for k, v in ep["bindings"].items()}
        ep["content_donor_values"] = {k: int(v) for k, v in ep["content_donor_values"].items()}
        for op in ops:
            for ad in addrs:
                out.append(dict(key=f"u{int(ep['unit'])}|{op}|{ad}", unit=int(ep["unit"]), op=op, ad=ad, ep=ep))
    return out


def build_receiver(eng, model, tok, case, device):
    ep, op, ad = case["ep"], case["op"], case["ad"]
    return eng._build_control(model, tok, ep["bindings"], op, ad, ep["content_donor_values"][ad], device)


def get_probe_suffix(rt, tok, receiver, ad):
    _, _, suffix = rt._probe_prompt(tok, receiver["presentation"], receiver["instruction"], ad)
    return [int(x) for x in suffix]


# --------------------------------------------------------------------- timing
def timing(model, base, prep, layer, device, reps=8):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    c = base.clone_cache(prep["cache"]); last = prep["last_token"]
    t = time.perf_counter()
    for _ in range(reps):
        model(input_ids=torch.tensor([[last]], device=device), attention_mask=torch.ones((1, prep["n"] + 1), dtype=torch.long, device=device),
              past_key_values=c, use_cache=True)
        Q.crop_cache(c, prep["n"])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    full = (time.perf_counter() - t) / reps
    prep["layer_module"] = Q.decoder(model).layers[layer + 1]
    t = time.perf_counter()
    Q.local_run(model, prep, prep["x_l"].repeat(reps, 1, 1), layer, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    local = (time.perf_counter() - t) / reps
    del c
    return dict(full_single_token_forward_s=full, local_single_layer_run_s=local, speedup=full / max(local, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True); ap.add_argument("--schedule", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--case", default="u15|SUBSTITUTE|H1"); ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--timing-only", action="store_true"); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    base, rt, eng = _imports(a.engine_dir)
    schedule = json.loads(Path(a.schedule).read_text())
    unit = int(a.case.split("|")[0].lstrip("u"))
    cases = [c for c in schedule_cases(schedule, [unit], ["SUBSTITUTE", "REVOKE"], ["H1", "H2"]) if c["key"] == a.case]
    if not cases:
        raise SystemExit(f"case {a.case} not found")
    case = cases[0]
    model, tok = load_model(a, rt)
    rec0 = build_receiver(eng, model, tok, case, a.device); suffix = get_probe_suffix(rt, tok, rec0, case["ad"]); del rec0

    def build_control_cache():
        return build_receiver(eng, model, tok, case, a.device)["control_cache"]

    out = dict(schema="co-gamma-local-turn121-diagnostic-v1", status="DIAGNOSTIC_ONLY", computes_reentry=False, case=a.case, layer=a.layer)
    prep = Q.prepare_active_position(model, base, build_control_cache(), suffix, a.layer, a.device)
    prep["last_token"] = suffix[-1]
    ok, err = Q.verify_local_equals_full(model, prep, a.layer, a.device)
    out["local_equals_full_bit_identical"] = ok; out["local_vs_full_max_abs_diff"] = err
    print(f"[gamma-local] local layer run bit-identical to full forward: {ok} (max abs diff {err:.3g})", flush=True)
    out["timing"] = timing(model, base, prep, a.layer, a.device)
    print(f"[gamma-local] timing: {json.dumps(out['timing'])}", flush=True)
    del prep
    if device_is_cuda := a.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not a.timing_only:
        if not ok:
            raise SystemExit("local run not bit-identical: refusing to compute Gamma")
        g = Q.gamma_local(model, base, build_control_cache, suffix, a.layer, a.device, log=lambda s: print(s, flush=True))
        g["units"] = [list(map(lambda x: list(x) if isinstance(x, tuple) else x, u)) for u in g["units"]]
        for r in g["partitions"]:
            r["A"] = str(r["A"]); r["B"] = str(r["B"])
        out["gamma"] = g
        out["interpretation"] = ("L_Gamma is NA. U_tested(delta) is the minimum over the DECLARED tested partition family and therefore only an upper bound on the true Gamma partition minimum. U_tested(2delta) is a resolution-sensitivity companion upper bound, not a lower Gamma bound. Role quotient not certified: diagnostic, not a C_O value.")
    Path(a.output).write_text(json.dumps(out, indent=1, default=float)); print("written", a.output)


if __name__ == "__main__":
    main()
