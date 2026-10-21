#!/usr/bin/env python3
"""T346 targeted calibration of FCA_E1 (FCA-Bench v0.2.7, test E1) at the mlp0@L24 site: does an E1 change obtained with ZERO persist with MEAN, with a
matched conditional substitution, and beyond an untargeted perturbation of matched strength? One world (u26), descriptive only.

All interventions act on activations, never on weights, on the input of layers[L].mlp.down_proj ('mlp' sites) or of layers[L].self_attn.o_proj ('attn');
'block0' = channels [0, I/4) of the MLP (the mlp0 role of C_L24), 'all' = every channel.
  ZERO            activations of the site set to 0 at every processed token (the T346 pilot intervention, dose 1)
  MEAN            activations replaced by their per-channel mean over reference E1 runs on disjoint instances, at every processed token
  PATCH           matched conditional substitution: in prompts that contain the E1 context sentence "Auxiliary identifier for OBJ: CODE. Semantic category: CAT.",
                  the site activations of the prompt (prefill) are replaced by those computed on the same prompt in which CODE and CAT are replaced by another listed
                  code and category whose prompt has the same token length and differs only inside the span of that sentence (decoding steps are not patched; prompts without
                  the sentence - clean and placebo conditions - are left untouched). Every application or skip is counted with its reason.
  RANDOM_MATCHED  untargeted perturbation of matched strength: at every processed token the output of layers[24].mlp.down_proj receives a vector along one fixed
                  random unit direction whose norm equals, token by token, the norm of the change that ZERO on the site would produce at that output.
  SHAM            the MEAN code path with dose 0 on the site (must reproduce BASELINE exactly; the run stops otherwise)
Default arms (6 instances, seeds 2026101801-2026101806): CAL_BASELINE, CAL_SHAM, MLP0_L24_ZERO, MLP0_L24_MEAN, MLP_L24_MEAN, MLP0_L24_PATCH_MATCHED,
STRENGTH_MATCHED_RANDOM. Optional (--arms): MLP_L24_ZERO, ATTN_L24_MEAN, MLP_L20_28_MEAN, MLP_L16_31_MEAN (strong-perturbation controls).
Strength of every arm, measured against the unhooked model on (a) four neutral texts and (b) E1 prompts captured from the reference runs: relative change of the
hidden state after layer 27, of the last hidden state, KL(base || arm) of the next-token distribution (mean over positions and at the last position), top-1 agreement.
Numerical regime: historical FCA R6 (bnb8, float16 compute) by default; --quantization / --dtype set another regime (recorded as deviations).
Not measured here: Gamma, Delta, R, Lambda on the modified states. No inference.

Modes: --selftest (CPU, tiny random 36-layer Qwen3 behind a fake adapter; no GPU, no Qwen) ; --run (GPU, needs --authorize-gpu-run) ; --analyze (CPU).
Requires t346_fca_e1_pilot.py and the FCA-Bench v0.2.7 package (FCA027/) in the working folder.
"""
import argparse, copy, csv, hashlib, importlib.util, json, random, sys, tempfile, time
from pathlib import Path

import numpy as np

WORK = Path('/mnt/c/Users/Merien/pycharm/llama/quantum')
OUTNAME = 'FAST_GPU_RESULTS/CLAUDE_T346_FCA_E1_CALIBRATION'
BASE_SCRIPT = 't346_fca_e1_pilot.py'
CONFIRM = 'I_AUTHORIZE_T346_FCA_CALIBRATION_GPU_RUN'
TASK_SEED, REF_SEED, BOOTSTRAP_SEED, DIRECTION_SEED = 2026101801, 2026102001, 2026101804, 2026101805
N_INSTANCES, N_REF, BOOTSTRAP_REPS, BAND_E1, N_E1_PROMPTS = 6, 2, 5000, 0.05, 13
AMP_LAYER = 27
MLP_ALL = lambda layers: tuple(('mlp', L, 'all') for L in layers)
B0 = (('mlp', 24, 'block0'),)
ALL_ARMS = (('CAL_BASELINE', 'NO_HOOK', ()),
            ('CAL_SHAM', 'SHAM', (('mlp', 24, 'all'),)),
            ('MLP0_L24_ZERO', 'ZERO', B0),
            ('MLP0_L24_MEAN', 'MEAN', B0),
            ('MLP_L24_MEAN', 'MEAN', (('mlp', 24, 'all'),)),
            ('MLP0_L24_PATCH_MATCHED', 'PATCH', B0),
            ('STRENGTH_MATCHED_RANDOM', 'RANDOM_MATCHED', B0),
            ('MLP_L24_ZERO', 'ZERO', (('mlp', 24, 'all'),)),
            ('ATTN_L24_MEAN', 'MEAN', (('attn', 24, 'all'),)),
            ('MLP_L20_28_MEAN', 'MEAN', MLP_ALL(range(20, 29))),
            ('MLP_L16_31_MEAN', 'MEAN', MLP_ALL(range(16, 32))))
DEFAULT_ARMS = ('CAL_BASELINE', 'CAL_SHAM', 'MLP0_L24_ZERO', 'MLP0_L24_MEAN', 'MLP_L24_MEAN', 'MLP0_L24_PATCH_MATCHED', 'STRENGTH_MATCHED_RANDOM')
AMP_TEXTS = ('The library opens at nine in the morning and closes at six in the evening. Visitors can borrow up to five books at a time.',
             'A small boat crossed the lake slowly while the wind changed direction twice before noon.',
             'To prepare the soil, remove the stones, add compost, and water the bed lightly for three days.',
             'The committee reviewed the budget, postponed two projects, and asked for a new estimate by Friday.')
DETAIL_KEYS = ('distributional_effect', 'residual_effect', 's_distributional', 's_direct', 'poison_direct_leak', 'erase_direct_leak', 'leak_reinjection_diagnostic')
E1_CODES = ["NIMBUS", "CEDAR", "ATLAS", "MERIDIAN", "ORBIT", "FALCON", "QUARTZ", "EMBER"]
E1_CATS = ["PLANT", "PLACE", "MATERIAL", "ANIMAL", "DEVICE", "COLOR"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def layers_of(model):
    m = model
    for attr in ('model', 'layers'):
        if hasattr(m, attr):
            m = getattr(m, attr)
    return m


def site_module(model, site):
    layer = layers_of(model)[site[1]]
    return layer.mlp.down_proj if site[0] == 'mlp' else layer.self_attn.o_proj


def site_width(model, site):
    c = model.config
    return int(c.intermediate_size) if site[0] == 'mlp' else int(c.num_attention_heads) * int(getattr(c, 'head_dim', c.hidden_size // c.num_attention_heads))


def site_span(model, site):
    w = site_width(model, site)
    return (0, w // 4) if site[2] == 'block0' else (0, w)


def site_key(site):
    return f'{site[0]}_L{site[1]}'


def tok_encode(tok, text):
    return list(tok(text, add_special_tokens=False)['input_ids'])


def tok_decode(tok, ids):
    return tok.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def context_sentence(meta, code=None, cat=None):
    return f"Auxiliary identifier for {meta['obj']}: {code or meta['code']}. Semantic category: {cat or meta['cat']}."


class MeanRecorder:
    """Per-channel sums of the full input of the reference sites over every processed token."""

    def __init__(self, model, sites):
        self.model, self.sites, self.handles, self.sum, self.n = model, sorted({(s[0], s[1], 'all') for s in sites}), [], {}, {}

    def __enter__(self):
        for s in self.sites:
            k = site_key(s)

            def pre(module, args, k=k):
                import torch
                x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float64)
                self.sum[k] = x.sum(0) if k not in self.sum else self.sum[k] + x.sum(0); self.n[k] = self.n.get(k, 0) + x.shape[0]
            self.handles.append(site_module(self.model, s).register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def means(self):
        return {k: (self.sum[k] / self.n[k]).float().cpu().numpy() for k in self.sum}, dict(self.n)


class PromptRecorder:
    """Keeps the first row of the input ids of every generate call (the rendered E1 prompt), with the instance metadata."""

    def __init__(self, model):
        self.model, self.prompts, self.meta = model, [], None

    def __enter__(self):
        self.orig = self.model.generate; outer = self

        def wrapped(*a, **k):
            ids = a[0] if a else k['input_ids']
            outer.prompts.append(dict(ids=[int(t) for t in ids[0].tolist()], meta=dict(outer.meta or {})))
            return outer.orig(*a, **k)
        self.model.generate = wrapped
        return self

    def __exit__(self, *exc):
        self.model.generate = self.orig


class Patcher:
    """Matched conditional substitution of the site activations over the prompt positions (see module docstring)."""

    def __init__(self, model, tokenizer, site):
        self.model, self.tok, self.site = model, tokenizer, site
        self.lo, self.hi = site_span(model, site); self.donor = None; self.capturing = False; self.meta = None
        self.counts = dict(prompts=0, applied=0, skipped_no_context=0, skipped_roundtrip=0, skipped_no_matched_pair=0, skipped_rows_differ=0, patched_forwards=0)
        self.pairs = []

    def donor_ids(self, ids):
        if self.tok is None or not self.meta:
            return None, 'skipped_no_context'
        text = tok_decode(self.tok, ids)
        if tok_encode(self.tok, text) != list(ids):
            return None, 'skipped_roundtrip'
        sent = context_sentence(self.meta)
        if sent not in text:
            return None, 'skipped_no_context'
        rng = random.Random(f"{self.meta.get('seed')}|{self.meta['code']}|{self.meta['cat']}")
        cands = [(c, k) for c in E1_CODES for k in E1_CATS if c != self.meta['code'] and k != self.meta['cat']]
        rng.shuffle(cands)
        for c, k in cands:
            ids2 = tok_encode(self.tok, text.replace(sent, context_sentence(self.meta, c, k), 1))
            if len(ids2) != len(ids):
                continue
            diff = [i for i, (x, y) in enumerate(zip(ids, ids2)) if x != y]
            if diff and diff[-1] - diff[0] + 1 <= len(tok_encode(self.tok, sent)) + 2:
                self.pairs.append(dict(code=c, cat=k, span=[diff[0], diff[-1] + 1]))
                return ids2, 'applied'
        return None, 'skipped_no_matched_pair'

    def prepare(self, ids_row):
        """Compute the donor activations for one prompt; returns True when the patch will be applied to the next prefill of that length."""
        import torch
        self.counts['prompts'] += 1; self.donor = None
        ids2, why = self.donor_ids([int(t) for t in ids_row.tolist()])
        self.counts[why] += 1
        if ids2 is None:
            return False
        got = {}

        def cap(module, args):
            got['x'] = args[0][0, :, self.lo:self.hi].detach().clone()
        h = site_module(self.model, self.site).register_forward_pre_hook(cap)
        self.capturing = True
        try:
            with torch.inference_mode():
                self.model(torch.tensor([ids2], dtype=torch.long, device=ids_row.device))
        finally:
            self.capturing = False; h.remove()
        self.donor = got['x']
        return True

    def pre_hook(self, module, args):
        x = args[0]
        if self.capturing or self.donor is None or x.shape[1] != self.donor.shape[0]:
            return None
        y = x.clone(); y[..., self.lo:self.hi] = self.donor.to(dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        self.counts['patched_forwards'] += 1
        return (y,) + tuple(args[1:])


class Intervention:
    """NO_HOOK, SHAM, ZERO, MEAN, PATCH or RANDOM_MATCHED on a set of sites (see module docstring); activations only."""

    def __init__(self, model, mode, sites, means=None, tokenizer=None, meta=None, dose=1.0):
        self.model, self.mode, self.sites, self.means, self.tok = model, mode, tuple(sites), means or {}, tokenizer
        self.dose = 0.0 if mode == 'SHAM' else float(dose); self.handles, self.calls, self.ss, self.cnt = [], 0, 0.0, 0
        self.patcher = None; self.meta = meta; self.orig_generate = None

    def __enter__(self):
        import torch
        if self.mode == 'NO_HOOK':
            return self
        if self.mode == 'PATCH':
            (site,) = self.sites
            self.patcher = Patcher(self.model, self.tok, site); self.patcher.meta = self.meta
            self.handles.append(site_module(self.model, site).register_forward_pre_hook(self.patcher.pre_hook))
            self.orig_generate = self.model.generate; outer = self

            def wrapped(*a, **k):
                ids = a[0] if a else k['input_ids']
                if not bool((ids == ids[0:1]).all()):
                    outer.patcher.counts['skipped_rows_differ'] += 1; outer.patcher.donor = None
                else:
                    outer.patcher.prepare(ids[0])
                try:
                    return outer.orig_generate(*a, **k)
                finally:
                    outer.patcher.donor = None
            self.model.generate = wrapped
            return self
        if self.mode == 'RANDOM_MATCHED':
            (site,) = self.sites
            lo, hi = site_span(self.model, site); mod = site_module(self.model, site)
            g = torch.Generator().manual_seed(DIRECTION_SEED); u = torch.randn(int(self.model.config.hidden_size), generator=g); u = u / u.norm()
            cache = {}

            def fwd(module, args, output, lo=lo, hi=hi):
                x = args[0]; x0 = x.clone(); x0[..., lo:hi] = 0
                delta = output - module.forward(x0)
                n = delta.float().norm(dim=-1, keepdim=True)
                key = (output.device, output.dtype)
                if key not in cache:
                    cache[key] = u.to(device=output.device)
                add = n * cache[key]
                self.ss += float((add * add).sum()); self.cnt += add.shape[:-1].numel(); self.calls += 1
                return output + add.to(output.dtype)
            self.handles.append(mod.register_forward_hook(fwd))
            return self
        for s in self.sites:
            lo, hi = site_span(self.model, s); mod = site_module(self.model, s)
            use_mean = self.mode in ('MEAN', 'SHAM')
            if use_mean and site_key(s) not in self.means:
                raise RuntimeError('no reference mean for ' + site_key(s))
            mu_full = torch.as_tensor(self.means[site_key(s)]) if use_mean else None
            cache = {}

            def pre(module, args, lo=lo, hi=hi, mu_full=mu_full, cache=cache):
                x = args[0]; y = x.clone(); blk = x[..., lo:hi]
                if mu_full is not None:
                    key = (x.device, x.dtype)
                    if key not in cache:
                        cache[key] = mu_full[lo:hi].to(device=x.device, dtype=x.dtype)
                    new = blk * (1.0 - self.dose) + cache[key] * self.dose
                else:
                    new = blk * (1.0 - self.dose)
                y[..., lo:hi] = new
                r = (blk - new).detach().float(); self.ss += float((r * r).sum()); self.cnt += r.numel(); self.calls += 1
                return (y,) + tuple(args[1:])
            self.handles.append(mod.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self.orig_generate is not None:
            self.model.generate = self.orig_generate; self.orig_generate = None

    def summary(self):
        s = dict(calls=self.calls, change_rms=(self.ss / self.cnt) ** 0.5 if self.cnt else None)
        if self.patcher is not None:
            s['patch'] = dict(self.patcher.counts); s['patch_pairs'] = self.patcher.pairs[:3]
        return s


def tokenize(adapter, model, text):
    import torch
    tok = getattr(adapter, 'tokenizer', None)
    ids = tok_encode(tok, text) if tok is not None else [ord(c) % int(model.config.vocab_size) for c in text]
    return torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)


def amplitude_on(model, adapter, mode, sites, means, items):
    """items: list of (ids tensor [1, T], meta). Strength of one arm against the unhooked model."""
    import torch
    rel_l, rel_f, kl_all, kl_last, top1, applied = [], [], [], [], [], 0
    for ids, meta in items:
        with torch.inference_mode():
            b = model(ids, output_hidden_states=True)
            with Intervention(model, mode, sites, means, getattr(adapter, 'tokenizer', None), meta) as hk:
                if mode == 'PATCH':
                    applied += int(hk.patcher.prepare(ids[0]))
                a = model(ids, output_hidden_states=True)
        hb, ha = b.hidden_states[AMP_LAYER + 1].float(), a.hidden_states[AMP_LAYER + 1].float()
        fb, fa = b.hidden_states[-1].float(), a.hidden_states[-1].float()
        rel_l.append(float((ha - hb).norm() / hb.norm())); rel_f.append(float((fa - fb).norm() / fb.norm()))
        del hb, ha, fb, fa
        lb, la = b.logits[0], a.logits[0]
        kl_parts, top_parts = [], []
        for s0 in range(0, lb.shape[0], 64):          # chunked over positions: the vocabulary-sized float32 tensors stay small
            lpb, lpa = torch.log_softmax(lb[s0:s0 + 64].float(), -1), torch.log_softmax(la[s0:s0 + 64].float(), -1)
            kl_parts.append((lpb.exp() * (lpb - lpa)).sum(-1)); top_parts.append((lpb.argmax(-1) == lpa.argmax(-1)).float())
            del lpb, lpa
        kl = torch.cat(kl_parts); tp = torch.cat(top_parts)
        kl_all.append(float(kl.mean())); kl_last.append(float(kl[-1])); top1.append(float(tp.mean()))
        del b, a, lb, la, kl, tp, kl_parts, top_parts
    out = dict(rel_change_hidden_after_L27=float(np.mean(rel_l)), rel_change_last_hidden=float(np.mean(rel_f)), mean_kl_nats=float(np.mean(kl_all)),
               last_position_kl_nats=float(np.mean(kl_last)), top1_agreement=float(np.mean(top1)), n_items=len(items))
    if mode == 'PATCH':
        out['patch_applied_items'] = applied
    return out


def free_cuda(log=print, tag=''):
    """Release cached CUDA blocks; log allocated / reserved / free so that a near-full device (spill into shared memory) is visible."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        log(f'[T346-CAL] cuda memory {tag}: allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB '
            f'free={free / 2**30:.2f} of {total / 2**30:.2f} GiB')


def means_sha(means):
    h = hashlib.sha256()
    for k in sorted(means):
        h.update(k.encode()); h.update(np.ascontiguousarray(means[k], dtype=np.float32).tobytes())
    return h.hexdigest()


def needed_sites(arms):
    return sorted({(s[0], s[1], 'all') for _, m, ss in arms if m in ('MEAN', 'SHAM') for s in ss})


def prepare(rt, out, header, arms, n_ref, resume, log=print):
    """Reference means and E1 prompts (disjoint instances), then the amplitudes of every arm; all reused on --resume after a checksum check."""
    import torch
    out = Path(out); model = rt['adapter'].model
    if out.exists() and not resume:
        raise SystemExit('Output folder already exists: move it or use --resume: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    hdr_path, mpath, ppath, apath = out / 'RUN_HEADER.json', out / 'REFERENCE_MEANS.npz', out / 'REFERENCE_E1_PROMPTS.json', out / 'AMPLITUDE.json'
    old = json.loads(hdr_path.read_text(encoding='utf-8')) if hdr_path.exists() else None
    if old is not None and old.get('provenance') != header.get('provenance'):
        raise SystemExit('RUN_HEADER provenance differs from this session: refuse to resume')
    if mpath.exists() and ppath.exists():
        with np.load(mpath, allow_pickle=False) as z:
            means = {k: z[k].copy() for k in z.files}
        prompts = json.loads(ppath.read_text(encoding='utf-8'))
        if old is None or old.get('reference_means_sha256') != means_sha(means) or old.get('reference_prompts_sha256') != hashlib.sha256(ppath.read_bytes()).hexdigest():
            raise SystemExit('REFERENCE_MEANS.npz or REFERENCE_E1_PROMPTS.json does not match RUN_HEADER: refuse to resume')
        log('[T346-CAL] reference means and prompts reused')
    else:
        t0 = time.perf_counter(); rec = MeanRecorder(model, needed_sites(arms)); pr = PromptRecorder(model)
        with rec, pr:
            for k in range(n_ref):
                inst = rt['test'].generate(REF_SEED + k); pr.meta = dict(inst.meta, seed=inst.seed); rt['test'].run(inst, rt['adapter'])
        means, counts = rec.means()
        uniq, seen = [], set()
        for p in pr.prompts:
            key = tuple(p['ids'])
            if key not in seen:
                seen.add(key); uniq.append(p)
        prompts = uniq[:N_E1_PROMPTS]
        np.savez(mpath, **means); ppath.write_text(json.dumps(prompts), encoding='utf-8')
        header = dict(header, reference_means_sha256=means_sha(means), reference_prompts_sha256=hashlib.sha256(ppath.read_bytes()).hexdigest(),
                      reference_token_counts=counts, reference_seconds=round(time.perf_counter() - t0, 2), reference_seeds=[REF_SEED + k for k in range(n_ref)],
                      n_reference_e1_prompts=len(prompts))
        hdr_path.write_text(json.dumps(header, indent=2, sort_keys=True), encoding='utf-8')
        log(f'[T346-CAL] reference means and {len(prompts)} E1 prompts from {n_ref} disjoint instances in {header["reference_seconds"]} s')
    if apath.exists():
        amp = json.loads(apath.read_text(encoding='utf-8'))
    else:
        t0 = time.perf_counter(); amp = {}
        dev = next(model.parameters()).device
        neutral = [(tokenize(rt['adapter'], model, t), None) for t in AMP_TEXTS]
        e1 = [(torch.tensor([p['ids']], dtype=torch.long, device=dev), p['meta']) for p in prompts]
        for arm, mode, sites in arms:
            if mode == 'NO_HOOK':
                amp[arm] = dict(neutral_texts=None, e1_prompts=None, note='reference arm')
                continue
            amp[arm] = dict(neutral_texts=amplitude_on(model, rt['adapter'], mode, sites, means, neutral),
                            e1_prompts=amplitude_on(model, rt['adapter'], mode, sites, means, e1) if e1 else None)
        amp['_seconds'] = round(time.perf_counter() - t0, 2)
        apath.write_text(json.dumps(amp, indent=2, sort_keys=True), encoding='utf-8')
        log(f'[T346-CAL] amplitudes of {len(arms)} arms in {amp["_seconds"]} s')
    return means, amp


def run_calibration(rt, out, header, arms, n_instances=N_INSTANCES, n_ref=N_REF, resume=False, stop_if_sham_differs=True, log=print):
    base = rt['base']; out = Path(out)
    means, amp = prepare(rt, out, header, arms, n_ref, resume, log)
    free_cuda(log, 'after reference and amplitude stages')
    rec_path = out / 'records.jsonl'
    adapter, test, stats = rt['adapter'], rt['test'], rt['stats']; tok = getattr(adapter, 'tokenizer', None)
    done = {(r['arm'], r['instance_id']) for r in base.read_records(rec_path)}
    total = len(arms) * n_instances; k = len(done); t_all = time.perf_counter()
    for i in range(n_instances):
        inst = test.generate(TASK_SEED + i)
        for arm, mode, sites in arms:
            if (arm, inst.id) in done:
                continue
            stats.reset(); t0 = time.perf_counter()
            with Intervention(adapter.model, mode, sites, means, tok, dict(inst.meta, seed=inst.seed)) as hk:
                recs = test.run(inst, adapter)
            res = test.score(inst, recs); summ = hk.summary()
            rec = dict(arm=arm, mode=mode, sites=[list(s) for s in sites], instance_id=inst.id, seed=inst.seed, result=res.__dict__,
                       records=[base.rec_to_dict(r) for r in recs], gen=stats.snapshot(), seconds=round(time.perf_counter() - t0, 2),
                       hook_calls=summ['calls'], hook_change_rms=summ['change_rms'], patch=summ.get('patch'), patch_pairs=summ.get('patch_pairs'))
            base.append_record(rec_path, rec); k += 1
            extra = f" patched={summ['patch']['applied']}/{summ['patch']['prompts']}" if summ.get('patch') else ''
            log(f'[T346-CAL] {k}/{total} {arm} {inst.id} score={res.score:.4f} valid={res.valid} gen_calls={rec["gen"]["generate_calls"]}{extra} {rec["seconds"]} s')
        free_cuda(log, 'after instance ' + inst.id)
        if stop_if_sham_differs:
            ident = sham_identity([r for r in base.read_records(rec_path) if r['instance_id'] == inst.id], base)
            if ident is not None and not ident['identical']:
                raise SystemExit('CAL_SHAM output differs from CAL_BASELINE on ' + inst.id + ': stop; nothing further is acquired')
    log(f'[T346-CAL] acquisition loop finished in {time.perf_counter() - t_all:.1f} s')
    return base.read_records(rec_path), amp


def sham_identity(records, base):
    by = {(r['arm'], r['instance_id']): r for r in records}
    pairs = [(by[k], by[('CAL_SHAM', k[1])]) for k in by if k[0] == 'CAL_BASELINE' and ('CAL_SHAM', k[1]) in by]
    if not pairs:
        return None
    n_diff = sum(1 for x, y in pairs if base.raws(x) != base.raws(y) or x['result']['score'] != y['result']['score'])
    return dict(n_instances_compared=len(pairs), n_instances_different=n_diff, identical=n_diff == 0)


def analyze(records, amp, base, reps=BOOTSTRAP_REPS):
    order = [a for a, _, _ in ALL_ARMS if any(r['arm'] == a for r in records)]
    by = {(r['arm'], r['instance_id']): r for r in records}
    res = dict(use='EXPLORATORY_TARGETED_CALIBRATION_u26_DESCRIPTIVE_NO_INFERENCE', task_seed_base=TASK_SEED, descriptive_band_FCA_E1=BAND_E1,
               co_change_measured=False, delta_on_modified_states='NOT_MEASURED_IN_THIS_SCRIPT',
               bootstrap=dict(unit='instance', reps=reps, seed=BOOTSTRAP_SEED, statistic='mean'), sham_vs_baseline=sham_identity(records, base), arms={})
    for a in order:
        rs = [r for r in records if r['arm'] == a]; ok = [r for r in rs if r['result']['valid']]
        sc = [r['result']['score'] for r in ok]
        det = {k: (float(np.mean([r['result']['details'][k] for r in ok if k in r['result']['details']])) if ok and any(k in r['result']['details'] for r in ok) else None)
               for k in DETAIL_KEYS}
        entry = dict(mode=rs[0]['mode'], sites=rs[0]['sites'], n_instances=len(rs), n_valid=len(ok), n_invalid=len(rs) - len(ok),
                     invalid_reasons=sorted({str(r['result']['details'].get('reason')) for r in rs if not r['result']['valid']}),
                     FCA_E1=float(np.mean(sc)) if sc else None, FCA_E1_ci95_descriptive=base.boot_ci(sc, reps) if sc else None, mean_details=det,
                     amplitude=amp.get(a), median_hook_change_rms=float(np.median([r['hook_change_rms'] or 0.0 for r in rs])),
                     total_instance_seconds=float(sum(r['seconds'] for r in rs)), median_instance_seconds=float(np.median([r['seconds'] for r in rs])),
                     total_generated_tokens=int(sum(r['gen']['generated_tokens'] for r in rs)))
        if rs[0].get('patch'):
            entry['patch_counts_total'] = {k: int(sum(r['patch'][k] for r in rs)) for k in rs[0]['patch']}
        if a != 'CAL_SHAM' and any(x == 'CAL_SHAM' for x, _ in by):
            ids = sorted(i for (x, i), r in by.items() if x == a and r['result']['valid'] and by.get(('CAL_SHAM', i), {}).get('result', {}).get('valid'))
            d = [by[(a, i)]['result']['score'] - by[('CAL_SHAM', i)]['result']['score'] for i in ids]
            if d:
                m = float(np.mean(d))
                entry['paired_vs_sham'] = dict(n_pairs=len(d), mean_diff=m, mean_diff_ci95_descriptive=base.boot_ci(d, reps), diff_by_instance=dict(zip(ids, d)),
                                               n_diff_positive=int(sum(x > 0 for x in d)), n_diff_negative=int(sum(x < 0 for x in d)),
                                               n_diff_beyond_band=int(sum(abs(x) > BAND_E1 for x in d)))
            entry['descriptive_reading'] = ('E1_INVALIDATED' if entry['n_invalid'] * 2 >= len(rs) else
                                            'MOVES_FCA_E1_BEYOND_BAND' if d and abs(float(np.mean(d))) > BAND_E1 else 'NO_DETECTABLE_CHANGE_OF_FCA_E1')
        res['arms'][a] = entry
    return res


def write_outputs(out, res):
    out = Path(out)
    with open(out / 'RESULT.json', 'x', encoding='utf-8') as f:
        json.dump(res, f, indent=2, sort_keys=True)
    with open(out / 'calibration_table.csv', 'x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'mode', 'n_valid', 'n_invalid', 'FCA_E1', 'mean_diff_vs_sham', 'n_pos', 'n_neg', 'e1_rel_change_L27', 'e1_last_kl_nats', 'neutral_rel_change_L27', 'reading'])
        for a, e in res['arms'].items():
            amp = e.get('amplitude') or {}; e1 = amp.get('e1_prompts') or {}; ne = amp.get('neutral_texts') or {}; p = e.get('paired_vs_sham') or {}
            w.writerow([a, e['mode'], e['n_valid'], e['n_invalid'], e['FCA_E1'], p.get('mean_diff'), p.get('n_diff_positive'), p.get('n_diff_negative'),
                        e1.get('rel_change_hidden_after_L27'), e1.get('last_position_kl_nats'), ne.get('rel_change_hidden_after_L27'), e.get('descriptive_reading')])


def build_runtime(a, work, base):
    fca_dir = base.find_fca_dir(work, a.fca_dir)
    fca, REG = base.import_fca(fca_dir)
    from fca_bench.adapters import adapter_from_config, load_config
    cfg_file = fca_dir / 'config.qwen3_8b_gpu8.json'
    cfg = load_config(str(cfg_file)) if cfg_file.is_file() else copy.deepcopy(base.HISTORICAL_R6_CONFIG)
    cfg['model_path'] = str(a.model_dir); cfg['progress'] = False; deviations = []
    if a.quantization:
        cfg['quantization'] = a.quantization; deviations.append(f'quantization={a.quantization} (historical bnb8)')
    if a.dtype:
        cfg['dtype'] = a.dtype; deviations.append(f'dtype={a.dtype} (historical float16 compute)')
    adapter = adapter_from_config(cfg)
    tree, files = base.fca_tree_sha(fca_dir)
    return dict(base=base, adapter=adapter, test=REG['E1'], stats=base.GenStats(adapter.model), config=cfg, deviations=deviations,
                provenance=dict(script=base.sha_file(Path(__file__)), base_script=base.sha_file(work / BASE_SCRIPT), fca_tree_sha256=tree, fca_files=files,
                                regime=dict(quantization=cfg.get('quantization'), dtype=cfg.get('dtype'))))


def select_arms(spec):
    want = list(DEFAULT_ARMS) if not spec else spec.split(',')
    for must in ('CAL_SHAM', 'CAL_BASELINE'):
        if must not in want:
            want.insert(0, must)
    bad = [w for w in want if w not in {x[0] for x in ALL_ARMS}]
    if bad:
        raise SystemExit('unknown arms: ' + ','.join(bad))
    return tuple(x for x in ALL_ARMS if x[0] in want)


def do_run(a):
    if a.authorize_gpu_run != CONFIRM:
        raise SystemExit('GPU run refused: pass --authorize-gpu-run ' + CONFIRM)
    base = load_module('t346_fca_e1_pilot', a.work / BASE_SCRIPT)
    rt = build_runtime(a, a.work, base); arms = select_arms(a.arms); n = a.instances or N_INSTANCES
    out = a.work / (OUTNAME + a.out_suffix)
    header = dict(provenance=rt['provenance'], config=rt['config'], deviations_from_historical_R6=rt['deviations'],
                  arms={x[0]: dict(mode=x[1], sites=[list(s) for s in x[2]]) for x in arms}, n_instances=n, task_seed_base=TASK_SEED,
                  hook_positions='ALL_PROCESSED_TOKENS (PATCH: prompt positions only)', amplitude_layer=AMP_LAYER, direction_seed=DIRECTION_SEED,
                  runtime=rt['adapter'].runtime_metadata())
    recs, amp = run_calibration(rt, out, header, arms, n, a.n_ref, a.resume, not a.continue_if_sham_differs)
    write_outputs(out, analyze(recs, amp, base, a.bootstrap_reps))
    print('[T346-CAL] done ;', out / 'RESULT.json')


def do_analyze(a):
    base = load_module('t346_fca_e1_pilot', a.work / BASE_SCRIPT)
    out = a.work / (OUTNAME + a.out_suffix)
    recs = base.read_records(out / 'records.jsonl')
    if not recs:
        raise SystemExit('No records in ' + str(out))
    amp = json.loads((out / 'AMPLITUDE.json').read_text(encoding='utf-8')) if (out / 'AMPLITUDE.json').is_file() else {}
    for p in ('RESULT.json', 'calibration_table.csv'):
        if (out / p).exists():
            (out / p).rename(out / (p + f'.previous_{int(time.time())}'))
    write_outputs(out, analyze(recs, amp, base, a.bootstrap_reps)); print('[T346-CAL] RESULT', out / 'RESULT.json')


class CharTok:
    """Invertible character tokenizer for the selftest (ASCII text, ids < 128)."""

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {'input_ids': [ord(c) for c in text]}

    def decode(self, ids, **_):
        return ''.join(chr(int(i)) for i in ids)


def make_fake_adapter(base, model, e1_samples=4, window=420):
    """The fake adapter of the pilot, with an invertible tokenizer and a wider context window so that the E1 context sentence reaches the model."""
    import torch
    fa = base.FakeAdapter(model, e1_samples=e1_samples); fa.tokenizer = CharTok()

    def new_session(system=None, instance=None):
        outer = fa

        class S:
            def __init__(self):
                self.msgs = [system or '']; self.n = 0

            def _ids(self):
                txt = '\n'.join(self.msgs)[-window:]
                return torch.tensor([[ord(c) % 128 for c in txt]], dtype=torch.long)

            def send(self, text, channel='work', expect_reply=True, sample=False):
                self.msgs.append(('[CONTROL] ' if channel == 'control' else '[WORK] ') + text)
                if not expect_reply:
                    return ''
                ids = self._ids(); torch.manual_seed(outer._stable_seed(instance, f'single|{self.n}|{text[:40]}'))
                with torch.inference_mode():
                    out = outer.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=5, do_sample=True, temperature=1.5, pad_token_id=0, use_cache=True)
                return outer._json(text, int(out[0, ids.shape[1]:].sum()), '\n'.join(self.msgs))

            def send_many(self, text, n, channel='work', sample=True, seed_key='batch'):
                self.msgs.append(('[CONTROL] ' if channel == 'control' else '[WORK] ') + text)
                ids = self._ids(); outs = []
                for start in range(0, n, outer.e1_batch_size):
                    b = min(outer.e1_batch_size, n - start); torch.manual_seed(outer._stable_seed(instance, f'{seed_key}|{start}|{b}'))
                    with torch.inference_mode():
                        g = outer.model.generate(ids.repeat(b, 1), attention_mask=torch.ones_like(ids.repeat(b, 1)), max_new_tokens=5, do_sample=True, temperature=1.5,
                                                 pad_token_id=0, use_cache=True)
                    outs += [outer._json(text, int(g[r, ids.shape[1]:].sum()), '\n'.join(self.msgs)) for r in range(b)]
                return outs
        return S()
    fa.new_session = new_session
    return fa


def selftest(a):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    t_start = time.perf_counter(); work = a.work
    base = load_module('t346_fca_e1_pilot', work / BASE_SCRIPT)
    fca_dir = base.find_fca_dir(work, a.fca_dir); fca, REG = base.import_fca(fca_dir)
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=1024, use_cache=True)
    model = Qwen3ForCausalLM(cfg).eval()
    adapter = make_fake_adapter(base, model)
    rt = dict(base=base, adapter=adapter, test=REG['E1'], stats=base.GenStats(model))
    quiet = lambda *_: None
    seen = {}

    def spy(name):
        def f(module, args):
            seen.setdefault(name, []).append(args[0].detach().clone())
        return f
    # 1. MEAN / ZERO / SHAM arithmetic on what down_proj and o_proj receive
    mu = {site_key(s): np.random.default_rng(1).normal(size=site_width(model, s)).astype(np.float32) for s in (('mlp', 24, 'all'), ('attn', 24, 'all'))}
    ids = torch.randint(0, 128, (2, 9))
    for mode, site, check in (('MEAN', ('mlp', 24, 'all'), 'mean_all'), ('MEAN', ('mlp', 24, 'block0'), 'mean_block'), ('ZERO', ('mlp', 24, 'block0'), 'zero_block'),
                              ('MEAN', ('attn', 24, 'all'), 'mean_attn'), ('SHAM', ('mlp', 24, 'all'), 'sham')):
        seen.clear()
        with torch.inference_mode():
            h0 = site_module(model, site).register_forward_pre_hook(spy('ref')); model(ids); h0.remove()
            with Intervention(model, mode, [site], mu):
                h1 = site_module(model, site).register_forward_pre_hook(spy('got')); model(ids); h1.remove()
        ref, got = seen['ref'][0], seen['got'][0]; lo, hi = site_span(model, site); m = torch.as_tensor(mu[site_key(site)])
        if check in ('mean_all', 'mean_attn'):
            assert torch.allclose(got, m.expand_as(got)), check
        elif check == 'mean_block':
            assert torch.allclose(got[..., lo:hi], m[lo:hi].expand_as(got[..., lo:hi])) and torch.equal(got[..., hi:], ref[..., hi:]) and hi == 32, check
        elif check == 'zero_block':
            assert torch.count_nonzero(got[..., lo:hi]) == 0 and torch.equal(got[..., hi:], ref[..., hi:]), check
        else:
            assert torch.equal(got, ref), check
    # 2. RANDOM_MATCHED: per-token norm of the added vector equals the norm of the change ZERO would produce, along one fixed direction
    dn = site_module(model, ('mlp', 24, 'block0')); outs = {}
    with torch.inference_mode():
        hh = dn.register_forward_hook(lambda m, a_, o: outs.__setitem__('base', o.detach().clone())); model(ids); hh.remove()
        with Intervention(model, 'ZERO', B0):
            hh = dn.register_forward_hook(lambda m, a_, o: outs.__setitem__('zero', o.detach().clone())); model(ids); hh.remove()
        with Intervention(model, 'RANDOM_MATCHED', B0):
            hh = dn.register_forward_hook(lambda m, a_, o: outs.__setitem__('rand', o.detach().clone())); model(ids); hh.remove()
    added = outs['rand'] - outs['base']; target = (outs['base'] - outs['zero']).norm(dim=-1)
    assert torch.allclose(added.norm(dim=-1), target, rtol=1e-4, atol=1e-6)
    dirs = added / added.norm(dim=-1, keepdim=True); assert torch.allclose(dirs, dirs[0, 0].expand_as(dirs), atol=1e-5)
    # 3. PATCH: prompt positions of the site replaced by the activations of the matched prompt; decoding steps and prompts without the sentence untouched
    meta = dict(obj='object-123', code='ATLAS', cat='PLANT', seed=5)
    text = 'system\n[WORK] ' + context_sentence(meta) + '\n[CONTROL] Revoke it.\n[WORK] What identifier is available?'
    tok = CharTok(); ids1 = torch.tensor([tok_encode(tok, text)])
    pt = Patcher(model, tok, B0[0]); pt.meta = meta
    assert pt.prepare(ids1[0]) and pt.counts['applied'] == 1
    donor_text = tok_decode(tok, tok_encode(tok, text.replace(context_sentence(meta), context_sentence(meta, pt.pairs[0]['code'], pt.pairs[0]['cat']))))
    seen.clear()
    with torch.inference_mode():
        h = dn.register_forward_pre_hook(spy('donor')); model(torch.tensor([tok_encode(tok, donor_text)])); h.remove()
        hp = dn.register_forward_pre_hook(pt.pre_hook); h = dn.register_forward_pre_hook(spy('patched'))
        model(ids1.repeat(2, 1)); model(ids1[:, :1]); h.remove(); hp.remove()
        h = dn.register_forward_pre_hook(spy('plain')); model(ids1.repeat(2, 1)); h.remove()
    d, p, q = seen['donor'][0], seen['patched'][0], seen['plain'][0]
    assert torch.allclose(p[:, :, :32], d[0, :, :32].expand(2, -1, -1)) and torch.equal(p[:, :, 32:], q[:, :, 32:]) and not torch.allclose(p[:, :, :32], q[:, :, :32])
    assert pt.counts['patched_forwards'] == 1, pt.counts
    pt2 = Patcher(model, tok, B0[0]); pt2.meta = meta
    assert not pt2.prepare(torch.tensor(tok_encode(tok, 'system\n[WORK] Neutral note.'))) and pt2.counts['skipped_no_context'] == 1
    # 4. the full loop on the fake adapter: default and optional arms
    w0 = {n: p_.detach().clone() for n, p_ in model.named_parameters()}
    arms = select_arms(','.join(x[0] for x in ALL_ARMS))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'run'
        recs, amp = run_calibration(rt, out, dict(provenance='x'), arms, 2, 1, log=quiet)
        assert all(torch.equal(w0[n], p_) for n, p_ in model.named_parameters()), 'weights changed'
        assert len(recs) == 2 * len(ALL_ARMS) and (out / 'REFERENCE_MEANS.npz').is_file() and (out / 'REFERENCE_E1_PROMPTS.json').is_file()
        hdr = json.loads((out / 'RUN_HEADER.json').read_text())
        assert hdr['reference_seeds'] == [REF_SEED] and hdr['n_reference_e1_prompts'] == N_E1_PROMPTS
        assert not set(hdr['reference_seeds']) & {TASK_SEED + i for i in range(N_INSTANCES)}
        ident = sham_identity(recs, base); assert ident['identical'] and ident['n_instances_compared'] == 2, ident
        assert amp['CAL_SHAM']['neutral_texts']['mean_kl_nats'] == 0.0 and amp['CAL_SHAM']['e1_prompts']['rel_change_hidden_after_L27'] == 0.0
        for arm in [x[0] for x in ALL_ARMS[2:]]:
            assert amp[arm]['e1_prompts']['rel_change_hidden_after_L27'] > 0, arm
        assert amp['MLP0_L24_PATCH_MATCHED']['neutral_texts']['rel_change_hidden_after_L27'] == 0.0 and amp['MLP0_L24_PATCH_MATCHED']['e1_prompts']['patch_applied_items'] > 0
        by = {(r['arm'], r['instance_id']): r for r in recs}
        assert all(by[('CAL_BASELINE', i)]['hook_calls'] == 0 and by[('CAL_SHAM', i)]['hook_calls'] > 0 for (x, i) in by if x == 'CAL_BASELINE')
        pr = [by[(x, i)]['patch'] for (x, i) in by if x == 'MLP0_L24_PATCH_MATCHED']
        assert all(p_['applied'] > 0 and p_['skipped_no_context'] > 0 and p_['patched_forwards'] == p_['applied'] for p_ in pr), pr
        assert any(base.raws(by[('MLP_L16_31_MEAN', i)]) != base.raws(by[('CAL_SHAM', i)]) for (x, i) in by if x == 'CAL_SHAM'), 'strongest arm must change outputs'
        res = analyze(recs, amp, base, 200); write_outputs(out, res)
        assert res['sham_vs_baseline']['identical'] and set(res['arms']) == {x[0] for x in ALL_ARMS} and res['co_change_measured'] is False
        assert res['arms']['MLP0_L24_PATCH_MATCHED']['patch_counts_total']['applied'] > 0
        # 5. resume reproduces the uninterrupted run; refusals
        full = base.read_records(out / 'records.jsonl')
        r_out = Path(tmp) / 'resume'; r_out.mkdir()
        for p_ in ('RUN_HEADER.json', 'REFERENCE_MEANS.npz', 'REFERENCE_E1_PROMPTS.json', 'AMPLITUDE.json'):
            (r_out / p_).write_bytes((out / p_).read_bytes())
        for r in full[:9]:
            base.append_record(r_out / 'records.jsonl', r)
        again, _ = run_calibration(rt, r_out, dict(provenance='x'), arms, 2, 1, resume=True, log=quiet)
        strip = lambda rs: [{k: v for k, v in r.items() if k not in ('seconds', 'gen')} for r in rs]
        assert strip(again) == strip(full), 'resume must reproduce the uninterrupted run'
        for bad, kw in (('existing folder', dict(out=out, header=dict(provenance='x'), resume=False)), ('other provenance', dict(out=r_out, header=dict(provenance='y'), resume=True))):
            try:
                run_calibration(rt, kw['out'], kw['header'], arms, 2, 1, resume=kw['resume'], log=quiet); raise AssertionError('accepted ' + bad)
            except SystemExit:
                pass
        (r_out / 'REFERENCE_E1_PROMPTS.json').write_text('[]')
        try:
            run_calibration(rt, r_out, dict(provenance='x'), arms, 2, 1, resume=True, log=quiet); raise AssertionError('accepted tampered prompts')
        except SystemExit as e:
            assert 'REFERENCE' in str(e)
        # 6. the loop stops when SHAM differs from BASELINE
        class Bad(Intervention):
            def __enter__(self):
                super().__enter__()
                if self.mode == 'SHAM':
                    self.dose = 0.5
                return self
        g = run_calibration.__globals__; orig = g['Intervention']; g['Intervention'] = Bad
        try:
            run_calibration(rt, Path(tmp) / 'bad', dict(provenance='x'), select_arms(None), 1, 1, log=quiet); raise AssertionError('did not stop on a non-identical SHAM')
        except SystemExit as e:
            assert 'CAL_SHAM output differs' in str(e), e
        finally:
            g['Intervention'] = orig
        assert [x[0] for x in select_arms(None)] == list(DEFAULT_ARMS) and [x[0] for x in select_arms('MLP_L24_MEAN')] == ['CAL_BASELINE', 'CAL_SHAM', 'MLP_L24_MEAN']
    print(f'[T346-CAL] selftest OK in {time.perf_counter() - t_start:.1f} s (CPU, tiny random model: MEAN/ZERO/SHAM arithmetic, RANDOM_MATCHED norm and direction, PATCH on prompt '
          'positions only and only with the context sentence, weights untouched, reference means and E1 prompts from disjoint instances, amplitudes on neutral texts and '
          'E1 prompts, SHAM identical to BASELINE, resume, refusals, stop on non-identical SHAM, analysis)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, default=WORK); ap.add_argument('--model-dir', type=Path, default=None); ap.add_argument('--fca-dir', type=Path, default=None)
    ap.add_argument('--selftest', action='store_true'); ap.add_argument('--run', action='store_true'); ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--authorize-gpu-run'); ap.add_argument('--arms'); ap.add_argument('--instances', type=int); ap.add_argument('--n-ref', type=int, default=N_REF)
    ap.add_argument('--quantization', default=None, help='override the historical bnb8 (e.g. none)'); ap.add_argument('--dtype', default=None, help='override the historical float16')
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--out-suffix', default=''); ap.add_argument('--continue-if-sham-differs', action='store_true')
    ap.add_argument('--bootstrap-reps', type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args()
    if a.model_dir is None:
        a.model_dir = a.work / 'Qwen3-8B'
    if a.selftest:
        selftest(a)
    elif a.run:
        do_run(a)
    elif a.analyze:
        do_analyze(a)
    else:
        ap.error('choose --selftest, --run or --analyze')


if __name__ == '__main__':
    main()
