#!/usr/bin/env python3
"""Activation-attenuation hook for the T346 pilot: block j of the MLP intermediate activation of one decoder layer (default L24, mlp0).

Site (as defined by candidate_registry in role_quotient.py): the input of down_proj in layers[layer].mlp, columns
[j*bs, (j+1)*bs) with bs = intermediate_size // n_blocks (Qwen3-8B: 12288 // 4 = 3072, so mlp0 = columns 0..3071). The hook is a
forward pre-hook on layers[layer].mlp.down_proj that multiplies those columns by (1 - dose). Weights are never edited.
Modes (the same code path runs in SHAM and ATTENUATE, so instrumentation cost is identical):
  NO_HOOK   no hook is registered (the unmodified model);
  SHAM      the hook runs with dose = 0: the multiplier is exactly 1.0, so the outputs are bit-identical to NO_HOOK;
  ATTENUATE dose in (0, 1]; dose = 1 sets the block to zero.
Positions (must be sealed before any acquisition; the module refuses an unknown value):
  ALL_PROCESSED_TOKENS        every token that goes through the layer in a forward pass (prefill and each decoding step);
  LAST_TOKEN_OF_EACH_FORWARD  only the last position of each forward pass (the 'active position' of the cube measurements).
While active, the hook records the RMS of the removed activation (dose x activation) and the RMS of the corresponding residual write
(W_down[:, block] @ removed activation) per call, to serve the 'intervention_rms' covariate; nothing is estimated beyond these.
Usage as a module: with BlockAttenuator(model, layer=24, block=0, dose=0.5, mode='ATTENUATE', positions='ALL_PROCESSED_TOKENS') as hk: ...
Self-test (CPU, tiny random Qwen3, no GPU, no Qwen weights):  python -I -u l24_mlp0_ablation_hook.py --selftest
"""
import argparse, hashlib, math
from pathlib import Path

import torch

MODES = ('NO_HOOK', 'SHAM', 'ATTENUATE')
POSITIONS = ('ALL_PROCESSED_TOKENS', 'LAST_TOKEN_OF_EACH_FORWARD')


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class BlockAttenuator:
    def __init__(self, model, layer=24, block=0, n_blocks=4, dose=0.0, mode='ATTENUATE', positions='ALL_PROCESSED_TOKENS', record=True):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if positions not in POSITIONS:
            raise ValueError(f'positions must be one of {POSITIONS}')
        if mode == 'ATTENUATE' and not (0.0 < dose <= 1.0):
            raise ValueError('ATTENUATE needs 0 < dose <= 1')
        if mode in ('NO_HOOK', 'SHAM') and dose != 0.0:
            raise ValueError(f'{mode} requires dose = 0')
        dec = model.model if hasattr(model, 'model') else model
        if not 0 <= layer < len(dec.layers):
            raise ValueError(f'layer {layer} outside 0..{len(dec.layers) - 1}')
        self.down = dec.layers[layer].mlp.down_proj
        dff = self.down.weight.shape[1]
        if dff % n_blocks or not 0 <= block < n_blocks:
            raise ValueError('intermediate size must be divisible by n_blocks and block must be in range')
        self.bs = dff // n_blocks; self.lo, self.hi = block * self.bs, (block + 1) * self.bs
        self.layer, self.block, self.dose, self.mode, self.positions, self.record = layer, block, float(dose), mode, positions, record
        self.handle = None; self._w = None; self.calls = 0; self.removed_ss = 0.0; self.removed_n = 0; self.write_ss = 0.0; self.write_n = 0

    def _pre_hook(self, module, args):
        x = args[0]
        scale = 1.0 - self.dose
        y = x.clone()
        sl = (slice(None), slice(-1, None), slice(self.lo, self.hi)) if self.positions == 'LAST_TOKEN_OF_EACH_FORWARD' else (slice(None), slice(None), slice(self.lo, self.hi))
        blk = y[sl]
        removed = blk * self.dose
        if self.record:
            r = removed.detach().float()
            self.removed_ss += float((r * r).sum()); self.removed_n += r.numel()
            if self._w is None or self._w.device != r.device:
                self._w = module.weight[:, self.lo:self.hi].detach().float()
            w = torch.nn.functional.linear(r, self._w)
            self.write_ss += float((w * w).sum()); self.write_n += w.numel()
        y[sl] = blk * scale
        self.calls += 1
        return (y,) + tuple(args[1:])

    def __enter__(self):
        if self.mode != 'NO_HOOK':
            self.handle = self.down.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        return False

    def summary(self):
        rms = lambda ss, n: math.sqrt(ss / n) if n else None
        return dict(mode=self.mode, layer=self.layer, block=self.block, columns=[self.lo, self.hi], dose=self.dose, positions=self.positions, calls=self.calls,
                    removed_activation_rms=rms(self.removed_ss, self.removed_n), removed_residual_write_rms=rms(self.write_ss, self.write_n),
                    hook_source_sha256=source_sha256())


def selftest():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=64, use_cache=True)
    for dtype in (torch.float32, torch.bfloat16):
        model = Qwen3ForCausalLM(cfg).to(dtype).eval()
        ids = torch.randint(0, 100, (2, 9))
        with torch.no_grad():
            base = model(ids).logits.clone()
            with BlockAttenuator(model, layer=2, block=0, mode='SHAM') as hk:
                sham = model(ids).logits.clone(); s = hk.summary()
            assert torch.equal(base, sham), f'SHAM not bit-identical to NO_HOOK ({dtype})'
            assert s['calls'] == 1 and s['removed_activation_rms'] == 0.0 and s['removed_residual_write_rms'] == 0.0
            with BlockAttenuator(model, layer=2, block=0, mode='NO_HOOK') as hk:
                nohook = model(ids).logits.clone(); assert hk.calls == 0 and hk.handle is None
            assert torch.equal(base, nohook)
            with BlockAttenuator(model, layer=2, block=0, dose=0.5, mode='ATTENUATE') as hk:
                half = model(ids).logits.clone(); sh = hk.summary()
            with BlockAttenuator(model, layer=2, block=0, dose=1.0, mode='ATTENUATE') as hk:
                full = model(ids).logits.clone(); sf = hk.summary()
            assert not torch.equal(base, half) and not torch.equal(half, full) and sf['removed_activation_rms'] > sh['removed_activation_rms'] > 0
            assert abs(sf['removed_activation_rms'] / sh['removed_activation_rms'] - 2.0) < 1e-6, 'removed RMS must scale with the dose'
            ref = Qwen3ForCausalLM(cfg).to(dtype).eval(); ref.load_state_dict(model.state_dict())
            ref.model.layers[2].mlp.down_proj.weight[:, 0:32] = 0
            assert torch.allclose(full.float(), ref(ids).logits.float(), atol=1e-5 if dtype == torch.float32 else 2e-2), 'dose 1 must equal zeroed columns'
            with BlockAttenuator(model, layer=2, block=1, dose=1.0, mode='ATTENUATE') as hk:
                blk1 = model(ids).logits.clone()
            assert not torch.equal(blk1, full), 'different blocks must act differently'
            with BlockAttenuator(model, layer=2, block=0, dose=1.0, mode='ATTENUATE', positions='LAST_TOKEN_OF_EACH_FORWARD') as hk:
                last = model(ids).logits.clone()
            assert not torch.equal(last[:, -1], base[:, -1]), 'LAST mode must change the last position'
            assert torch.equal(last[:, 0], base[:, 0]), 'LAST mode must leave the first position untouched (causal attention)'
            assert hk.handle is None
            g0 = model.generate(ids, max_new_tokens=6, do_sample=False)
            with BlockAttenuator(model, layer=2, block=0, mode='SHAM'):
                g1 = model.generate(ids, max_new_tokens=6, do_sample=False)
            assert torch.equal(g0, g1), 'SHAM must reproduce greedy generation exactly'
            with BlockAttenuator(model, layer=2, block=0, dose=1.0, mode='ATTENUATE') as hk:
                g2 = model.generate(ids, max_new_tokens=6, do_sample=False)
            assert hk.calls == 6, f'hook must run on prefill and each decoding step (calls={hk.calls})'
    for bad in (dict(mode='ATTENUATE', dose=0.0), dict(mode='SHAM', dose=0.3), dict(mode='X'), dict(positions='FIRST'), dict(layer=9), dict(block=4)):
        try:
            BlockAttenuator(model, **{**dict(layer=2, block=0, dose=0.5, mode='ATTENUATE'), **bad}); raise AssertionError(f'accepted {bad}')
        except ValueError:
            pass
    print('[l24-hook] selftest OK (SHAM bit-identical in fp32 and bf16, NO_HOOK registers nothing, dose 1 equals zeroed columns, RMS scales with dose, '
          'LAST mode, generation with cache, refusal of invalid settings)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    selftest()
    print('[l24-hook] source sha256', source_sha256())
