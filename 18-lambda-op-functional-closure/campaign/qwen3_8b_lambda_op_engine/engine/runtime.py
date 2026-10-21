#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_GROUPS = tuple(tuple(range(i, i + 3)) for i in range(0, 36, 3))
PRIMARY_GROUP = 7
PRIMARY_LAYERS = (21, 22, 23)
REFERENCE_GROUP = 10
REFERENCE_LAYERS = (30, 31, 32)
STRUCTURAL_ROLE = 8
SELECT_HANDLES = ("H1", "H2", "H3")
SELECT_PROBE_TEMPLATE = (
    "Report the currently active value associated with TARGET_HANDLE={handle}. "
    "Answer with only the integer, or NONE if that handle has no active binding."
)


def dtype_from_name(name: str):
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[key]


def load_model(model_dir: Path, device: str = "cuda", dtype: str = "bfloat16"):
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype=dtype_from_name(dtype),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def render_chat(tok, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode(tok, text: str) -> list[int]:
    return [int(x) for x in tok.encode(text, add_special_tokens=False)]


def split_append(tok, prefix_msgs, later_msgs, *, final_generation_prompt: bool = False):
    prefix = encode(tok, render_chat(tok, prefix_msgs, False))
    full = encode(tok, render_chat(tok, prefix_msgs + later_msgs, final_generation_prompt))
    n = 0
    for a, b in zip(prefix, full):
        if a != b:
            break
        n += 1
    if n != len(prefix):
        raise RuntimeError("prefix mismatch while appending messages")
    return prefix, full[len(prefix):]


def select_presentation(bindings: dict[str, int]) -> str:
    parts = [f"HANDLE={handle} ACTIVE=YES VALUE={bindings[handle]}" for handle in SELECT_HANDLES]
    return "Episode bindings. " + ". ".join(parts) + ". Retain all active bindings for this episode."


def select_instruction(revoked: Iterable[str]) -> str:
    revoked_set = set(revoked)
    status = ",".join(
        f"{handle}:{'REVOKE' if handle in revoked_set else 'KEEP'}" for handle in SELECT_HANDLES
    )
    return f"CONTROL_MAP={status}. Apply each control only to its named handle."


def prefix_ids(tok, bindings: dict[str, int]) -> list[int]:
    msg = select_presentation(bindings)
    return encode(tok, render_chat(tok, [{"role": "user", "content": msg}], False))


def donor_shift(bindings: dict[str, int], shift: int) -> dict[str, int]:
    return {h: 30 + ((int(v) - 30 + int(shift)) % 30) for h, v in bindings.items()}


def common_start(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    if n >= len(a):
        n = max(0, len(a) - 1)
    return n


def cache_seq_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    try:
        kv = cache[0]
        return int(kv[0].shape[-2])
    except Exception as exc:
        raise RuntimeError("Cannot determine cache sequence length") from exc


def layer_kv(cache, idx: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        for ka, va in (("keys", "values"), ("key_cache", "value_cache"), ("key_states", "value_states")):
            if hasattr(layer, ka) and hasattr(layer, va):
                k = getattr(layer, ka)
                v = getattr(layer, va)
                if torch.is_tensor(k) and torch.is_tensor(v):
                    return k, v
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        k = cache.key_cache[idx]
        v = cache.value_cache[idx]
        if torch.is_tensor(k) and torch.is_tensor(v):
            return k, v
    item = cache[idx]
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        if torch.is_tensor(item[0]) and torch.is_tensor(item[1]):
            return item[0], item[1]
    raise RuntimeError(f"Cannot access K/V tensors for cache layer {idx}")


def clone_cache(cache):
    cloned = copy.deepcopy(cache)
    k0, _ = layer_kv(cache, 0)
    kc, _ = layer_kv(cloned, 0)
    if k0.data_ptr() == kc.data_ptr():
        export_name = "to_" + "lega" + "cy_cache"
        import_name = "from_" + "lega" + "cy_cache"
        export_cache = getattr(cache, export_name, None)
        import_cache = getattr(type(cache), import_name, None)
        if callable(export_cache) and callable(import_cache):
            packed = export_cache()
            packed_clone = tuple((k.clone(), v.clone()) for k, v in packed)
            cloned = import_cache(packed_clone)
            kc, _ = layer_kv(cloned, 0)
        if k0.data_ptr() == kc.data_ptr():
            raise RuntimeError("Cache clone aliases original tensor storage")
    return cloned


def run_prefix(model, ids: list[int], device: str):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    with torch.inference_mode():
        out = model(
            input_ids=x,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    return out.past_key_values


def apply_scaled_suffix_intervention(
    receiver_cache,
    donor_cache,
    layer_indices: Iterable[int],
    start: int,
    alpha: float,
):
    if not (0.0 < alpha <= 1.0):
        raise ValueError(alpha)
    c = clone_cache(receiver_cache)
    for li in layer_indices:
        k, v = layer_kv(c, int(li))
        dk, dv = layer_kv(donor_cache, int(li))
        if dk.shape != k.shape or dv.shape != v.shape:
            raise RuntimeError("Donor/receiver cache shapes differ")
        if start < 0 or start >= k.shape[-2]:
            raise RuntimeError(f"Invalid suffix start={start} for layer {li}, shape={tuple(k.shape)}")
        ks0 = k[..., start:, :].clone()
        vs0 = v[..., start:, :].clone()
        k[..., start:, :].copy_(ks0 + (dk[..., start:, :] - ks0) * alpha)
        v[..., start:, :].copy_(vs0 + (dv[..., start:, :] - vs0) * alpha)
    return c


def build_native_prefix(model, tok, bindings: dict[str, int], device: str):
    ids = prefix_ids(tok, bindings)
    return run_prefix(model, ids, device), ids


def build_perturbed_prefix(
    model,
    tok,
    bindings: dict[str, int],
    device: str,
    donor_shift_value: int = 13,
    alpha: float = 0.82,
):
    donor = donor_shift(bindings, donor_shift_value)
    receiver_ids = prefix_ids(tok, bindings)
    donor_ids = prefix_ids(tok, donor)
    if len(receiver_ids) != len(donor_ids):
        raise RuntimeError("receiver/donor token length mismatch")
    receiver_cache = run_prefix(model, receiver_ids, device)
    donor_cache = run_prefix(model, donor_ids, device)
    start = common_start(receiver_ids, donor_ids)
    cache = apply_scaled_suffix_intervention(
        receiver_cache,
        donor_cache,
        LAYER_GROUPS[STRUCTURAL_ROLE],
        start,
        alpha,
    )
    return cache, receiver_ids


def advance_control(model, cache, suffix_ids: list[int], context_len: int, device: str):
    c = clone_cache(cache)
    x = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    total = context_len + len(suffix_ids)
    with torch.inference_mode():
        out = model(
            input_ids=x,
            attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
            past_key_values=c,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    return out.past_key_values


def control_cache(model, tok, prefix_cache, prefix_ids_: list[int], presentation: str, instruction: str, device: str):
    _, suffix = split_append(
        tok,
        [{"role": "user", "content": presentation}],
        [{"role": "user", "content": instruction}],
        final_generation_prompt=False,
    )
    return advance_control(model, prefix_cache, suffix, len(prefix_ids_), device), suffix


def patch_control_suffix(receiver_cache, donor_cache, layers: Iterable[int], start: int):
    c = clone_cache(receiver_cache)
    for li in layers:
        k, v = layer_kv(c, int(li))
        dk, dv = layer_kv(donor_cache, int(li))
        if k.shape != dk.shape or v.shape != dv.shape:
            raise RuntimeError(f"cache shape mismatch at layer {li}: {tuple(k.shape)} vs {tuple(dk.shape)}")
        if start < 0 or start >= k.shape[-2]:
            raise RuntimeError(f"invalid control suffix start={start} for layer {li}, shape={tuple(k.shape)}")
        k[..., start:, :].copy_(dk[..., start:, :])
        v[..., start:, :].copy_(dv[..., start:, :])
    return c


def advance_cache(model, cache, suffix_ids: list[int], context_len: int, device: str):
    c = clone_cache(cache)
    if not suffix_ids:
        raise RuntimeError("empty suffix")
    x = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    total = context_len + len(suffix_ids)
    with torch.inference_mode():
        out = model(
            input_ids=x,
            attention_mask=torch.ones((1, total), dtype=torch.long, device=device),
            past_key_values=c,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    return {
        "cache": out.past_key_values,
        "logits": out.logits[:, -1, :].detach(),
        "context_len": total,
    }


def answer_token_ids(tok, prompt_messages: list[dict[str, str]], response: str) -> list[int]:
    p = encode(tok, render_chat(tok, prompt_messages, True))
    f = encode(tok, render_chat(tok, prompt_messages + [{"role": "assistant", "content": str(response)}], False))
    if len(f) < len(p) or f[:len(p)] != p:
        ids = encode(tok, str(response))
        if not ids:
            raise RuntimeError(f"empty tokenization for response={response!r}")
        return ids
    suffix = f[len(p):]
    specials = {int(x) for x in getattr(tok, "all_special_ids", []) if isinstance(x, int)}
    content = []
    for tid in suffix:
        if int(tid) in specials:
            break
        content.append(int(tid))
    raw = encode(tok, str(response))
    if raw and len(content) >= len(raw) and content[:len(raw)] == raw:
        content = raw
    if not content:
        content = raw
    if not content:
        raise RuntimeError(f"could not isolate assistant tokens for response={response!r}")
    return content


def sequence_logprob(model, prompt_state: dict[str, Any], token_ids: list[int], device: str) -> float:
    if not token_ids:
        raise ValueError("token_ids cannot be empty")
    logits = prompt_state["logits"]
    cache = prompt_state["cache"]
    context_len = int(prompt_state["context_len"])
    total = 0.0
    for i, tid in enumerate(token_ids):
        lp = torch.log_softmax(logits.float().reshape(-1), dim=-1)
        total += float(lp[int(tid)].item())
        if i + 1 < len(token_ids):
            nxt = advance_cache(model, cache, [int(tid)], context_len, device)
            logits = nxt["logits"]
            cache = nxt["cache"]
            context_len = int(nxt["context_len"])
    return float(total)


def probe_margin(model, tok, control_cache_, presentation: str, instruction: str, handle: str, value: int, device: str):
    probe = SELECT_PROBE_TEMPLATE.format(handle=handle)
    prefix_msgs = [
        {"role": "user", "content": presentation},
        {"role": "user", "content": instruction},
    ]
    _, probe_suffix = split_append(
        tok,
        prefix_msgs,
        [{"role": "user", "content": probe}],
        final_generation_prompt=True,
    )
    prompt_state = advance_cache(
        model,
        control_cache_,
        probe_suffix,
        cache_seq_length(control_cache_),
        device,
    )
    full_prompt_msgs = prefix_msgs + [{"role": "user", "content": probe}]
    none_ids = answer_token_ids(tok, full_prompt_msgs, "NONE")
    value_ids = answer_token_ids(tok, full_prompt_msgs, str(int(value)))
    lp_none = sequence_logprob(model, prompt_state, none_ids, device)
    lp_value = sequence_logprob(model, prompt_state, value_ids, device)
    return float(lp_none - lp_value)
