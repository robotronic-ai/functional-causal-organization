#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.expanduser("~/.cache/quantum_inductor"),
)

import numpy as np
import torch


DEFAULT_PROMPT_PAIRS = [
    ("red", "blue"),
    ("cat", "dog"),
    ("north", "south"),
    ("summer", "winter"),
    ("yes", "no"),
    ("three", "seven"),
    ("circle", "square"),
    ("coffee", "tea"),
    ("morning", "evening"),
    ("left", "right"),
    ("open", "closed"),
    ("large", "small"),
]


@dataclass
class CacheSnapshot:
    length: int
    k: list[torch.Tensor]
    v: list[torch.Tensor]


@dataclass
class ContextSpec:
    context_id: int
    word_a: str
    word_b: str
    prompt_a: str
    prompt_b: str
    token_length: int
    probe_token_id: int


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def patch_streaming_cache() -> None:
    try:
        import streaming_kv_cache as kv_module
    except ImportError:
        return

    original = kv_module.StreamingKVCache

    class SmallStreamingKVCache(original):
        def __init__(
            self,
            window_size,
            chunk_size,
            quant_mode="bf16",
            offload_dir=None,
            **kwargs,
        ):
            super().__init__(
                window_size=8192,
                chunk_size=4096,
                quant_mode=quant_mode,
                offload_dir=offload_dir,
                **kwargs,
            )

    kv_module.StreamingKVCache = SmallStreamingKVCache


def load_model(model_dir: Path, root: Path, device: str):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    patch_streaming_cache()

    from quantum_model import QuantumModelForCausalLM
    from quantum_config import QuantumConfig
    from quantum_tokenizer import QuantumTokenizer

    tokenizer = QuantumTokenizer.from_pretrained(
        str(model_dir),
        use_fast=False,
    )
    tokenizer.bos_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.additional_eos_ids = [11, 17]
    tokenizer.pad_token_id = 2
    tokenizer.unk_token_id = 3

    config = QuantumConfig.from_pretrained(str(model_dir))

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = QuantumModelForCausalLM(config)
    torch.set_default_dtype(old_dtype)

    QuantumModelForCausalLM.load_quantum_weights(
        model,
        str(model_dir),
        device=device,
    )

    if hasattr(model, "virtual_mask") and model.virtual_mask is not None:
        with torch.no_grad():
            model.virtual_mask.fill_(False)

    model.eval()
    cleanup_cuda()
    return model, tokenizer, config


def reset_transient_state(model) -> None:
    if hasattr(model, "reset_fusion_buffer"):
        try:
            model.reset_fusion_buffer()
        except Exception:
            pass

    for layer in getattr(model, "layers", []):
        for method_name in (
            "reset_for_new_sequence",
            "reset_routing_state",
            "reset_state",
        ):
            method = getattr(layer, method_name, None)
            if callable(method):
                try:
                    method()
                except TypeError:
                    pass
                except Exception:
                    pass

        if hasattr(layer, "_cached_skip_decision"):
            layer._cached_skip_decision = False
        if hasattr(layer, "_cached_sign_skip"):
            layer._cached_sign_skip = False
        if hasattr(layer, "_cached_head_mask"):
            layer._cached_head_mask = None
        if hasattr(layer, "_ar_token_counter"):
            layer._ar_token_counter = 0


def encode(tokenizer, text: str, device: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, torch.Tensor):
        tensor = ids.detach().clone().long()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.tensor([ids], dtype=torch.long)
    return tensor.to(device)


def build_prompt(word: str) -> str:
    return (
        "<|im_start|>user\n"
        f"Keep the keyword {word} in mind and wait."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim >= 3:
                return item

    raise RuntimeError("Could not locate logits in model output.")


def prefill(
    model,
    input_ids: torch.Tensor,
    flow: str,
) -> tuple[Any, int]:
    model.kv_cache.reset()
    reset_transient_state(model)

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            current_flow=flow,
        )

    logits = extract_logits(output)
    probe_token_id = int(
        torch.argmax(logits[:, -1, :], dim=-1).item()
    )
    return output, probe_token_id


def snapshot_cache(cache, n_layers: int) -> CacheSnapshot:
    lengths = [
        int(cache.get_seq_length(layer_idx))
        for layer_idx in range(n_layers)
    ]

    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"Layer cache lengths differ: {lengths}"
        )

    length = lengths[0]
    if length <= 0:
        raise RuntimeError("Cannot snapshot an empty cache.")

    k_states = []
    v_states = []

    for layer_idx in range(n_layers):
        if layer_idx not in cache._buf_k:
            raise RuntimeError(
                f"Layer {layer_idx} is missing from the cache."
            )
        k_states.append(
            cache._buf_k[layer_idx][
                :, :, :length, :
            ].detach().clone()
        )
        v_states.append(
            cache._buf_v[layer_idx][
                :, :, :length, :
            ].detach().clone()
        )

    return CacheSnapshot(
        length=length,
        k=k_states,
        v=v_states,
    )


def restore_cache(cache, snapshot: CacheSnapshot) -> None:
    cache.reset()
    length = snapshot.length

    for layer_idx, (k_source, v_source) in enumerate(
        zip(snapshot.k, snapshot.v)
    ):
        k = k_source.to(cache.device)
        v = v_source.to(cache.device)

        batch, heads, _, head_dim = k.shape
        if batch != 1:
            raise RuntimeError("Only batch size 1 is supported.")

        capacity = max(256, length + 8)

        buf_k = k.new_empty(
            batch,
            heads,
            capacity,
            head_dim,
        )
        buf_v = v.new_empty(
            batch,
            heads,
            capacity,
            head_dim,
        )
        buf_k[:, :, :length, :] = k
        buf_v[:, :, :length, :] = v

        cache._buf_k[layer_idx] = buf_k
        cache._buf_v[layer_idx] = buf_v
        cache._base[layer_idx] = 0
        cache._total[layer_idx] = length
        cache._pages[layer_idx] = []
        cache._lru[layer_idx] = {}
        cache._small_mode[layer_idx] = True

        cache._k_min[layer_idx] = torch.empty(
            0,
            heads,
            head_dim,
            dtype=cache._meta_dt,
            device=k.device,
        )
        cache._k_max[layer_idx] = torch.empty(
            0,
            heads,
            head_dim,
            dtype=cache._meta_dt,
            device=k.device,
        )

        cache.last_positions[layer_idx] = torch.arange(
            length,
            device=k.device,
        )

    cache.seen_tokens = length
    cache._step = 0


def patch_last_kv(
    cache,
    donor: CacheSnapshot,
    physical_layers: list[int],
) -> None:
    for layer_idx in physical_layers:
        target_length = int(cache._total[layer_idx])
        if target_length != donor.length:
            raise RuntimeError(
                "Donor and recipient cache lengths do not match."
            )

        position = target_length - 1
        cache._buf_k[layer_idx][
            :, :, position:position + 1, :
        ].copy_(
            donor.k[layer_idx][
                :, :, -1:, :
            ].to(cache.device)
        )
        cache._buf_v[layer_idx][
            :, :, position:position + 1, :
        ].copy_(
            donor.v[layer_idx][
                :, :, -1:, :
            ].to(cache.device)
        )


def decode_one(
    model,
    token_id: int,
    flow: str,
    device: str,
) -> Any:
    token = torch.tensor(
        [[token_id]],
        dtype=torch.long,
        device=device,
    )

    reset_transient_state(model)

    with torch.inference_mode():
        return model(
            input_ids=token,
            past_key_values=model.kv_cache,
            use_cache=True,
            current_flow=flow,
        )


def response_vectors(
    cache,
    expected_length: int,
    n_layers: int,
) -> list[torch.Tensor]:
    vectors = []

    for layer_idx in range(n_layers):
        length = int(cache.get_seq_length(layer_idx))
        if length != expected_length:
            raise RuntimeError(
                f"Layer {layer_idx}: expected cache length "
                f"{expected_length}, got {length}."
            )

        k = cache._buf_k[layer_idx][
            :, :, length - 1:length, :
        ].detach().float().cpu().reshape(-1)
        v = cache._buf_v[layer_idx][
            :, :, length - 1:length, :
        ].detach().float().cpu().reshape(-1)

        vectors.append(torch.cat([k, v], dim=0))

    return vectors


def rms(vector: torch.Tensor) -> float:
    return float(
        torch.sqrt(torch.mean(vector.float() ** 2)).item()
    )


def normalized_distance(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float:
    numerator = rms(left - right)
    denominator = (
        0.5 * (rms(left) + rms(right)) + 1e-12
    )
    return float(numerator / denominator)


def classify_response(
    value: torch.Tensor,
    pole_zero: torch.Tensor,
    pole_one: torch.Tensor,
) -> tuple[int, float, float, float]:
    d0 = float(
        torch.mean((value - pole_zero) ** 2).item()
    )
    d1 = float(
        torch.mean((value - pole_one) ** 2).item()
    )
    bit = int(d1 < d0)
    margin = abs(d0 - d1) / (d0 + d1 + 1e-20)
    return bit, float(margin), d0, d1


def candidate_contexts(
    tokenizer,
    device: str,
    max_contexts: int,
    max_tokens: int,
) -> list[tuple[str, str, torch.Tensor, torch.Tensor]]:
    selected = []

    for word_a, word_b in DEFAULT_PROMPT_PAIRS:
        prompt_a = build_prompt(word_a)
        prompt_b = build_prompt(word_b)

        ids_a = encode(tokenizer, prompt_a, device)
        ids_b = encode(tokenizer, prompt_b, device)

        if ids_a.shape[1] != ids_b.shape[1]:
            continue
        if ids_a.shape[1] > max_tokens:
            continue
        if torch.equal(ids_a, ids_b):
            continue

        selected.append(
            (word_a, word_b, ids_a, ids_b)
        )

        if len(selected) >= max_contexts:
            break

    if len(selected) < 2:
        raise RuntimeError(
            "Fewer than two equal-length prompt pairs were found."
        )

    return selected


def prepare_context(
    model,
    tokenizer,
    word_a: str,
    word_b: str,
    ids_a: torch.Tensor,
    ids_b: torch.Tensor,
    context_id: int,
    flow: str,
    n_layers: int,
    device: str,
) -> dict:
    prefill(model, ids_a, flow)
    snapshot_a = snapshot_cache(model.kv_cache, n_layers)
    output_a = model(
        input_ids=torch.tensor(
            [[0]],
            dtype=torch.long,
            device=device,
        ),
        past_key_values=None,
        use_cache=False,
        current_flow="s_attn",
    ) if False else None

    # Re-run A only to obtain the clean next-token probe ID from logits.
    model.kv_cache.reset()
    reset_transient_state(model)
    with torch.inference_mode():
        prefill_a = model(
            input_ids=ids_a,
            use_cache=True,
            current_flow=flow,
        )
    probe_token_id = int(
        torch.argmax(
            extract_logits(prefill_a)[:, -1, :],
            dim=-1,
        ).item()
    )
    snapshot_a = snapshot_cache(model.kv_cache, n_layers)

    prefill(model, ids_b, flow)
    snapshot_b = snapshot_cache(model.kv_cache, n_layers)

    if snapshot_a.length != snapshot_b.length:
        raise RuntimeError("Prompt pair cache lengths differ.")

    return {
        "context_id": context_id,
        "word_a": word_a,
        "word_b": word_b,
        "prompt_a": build_prompt(word_a),
        "prompt_b": build_prompt(word_b),
        "token_length": snapshot_a.length,
        "probe_token_id": probe_token_id,
        "snapshot_a": snapshot_a,
        "snapshot_b": snapshot_b,
    }


def run_assignment(
    model,
    context: dict,
    assignment: np.ndarray,
    physical_roles: list[list[int]],
    flow: str,
    n_layers: int,
    device: str,
) -> list[torch.Tensor]:
    restore_cache(model.kv_cache, context["snapshot_a"])

    for role_index, bit in enumerate(assignment.tolist()):
        if int(bit) == 1:
            patch_last_kv(
                model.kv_cache,
                context["snapshot_b"],
                physical_roles[role_index],
            )

    before_lengths = [
        model.kv_cache.get_seq_length(i)
        for i in range(n_layers)
    ]

    decode_one(
        model,
        context["probe_token_id"],
        flow,
        device,
    )

    after_lengths = [
        model.kv_cache.get_seq_length(i)
        for i in range(n_layers)
    ]

    expected_before = context["token_length"]
    expected_after = expected_before + 1

    if any(length != expected_before for length in before_lengths):
        raise RuntimeError(
            f"Cache transport error before decode: {before_lengths}"
        )
    if any(length != expected_after for length in after_lengths):
        raise RuntimeError(
            f"Cache transport error after decode: {after_lengths}"
        )

    return response_vectors(
        model.kv_cache,
        expected_after,
        n_layers,
    )


def build_pilot(
    model,
    tokenizer,
    flow: str,
    n_contexts: int,
    n_random_trials: int,
    seed: int,
    max_prompt_tokens: int,
    device: str,
) -> dict:
    n_layers = len(model.layers)
    raw_contexts = candidate_contexts(
        tokenizer,
        device,
        n_contexts,
        max_prompt_tokens,
    )

    contexts = [
        prepare_context(
            model,
            tokenizer,
            word_a,
            word_b,
            ids_a,
            ids_b,
            context_id,
            flow,
            n_layers,
            device,
        )
        for context_id, (
            word_a,
            word_b,
            ids_a,
            ids_b,
        ) in enumerate(raw_contexts)
    ]

    physical_roles = [[i] for i in range(n_layers)]
    rng = np.random.default_rng(seed)

    context_results = []
    source_effects = np.zeros(
        (len(contexts), n_layers, n_layers),
        dtype=np.float64,
    )
    separations = np.zeros(
        (len(contexts), n_layers),
        dtype=np.float64,
    )
    repeat_noise = np.zeros(
        (len(contexts), n_layers),
        dtype=np.float64,
    )
    random_margins = []

    all_zero = np.zeros(n_layers, dtype=np.int8)
    all_one = np.ones(n_layers, dtype=np.int8)

    for context in contexts:
        context_id = context["context_id"]

        ref_zero = run_assignment(
            model,
            context,
            all_zero,
            physical_roles,
            flow,
            n_layers,
            device,
        )
        ref_zero_repeat = run_assignment(
            model,
            context,
            all_zero,
            physical_roles,
            flow,
            n_layers,
            device,
        )
        ref_one = run_assignment(
            model,
            context,
            all_one,
            physical_roles,
            flow,
            n_layers,
            device,
        )

        for target in range(n_layers):
            separations[context_id, target] = normalized_distance(
                ref_zero[target],
                ref_one[target],
            )
            repeat_noise[context_id, target] = normalized_distance(
                ref_zero[target],
                ref_zero_repeat[target],
            )

        for source in range(n_layers):
            assignment = all_zero.copy()
            assignment[source] = 1

            response = run_assignment(
                model,
                context,
                assignment,
                physical_roles,
                flow,
                n_layers,
                device,
            )

            for target in range(n_layers):
                source_effects[
                    context_id,
                    source,
                    target,
                ] = normalized_distance(
                    response[target],
                    ref_zero[target],
                )

        context_trial_margins = []
        for _ in range(n_random_trials):
            assignment = rng.integers(
                0,
                2,
                size=n_layers,
                dtype=np.int8,
            )
            response = run_assignment(
                model,
                context,
                assignment,
                physical_roles,
                flow,
                n_layers,
                device,
            )

            margins = []
            for target in range(n_layers):
                _, margin, _, _ = classify_response(
                    response[target],
                    ref_zero[target],
                    ref_one[target],
                )
                margins.append(margin)

            context_trial_margins.append(margins)
            random_margins.extend(margins)

        context_results.append(
            {
                "context_id": context_id,
                "word_a": context["word_a"],
                "word_b": context["word_b"],
                "token_length": context["token_length"],
                "probe_token_id": context["probe_token_id"],
                "minimum_pole_separation": float(
                    separations[context_id].min()
                ),
                "maximum_repeat_noise": float(
                    repeat_noise[context_id].max()
                ),
                "median_random_classification_margin": float(
                    np.median(context_trial_margins)
                ),
            }
        )

    noise_floor = max(
        1e-7,
        10.0 * float(np.quantile(repeat_noise, 0.99)),
    )
    identification_threshold = max(
        1e-5,
        noise_floor,
    )

    mean_separation = separations.mean(axis=0)
    mean_source_effect = source_effects.mean(axis=0)
    max_source_effect = mean_source_effect.max(axis=1)

    identifiable = [
        i
        for i in range(n_layers)
        if mean_separation[i] > identification_threshold
    ]
    essential = [
        i
        for i in range(n_layers)
        if max_source_effect[i] > identification_threshold
    ]
    retained = sorted(set(identifiable) & set(essential))

    passed = (
        len(retained) >= 4
        and float(np.max(repeat_noise)) <= max(
            1e-4,
            identification_threshold,
        )
    )

    return {
        "experiment": "quantum_gamma_kv_last_token_probe_v1",
        "flow": flow,
        "physical_layer_count": n_layers,
        "context_count": len(contexts),
        "random_trials_per_context": n_random_trials,
        "intervention": (
            "Binary interchange of the final persistent K/V position "
            "between equal-length donor contexts, independently by layer."
        ),
        "response": (
            "Newly written K/V position one decode cycle later, "
            "classified against frozen all-zero and all-one reference poles."
        ),
        "scope": (
            "Finite-battery probe quotient over layer-local persistent K/V "
            "roles. This is not asserted to be the universal full-state "
            "causal quotient."
        ),
        "noise_floor": float(noise_floor),
        "identification_threshold": float(
            identification_threshold
        ),
        "retained_physical_layers": retained,
        "identifiable_response_layers": identifiable,
        "causally_essential_source_layers": essential,
        "mean_pole_separation_by_layer": mean_separation.tolist(),
        "mean_max_source_effect_by_layer": max_source_effect.tolist(),
        "mean_single_flip_effect_matrix": mean_source_effect.tolist(),
        "repeat_noise_by_context_layer": repeat_noise.tolist(),
        "classification_margin_summary": {
            "minimum": float(np.min(random_margins)),
            "median": float(np.median(random_margins)),
            "p05": float(np.quantile(random_margins, 0.05)),
        },
        "contexts": context_results,
        "quotient_gate": {
            "passed": bool(passed),
            "method": (
                "finite_battery_kv_last_position_interchange_v1"
            ),
            "scope": (
                "probe_quotient_not_full_state_quotient"
            ),
            "holdout_factorization_violations": None,
            "retained_role_count": len(retained),
            "physical_layer_count": n_layers,
            "determinism_noise_floor": float(noise_floor),
            "identification_threshold": float(
                identification_threshold
            ),
        },
    }


def prepare_collection_contexts(
    model,
    tokenizer,
    pilot: dict,
    flow: str,
    max_prompt_tokens: int,
    device: str,
) -> list[dict]:
    pairs = [
        (
            item["word_a"],
            item["word_b"],
        )
        for item in pilot["contexts"]
    ]

    raw = []
    for word_a, word_b in pairs:
        ids_a = encode(
            tokenizer,
            build_prompt(word_a),
            device,
        )
        ids_b = encode(
            tokenizer,
            build_prompt(word_b),
            device,
        )
        if (
            ids_a.shape[1] != ids_b.shape[1]
            or ids_a.shape[1] > max_prompt_tokens
        ):
            raise RuntimeError(
                f"Context {word_a}/{word_b} changed tokenization."
            )
        raw.append((word_a, word_b, ids_a, ids_b))

    return [
        prepare_context(
            model,
            tokenizer,
            word_a,
            word_b,
            ids_a,
            ids_b,
            context_id,
            flow,
            len(model.layers),
            device,
        )
        for context_id, (
            word_a,
            word_b,
            ids_a,
            ids_b,
        ) in enumerate(raw)
    ]


def collect_trials(
    model,
    tokenizer,
    pilot: dict,
    flow: str,
    trials_per_split: int,
    seed: int,
    max_prompt_tokens: int,
    device: str,
) -> tuple[dict, dict]:
    retained = [
        int(value)
        for value in pilot["retained_physical_layers"]
    ]
    if len(retained) < 4:
        raise RuntimeError(
            "Pilot retained fewer than four causal roles."
        )

    n_layers = len(model.layers)
    physical_roles = [[layer] for layer in retained]
    n_roles = len(physical_roles)

    contexts = prepare_collection_contexts(
        model,
        tokenizer,
        pilot,
        flow,
        max_prompt_tokens,
        device,
    )

    references = {}
    all_zero = np.zeros(n_roles, dtype=np.int8)
    all_one = np.ones(n_roles, dtype=np.int8)

    for context in contexts:
        ref_zero_all = run_assignment(
            model,
            context,
            all_zero,
            physical_roles,
            flow,
            n_layers,
            device,
        )
        ref_one_all = run_assignment(
            model,
            context,
            all_one,
            physical_roles,
            flow,
            n_layers,
            device,
        )

        references[context["context_id"]] = {
            "zero": [
                ref_zero_all[layer]
                for layer in retained
            ],
            "one": [
                ref_one_all[layer]
                for layer in retained
            ],
        }

    all_x = []
    all_y = []
    all_context = []
    all_split = []
    margin_records = []

    split_seeds = [
        seed + 1001,
        seed + 2001,
        seed + 3001,
        seed + 4001,
    ]

    for split_id, split_seed in enumerate(split_seeds):
        rng = np.random.default_rng(split_seed)

        for trial_index in range(trials_per_split):
            context = contexts[
                trial_index % len(contexts)
            ]
            context_id = context["context_id"]
            assignment = rng.integers(
                0,
                2,
                size=n_roles,
                dtype=np.int8,
            )

            response_all = run_assignment(
                model,
                context,
                assignment,
                physical_roles,
                flow,
                n_layers,
                device,
            )

            y = np.zeros(n_roles, dtype=np.int8)
            margins = []

            for role_index, physical_layer in enumerate(retained):
                bit, margin, _, _ = classify_response(
                    response_all[physical_layer],
                    references[context_id]["zero"][role_index],
                    references[context_id]["one"][role_index],
                )
                y[role_index] = bit
                margins.append(margin)

            all_x.append(assignment)
            all_y.append(y)
            all_context.append(context_id)
            all_split.append(split_id)
            margin_records.extend(margins)

    arrays = {
        "x": np.asarray(all_x, dtype=np.int8),
        "y": np.asarray(all_y, dtype=np.int8),
        "context": np.asarray(
            all_context,
            dtype=np.int64,
        ),
        "split": np.asarray(
            all_split,
            dtype=np.int8,
        ),
    }

    metadata = {
        "experiment": "quantum_gamma_kv_last_token_probe_v1",
        "flow": flow,
        "physical_layers": retained,
        "role_count": n_roles,
        "context_count": len(contexts),
        "trials_per_split": trials_per_split,
        "total_trials": int(len(all_x)),
        "split_seeds": split_seeds,
        "classification_margin_summary": {
            "minimum": float(np.min(margin_records)),
            "median": float(np.median(margin_records)),
            "p05": float(np.quantile(margin_records, 0.05)),
        },
        "scope": (
            "Gamma probe on the finite binary K/V-last-position "
            "causal quotient."
        ),
    }

    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real Quantum recurrent-memory Gamma K/V "
            "intervention experiment."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("pilot", "collect", "all"),
        default="pilot",
    )
    parser.add_argument(
        "--root",
        default="/mnt/c/Users/Merien/pycharm/llama/quantum",
    )
    parser.add_argument(
        "--model-dir",
        default=(
            "/mnt/c/Users/Merien/pycharm/llama/quantum/"
            "quantum_vibe_thinker"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--flow",
        choices=("s_attn", "fused"),
        default="fused",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=28017,
    )
    parser.add_argument(
        "--contexts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pilot-random-trials",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--trials-per-split",
        type=int,
        default=2400,
    )
    parser.add_argument(
        "--pilot-output",
        default="quantum_gamma_kv_pilot_v1.json",
    )
    parser.add_argument(
        "--trials-output",
        default="target_trials.npz",
    )
    parser.add_argument(
        "--gate-output",
        default="quotient_gate.json",
    )
    parser.add_argument(
        "--metadata-output",
        default="target_trials_metadata.json",
    )
    args = parser.parse_args()

    set_determinism(args.seed)

    root = Path(args.root).resolve()
    model_dir = Path(args.model_dir).resolve()

    model, tokenizer, config = load_model(
        model_dir,
        root,
        args.device,
    )

    print(
        json.dumps(
            {
                "model_class": model.__class__.__name__,
                "layer_count": len(model.layers),
                "kv_cache_class": model.kv_cache.__class__.__name__,
                "flow": args.flow,
                "use_adaptive_routing": bool(
                    getattr(
                        config,
                        "use_adaptive_routing",
                        False,
                    )
                ),
            },
            indent=2,
        )
    )

    pilot = None

    if args.mode in ("pilot", "all"):
        start = time.time()
        pilot = build_pilot(
            model=model,
            tokenizer=tokenizer,
            flow=args.flow,
            n_contexts=args.contexts,
            n_random_trials=args.pilot_random_trials,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            device=args.device,
        )

        Path(args.pilot_output).write_text(
            json.dumps(pilot, indent=2)
        )
        Path(args.gate_output).write_text(
            json.dumps(
                pilot["quotient_gate"],
                indent=2,
            )
        )

        print(
            json.dumps(
                {
                    "pilot_output": args.pilot_output,
                    "quotient_gate_output": args.gate_output,
                    "quotient_gate_passed": pilot[
                        "quotient_gate"
                    ]["passed"],
                    "retained_role_count": len(
                        pilot["retained_physical_layers"]
                    ),
                    "retained_physical_layers": pilot[
                        "retained_physical_layers"
                    ],
                    "noise_floor": pilot["noise_floor"],
                    "identification_threshold": pilot[
                        "identification_threshold"
                    ],
                    "classification_margin_summary": pilot[
                        "classification_margin_summary"
                    ],
                    "elapsed_seconds": time.time() - start,
                },
                indent=2,
            )
        )

        if (
            args.mode == "all"
            and not pilot["quotient_gate"]["passed"]
        ):
            raise SystemExit(
                "Pilot quotient gate failed; collection was not started."
            )

    if args.mode == "collect":
        pilot = json.loads(
            Path(args.pilot_output).read_text()
        )
        if not pilot["quotient_gate"]["passed"]:
            raise SystemExit(
                "Pilot quotient gate did not pass."
            )

    if args.mode in ("collect", "all"):
        start = time.time()
        arrays, metadata = collect_trials(
            model=model,
            tokenizer=tokenizer,
            pilot=pilot,
            flow=args.flow,
            trials_per_split=args.trials_per_split,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            device=args.device,
        )

        np.savez_compressed(
            args.trials_output,
            **arrays,
        )
        Path(args.metadata_output).write_text(
            json.dumps(metadata, indent=2)
        )

        print(
            json.dumps(
                {
                    "trials_output": args.trials_output,
                    "metadata_output": args.metadata_output,
                    "shape_x": list(arrays["x"].shape),
                    "shape_y": list(arrays["y"].shape),
                    "split_counts": {
                        str(split_id): int(
                            np.sum(
                                arrays["split"]
                                == split_id
                            )
                        )
                        for split_id in range(4)
                    },
                    "classification_margin_summary": metadata[
                        "classification_margin_summary"
                    ],
                    "elapsed_seconds": time.time() - start,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
