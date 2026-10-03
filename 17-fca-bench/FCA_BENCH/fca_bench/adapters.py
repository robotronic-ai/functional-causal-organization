from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from .core import CONTROL, WORK, Instance


class UnsupportedCapability(RuntimeError):
    pass


def resolve_model_source(model_path: str, local_files_only: bool = True) -> str:
    raw_model_path = str(model_path).strip()
    if not raw_model_path:
        raise ValueError("HF backend requires a non-empty model_path.")
    if raw_model_path.startswith("/absolute/path/to/"):
        raise ValueError("model_path is still an example placeholder.")

    expanded = Path(os.path.expandvars(raw_model_path)).expanduser()
    resolved_local: Path | None = None
    if expanded.exists():
        resolved_local = expanded.resolve()
    elif not expanded.is_absolute():
        candidate = (Path.cwd() / expanded).resolve()
        if candidate.exists():
            resolved_local = candidate

    if local_files_only and resolved_local is None:
        attempted = (Path.cwd() / expanded).resolve() if not expanded.is_absolute() else expanded
        raise FileNotFoundError(
            "Local model directory not found. "
            f"Configured model_path={raw_model_path!r}; working_directory={str(Path.cwd())!r}; "
            f"resolved_candidate={str(attempted)!r}."
        )
    return str(resolved_local) if resolved_local is not None else raw_model_path


class BaseAdapter:
    name = "base"
    capabilities: set[str] = {"chat"}

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


class HFLocalAdapter(BaseAdapter):
    """Strict local GPU adapter for the FCA v0.2.7 Qwen baseline.

    When strict_gpu is true, CPU/disk offload is treated as an error. Optional
    bitsandbytes quantization is supported so an 8B model can remain entirely on a
    16 GiB GPU. Quantization changes the evaluated deployed system and is recorded
    in result metadata.
    """

    capabilities = {"chat"}

    def __init__(
        self,
        model_path: str,
        device_map: Any = None,
        dtype: str = "float16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        sample_temperature: float = 0.8,
        top_p: float = 0.95,
        enable_thinking: bool = False,
        seed: int = 1234,
        e1_samples: int = 20,
        e1_batch_size: int = 4,
        lazy_context: bool = True,
        local_files_only: bool = True,
        system_id: str | None = None,
        name: str | None = None,
        progress: bool = True,
        strict_gpu: bool = True,
        gpu_index: int = 0,
        quantization: str = "bnb8",
        attn_implementation: str = "sdpa",
    ):
        effective_model_path = resolve_model_source(model_path, local_files_only=bool(local_files_only))

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("HFLocalAdapter requires torch and transformers.") from exc

        self._torch = torch
        self.name = name or f"hf:{effective_model_path}"
        self.system_id = system_id or self.name
        self.model_path = effective_model_path
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.sample_temperature = float(sample_temperature)
        self.top_p = float(top_p)
        self.enable_thinking = bool(enable_thinking)
        self.seed = int(seed)
        self.e1_samples = int(e1_samples)
        self.e1_batch_size = max(1, int(e1_batch_size))
        self.lazy_context = bool(lazy_context)
        self.progress = bool(progress)
        self.strict_gpu = bool(strict_gpu)
        self.gpu_index = int(gpu_index)
        self.quantization = str(quantization).lower().strip()
        self.attn_implementation = str(attn_implementation).strip()
        self._progress_total_sequences: int | None = None
        self._progress_done_sequences = 0
        self._progress_batches = 0
        self._progress_started = time.perf_counter()
        self._progress_context = ""
        self._progress_file: Path | None = None

        if self.strict_gpu and not torch.cuda.is_available():
            raise RuntimeError("strict_gpu=true but CUDA is not available.")
        if self.strict_gpu and self.gpu_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{self.gpu_index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_index)
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        model_dtype: Any = dtype
        if dtype == "auto":
            model_dtype = "auto"
        elif hasattr(torch, dtype):
            model_dtype = getattr(torch, dtype)

        quantization_config = None
        if self.quantization in {"bnb8", "8bit", "int8"}:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "quantization=bnb8 requires bitsandbytes. Install it in the active environment before rerunning."
                ) from exc
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        elif self.quantization in {"bnb4", "4bit", "nf4"}:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "quantization=bnb4 requires bitsandbytes. Install it in the active environment before rerunning."
                ) from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization not in {"none", "off", "false", ""}:
            raise ValueError(f"Unknown quantization mode: {self.quantization}")

        if device_map is None:
            device_map = {"": self.gpu_index} if self.strict_gpu else "auto"

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.gpu_index)
            total_gib = props.total_memory / (1024 ** 3)
            print(
                f"[GPU] cuda:{self.gpu_index} {props.name} total_vram={total_gib:.2f} GiB "
                f"quantization={self.quantization or 'none'} attention={self.attn_implementation}",
                flush=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            effective_model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=self.local_files_only,
        )

        load_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": model_dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(effective_model_path, **load_kwargs)
        self.model.eval()

        if self.strict_gpu:
            self._assert_gpu_only()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(self.gpu_index) / (1024 ** 3)
            print(f"[GPU] model ready; allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB; CPU offload=DISALLOWED", flush=True)

    def _assert_gpu_only(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            bad = {k: v for k, v in device_map.items() if str(v).lower() in {"cpu", "disk", "meta"}}
            if bad:
                raise RuntimeError(
                    "GPU-only mode refused the loaded model because Hugging Face offloaded modules: "
                    + repr(bad)
                )
        try:
            first_device = next(self.model.parameters()).device
        except StopIteration:
            first_device = None
        if first_device is not None and getattr(first_device, "type", None) != "cuda":
            raise RuntimeError(f"GPU-only mode expected CUDA parameters but found {first_device}.")

    def runtime_metadata(self) -> dict[str, Any]:
        out = {
            "strict_gpu": self.strict_gpu,
            "gpu_index": self.gpu_index,
            "quantization": self.quantization,
            "attention_implementation": self.attn_implementation,
            "e1_batch_size": self.e1_batch_size,
        }
        if self._torch.cuda.is_available():
            props = self._torch.cuda.get_device_properties(self.gpu_index)
            out.update({
                "gpu_name": props.name,
                "gpu_total_vram_gib": round(props.total_memory / (1024 ** 3), 3),
            })
        return out

    def configure_progress(self, total_calls: int | None = None, enabled: bool | None = None) -> None:
        if enabled is not None:
            self.progress = bool(enabled)
        self._progress_total_sequences = int(total_calls) if total_calls is not None else None
        self._progress_done_sequences = 0
        self._progress_batches = 0
        self._progress_started = time.perf_counter()

    def set_progress_file(self, path: str | Path | None) -> None:
        self._progress_file = Path(path) if path else None

    def set_progress_context(self, test_code: str, instance_id: str) -> None:
        self._progress_context = f"{test_code}/{instance_id}"

    def _gpu_memory_text(self) -> str:
        if not self._torch.cuda.is_available():
            return ""
        alloc = self._torch.cuda.memory_allocated(self.gpu_index) / (1024 ** 3)
        reserved = self._torch.cuda.memory_reserved(self.gpu_index) / (1024 ** 3)
        return f" vram={alloc:.2f}/{reserved:.2f}GiB"

    def _write_progress_json(self, payload: dict[str, Any]) -> None:
        if self._progress_file is None:
            return
        self._progress_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._progress_file.with_suffix(self._progress_file.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self._progress_file)

    def _report_generation(
        self,
        seconds: float,
        input_tokens: int,
        output_tokens: int,
        sequences: int = 1,
    ) -> None:
        sequences = max(1, int(sequences))
        self._progress_done_sequences += sequences
        self._progress_batches += 1
        done = self._progress_done_sequences
        total = self._progress_total_sequences
        elapsed = max(1e-9, time.perf_counter() - self._progress_started)
        rate = done / elapsed
        eta = ((total - done) / rate) if total and total > done and rate > 0 else 0.0
        payload = {
            "context": self._progress_context,
            "completed_sequences": done,
            "total_sequences": total,
            "generation_batches": self._progress_batches,
            "last_batch_sequences": sequences,
            "last_batch_seconds": round(seconds, 6),
            "elapsed_seconds": round(elapsed, 3),
            "sequences_per_second": round(rate, 6),
            "eta_seconds": round(eta, 3) if total else None,
        }
        self._write_progress_json(payload)
        if not self.progress:
            return
        total_text = str(total) if total is not None else "?"
        ctx = f" {self._progress_context}" if self._progress_context else ""
        print(
            f"[SEQ {done}/{total_text}] batch={sequences} batch_dt={seconds:.2f}s{ctx} "
            f"in={input_tokens} out={output_tokens} rate={rate:.3f}seq/s "
            f"ETA~{eta/60:.1f}m{self._gpu_memory_text()}",
            flush=True,
        )

    def _stable_seed(self, instance: Instance | None, key: str) -> int:
        inst_seed = instance.seed if instance is not None else 0
        raw = f"{self.seed}|{inst_seed}|{key}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)

    class Session:
        def __init__(self, outer: "HFLocalAdapter", system: str | None, instance: Instance | None):
            self.outer = outer
            self.instance = instance
            self.messages: list[dict[str, str]] = []
            if system:
                self.messages.append({"role": "system", "content": system})
            self.pending: list[str] = []
            self.call_index = 0

        def _render(self):
            tok = self.outer.tokenizer
            kwargs = dict(add_generation_prompt=True, return_tensors="pt")
            try:
                return tok.apply_chat_template(
                    self.messages,
                    enable_thinking=self.outer.enable_thinking,
                    **kwargs,
                )
            except TypeError:
                return tok.apply_chat_template(self.messages, **kwargs)

        def _device_ids(self, ids):
            return ids.to(self.outer._torch.device(f"cuda:{self.outer.gpu_index}"))

        def _generation_kwargs(self, do_sample: bool) -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.outer.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.outer.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if do_sample:
                kwargs["temperature"] = self.outer.sample_temperature
                kwargs["top_p"] = self.outer.top_p
            return kwargs

        def _append_user(self, text: str, channel: str) -> None:
            tag = "[CONTROL] " if channel == CONTROL else "[WORK] "
            block = tag + text
            if self.pending:
                block = "\n\n".join(self.pending + [block])
                self.pending.clear()
            self.messages.append({"role": "user", "content": block})

        def send(
            self,
            text: str,
            channel: str = WORK,
            expect_reply: bool = True,
            sample: bool = False,
        ) -> str:
            tag = "[CONTROL] " if channel == CONTROL else "[WORK] "
            block = tag + text
            if self.outer.lazy_context and not expect_reply:
                self.pending.append(block)
                return ""
            self._append_user(text, channel)

            ids = self._device_ids(self._render())
            seed_key = f"single|{self.call_index}|{len(self.messages)}|{text[:80]}"
            seed = self.outer._stable_seed(self.instance, seed_key)
            self.call_index += 1
            self.outer._torch.manual_seed(seed)
            self.outer._torch.cuda.manual_seed_all(seed)

            do_sample = bool(sample or self.outer.temperature > 0)
            kwargs = self._generation_kwargs(do_sample)
            attention_mask = self.outer._torch.ones_like(ids, dtype=self.outer._torch.long)
            t0 = time.perf_counter()
            with self.outer._torch.inference_mode():
                generated = self.outer.model.generate(ids, attention_mask=attention_mask, **kwargs)
            dt = time.perf_counter() - t0
            new_tokens = generated[0][ids.shape[-1] :]
            self.outer._report_generation(dt, int(ids.shape[-1]), int(new_tokens.shape[-1]), sequences=1)
            output = self.outer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            self.messages.append({"role": "assistant", "content": output})
            return output

        def send_many(
            self,
            text: str,
            n: int,
            channel: str = WORK,
            sample: bool = True,
            seed_key: str = "batch",
        ) -> list[str]:
            """Generate independent fresh-context samples in GPU micro-batches.

            The same rendered prompt is replicated across the batch. There is no
            cross-sequence attention, and each micro-batch receives a deterministic,
            distinct RNG seed. This is an execution optimization for E1; it does not
            change the E1 scoring definition.
            """
            n = int(n)
            if n <= 0:
                return []
            self._append_user(text, channel)
            base_ids = self._device_ids(self._render())
            outputs: list[str] = []
            batch_size = max(1, self.outer.e1_batch_size)
            do_sample = bool(sample)
            kwargs = self._generation_kwargs(do_sample)

            for start in range(0, n, batch_size):
                b = min(batch_size, n - start)
                ids = base_ids.repeat(b, 1)
                attention_mask = self.outer._torch.ones_like(ids, dtype=self.outer._torch.long)
                seed = self.outer._stable_seed(self.instance, f"{seed_key}|{start}|{b}")
                self.outer._torch.manual_seed(seed)
                self.outer._torch.cuda.manual_seed_all(seed)
                t0 = time.perf_counter()
                with self.outer._torch.inference_mode():
                    generated = self.outer.model.generate(ids, attention_mask=attention_mask, **kwargs)
                dt = time.perf_counter() - t0
                out_token_total = 0
                for row in range(b):
                    new_tokens = generated[row][ids.shape[-1] :]
                    out_token_total += int(new_tokens.shape[-1])
                    outputs.append(self.outer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
                self.outer._report_generation(
                    dt,
                    int(ids.shape[-1]) * b,
                    out_token_total,
                    sequences=b,
                )
            return outputs

        def mechanistic(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
            raise UnsupportedCapability(operation)

    def new_session(self, system: str | None = None, instance: Instance | None = None):
        return HFLocalAdapter.Session(self, system, instance)


class ExternalAdapter(BaseAdapter):
    def __init__(self, module_path: str, class_name: str = "CustomAdapter", config: dict[str, Any] | None = None):
        path = Path(module_path).resolve()
        spec = importlib.util.spec_from_file_location("fca_external_adapter", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load adapter module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        self.impl = cls(config or {})
        self.name = getattr(self.impl, "name", class_name)
        self.system_id = getattr(self.impl, "system_id", self.name)

    def supports(self, capability: str) -> bool:
        return bool(self.impl.supports(capability))

    def new_session(self, system: str | None = None, instance: Instance | None = None):
        return self.impl.new_session(system=system, instance=instance)


def adapter_from_config(config: dict[str, Any]):
    backend = config.get("backend", "hf")
    if backend == "hf":
        allowed = {
            "model_path", "device_map", "dtype", "trust_remote_code", "max_new_tokens",
            "temperature", "sample_temperature", "top_p", "enable_thinking", "seed",
            "e1_samples", "e1_batch_size", "lazy_context", "local_files_only", "system_id",
            "name", "progress", "strict_gpu", "gpu_index", "quantization", "attn_implementation",
        }
        args = {k: v for k, v in config.items() if k in allowed}
        if "model_path" not in args:
            raise ValueError("HF backend requires model_path in the JSON config.")
        return HFLocalAdapter(**args)
    if backend == "external":
        return ExternalAdapter(
            module_path=config["module_path"],
            class_name=config.get("class_name", "CustomAdapter"),
            config=config.get("adapter_config", {}),
        )
    raise ValueError(f"Unknown backend: {backend}")


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
