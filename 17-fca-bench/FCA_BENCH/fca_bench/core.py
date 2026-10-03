from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

WORK = "work"
CONTROL = "control"


@dataclass
class Turn:
    content: str
    channel: str = WORK


@dataclass
class Probe:
    id: str
    prompt: str
    kind: str
    gold: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Instance:
    id: str
    dim: str
    test: str
    seed: int
    context: list[Turn] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    probe_id: str
    raw: str = ""
    parsed: Any = None
    arm: str = "main"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    instance_id: str
    dim: str
    test: str
    score: float
    raw_score: float = 0.0
    valid: bool = True
    status: str = "executed"
    details: dict[str, Any] = field(default_factory=dict)


class Session(Protocol):
    def send(
        self,
        text: str,
        channel: str = WORK,
        expect_reply: bool = True,
        sample: bool = False,
    ) -> str: ...


class ModelAdapter(Protocol):
    name: str

    def new_session(self, system: str | None = None, instance: Instance | None = None) -> Session: ...

    def supports(self, capability: str) -> bool: ...


@dataclass
class Test:
    code: str
    dim: str
    label: str
    generate: Callable[[int], Instance]
    run: Callable[[Instance, ModelAdapter], list[Record]]
    score: Callable[[Instance, list[Record]], Result]
    track: str = "behavioral"
    notes: str = ""


REGISTRY: dict[str, Test] = {}


def register(test: Test) -> Test:
    if test.code in REGISTRY:
        raise ValueError(f"Duplicate test code: {test.code}")
    REGISTRY[test.code] = test
    return test


def rng(seed: int, salt: str = "") -> random.Random:
    return random.Random(f"{seed}|{salt}")


JSON_INSTRUCTION = "Return only one valid JSON object. Do not add prose or Markdown fences."
_FENCE = re.compile(r"```(?:json)?(.*?)```", re.S | re.I)


def parse_json(raw: str | None) -> Any:
    if not raw:
        return None
    match = _FENCE.search(raw)
    if match:
        raw = match.group(1)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return None
    return None


def ask_json(
    session: Session,
    probe: Probe,
    schema: str,
    channel: str = WORK,
    sample: bool = False,
) -> Record:
    prompt = (
        f"{probe.prompt}\n\n"
        f'Expected schema: {{"probe_id": "{probe.id}", {schema}}}\n'
        f"{JSON_INSTRUCTION}"
    )
    raw = session.send(prompt, channel=channel, expect_reply=True, sample=sample)
    return Record(probe.id, raw=raw, parsed=parse_json(raw))



def ask_json_many(
    session: Session,
    probe: Probe,
    schema: str,
    n: int,
    channel: str = WORK,
    sample: bool = True,
    seed_key: str = "batch",
) -> list[Record]:
    """Request multiple independent samples from an adapter that supports batching.

    Falls back to repeated scalar sends for external adapters. The bundled GPU
    adapter implements send_many with deterministic micro-batching.
    """
    prompt = (
        f"{probe.prompt}\n\n"
        f'Expected schema: {{"probe_id": "{probe.id}", {schema}}}\n'
        f"{JSON_INSTRUCTION}"
    )
    if hasattr(session, "send_many"):
        raws = session.send_many(prompt, n=n, channel=channel, sample=sample, seed_key=seed_key)
    else:
        raws = [session.send(prompt, channel=channel, expect_reply=True, sample=sample) for _ in range(int(n))]
    return [Record(probe.id, raw=raw, parsed=parse_json(raw)) for raw in raws]

def field_of(record: Record | None, key: str, default: Any = None) -> Any:
    if record is not None and isinstance(record.parsed, dict):
        return record.parsed.get(key, default)
    return default


def clipped(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def structural_absent(instance: Instance, capability: str, note: str = "") -> Result:
    """Return an architecture-absence result, not an executed performance score.

    The numeric zero is retained for machine aggregation of causal availability,
    but the status MUST be reported separately from executed scores.
    """
    details = {"missing_capability": capability}
    if note:
        details["note"] = note
    return Result(
        instance.id,
        instance.dim,
        instance.test,
        score=0.0,
        raw_score=0.0,
        valid=True,
        status="structural_absent",
        details=details,
    )


# Compatibility alias for turn-1/turn-4 test modules. New code should call
# structural_absent; both names now produce status="structural_absent".
def structural_zero(instance: Instance, capability: str, note: str = "") -> Result:
    return structural_absent(instance, capability, note)
