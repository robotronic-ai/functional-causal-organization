from __future__ import annotations

from typing import Any

from .core import Record
from .adapters import UnsupportedCapability


def unsupported_record(capability: str) -> list[Record]:
    return [Record("__structural__", parsed={"unsupported": capability}, meta={"status": "structural_absent"})]


def missing_capability(records: list[Record]) -> str | None:
    if records and isinstance(records[0].parsed, dict):
        return records[0].parsed.get("unsupported")
    return None


def op(session, operation: str, payload: dict[str, Any] | None = None) -> Any:
    # Prefer the causal naming.
    # Fall back to the turn-1 mechanistic hook for compatibility.
    fn = getattr(session, "causal", None)
    if fn is None:
        fn = getattr(session, "mechanistic", None)
    if fn is None:
        raise UnsupportedCapability(operation)
    return fn(operation, payload or {})
