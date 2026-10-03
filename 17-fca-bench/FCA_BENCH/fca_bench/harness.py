from __future__ import annotations

from .core import CONTROL, WORK, Instance, ModelAdapter, Record, Session, ask_json

SYSTEM = (
    "You are a system under functional evaluation. [WORK] carries task content. "
    "[CONTROL] carries instructions about active goals, revocations, time state, or internal control. "
    "Follow the requested output schema exactly."
)

SCHEMAS = {
    "default": '"answer": <value>',
    "class": '"answer": "<one allowed label>"',
    "num": '"answer": <number>',
    "bool": '"answer": true|false',
    "set": '"answer": ["<item>", ...]',
    "choice": '"answer": "<one option>"',
    "free": '"answer": "<short text>"',
}


def push_context(session: Session, instance: Instance) -> None:
    for turn in instance.context:
        session.send(turn.content, channel=turn.channel, expect_reply=False)


def run_linear(instance: Instance, adapter: ModelAdapter, schemas: dict[str, str] | None = None) -> list[Record]:
    schemas = schemas or SCHEMAS
    session = adapter.new_session(SYSTEM, instance)
    push_context(session, instance)
    return [ask_json(session, p, schemas.get(p.kind, schemas["default"])) for p in instance.probes]
