from __future__ import annotations

from typing import Any

from fca_bench.adapters import UnsupportedCapability
from fca_bench.core import CONTROL, WORK, Instance


class CustomAdapter:
    """Template for a recurrent, dual-stream, or otherwise modified architecture."""

    name = "custom-architecture"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.name)
        self.system_id = config.get("system_id", self.name)
        self.capabilities = {
            "chat",
            # Uncomment only capabilities that are implemented below.
            # "silent_steps",
            # "adaptive_compute",
            # "idle_dynamics",
            # "persistent_latent_state",
            # "dual_reset",
            # "state_write",
            # "checkpoint_rollback",
            # "semantic_gc",
            # "self_interrupt",
            # "async_preemption",
            # "independent_stream_clocks",
            # "workspace_broadcast",
            # "workspace_lesion",
            # "workspace_competition",
            # "attention_state",
            # "attention_reallocation",
            # "attention_schema_lesion",
        }
        # Load the user's model and tokenizer here.

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    class Session:
        def __init__(self, outer: "CustomAdapter", system: str | None, instance: Instance | None):
            self.outer = outer
            self.system = system
            self.instance = instance

        def send(
            self,
            text: str,
            channel: str = WORK,
            expect_reply: bool = True,
            sample: bool = False,
        ) -> str:
            if channel == CONTROL:
                # Route to the architecture's control stream when available.
                pass
            # Route WORK input to the normal work stream and return generated text.
            raise NotImplementedError

        def causal(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
            payload = payload or {}
            # Map FCA causal/interventional operations to native architecture primitives here.
            # Each causal test documents its expected operation and return shape.
            raise UnsupportedCapability(operation)

        def mechanistic(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
            # Backward-compatible alias for turn-1 adapters.
            return self.causal(operation, payload)

    def new_session(self, system: str | None = None, instance: Instance | None = None):
        return CustomAdapter.Session(self, system, instance)
