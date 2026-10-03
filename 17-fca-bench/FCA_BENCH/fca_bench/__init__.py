"""FCA-Bench: Functional Causal Access Benchmark.

The name refers to causal accessibility and causal governance of functional state.
It is not a consciousness score, a general-intelligence score, or a sophistication
score, and it is independent from A_DAGGER.
"""

from .core import REGISTRY, Instance, Probe, Record, Result, Test, Turn

__version__ = "0.2.7"
__runner_id__ = "FCA027_GPU_R6"

BEHAVIORAL_DIMENSIONS = {
    "T": "Temporal Grounding",
    "B": "Internal/External Boundary",
    "E": "Context Governance / Erasure",
    "M": "Metacognitive Monitoring",
    "C": "Cognitive Control / Replanning",
    "A": "Causal Agency",
    "P": "Functional Self-Prediction",
}

MECHANISTIC_DIMENSIONS = {
    "R": "Latent Recurrence",
    "D": "Autonomous Temporal Dynamics",
    "L": "Persistent Latent State",
    "U": "Mutable Internal State",
    "I": "Endogenous Interrupt / Arbitration",
    "G": "Global Workspace / Broadcast",
    "H": "Higher-Order Attention Control",
}

CAUSAL_DIMENSIONS = MECHANISTIC_DIMENSIONS

DIMENSIONS = {**BEHAVIORAL_DIMENSIONS, **CAUSAL_DIMENSIONS}

from . import tests_time, tests_boundary, tests_erasure, tests_meta
from . import tests_control, tests_agency, tests_selfpred
from . import tests_recurrence, tests_temporal_dynamics, tests_latent_state
from . import tests_mutable_state, tests_interrupt, tests_workspace, tests_attention_schema

__all__ = [
    "REGISTRY", "DIMENSIONS", "BEHAVIORAL_DIMENSIONS", "CAUSAL_DIMENSIONS", "MECHANISTIC_DIMENSIONS",
    "Instance", "Probe", "Record", "Result", "Test", "Turn", "__version__", "__runner_id__",
]
