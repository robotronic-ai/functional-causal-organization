#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialBatch:
    x: np.ndarray
    y: np.ndarray
    context: np.ndarray


class GammaTargetAdapter(ABC):
    """
    Model-specific contract for the recurrent-memory transformer experiment.

    The architecture-specific layer must produce only frozen macro-role trials.
    The Gamma certifier must not import or inspect the model implementation.
    """

    @abstractmethod
    def load_frozen_model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def role_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def collect_trials(
        self,
        split_name: str,
        n_trials: int,
        seed: int,
    ) -> TrialBatch:
        """
        Return binary macro-role source states x, binary macro-role next states y,
        and a discrete context identifier for every trial.

        The intervention battery must be balanced and independent of context.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_causal_minimal_quotient(
        self,
        discovery_seed: int,
        holdout_seed: int,
    ) -> dict:
        """
        Return a JSON-serializable gate.

        Required field:
            passed: bool

        The gate must be evaluated before Gamma certification.
        """
        raise NotImplementedError
