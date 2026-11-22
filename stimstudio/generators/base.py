"""Generator base class, parameter specs, and a small registry.

A Generator turns parameters + a pulse count into a :class:`PulseTrack`.
Stochastic generators must be deterministic for a given ``seed`` so that a
cached render matches a later re-render by the sequencer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.params import ParamSpec
from ..core.pulse import PulseTrack


class Generator(ABC):
    type_id: str = "base"
    label: str = "Base"
    #: True if output depends on ``seed`` (enables the "new seed" button).
    stochastic: bool = False

    #: Short category for the browser ("activity", "shape", "rhythm", "manual", "import").
    category: str = "shape"
    #: "pulse" (amp/freq L/R) or "triphase" (volume, x, y, freq — triphase tracks only).
    domain: str = "pulse"
    #: True if the generator follows the project tempo. The renderer then passes
    #: the project BPM in ``params["_bpm"]`` (and re-renders when it changes).
    tempo_synced: bool = False

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return []

    @classmethod
    def default_params(cls) -> dict:
        return {spec.key: spec.default for spec in cls.param_specs()}

    def merged_params(self, params: dict | None) -> dict:
        merged = self.default_params()
        if params:
            merged.update(params)
        return merged

    @abstractmethod
    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        """Return a PulseTrack of exactly *n* pulses."""

    def segment_boundaries(self, n: int, params: dict, seed: int = 0) -> list[float]:
        """Times (s) where an internal segment/stage changes. Empty by default."""
        return []

    def manual_params_at(self, n: int, params: dict, seed: int = 0) -> dict:
        """Controllable parameter values after simulating *n* pulses. Empty by default."""
        return {}


# -- registry -------------------------------------------------------------
_REGISTRY: dict[str, type[Generator]] = {}


def register(cls: type[Generator]) -> type[Generator]:
    _REGISTRY[cls.type_id] = cls
    return cls


def get_generator(type_id: str) -> Generator:
    try:
        return _REGISTRY[type_id]()
    except KeyError:
        raise KeyError(f"unknown generator type: {type_id!r}") from None


def list_generators() -> list[type[Generator]]:
    return list(_REGISTRY.values())
