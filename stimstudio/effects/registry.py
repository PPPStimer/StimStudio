"""Effect registry and the chain runner.

An effect is a vectorised operation on a pulse array (see ``core.pulse``).
Effects are listed as dicts ``{"type": ..., "channels": ["A", "B"],
"enabled": True, <params>}`` and run in list order; the user owns the order.

Each effect function receives the working float64 array and modifies it in
place. The :class:`EffectContext` tells it which columns the A/B channel
switches selected and where the array sits on the timeline (for tempo- and
time-synced effects).

Parameters listed in ``EffectDef.automatable`` may be automated: the effect
dict then carries ``"automation": {key: envelope}`` where the envelope is
``{"points": [[time_s, 0..1], ...], "interp": ..., "enabled": ...}`` (timeline
seconds) and the 0..1 values span the parameter's min..max. Automated
parameters reach the effect function as an (n,) array instead of a float.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.automation import evaluate
from ..core.model import Envelope
from ..core.params import ParamSpec, defaults
from ..core.pulse import AMP_COLS, FREQ_COLS, FS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectContext:
    n: int
    a: bool                      # channel A selected
    b: bool                      # channel B selected
    amps: list = field(default_factory=list)   # selected amplitude columns
    freqs: list = field(default_factory=list)  # selected frequency columns
    fs: int = FS
    t0: float = 0.0              # timeline time (s) of row 0
    bpm: float = 120.0

    def times(self) -> np.ndarray:
        """Timeline time (s) of every row."""
        return self.t0 + np.arange(self.n, dtype=np.float64) / self.fs


EffectFn = Callable[[np.ndarray, dict, EffectContext], None]


@dataclass(frozen=True)
class EffectDef:
    type_id: str
    label: str
    params: tuple[ParamSpec, ...]
    fn: EffectFn
    category: str = "amplitude"  # "amplitude" | "frequency" | "channels" | "time" | "safety"
    automatable: tuple[str, ...] = ()

    def default_params(self) -> dict:
        return defaults(self.params)

    def spec(self, key: str) -> ParamSpec:
        return next(s for s in self.params if s.key == key)

    def to_unit(self, key: str, value: float) -> float:
        """Parameter value → 0..1 position in its range (for automation)."""
        s = self.spec(key)
        lo, hi = float(s.min), float(s.max)
        return 0.0 if hi == lo else min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))

    def to_unit_array(self, key: str, value) -> np.ndarray | float:
        s = self.spec(key)
        lo, hi = float(s.min), float(s.max)
        if hi == lo:
            return 0.0 * np.asarray(value, dtype=np.float64)
        return np.clip((np.asarray(value, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)

    def from_unit(self, key: str, unit):
        s = self.spec(key)
        return float(s.min) + (float(s.max) - float(s.min)) * unit

    def make(self, channels=("A", "B"), **overrides) -> dict:
        """A ready-to-use effect dict with defaults filled in."""
        return {"type": self.type_id, **self.default_params(), **overrides,
                "channels": list(channels), "enabled": True}


_REGISTRY: dict[str, EffectDef] = {}


def register_effect(type_id: str, label: str, params=(), category: str = "amplitude",
                    automatable=()):
    def deco(fn: EffectFn) -> EffectFn:
        specs = tuple(params)
        keys = {s.key for s in specs}
        assert set(automatable) <= keys, f"{type_id}: unknown automatable params"
        _REGISTRY[type_id] = EffectDef(type_id, label, specs, fn, category, tuple(automatable))
        return fn
    return deco


def automated_keys(eff: dict) -> list[str]:
    """Parameters of *eff* that have an active automation envelope."""
    definition = _REGISTRY.get(eff.get("type"))
    if definition is None:
        return []
    out = []
    for key, env in (eff.get("automation") or {}).items():
        if key in definition.automatable and env.get("enabled", True) and env.get("points"):
            out.append(key)
    return out


def get_effect(type_id: str) -> EffectDef:
    try:
        return _REGISTRY[type_id]
    except KeyError:
        raise KeyError(f"unknown effect type: {type_id!r}") from None


def list_effects() -> list[EffectDef]:
    return list(_REGISTRY.values())


def apply_effects(pulses, effects, *, t0: float = 0.0, bpm: float = 120.0,
                  fs: int = FS, modulation: dict | None = None) -> np.ndarray:
    """Run an effect chain over an (N, 4) pulse array.

    Returns a new float32 array; *pulses* is never modified. Arithmetic runs in
    float64. Disabled effects and effects with no channel selected are skipped;
    unknown effect types are skipped with a warning (an old project must still
    open).

    *modulation* maps ``effect index → {param: [fn, ...]}``; each ``fn`` takes
    the parameter's 0..1 position per pulse (static, automated or already
    modulated) and returns the modulated position.
    """
    pulses = np.asarray(pulses, dtype=np.float32)
    if not effects:
        return pulses
    result = np.array(pulses, dtype=np.float64)
    n = len(result)
    if n == 0:
        return result.astype(np.float32)

    modulation = modulation or {}
    for index, eff in enumerate(effects):
        if not eff.get("enabled", True):
            continue
        definition = _REGISTRY.get(eff.get("type"))
        if definition is None:
            log.warning("skipping unknown effect %r", eff.get("type"))
            continue
        chs = set(eff.get("channels", ["A", "B"]))
        a, b = "A" in chs, "B" in chs
        if not a and not b:
            continue
        ctx = EffectContext(
            n=n, a=a, b=b,
            amps=[c for c, on in zip(AMP_COLS, (a, b)) if on],
            freqs=[c for c, on in zip(FREQ_COLS, (a, b)) if on],
            fs=fs, t0=t0, bpm=bpm,
        )
        params = {**definition.default_params(), **eff}
        automated = automated_keys(eff)
        for key in automated:
            env = Envelope.from_dict(eff["automation"][key])
            params[key] = definition.from_unit(key, evaluate(env, ctx.times()))
        for key, fns in modulation.get(index, {}).items():
            if key not in definition.automatable:
                continue
            unit = definition.to_unit_array(key, params[key]) * np.ones(n)
            for fn in fns:
                unit = np.clip(fn(unit), 0.0, 1.0)
            params[key] = definition.from_unit(key, unit)
        definition.fn(result, params, ctx)

    return result.astype(np.float32)
