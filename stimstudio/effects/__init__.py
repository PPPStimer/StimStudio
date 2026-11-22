"""Effect registry. Importing this package registers all built-in effects."""

from .registry import (
    EffectContext,
    EffectDef,
    apply_effects,
    automated_keys,
    get_effect,
    list_effects,
    register_effect,
)
from . import builtin  # noqa: F401  (registers effects)
from .builtin import LEGACY_EFFECTS
from .fades import crossfade_edges, fade_in, fade_out

__all__ = [
    "EffectContext",
    "EffectDef",
    "LEGACY_EFFECTS",
    "apply_effects",
    "automated_keys",
    "crossfade_edges",
    "fade_in",
    "fade_out",
    "get_effect",
    "list_effects",
    "register_effect",
]
