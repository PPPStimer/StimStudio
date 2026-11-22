"""Parameter schema shared by generators and effects (drives the GUI panels)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParamSpec:
    """Describes one tunable parameter."""

    key: str
    label: str
    kind: str = "float"  # "float" | "int" | "bool" | "choice" | "str" | "path"
    default: object = 0.0
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple = field(default_factory=tuple)


def defaults(specs) -> dict:
    """``{key: default}`` for a list of specs."""
    return {spec.key: spec.default for spec in specs}
