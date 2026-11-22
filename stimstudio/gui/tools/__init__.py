"""Tools menu: stand-alone utilities, each in its own window.

A utility is one ``ToolInfo`` in ``TOOLS``; the main window builds the Tools
menu from that list and keeps one window per tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolInfo:
    id: str
    label: str
    tooltip: str
    factory: Callable                  # factory(main_window) -> ToolWindow


def _color_rhythm(main_window):
    from .color_rhythm import ColorRhythmWindow
    return ColorRhythmWindow(main_window)


def _triphase_audio(main_window):
    from .triphase_audio import TriphaseAudioWindow
    return TriphaseAudioWindow(main_window)


TOOLS: tuple[ToolInfo, ...] = (
    ToolInfo("color_rhythm", "&Color rhythm from video…",
             "How much of a region chosen colours fill → a clip (then: Beat marks from peaks)",
             _color_rhythm),
    ToolInfo("triphase_audio", "&Triphase audio → stim clip…",
             "A phase-coded three-phase stereo file → an A/B amplitude + frequency clip "
             "(or a triphase clip)",
             _triphase_audio),
)


def tool_info(tool_id: str) -> ToolInfo:
    for t in TOOLS:
        if t.id == tool_id:
            return t
    raise KeyError(tool_id)
