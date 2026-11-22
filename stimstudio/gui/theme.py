"""Palette, stylesheet loading and time formatting (VSCode Dark look)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QIcon

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
ICON = Path(__file__).resolve().parent / "ss.svg"

BG = "#1e1e1e"
BG_ALT = "#232324"       # alternate track rows
PANEL_BG = "#252526"
BORDER = "#3c3c3c"
GRID = "#2e2e2e"
GRID_STRONG = "#3a3a3a"
TEXT = "#d4d4d4"
TEXT_DIM = "#858585"
ACCENT = "#007acc"
HIGHLIGHT = "#264f78"
DANGER = "#f14c4c"       # playhead, panic
CURSOR = "#e8e8e8"       # edit cursor

#: Lane colours in column order (amp L, amp R, freq L, freq R).
LANE_COLORS = ("#4ec9b0", "#569cd6", "#dcdcaa", "#ce9178")
LANE_NAMES = ("Amp L", "Amp R", "Freq L", "Freq R")

#: Colours offered for new tracks (cycled).
TRACK_COLORS = ("#4ec9b0", "#569cd6", "#c586c0", "#dcdcaa", "#ce9178", "#9cdcfe",
                "#b5cea8", "#d16969")


def load_stylesheet() -> str:
    return (RESOURCES / "qss" / "vscode_dark.qss").read_text(encoding="utf-8")


def app_icon() -> QIcon:
    """The application icon: every window's title bar, the taskbar, the About box."""
    return QIcon(str(ICON))


def format_time(seconds: float, decimals: int = 3) -> str:
    """Seconds as mm:ss.mmm, growing an h: prefix only once past an hour."""
    value = round(abs(float(seconds)), decimals)
    hours, rest = divmod(value, 3600.0)
    minutes, sec = divmod(rest, 60.0)
    width = 2 + (decimals + 1 if decimals else 0)
    out = f"{int(minutes):02d}:{sec:0{width}.{decimals}f}"
    if hours:
        out = f"{int(hours)}:{out}"
    return ("-" + out) if seconds < 0 else out
