"""Shared pieces of the pattern editors: the drawable plot and curve helpers.

``DrawablePlot`` is ported from WaveHowl's ``_DrawablePlot``; like every plot
in the app it renders through OpenGL (pyqtgraph ``useOpenGL``).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal

from .. import theme


#: Sparse 0..1 ticks: short plots stay readable.
Y_TICKS = [(0.0, "0"), (0.5, ".5"), (1.0, "1")]


def encode_csv(arr) -> str:
    return ",".join(f"{v:.4f}" for v in arr)


def decode_csv(raw, n: int | None = None, default: float = 0.5) -> np.ndarray:
    """CSV of 0..1 values; resampled to *n* points when given."""
    try:
        pts = [float(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        pts = []
    if not pts:
        return np.full(n or 1, default, dtype=np.float32)
    src = np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)
    if n is None or len(src) == n:
        return src
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(src)), src).astype(np.float32)


def gaussian_smooth(arr: np.ndarray, sigma: int) -> np.ndarray:
    half = max(1, sigma * 2)
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma, 1)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, half, mode="edge")
    return np.clip(np.convolve(padded, kernel, mode="valid")[:len(arr)], 0, 1).astype(np.float32)


def staircase(values: np.ndarray, x0: float, x1: float) -> tuple[np.ndarray, np.ndarray]:
    """Step values as (x, y) arrays of a staircase spanning [x0, x1]."""
    n = len(values)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    edges = np.linspace(x0, x1, n + 1)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(values, 2)
    return x, y


class DrawablePlot(pg.PlotWidget):
    """PlotWidget that reports left-drag positions in data coordinates."""

    pointDrawn = pyqtSignal(float, float)
    strokeStarted = pyqtSignal()
    strokeEnded = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent=parent, background=theme.BG)
        self._pressing = False
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setYRange(0.0, 1.0, padding=0.02)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setLabel("left", title, color=theme.TEXT_DIM, size="8pt")
        self.getAxis("left").setWidth(44)
        for side in ("left", "bottom"):
            self.getAxis(side).setPen(pg.mkPen(theme.BORDER))
            self.getAxis(side).setTextPen(pg.mkPen(theme.TEXT_DIM))
        self.getAxis("left").setTicks([Y_TICKS])
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._pressing = True
            self.strokeStarted.emit()
            self._emit(ev.position())
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._pressing:
            self._emit(ev.position())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._pressing:
            self._pressing = False
            self.strokeEnded.emit()
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def _emit(self, pos):
        p = self.getViewBox().mapSceneToView(self.mapToScene(pos.toPoint()))
        self.pointDrawn.emit(float(p.x()), float(p.y()))

    @property
    def drawing(self) -> bool:
        return self._pressing


def lane_pen(col: int, dashed: bool = False, width: float = 2.0):
    style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
    return pg.mkPen(theme.LANE_COLORS[col], width=1.0 if dashed else width, style=style)
