"""Draw and step editors for the ``draw`` / ``steps`` generators.

Both show two plots (amplitude, frequency) with channel A and B curves, the
clip's rendered output as dashed lines, and shade the part of the source
hidden by the clip's trim. A stroke edits a local copy; releasing the mouse
emits ``committed`` once, which becomes one undoable command.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

from ...core.pulse import FS
from ...generators.steps import decode_step_vals
from .curves import DrawablePlot, decode_csv, encode_csv, gaussian_smooth, lane_pen, staircase

DRAW_POINTS = 128
SIDES = ("a", "b")


class _CurveEditor(QWidget):
    committed = pyqtSignal(dict)   # param key -> CSV string

    #: param key per (kind, side), e.g. ("amp", "a") -> "amp_a"
    KEY_FORMAT = "{kind}_{side}"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total_s = 8.0          # source span shown: trim + clip length
        self.trim_s = 0.0
        self.data: dict[str, np.ndarray] = {}
        self._last: dict[str, tuple[int, float] | None] = {"amp": None, "freq": None}
        self._stroke_dirty = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)
        self.toolbar = QHBoxLayout()
        self.toolbar.addWidget(QLabel("Draw on:"))
        self.side_group = QButtonGroup(self)
        for i, (label, tip) in enumerate((("A", "Channel A only"), ("B", "Channel B only"),
                                          ("A+B", "Both channels at once"))):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setObjectName("secondary")
            b.setMinimumWidth(44)
            b.setToolTip(tip)
            self.side_group.addButton(b, i)
            self.toolbar.addWidget(b)
        self.side_group.button(2).setChecked(True)
        self.toolbar.addSpacing(12)
        lay.addLayout(self.toolbar)

        self.plots = {"amp": DrawablePlot("Amplitude"), "freq": DrawablePlot("Frequency")}
        self.plots["freq"].setXLink(self.plots["amp"])
        self.lines, self.rendered, self.shades = {}, {}, {}
        for kind, plot in self.plots.items():
            lay.addWidget(plot, 1)
            base = 0 if kind == "amp" else 2
            for j, side in enumerate(SIDES):
                self.rendered[(kind, side)] = plot.plot([], [], pen=lane_pen(base + j, dashed=True))
                self.lines[(kind, side)] = plot.plot([], [], pen=lane_pen(base + j))
            shade = pg.LinearRegionItem(values=(0, 0), movable=False,
                                        brush=pg.mkBrush(0, 0, 0, 110))
            shade.setZValue(-5)
            for line in shade.lines:
                line.setPen(pg.mkPen(None))
            plot.addItem(shade)
            self.shades[kind] = shade
            plot.pointDrawn.connect(lambda x, y, k=kind: self._on_draw(k, x, y))
            plot.strokeStarted.connect(lambda k=kind: self._on_stroke_start(k))
            plot.strokeEnded.connect(lambda k=kind: self._on_stroke_end(k))

    # ── API ─────────────────────────────────────────────────────────────────
    @property
    def drawing(self) -> bool:
        return any(p.drawing for p in self.plots.values())

    def active_sides(self) -> tuple[str, ...]:
        return {0: ("a",), 1: ("b",), 2: ("a", "b")}[self.side_group.checkedId()]

    def key(self, kind: str, side: str) -> str:
        return self.KEY_FORMAT.format(kind=kind, side=side)

    def set_span(self, trim_s: float, length_s: float) -> None:
        self.trim_s = trim_s
        self.total_s = max(0.1, trim_s + length_s)
        self.plots["amp"].setXRange(0.0, self.total_s, padding=0.01)
        for shade in self.shades.values():
            shade.setRegion((0.0, trim_s))
            shade.setVisible(trim_s > 0)
        self.refresh_lines()

    def set_rendered(self, pulses: np.ndarray | None) -> None:
        """Dashed overlay: the clip's final output (after effects and fades)."""
        if pulses is None or not len(pulses):
            for line in self.rendered.values():
                line.setData([], [])
            return
        t = self.trim_s + np.arange(len(pulses)) / FS
        for (kind, side), line in self.rendered.items():
            col = (0 if kind == "amp" else 2) + SIDES.index(side)
            line.setData(t, pulses[:, col])

    def encode(self) -> dict:
        return {self.key(k, s): encode_csv(self.data[self.key(k, s)])
                for k in ("amp", "freq") for s in SIDES}

    # ── drawing ─────────────────────────────────────────────────────────────
    def _index(self, x: float) -> int:
        raise NotImplementedError

    def _on_stroke_start(self, kind: str) -> None:
        self._last[kind] = None
        self._stroke_dirty = False

    def _on_draw(self, kind: str, x: float, y: float) -> None:
        i, v = self._index(x), float(np.clip(y, 0.0, 1.0))
        last = self._last[kind]
        for side in self.active_sides():
            arr = self.data[self.key(kind, side)]
            if last is not None and abs(i - last[0]) > 1:
                (i0, v0), (i1, v1) = sorted([last, (i, v)])
                arr[i0:i1 + 1] = np.linspace(v0, v1, i1 - i0 + 1)
            else:
                arr[i] = v
        self._last[kind] = (i, v)
        self._stroke_dirty = True
        self.refresh_lines()

    def _on_stroke_end(self, kind: str) -> None:
        self._last[kind] = None
        if self._stroke_dirty:
            self._stroke_dirty = False
            self.committed.emit(self.encode())

    def _apply(self, fn) -> None:
        """Run a bulk operation on the active channels' curves and commit it."""
        for kind in ("amp", "freq"):
            for side in self.active_sides():
                k = self.key(kind, side)
                self.data[k] = fn(self.data[k], kind).astype(np.float32)
        self.refresh_lines()
        self.committed.emit(self.encode())

    def copy_a_to_b(self) -> None:
        for kind in ("amp", "freq"):
            self.data[self.key(kind, "b")] = self.data[self.key(kind, "a")].copy()
        self.refresh_lines()
        self.committed.emit(self.encode())

    def refresh_lines(self) -> None:
        raise NotImplementedError


class DrawEditor(_CurveEditor):
    """Free-hand curves for the ``draw`` generator (128 points per curve)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sigma = QSpinBox()
        self.sigma.setRange(1, 10)
        self.sigma.setValue(3)
        self.sigma.setPrefix("σ ")
        self.sigma.setToolTip("Smoothing strength")
        tools = (("Smooth", "Gaussian smoothing of the active channels",
                  lambda: self._apply(lambda a, _k: gaussian_smooth(a, self.sigma.value()))),
                 ("Flat 0.5", "Reset the active channels to 0.5",
                  lambda: self._apply(lambda a, _k: np.full_like(a, 0.5))),
                 ("Invert", "Mirror the active channels (v → 1 − v)",
                  lambda: self._apply(lambda a, _k: 1.0 - a)),
                 ("A → B", "Copy channel A to B", self.copy_a_to_b))
        self.toolbar.addWidget(self.sigma)
        for label, tip, fn in tools:
            b = QPushButton(label)
            b.setObjectName("secondary")
            b.setToolTip(tip)
            b.clicked.connect(fn)
            self.toolbar.addWidget(b)
        self.toolbar.addStretch(1)
        for k in ("amp", "freq"):
            for s in SIDES:
                self.data[self.key(k, s)] = np.full(DRAW_POINTS, 0.5, np.float32)

    def load(self, params: dict) -> None:
        for k in ("amp", "freq"):
            for s in SIDES:
                raw = params.get(self.key(k, s), "")
                self.data[self.key(k, s)] = (decode_csv(raw, DRAW_POINTS) if raw
                                             else np.full(DRAW_POINTS, 0.5, np.float32))
        self.refresh_lines()

    def _index(self, x: float) -> int:
        return int(np.clip(round(x / self.total_s * (DRAW_POINTS - 1)), 0, DRAW_POINTS - 1))

    def refresh_lines(self) -> None:
        x = np.linspace(0.0, self.total_s, DRAW_POINTS)
        for (kind, side), line in self.lines.items():
            line.setData(x, self.data[self.key(kind, side)])


class StepEditor(_CurveEditor):
    """Per-step values for the ``steps`` generator."""

    KEY_FORMAT = "{kind}_{side}_vals"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps = 8
        tools = (("Flat 0.5", "Reset the active channels to 0.5",
                  lambda: self._apply(lambda a, _k: np.full_like(a, 0.5))),
                 ("Random", "Random values for the active channels",
                  lambda: self._apply(lambda a, _k: np.random.default_rng().random(len(a)))),
                 ("Shift →", "Rotate the active channels one step right",
                  lambda: self._apply(lambda a, _k: np.roll(a, 1))),
                 ("A → B", "Copy channel A to B", self.copy_a_to_b))
        for label, tip, fn in tools:
            b = QPushButton(label)
            b.setObjectName("secondary")
            b.setToolTip(tip)
            b.clicked.connect(fn)
            self.toolbar.addWidget(b)
        self.toolbar.addStretch(1)
        self.load({})

    def load(self, params: dict) -> None:
        self.steps = max(1, int(params.get("steps", 8)))
        for k in ("amp", "freq"):
            for s in SIDES:
                self.data[self.key(k, s)] = decode_step_vals(params.get(self.key(k, s), ""),
                                                             self.steps)
        self.refresh_lines()

    def _index(self, x: float) -> int:
        return int(np.clip(x / self.total_s * self.steps, 0, self.steps - 1))

    def refresh_lines(self) -> None:
        for (kind, side), line in self.lines.items():
            line.setData(*staircase(self.data[self.key(kind, side)], 0.0, self.total_s))
