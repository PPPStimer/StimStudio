"""Generator clip page: rendered preview plus stage navigation for Howl activities.

Many Howl activities run through internal stages. The page marks the stage
boundaries inside the clip and can snap the clip to one stage (trim + length),
step to the previous / next stage, and freeze the activity's state at the clip
into manual-mode parameters.
"""

from __future__ import annotations

import json

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.model import GeneratorSource
from ...core.pulse import FS
from ...core.render import manual_params_at, segment_stages
from .. import theme
from ..widgets.master_preview import TimeAxisItem
from .curves import Y_TICKS, lane_pen

#: How far into an activity stages are searched (s).
STAGE_HORIZON = 600.0


def _btn(text, tip, slot) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("secondary")
    b.setToolTip(tip)
    b.clicked.connect(slot)
    return b


class GeneratorPage(QWidget):
    #: (attribute changes, undo text) to apply to the clip
    editRequested = pyqtSignal(dict, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip = None
        self.stages: list[tuple[float, float]] = []
        self.snapshot: dict = {}
        self._stage_key = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)
        bar = QHBoxLayout()
        self.stage_lbl = QLabel()
        self.prev_btn = _btn("◀ Stage", "Snap the clip to the previous stage", lambda: self._step(-1))
        self.next_btn = _btn("Stage ▶", "Snap the clip to the next stage", lambda: self._step(+1))
        self.fit_btn = _btn("Fit stage", "Snap the clip to the stage it starts in",
                            lambda: self._step(0))
        self.full_btn = _btn("From start", "Remove the trim: play the activity from its start",
                             self._from_start)
        for w in (self.stage_lbl, self.prev_btn, self.fit_btn, self.next_btn, self.full_btn):
            bar.addWidget(w)
        bar.addSpacing(16)
        self.manual_lbl = QLabel()
        self.manual_lbl.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.manual_btn = _btn("Freeze as manual",
                               "Switch the activity to manual mode with its parameters at "
                               "the middle of the clip", self._freeze_manual)
        bar.addWidget(self.manual_lbl, 1)
        bar.addWidget(self.manual_btn)
        lay.addLayout(bar)

        self.plot_widget = pg.GraphicsLayoutWidget()
        lay.addWidget(self.plot_widget, 1)
        self.plots, self.curves = [], []
        for row, (label, cols) in enumerate((("Amp", (0, 1)), ("Freq", (2, 3)))):
            p = self.plot_widget.addPlot(row=row, col=0,
                                         axisItems={"bottom": TimeAxisItem(orientation="bottom")})
            p.setYRange(0, 1.02, padding=0)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("left", label, color=theme.TEXT_DIM, size="8pt")
            p.getAxis("left").setWidth(44)
            p.getAxis("left").setTicks([Y_TICKS])
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)
            if row:
                p.setXLink(self.plots[0])
            for col in cols:
                self.curves.append(p.plot(pen=lane_pen(col, width=1.2)))
            self.plots.append(p)
        self._markers: list[tuple[pg.PlotItem, pg.InfiniteLine]] = []

    # ── data ────────────────────────────────────────────────────────────────
    def show_clip(self, clip, rendered: np.ndarray) -> None:
        self.clip = clip
        t = clip.trim_s + np.arange(len(rendered)) / FS
        for col, curve in enumerate(self.curves):
            curve.setData(t, rendered[:, col], skipFiniteCheck=True)
        self.plots[0].setXRange(clip.trim_s, clip.trim_s + max(clip.length_s, 0.1), padding=0.0)
        self._update_stages()

    def _update_stages(self) -> None:
        clip = self.clip
        src = clip.source
        is_gen = isinstance(src, GeneratorSource)
        key = (src.type_id, json.dumps(src.params, sort_keys=True, default=str), src.seed) \
            if is_gen else None
        if key != self._stage_key:
            self._stage_key = key
            try:
                self.stages = segment_stages(src.type_id, src.params, src.seed,
                                             STAGE_HORIZON) if is_gen else []
            except Exception:  # noqa: BLE001 — a broken generator must not kill the panel
                self.stages = []
        for plot, line in self._markers:
            plot.removeItem(line)
        self._markers = []
        lo, hi = clip.trim_s, clip.trim_s + clip.length_s
        pen = pg.mkPen(theme.ACCENT, width=1.5, style=Qt.PenStyle.DashLine)
        for a, _b in self.stages[1:]:
            if lo < a < hi:
                for p in self.plots:
                    m = pg.InfiniteLine(pos=a, angle=90, movable=False, pen=pen)
                    p.addItem(m, ignoreBounds=True)
                    self._markers.append((p, m))

        has_stages = bool(self.stages)
        for w in (self.prev_btn, self.next_btn, self.fit_btn):
            w.setVisible(has_stages)
        self.full_btn.setVisible(is_gen and clip.trim_s > 0)
        if has_stages:
            idx = self._stage_index(lo)
            self.stage_lbl.setText(f"Stage {idx + 1} / {len(self.stages)}"
                                   f"{'+' if self.stages[-1][1] >= STAGE_HORIZON else ''}")
        else:
            self.stage_lbl.setText("No stages" if is_gen else "Pulse data")

        self.snapshot = {}
        if is_gen:
            try:
                self.snapshot = manual_params_at(src.type_id, src.params, src.seed,
                                                 lo + clip.length_s / 2)
            except Exception:  # noqa: BLE001
                self.snapshot = {}
        can_freeze = bool(self.snapshot.get("manual")) and not src.params.get("manual")
        self.manual_btn.setVisible(can_freeze)
        self.manual_lbl.setText(", ".join(f"{k}={_fmt(v)}" for k, v in self.snapshot.items()
                                          if k != "manual") if can_freeze else "")

    def _stage_index(self, t: float) -> int:
        for i, (a, b) in enumerate(self.stages):
            if a - 1e-6 <= t < b - 1e-6:
                return i
        return len(self.stages) - 1

    # ── actions ─────────────────────────────────────────────────────────────
    def _step(self, delta: int) -> None:
        if not self.stages or self.clip is None:
            return
        idx = max(0, min(len(self.stages) - 1, self._stage_index(self.clip.trim_s) + delta))
        a, b = self.stages[idx]
        self.editRequested.emit({"trim_s": a, "length_s": b - a,
                                 "fade_in_s": min(self.clip.fade_in_s, b - a),
                                 "fade_out_s": min(self.clip.fade_out_s, b - a)},
                                f"Clip to stage {idx + 1}")

    def _from_start(self) -> None:
        if self.clip is not None:
            self.editRequested.emit({"trim_s": 0.0}, "Clip from activity start")

    def _freeze_manual(self) -> None:
        clip = self.clip
        if clip is None or not self.snapshot.get("manual"):
            return
        src = clip.source.clone()
        src.params = {**src.params, **self.snapshot, "manual": True}
        self.editRequested.emit({"source": src}, "Freeze activity as manual")


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)
