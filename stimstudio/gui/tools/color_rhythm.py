"""Color rhythm: how much of a target something fills → a pulse clip.

Drag out the target the shapes fly into (a ring, a box) on a frame, Analyse,
and Create: a clip whose amplitude follows the fill. What counts as "fill":

* **anything unlike the surroundings** (default) — zone pixels that differ
  from the background colour around the target, whatever their colour or
  shape; a background that changes from scene to scene does not count;
* **chosen colours** — the share of the colours ticked in the table of top
  colours (or clicked on the frame).

Only the inner part of the target is measured (*Measure inside*), so the
target's own outline stays out. Beat marks then come from the clip as from any
other (clip right-click → Beat marks from peaks…).

The decoding pass runs on the thread pool behind a progress dialog; results
are cached per file, rate and zone, so switching the measure, ticking colours
or moving a spin box only recombines cached per-sample data.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout)

from ...core.pulse import FS
from ...io import video_colors as vc
from .. import metrics, theme
from ..video_analysis import RoiPicker
from .base import ToolWindow

_CACHE: "OrderedDict[tuple, vc.ColorAnalysis]" = OrderedDict()
_CACHE_MAX = 4

RATES = (0, 15, 25, 30, 60)        # 0 = every frame of the video
PREVIEW_HEIGHT = 1080                # preview frames this tall (or the video's own height)
ZOOM_MAX = 32.0
SIGNALS = (("fill", "Fill"),
           ("rises", "Arrivals (fill rises)"),
           ("drops", "Vanishing (fill drops)"))
SIGNAL_NAMES = {"fill": "fill", "rises": "arrivals", "drops": "vanishing"}
TOP_N = 40
MEASURES = (("difference", "Anything unlike the surroundings"),
            ("colours", "Chosen colours"))
SHAPE_LABELS = (("ellipse", "Ellipse"), ("rect", "Rectangle"))
COLUMNS = (("", ""),
           ("Colour", "Average real colour of the palette bin and its name"),
           ("Screen %", "Average share of the measured zone the colour covers"),
           ("Present %", "Share of the time the colour is there (at least the threshold)"),
           ("Appearances", "How many times the colour showed up (absent → present)"),
           ("Peak %", "Largest share of the measured zone the colour ever covered"))
DIFF_COLOUR = "#4ec9b0"


def _cache_key(path, rate, roi, shape, inner) -> tuple:
    path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (path, mtime, float(rate), tuple(roi) if roi else None, shape, round(inner, 4))


def cached_analysis(path, rate, roi, shape="rect", inner=1.0) -> vc.ColorAnalysis | None:
    key = _cache_key(path, rate, roi, shape, inner)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
    return hit


def store_analysis(path, rate, roi, shape, inner, analysis) -> None:
    _CACHE[_cache_key(path, rate, roi, shape, inner)] = analysis
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


class _Signals(QObject):
    progress = pyqtSignal(str, float)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class ColorJob(QRunnable):
    def __init__(self, path: str, rate: float, roi, shape: str, inner: float):
        super().__init__()
        self.setAutoDelete(False)
        self.path, self.rate, self.roi, self.shape, self.inner = path, rate, roi, shape, inner
        self.signals = _Signals()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = vc.analyse_video_colors(
                self.path, self.rate, self.roi,
                progress=lambda f: self.signals.progress.emit("Reading frames", f),
                cancel=lambda: self._cancel, shape=self.shape, inner=self.inner)
        except vc.Cancelled:
            self.signals.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit(result)


def analyse(parent, path, rate, roi, shape="rect", inner=1.0) -> vc.ColorAnalysis | None:
    """Cached analysis, or run it behind a progress dialog (None if cancelled)."""
    hit = cached_analysis(path, rate, roi, shape, inner)
    if hit is not None:
        return hit
    from ..audio_import import run_job
    result = run_job(parent, ColorJob(path, rate, roi, shape, inner),
                     os.path.basename(path), "Color rhythm")
    if result is not None:
        store_analysis(path, rate, roi, shape, inner, result)
    return result


def _swatch(rgb) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(QColor(*rgb))
    return QIcon(pm)


class ColorPicker(RoiPicker):
    """The target picker: drag the target's outline; the measured zone (dashed)
    and the background band's outer edge (dotted) are drawn inside / around it.
    A click without dragging is an eyedropper. With a highlight mask, every
    pixel outside it is dimmed. The wheel zooms around the cursor, the right
    or middle button pans, ``fit()`` shows the whole frame again."""

    colorPicked = pyqtSignal(int, int)        # pixel (x, y) in the frame
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rgb: np.ndarray | None = None
        self.highlight: np.ndarray | None = None
        self.shape = "ellipse"
        self.inner = 0.7
        self._press = None
        self.zoom = 1.0
        self._view_c = QPointF(0.5, 0.5)       # frame point (fractions) at the widget centre
        self._pan = None
        self.setToolTip("Drag around the outer edge of the target · click a colour to pick it · "
                        "wheel zooms · right / middle drag pans")

    def set_frame(self, rgb) -> None:
        self.rgb = None if rgb is None else np.ascontiguousarray(rgb)
        self.highlight = None
        self._refresh()

    def set_highlight(self, mask) -> None:
        self.highlight = mask
        self._refresh()

    def set_zone(self, shape: str, inner: float) -> None:
        self.shape, self.inner = shape, float(inner)
        self.update()

    def _refresh(self) -> None:
        img = self.rgb
        if img is not None and self.highlight is not None \
                and self.highlight.shape == img.shape[:2]:
            img = np.where(self.highlight[..., None], img, (img * 0.22).astype(np.uint8))
        super().set_frame(img)

    # ── zoom / pan ──────────────────────────────────────────────────────────
    def _frame_rect(self) -> QRectF:
        base = super()._frame_rect()
        if self.zoom <= 1.0:
            return base
        w, h = base.width() * self.zoom, base.height() * self.zoom
        return QRectF(self.width() / 2 - self._view_c.x() * w,
                      self.height() / 2 - self._view_c.y() * h, w, h)

    def _clamp_view(self) -> None:
        base = super()._frame_rect()
        c = []
        for v, size, full in ((self._view_c.x(), self.width(), base.width()),
                              (self._view_c.y(), self.height(), base.height())):
            half = size / 2 / max(1e-6, full * self.zoom)
            c.append(0.5 if half >= 0.5 else min(1 - half, max(half, v)))
        self._view_c = QPointF(*c)

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        """Zoom to *zoom* (1 = whole frame) keeping the frame point under the
        widget position *anchor* (default: the centre) where it is."""
        anchor = QPointF(self.width() / 2, self.height() / 2) if anchor is None else anchor
        r = self._frame_rect()
        fx = (anchor.x() - r.left()) / max(1e-6, r.width())
        fy = (anchor.y() - r.top()) / max(1e-6, r.height())
        self.zoom = float(min(ZOOM_MAX, max(1.0, zoom)))
        base = super()._frame_rect()
        self._view_c = QPointF(fx - (anchor.x() - self.width() / 2) / (base.width() * self.zoom),
                               fy - (anchor.y() - self.height() / 2) / (base.height() * self.zoom))
        self._clamp_view()
        self.update()
        self.zoomChanged.emit(self.zoom)

    def fit(self) -> None:
        self.zoom, self._view_c = 1.0, QPointF(0.5, 0.5)
        self.update()
        self.zoomChanged.emit(self.zoom)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps and self._image is not None:
            self.set_zoom(self.zoom * 1.25 ** steps, QPointF(event.position()))
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_view()

    def _scaled(self, box: QRectF, k: float) -> QRectF:
        c = box.center()
        w, h = box.width() * k, box.height() * k
        return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)

    def _draw(self, p: QPainter, rect: QRectF) -> None:
        if self.shape == "ellipse":
            p.drawEllipse(rect)
        else:
            p.drawRect(rect)

    def paintEvent(self, event):
        if self._image is None:
            super(RoiPicker, self).paintEvent(event)            # QLabel: the "no frame" text
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#000"))
        r = self._frame_rect()
        p.drawPixmap(r, self._image, QRectF(self._image.rect()))
        if self.roi is not None:
            x, y, w, h = self.roi
            box = QRectF(r.left() + x * r.width(), r.top() + y * r.height(),
                         w * r.width(), h * r.height())
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(theme.TEXT_DIM), 1, Qt.PenStyle.DotLine))
            self._draw(p, self._scaled(box, vc.BAND[1]))
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            self._draw(p, box)
            p.setPen(QPen(QColor("#ffcc00"), 1.5, Qt.PenStyle.DashLine))
            self._draw(p, self._scaled(box, self.inner))
        p.end()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = (QPointF(event.position()), QPointF(self._view_c), event.button())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = QPointF(event.position())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            start, c0, _button = self._pan
            base = super()._frame_rect()
            d = event.position() - start
            self._view_c = QPointF(c0.x() - d.x() / (base.width() * self.zoom),
                                   c0.y() - d.y() / (base.height() * self.zoom))
            self._clamp_view()
            self.update()
            return
        if self._anchor is None:
            return
        b = self._to_frac(event.position())
        a = self._anchor
        # small targets are fine: at least 3 frame pixels (zoom in to draw them)
        iw = self._image.width() if self._image is not None else 100
        ih = self._image.height() if self._image is not None else 100
        w, h = abs(a.x() - b.x()), abs(a.y() - b.y())
        if w * iw >= 3 and h * ih >= 3:
            self.roi = (round(min(a.x(), b.x()), 5), round(min(a.y(), b.y()), 5),
                        round(w, 5), round(h, 5))
            self.update()

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            if event.button() == self._pan[2]:
                self._pan = None
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
        press, self._press = self._press, None
        if (press is not None and self.rgb is not None
                and (event.position() - press).manhattanLength() < 4):
            self._anchor = None                    # a click: eyedropper, keep the region
            f = self._to_frac(event.position())
            h, w = self.rgb.shape[:2]
            self.colorPicked.emit(int(min(w - 1, f.x() * w)), int(min(h - 1, f.y() * h)))
            return
        super().mouseReleaseEvent(event)


class ColorRhythmWindow(ToolWindow):
    title = "Color rhythm from video"

    def __init__(self, main_window):
        super().__init__(main_window)
        self.path: str | None = None
        self.duration_s = 0.0
        self.fps = 0.0
        self.from_project = True
        self.analysis: vc.ColorAnalysis | None = None
        self._analysis_key = None
        self._stats: list[vc.ColorStat] | None = None
        self.selected: set[int] = set()
        self._filling = False
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(120)
        self._frame_timer.timeout.connect(self._load_frame)
        self._build()
        metrics.fit_to_screen(self, 1180, 820)

    # ── layout ──────────────────────────────────────────────────────────────
    def _build(self) -> None:
        lay = QVBoxLayout(metrics.scroll_body(self))
        src = QHBoxLayout()
        src.addWidget(QLabel("Video:"))
        self.source_label = QLabel("—")
        self.source_label.setTextFormat(Qt.TextFormat.PlainText)
        src.addWidget(self.source_label, 1)
        self.project_btn = QPushButton("Project video")
        self.project_btn.setToolTip("Analyse the video on the project's video track; the clip "
                                    "lands lined up with it")
        self.project_btn.clicked.connect(self.use_project_video)
        src.addWidget(self.project_btn)
        self.file_btn = QPushButton("Open file…")
        self.file_btn.setToolTip("Analyse another video file; the clip lands at the edit cursor")
        self.file_btn.clicked.connect(lambda: self.open_file())
        src.addWidget(self.file_btn)
        lay.addLayout(src)

        top = QHBoxLayout()
        left = QVBoxLayout()
        self.picker = ColorPicker()
        self.picker.setMinimumSize(480, 270)
        left.addWidget(self.picker, 1)
        trow = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setToolTip("Previous frame")
        self.prev_btn.setAutoRepeat(True)
        self.prev_btn.setFixedWidth(32)
        self.prev_btn.clicked.connect(lambda: self.step_frame(-1))
        trow.addWidget(self.prev_btn)
        self.next_btn = QPushButton("▶")
        self.next_btn.setToolTip("Next frame")
        self.next_btn.setAutoRepeat(True)
        self.next_btn.setFixedWidth(32)
        self.next_btn.clicked.connect(lambda: self.step_frame(+1))
        trow.addWidget(self.next_btn)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setToolTip("Frame to look at (does not limit the analysis)")
        self.time_slider.valueChanged.connect(self._time_moved)
        trow.addWidget(self.time_slider, 1)
        self.time_label = QLabel(theme.format_time(0.0))
        trow.addWidget(self.time_label)
        left.addLayout(trow)
        brow = QHBoxLayout()
        whole = QPushButton("Whole frame")
        whole.setToolTip("Measure the whole frame (no target)")
        whole.clicked.connect(lambda: self.picker.set_roi(None))
        brow.addWidget(whole)
        fit = QPushButton("Fit")
        fit.setToolTip("Show the whole frame again (wheel zooms, right / middle drag pans)")
        fit.clicked.connect(self.picker.fit)
        brow.addWidget(fit)
        self.zoom_label = QLabel("×1")
        self.zoom_label.setMinimumWidth(36)
        self.picker.zoomChanged.connect(
            lambda z: self.zoom_label.setText(f"×{z:.1f}" if z < 10 else f"×{z:.0f}"))
        brow.addWidget(self.zoom_label)
        self.mask_check = QCheckBox("Show what counts")
        self.mask_check.setToolTip("Dim everything on the frame that the measure does not count")
        self.mask_check.setChecked(True)
        self.mask_check.toggled.connect(lambda _on: self._update_mask())
        brow.addWidget(self.mask_check)
        brow.addStretch(1)
        legend = QLabel("<span style='color:#007acc'>━</span> target  "
                        "<span style='color:#ffcc00'>┅</span> measured  "
                        "<span style='color:#858585'>┈</span> background from here")
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setWordWrap(True)
        brow.addWidget(legend)
        left.addLayout(brow)
        top.addLayout(left, 5)

        right = QVBoxLayout()
        tgt = QGroupBox("Target")
        form = QFormLayout(tgt)
        self.shape = QComboBox()
        for key, label in SHAPE_LABELS:
            self.shape.addItem(label, key)
        self.shape.setToolTip("Shape of the target: a ring or circle → Ellipse")
        form.addRow("Shape", self.shape)
        self.inner = QSpinBox()
        self.inner.setRange(20, 100)
        self.inner.setSingleStep(5)
        self.inner.setValue(70)
        self.inner.setSuffix(" %")
        self.inner.setToolTip("Measure only this inner part of the target (dashed), so its own "
                              "outline — the ring — does not count")
        form.addRow("Measure inside", self.inner)
        self.rate = QComboBox()
        for r in RATES:
            self.rate.addItem(f"{r} / s" if r else "All frames", r)
        self.rate.setCurrentIndex(0)
        self.rate.setToolTip("Frames measured per second — All frames for fast rhythms "
                             "(up to ~10 shapes a second needs 30–60 frames a second)")
        form.addRow("Analysis rate", self.rate)
        self.run_btn = QPushButton("Analyse")
        self.run_btn.clicked.connect(self.run_analysis)
        form.addRow(self.run_btn)
        right.addWidget(tgt)

        meas = QGroupBox("Fill = ")
        mform = QFormLayout(meas)
        self.measure = QComboBox()
        for key, label in MEASURES:
            self.measure.addItem(label, key)
        self.measure.setToolTip(
            "Anything unlike the surroundings: whatever flies in, any colour or shape "
            "(the background colour is taken around the target, every frame).\n"
            "Chosen colours: only the colours ticked below.")
        mform.addRow(self.measure)
        self.difference = QSpinBox()
        self.difference.setRange(vc.DIST_STEP, 200)
        self.difference.setSingleStep(vc.DIST_STEP)
        self.difference.setValue(vc.DEFAULT_DIFFERENCE)
        self.difference.setToolTip("How far (RGB distance) a pixel must be from the background "
                                   "colour to count — raise it against noise and soft glows")
        mform.addRow("Difference from", self.difference)
        self.presence = QDoubleSpinBox()
        self.presence.setRange(0.05, 50.0)
        self.presence.setDecimals(2)
        self.presence.setSingleStep(0.25)
        self.presence.setValue(0.5)
        self.presence.setSuffix(" %")
        self.presence.setToolTip("A colour counts as present when it covers at least this "
                                 "share of the measured zone")
        self.presence.valueChanged.connect(lambda _v: self._stats_changed())
        mform.addRow("Present from", self.presence)
        right.addWidget(meas)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        right.addWidget(self.info)
        self.table = QTableWidget(0, len(COLUMNS))
        for i, (name, tip) in enumerate(COLUMNS):
            item = QTableWidgetItem(name)
            item.setToolTip(tip)
            self.table.setHorizontalHeaderItem(i, item)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.SortOrder.DescendingOrder)
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemDoubleClicked.connect(self._row_double_clicked)
        right.addWidget(self.table, 1)
        top.addLayout(right, 4)
        lay.addLayout(top, 3)

        # drawn like the pattern editor: every point, peak downsampling, clip to view
        self.plot = pg.PlotWidget(background=theme.BG)
        self.plot.setMinimumHeight(140)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setYRange(0.0, 1.02, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.setClipToView(True)
        self.plot.addLegend(offset=(8, 8))
        self.fill_curve = self.plot.plot(pen=pg.mkPen(theme.TEXT_DIM, width=1),
                                         name="Measured fill (per analysed frame)")
        self.amp_curve = self.plot.plot(pen=pg.mkPen(theme.LANE_COLORS[0], width=1.4),
                                        fillLevel=0.0, brush=pg.mkBrush(78, 201, 176, 40),
                                        name="Clip amplitude (what Create clip makes)")
        self.frame_line = pg.InfiniteLine(angle=90, movable=False,
                                          pen=pg.mkPen(theme.DANGER, width=1))
        self.plot.addItem(self.frame_line)
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)
        lay.addWidget(self.plot, 2)

        out = QGroupBox("Clip")
        orow = QHBoxLayout(out)
        f1 = QFormLayout()
        self.smooth = self._dspin(0.0, 2.0, 0.0, 0.02, " s")
        f1.addRow("Smoothing", self.smooth)
        self.normalize = QCheckBox("Scale the maximum to 1")
        self.normalize.setChecked(True)
        f1.addRow(self.normalize)
        orow.addLayout(f1)
        f2 = QFormLayout()
        rng = QHBoxLayout()
        self.floor = self._dspin(0.0, 1.0, 0.0, 0.05)
        self.ceiling = self._dspin(0.0, 1.0, 1.0, 0.05)
        rng.addWidget(self.floor)
        rng.addWidget(QLabel(".."))
        rng.addWidget(self.ceiling)
        f2.addRow("Amplitude", rng)
        self.freq = self._dspin(0.0, 1.0, 0.5, 0.05)
        f2.addRow("Frequency (0..1)", self.freq)
        self.signal = QComboBox()
        for key, label in SIGNALS:
            self.signal.addItem(label, key)
        self.signal.setToolTip(
            "Fill: how full the target is.\n"
            "Arrivals: only while the fill grows (a shape coming in).\n"
            "Vanishing: one sharp spike when the fill drops (a shape hits and disappears) — "
            "best for dense rhythms, where the next shape is already on its way.")
        f1.addRow("Amplitude from", self.signal)
        orow.addLayout(f2)
        col = QVBoxLayout()
        self.separate = QCheckBox("One track per colour")
        self.separate.setToolTip("Otherwise one clip follows all chosen colours together")
        col.addWidget(self.separate)
        self.chosen_label = QLabel()
        self.chosen_label.setWordWrap(True)
        col.addWidget(self.chosen_label)
        orow.addLayout(col, 1)
        self.create_btn = QPushButton("Create clip")
        self.create_btn.setDefault(True)
        self.create_btn.clicked.connect(self.create_clip)
        orow.addWidget(self.create_btn)
        lay.addWidget(out)

        for w in (self.smooth, self.floor, self.ceiling, self.freq):
            w.valueChanged.connect(lambda _v: self.update_preview())
        self.normalize.toggled.connect(lambda _on: self.update_preview())
        self.signal.currentIndexChanged.connect(lambda _i: self.update_preview())
        self.difference.valueChanged.connect(lambda _v: self._measure_changed())
        self.measure.currentIndexChanged.connect(lambda _i: self._measure_changed())
        self.picker.roiChanged.connect(lambda _r: self._sync())
        self.picker.colorPicked.connect(self.pick_color)
        self.rate.currentIndexChanged.connect(lambda _i: self._sync())
        for combo in (self.measure, self.signal, self.rate):
            self._narrow(combo)
        self.shape.currentIndexChanged.connect(lambda _i: self._zone_changed())
        self.inner.valueChanged.connect(lambda _v: self._zone_changed())
        self._zone_changed()

    @staticmethod
    def _narrow(combo: QComboBox, chars: int = 14) -> None:
        """Let a combo box shrink below its longest item (the popup still shows it)."""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(chars)

    @staticmethod
    def _dspin(lo, hi, value, step, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(2)
        w.setValue(value)
        if suffix:
            w.setSuffix(suffix)
        return w

    # ── source ──────────────────────────────────────────────────────────────
    def showEvent(self, event):
        if self.from_project:
            self.use_project_video()
        super().showEvent(event)

    def _project_video_clip(self):
        vt = self.doc.project.video_track()
        return vt.clips[0] if vt is not None and vt.clips else None

    def use_project_video(self) -> None:
        clip = self._project_video_clip()
        if clip is None:
            self.set_source(None, 0.0, True)
            return
        start = min(max(0.0, self.doc.cursor_s - clip.start_s), clip.source.duration_s)
        self.set_source(clip.source.path, clip.source.duration_s, True, start)

    def open_file(self, path: str | None = None) -> None:
        from ...io.video import VIDEO_EXTENSIONS, probe_duration_av
        if path is None:
            exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(self, "Video to analyse",
                                                  self.main_window._last_dir(),
                                                  f"Video ({exts});;All files (*)")
            if not path:
                return
        try:
            duration = probe_duration_av(path)
        except (ImportError, OSError) as exc:
            QMessageBox.warning(self, self.title, f"Could not read the video:\n{exc}")
            return
        self.set_source(path, duration, False, duration * 0.1)

    def set_source(self, path, duration_s: float, from_project: bool,
                   frame_s: float | None = None) -> None:
        changed = path != self.path
        self.path, self.duration_s, self.from_project = path, float(duration_s), from_project
        if changed:
            self.fps = 0.0
            if path:
                try:
                    from ...io.video import probe_fps
                    self.fps = probe_fps(path)
                except Exception:  # noqa: BLE001  (no PyAV: "All frames" falls back to 30)
                    self.fps = 0.0
            self.rate.setItemText(0, f"All frames ({self.fps:g} / s)" if self.fps
                                  else "All frames")
        self.source_label.setText(
            (Path(path).name + (" (project video)" if from_project else "")) if path else
            "no video — open one in the project (File → Open video…) or Open file…")
        if changed:
            self.analysis, self._analysis_key, self._stats = None, None, None
            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, max(0, int(self.duration_s * 1000)))
            self.time_slider.blockSignals(False)
            if frame_s is not None:
                self.time_slider.setValue(int(frame_s * 1000))
            self._time_moved(self.time_slider.value())
            self._load_frame()
            self._fill_table()
        self._sync()

    def _time_moved(self, ms: int) -> None:
        t = ms / 1000.0
        self.time_label.setText(theme.format_time(t))
        self.frame_line.setValue(t)
        self._frame_timer.start()

    def _plot_clicked(self, event) -> None:
        """A click on the graph shows that moment's frame."""
        if event.button() != Qt.MouseButton.LeftButton or not self.path:
            return
        pos = self.plot.getViewBox().mapSceneToView(event.scenePos())
        self.time_slider.setValue(int(max(0.0, pos.x()) * 1000))

    def frame_time(self) -> float:
        return self.time_slider.value() / 1000.0

    def step_frame(self, n: int) -> None:
        dt = 1.0 / (self.fps or 30.0)
        frame = round(self.frame_time() / dt) + n
        self.time_slider.setValue(int(round(max(0.0, frame * dt) * 1000)))
        self._frame_timer.stop()
        self._load_frame()

    def _load_frame(self) -> None:
        if not self.path:
            self.picker.set_frame(None)
            return
        try:
            from ...io.video import grab_frame
            self.picker.set_frame(grab_frame(self.path, self.frame_time(), PREVIEW_HEIGHT))
        except Exception:  # noqa: BLE001  (a preview frame is optional)
            self.picker.set_frame(None)
        self._update_mask()

    # ── analysis ────────────────────────────────────────────────────────────
    def zone(self) -> vc.Zone:
        return vc.Zone(self.picker.roi, self.shape.currentData(), self.inner.value() / 100.0)

    def analysis_rate(self) -> float:
        """Frames per second to measure ("All frames" = the video's own rate)."""
        return float(self.rate.currentData() or round(self.fps, 3) or 30.0)

    def settings_key(self) -> tuple:
        z = self.zone()
        return (self.analysis_rate(), z.roi, z.shape, z.inner)

    def _zone_changed(self) -> None:
        z = self.zone()
        self.picker.set_zone(z.shape, z.inner)
        self._sync()

    def run_analysis(self) -> bool:
        from ...io.video import have_pyav
        if not self.path:
            return False
        if not have_pyav():
            QMessageBox.information(self, self.title, "Video analysis needs PyAV:  pip install av")
            return False
        key = self.settings_key()
        try:
            result = analyse(self, self.path, *key)
        except RuntimeError as exc:
            QMessageBox.warning(self, self.title, str(exc))
            return False
        if result is None:
            return False
        self._set_analysis(result, key)
        return True

    def _set_analysis(self, analysis, key) -> None:
        self.analysis, self._analysis_key, self._stats = analysis, key, None
        self._fill_table()
        self._sync()

    def _stale(self) -> bool:
        return self.analysis is None or self._analysis_key != self.settings_key()

    def _sync(self) -> None:
        if self.path and self._stale():
            hit = cached_analysis(self.path, *self.settings_key())
            if hit is not None:
                self._set_analysis(hit, self.settings_key())
                return
        stale = self._stale()
        self.run_btn.setEnabled(bool(self.path))
        self.run_btn.setText("Analyse" if stale else "Analysed ✓")
        if not self.path:
            self.info.setText("")
        elif self.analysis is None:
            self.info.setText("Drag around the outer edge of the target the shapes fly into, "
                              "then Analyse.")
        else:
            a = self.analysis
            w, h = a.roi_size
            self.info.setText(f"{a.n} samples · {theme.format_time(a.duration_s, 1)} · "
                              f"measured zone {w}×{h} px" +
                              (" · settings changed: Analyse again" if stale else ""))
        self._measure_changed()

    # ── measure ─────────────────────────────────────────────────────────────
    def measure_key(self) -> str:
        return self.measure.currentData()

    def set_measure(self, key: str) -> None:
        self.measure.setCurrentIndex(self.measure.findData(key))

    def _measure_changed(self) -> None:
        colours = self.measure_key() == "colours"
        self.difference.setEnabled(not colours)
        self.presence.setEnabled(colours)
        self.table.setEnabled(colours)
        self.separate.setEnabled(colours)
        n = len(self.selected)
        if colours:
            self.chosen_label.setText(f"{n} colour{'s' if n != 1 else ''} chosen" if n
                                      else "Tick colours in the table or click them on the frame")
        else:
            self.chosen_label.setText("Fill = anything unlike the surroundings")
        self.create_btn.setEnabled(bool(self.path) and (n > 0 or not colours))
        self._update_mask()
        self.update_preview()

    def _update_mask(self) -> None:
        rgb = self.picker.rgb
        mask = None
        if rgb is not None and self.mask_check.isChecked():
            if self.measure_key() == "colours":
                if self.selected:
                    mask = vc.bin_mask(rgb, self.selected)
            elif self.picker.roi is not None:
                mask = vc.difference_mask(rgb, self.zone(), self.difference.value())
        self.picker.set_highlight(mask)

    def fill(self, bins=None) -> np.ndarray | None:
        """The fill signal of the current measure (*bins* overrides the chosen
        colours), at the analysis rate."""
        a = self.analysis
        if a is None:
            return None
        if self.measure_key() == "difference":
            return vc.difference_signal(a, self.difference.value())
        bins = self.selected if bins is None else bins
        return vc.fill_signal(a, bins) if bins else None

    # ── colours ─────────────────────────────────────────────────────────────
    def stats(self) -> list[vc.ColorStat]:
        if self.analysis is None:
            return []
        if self._stats is None:
            self._stats = vc.color_stats(self.analysis, self.presence.value() / 100.0)
        return self._stats

    def _stats_changed(self) -> None:
        self._stats = None
        self._fill_table()

    def _fill_table(self) -> None:
        stats = self.stats()
        rows = stats[:TOP_N] + [s for s in stats[TOP_N:] if s.bin in self.selected]
        t = self.table
        self._filling = True
        t.setSortingEnabled(False)
        t.setRowCount(len(rows))
        for r, s in enumerate(rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                         | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if s.bin in self.selected
                              else Qt.CheckState.Unchecked)
            chk.setData(Qt.ItemDataRole.UserRole, s.bin)
            t.setItem(r, 0, chk)
            name = QTableWidgetItem(f"{s.hex}  {s.label}")
            name.setIcon(_swatch(s.rgb))
            name.setData(Qt.ItemDataRole.UserRole, s.bin)
            t.setItem(r, 1, name)
            for c, v in enumerate((round(s.screen_pct, 2), round(s.frames_pct, 1),
                                   s.appearances, round(s.peak_pct, 1)), start=2):
                item = QTableWidgetItem()
                item.setData(Qt.ItemDataRole.DisplayRole, v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t.setItem(r, c, item)
        t.setSortingEnabled(True)
        self._filling = False

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 0:
            return
        b = int(item.data(Qt.ItemDataRole.UserRole))
        if item.checkState() == Qt.CheckState.Checked:
            self.selected.add(b)
        else:
            self.selected.discard(b)
        self._measure_changed()

    def _row_double_clicked(self, item: QTableWidgetItem) -> None:
        chk = self.table.item(item.row(), 0)
        if chk is not None and item.column() != 0:
            chk.setCheckState(Qt.CheckState.Unchecked if chk.checkState() == Qt.CheckState.Checked
                              else Qt.CheckState.Checked)

    def set_selected(self, bins) -> None:
        self.selected = set(int(b) for b in bins)
        self._fill_table()
        self._measure_changed()

    def clear_selection(self) -> None:
        self.set_selected(())

    def pick_color(self, x: int, y: int) -> int | None:
        """Eyedropper: toggle the colour around frame pixel (x, y); picking a
        colour switches the measure to Chosen colours."""
        if self.picker.rgb is None:
            return None
        b = vc.dominant_bin(self.picker.rgb, x, y)
        if b in self.selected:
            self.selected.discard(b)
        else:
            self.selected.add(b)
        self._fill_table()
        if self.measure_key() != "colours":
            self.set_measure("colours")            # → _measure_changed
        else:
            self._measure_changed()
        return b

    # ── output ──────────────────────────────────────────────────────────────
    def pulses_for(self, bins=None) -> np.ndarray | None:
        a = self.analysis
        sig = self.fill(bins)
        if sig is None:
            return None
        return vc.fill_to_pulses(
            sig, a.rate, a.duration_s, smooth_s=self.smooth.value(),
            normalize=self.normalize.isChecked(),
            floor=min(self.floor.value(), self.ceiling.value()),
            ceiling=max(self.floor.value(), self.ceiling.value()), freq=self.freq.value(),
            mode=self.signal.currentData())

    def update_preview(self) -> None:
        sig = self.fill()
        if sig is None:
            self.fill_curve.setData([], [])
            self.amp_curve.setData([], [])
            return
        self.fill_curve.setData(self.analysis.times, sig)
        p = self.pulses_for()
        self.amp_curve.setData(np.arange(len(p)) / FS, p[:, 0], skipFiniteCheck=True)

    def _groups(self) -> list[list[int]]:
        share = self.analysis.frac.mean(axis=0) if self.analysis is not None else None
        order = sorted(self.selected, key=lambda b: -share[b] if share is not None else b)
        return [[b] for b in order] if self.separate.isChecked() else [order]

    def result_data(self) -> dict:
        """``{"path", "from_project", "roi", "measure", "clips": [{"name", "color",
        "bins", "pulses"}]}`` for ``MainWindow.add_color_rhythm``."""
        clips = []
        suffix = SIGNAL_NAMES[self.signal.currentData()]
        if self.measure_key() == "difference":
            clips.append({"name": f"Target {suffix}", "color": DIFF_COLOUR, "bins": [],
                          "pulses": self.pulses_for()})
        else:
            for bins in self._groups():
                if not bins:
                    continue
                rgb = tuple(int(c) for c in self.analysis.mean_rgb[bins[0]])
                colour = "#{:02x}{:02x}{:02x}".format(*rgb)
                if len(bins) == 1:
                    name = vc.bin_label(bins[0]).capitalize()
                elif len(bins) <= 3:
                    name = " + ".join(vc.bin_label(b) for b in bins).capitalize()
                else:
                    name = f"Colours ({len(bins)})"
                if suffix != "fill":
                    name += f" ({suffix})"
                clips.append({"name": name, "color": colour, "bins": list(bins),
                              "pulses": self.pulses_for(bins)})
        return {"path": self.path, "from_project": self.from_project,
                "roi": self.picker.roi, "measure": self.measure_key(),
                "signal": self.signal.currentData(), "clips": clips}

    def create_clip(self):
        if self.measure_key() == "colours" and not self.selected:
            return None
        if self._stale() and not self.run_analysis():
            return None
        return self.main_window.add_color_rhythm(self.result_data())
