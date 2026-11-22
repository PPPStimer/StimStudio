"""Video analysis dialog: scene cuts → markers, motion → a stim clip.

The decoding pass runs on the thread pool behind a progress dialog
(``audio_import.run_job``); results are cached per file and settings, so moving
a slider only re-maps the cached signals.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

from ..core.pulse import FS
from ..io import video_analysis as va
from . import theme

_CACHE: "OrderedDict[tuple, va.VideoAnalysis]" = OrderedDict()
_CACHE_MAX = 4


def _cache_key(path, rate, roi, flow) -> tuple:
    path = os.path.abspath(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (path, mtime, float(rate), tuple(roi) if roi else None, bool(flow))


def cached_analysis(path, rate, roi, flow) -> va.VideoAnalysis | None:
    key = _cache_key(path, rate, roi, flow)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
    return hit


def store_analysis(path, rate, roi, flow, analysis) -> None:
    _CACHE[_cache_key(path, rate, roi, flow)] = analysis
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


class _Signals(QObject):
    progress = pyqtSignal(str, float)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class AnalysisJob(QRunnable):
    def __init__(self, path: str, rate: float, roi, flow: bool):
        super().__init__()
        self.setAutoDelete(False)
        self.path, self.rate, self.roi, self.flow = path, rate, roi, flow
        self.signals = _Signals()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = va.analyse_video(
                self.path, self.rate, self.roi, self.flow,
                progress=lambda f: self.signals.progress.emit("Reading frames", f),
                cancel=lambda: self._cancel)
        except va.Cancelled:
            self.signals.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit(result)


def analyse(parent, path, rate, roi, flow) -> va.VideoAnalysis | None:
    """Cached analysis, or run it behind a progress dialog (None if cancelled)."""
    hit = cached_analysis(path, rate, roi, flow)
    if hit is not None:
        return hit
    from .audio_import import run_job
    result = run_job(parent, AnalysisJob(path, rate, roi, flow),
                     os.path.basename(path), "Video analysis")
    if result is not None:
        store_analysis(path, rate, roi, flow, result)
    return result


class RoiPicker(QLabel):
    """A video frame where a rectangle (the region of interest) is dragged out."""

    roiChanged = pyqtSignal(object)          # (x, y, w, h) fractions or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi: tuple[float, float, float, float] | None = None
        self._anchor: QPointF | None = None
        self._image: QPixmap | None = None
        self.setMinimumSize(320, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #000;")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Drag a rectangle around the action — only motion inside it counts")

    def set_frame(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._image = None
            self.setText("No preview frame")
            return
        h, w, _ = rgb.shape
        img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._image = QPixmap.fromImage(img.copy())
        self.update()

    def _frame_rect(self) -> QRectF:
        if self._image is None:
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_frac(self, p: QPointF) -> QPointF:
        r = self._frame_rect()
        return QPointF(min(1.0, max(0.0, (p.x() - r.left()) / r.width())),
                       min(1.0, max(0.0, (p.y() - r.top()) / r.height())))

    def set_roi(self, roi) -> None:
        self.roi = tuple(roi) if roi else None
        self.update()
        self.roiChanged.emit(self.roi)

    def paintEvent(self, event):
        if self._image is None:
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000"))
        r = self._frame_rect()
        p.drawPixmap(r, self._image, QRectF(self._image.rect()))
        if self.roi is not None:
            x, y, w, h = self.roi
            box = QRectF(r.left() + x * r.width(), r.top() + y * r.height(),
                         w * r.width(), h * r.height())
            shade = QColor(0, 0, 0, 140)
            for part in (QRectF(r.left(), r.top(), r.width(), box.top() - r.top()),
                         QRectF(r.left(), box.bottom(), r.width(), r.bottom() - box.bottom()),
                         QRectF(r.left(), box.top(), box.left() - r.left(), box.height()),
                         QRectF(box.right(), box.top(), r.right() - box.right(), box.height())):
                p.fillRect(part, shade)
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.drawRect(box)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._anchor = self._to_frac(event.position())

    def mouseMoveEvent(self, event):
        if self._anchor is None:
            return
        b = self._to_frac(event.position())
        a = self._anchor
        x, y = min(a.x(), b.x()), min(a.y(), b.y())
        w, h = abs(a.x() - b.x()), abs(a.y() - b.y())
        if w > 0.02 and h > 0.02:
            self.roi = (round(x, 4), round(y, 4), round(w, 4), round(h, 4))
            self.update()

    def mouseReleaseEvent(self, event):
        if self._anchor is not None:
            self._anchor = None
            self.roiChanged.emit(self.roi)


class VideoAnalysisDialog(QDialog):
    """Pick what to make from the video; ``result_data()`` returns it after Accept."""

    def __init__(self, path: str, duration_s: float, parent=None, motion: bool = True,
                 cuts: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Video analysis — " + os.path.basename(path))
        self.path, self.duration_s = path, duration_s
        self.analysis: va.VideoAnalysis | None = None
        self._analysis_key = None
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.picker = RoiPicker()
        top.addWidget(self.picker, 3)
        side = QFormLayout()
        self.rate = QComboBox()
        for r in (10, 20, 30):
            self.rate.addItem(f"{r} / s", r)
        self.rate.setCurrentIndex(1)
        side.addRow("Analysis rate", self.rate)
        self.flow = QCheckBox("Optical flow (OpenCV)")
        self.flow.setEnabled(va.have_cv2())
        self.flow.setToolTip("More accurate direction of motion; needs opencv-python"
                             if va.have_cv2() else "Install opencv-python to enable")
        side.addRow(self.flow)
        reset = QPushButton("Whole frame")
        reset.clicked.connect(lambda: self.picker.set_roi(None))
        side.addRow("Region", reset)
        self.run_btn = QPushButton("Analyse")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.run_analysis)
        side.addRow(self.run_btn)
        self.info = QLabel("Drag a rectangle around the action, then Analyse.")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        side.addRow(self.info)
        sidew = QWidget()
        sidew.setLayout(side)
        top.addWidget(sidew, 2)
        lay.addLayout(top)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(150)
        self.plot.setBackground(theme.BG)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.amp_curve = self.plot.plot(pen=pg.mkPen(theme.ACCENT, width=1.5))
        self.freq_curve = self.plot.plot(pen=pg.mkPen("#c586c0", width=1))
        self.cut_lines: list[pg.InfiniteLine] = []
        lay.addWidget(self.plot)

        opts = QHBoxLayout()
        self.motion_box = QGroupBox("Motion → stim clip")
        self.motion_box.setCheckable(True)
        self.motion_box.setChecked(motion)
        mf = QFormLayout(self.motion_box)
        self.target = QComboBox()
        self.target.addItem("New track (A + B)", "standard")
        self.target.addItem("New triphase track (stroke position)", "triphase")
        mf.addRow("Target", self.target)
        self.smooth = self._dspin(0.0, 5.0, 0.3, 0.05, " s")
        mf.addRow("Smoothing", self.smooth)
        self.gamma = self._dspin(0.2, 4.0, 1.0, 0.1)
        mf.addRow("Curve (gamma)", self.gamma)
        rng = QHBoxLayout()
        self.floor = self._dspin(0.0, 1.0, 0.0, 0.05)
        self.ceiling = self._dspin(0.0, 1.0, 0.85, 0.05)
        rng.addWidget(self.floor)
        rng.addWidget(QLabel(".."))
        rng.addWidget(self.ceiling)
        mf.addRow("Amplitude", rng)
        self.freq_mode = QComboBox()
        self.freq_mode.addItem("Constant", "constant")
        self.freq_mode.addItem("Faster motion → higher", "speed")
        self.freq_mode.addItem("Upward motion → higher", "direction")
        mf.addRow("Frequency", self.freq_mode)
        self.freq = self._dspin(0.0, 1.0, 0.5, 0.05)
        mf.addRow("Constant freq (0..1)", self.freq)
        opts.addWidget(self.motion_box)

        self.cuts_box = QGroupBox("Scene cuts → markers")
        self.cuts_box.setCheckable(True)
        self.cuts_box.setChecked(cuts)
        cf = QFormLayout(self.cuts_box)
        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setValue(50)
        cf.addRow("Sensitivity", self.sensitivity)
        self.min_gap = self._dspin(0.2, 30.0, 1.0, 0.1, " s")
        cf.addRow("Min. gap", self.min_gap)
        self.cut_count = QLabel("—")
        cf.addRow("Found", self.cut_count)
        opts.addWidget(self.cuts_box)
        lay.addLayout(opts)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        for w in (self.smooth, self.gamma, self.floor, self.ceiling, self.freq, self.min_gap):
            w.valueChanged.connect(self.update_preview)
        for w in (self.freq_mode, self.target):
            w.currentIndexChanged.connect(self.update_preview)
        self.sensitivity.valueChanged.connect(self.update_preview)
        self.motion_box.toggled.connect(self._sync)
        self.cuts_box.toggled.connect(self._sync)
        self.picker.roiChanged.connect(lambda _r: self._sync())
        self.rate.currentIndexChanged.connect(lambda _i: self._sync())
        self.flow.toggled.connect(lambda _on: self._sync())
        self._load_frame()
        self.resize(820, 640)
        self._sync()

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

    def _load_frame(self) -> None:
        try:
            from ..io.video import iter_thumbnails
            t = max(0.0, self.duration_s * 0.1)
            frames = list(iter_thumbnails(self.path, [t], height=270))
            self.picker.set_frame(frames[0][1] if frames else None)
        except Exception:  # noqa: BLE001  (a preview frame is optional)
            self.picker.set_frame(None)

    # ── analysis ────────────────────────────────────────────────────────────
    def settings_key(self) -> tuple:
        return (self.rate.currentData(), self.picker.roi, self.flow.isChecked())

    def run_analysis(self) -> bool:
        rate, roi, flow = self.settings_key()
        try:
            result = analyse(self, self.path, rate, roi, flow)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Video analysis", str(exc))
            return False
        if result is None:
            return False
        self.analysis = result
        self._analysis_key = (rate, roi, flow)
        self._sync()
        return True

    def _sync(self) -> None:
        stale = self.analysis is None or self._analysis_key != self.settings_key()
        if not stale or cached_analysis(self.path, *self.settings_key()) is not None:
            if stale:
                self.analysis = cached_analysis(self.path, *self.settings_key())
                self._analysis_key = self.settings_key()
            stale = False
        self.run_btn.setText("Analyse" if stale else "Analysed ✓")
        any_output = self.motion_box.isChecked() or self.cuts_box.isChecked()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(any_output)
        self.update_preview()

    def cuts(self) -> list[float]:
        if self.analysis is None:
            return []
        return va.detect_cuts(self.analysis, self.sensitivity.value() / 100.0,
                              self.min_gap.value())

    def pulses(self) -> np.ndarray | None:
        if self.analysis is None:
            return None
        return va.motion_to_pulses(
            self.analysis, triphase=self.target.currentData() == "triphase",
            smooth_s=self.smooth.value(), gamma=self.gamma.value(),
            floor=min(self.floor.value(), self.ceiling.value()),
            ceiling=max(self.floor.value(), self.ceiling.value()),
            freq_mode=self.freq_mode.currentData(), freq=self.freq.value())

    def update_preview(self) -> None:
        for line in self.cut_lines:
            self.plot.removeItem(line)
        self.cut_lines = []
        if self.analysis is None:
            self.amp_curve.setData([], [])
            self.freq_curve.setData([], [])
            self.cut_count.setText("—")
            self.info.setText("Drag a rectangle around the action, then Analyse.")
            return
        self.info.setText(f"{self.analysis.n} samples, {self.analysis.duration_s:.1f} s "
                          f"({'optical flow' if self.analysis.method == 'flow' else 'phase'})")
        if self.motion_box.isChecked():
            p = self.pulses()
            step = max(1, len(p) // 4000)
            t = np.arange(0, len(p), step) / FS
            self.amp_curve.setData(t, p[::step, 0])
            self.freq_curve.setData(t, p[::step, 2])     # frequency, or triphase Y
        else:
            self.amp_curve.setData(self.analysis.times, self.analysis.cut_score)
            self.freq_curve.setData([], [])
        if self.cuts_box.isChecked():
            cuts = self.cuts()
            self.cut_count.setText(f"{len(cuts)} cut{'s' if len(cuts) != 1 else ''}")
            pen = pg.mkPen("#dcdcaa", width=1, style=Qt.PenStyle.DashLine)
            for c in cuts[:500]:
                line = pg.InfiniteLine(pos=c, angle=90, pen=pen)
                self.plot.addItem(line)
                self.cut_lines.append(line)
        else:
            self.cut_count.setText("—")

    def _accept(self) -> None:
        if (self.analysis is None or self._analysis_key != self.settings_key()) \
                and not self.run_analysis():
            return
        self.accept()

    def result_data(self) -> dict:
        """``{"pulses": (N,4) | None, "triphase": bool, "cuts": [s, …] | None}``."""
        return {"pulses": self.pulses() if self.motion_box.isChecked() else None,
                "triphase": self.target.currentData() == "triphase",
                "cuts": self.cuts() if self.cuts_box.isChecked() else None,
                "roi": self.picker.roi}
