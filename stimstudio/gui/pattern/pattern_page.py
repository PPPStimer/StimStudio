"""PatternPage: edit a beat-marks or pattern clip in the pattern editor dock.

* the marks plot: ticks on the source time axis, height = velocity, over the
  rendered level. Double-click adds a mark (at the clicked velocity), drag
  moves (snapped), drag the tick top (or Shift-drag) sets velocity, Del
  deletes, right-click pins a pool entry. Wheel zooms, dragging empty space
  pans.
* settings (pattern clips): pick mode, sequence, seed, fit, overlap, velocity
  depth; quantize; the pool with weights.

Edits are applied to a working copy of the source and committed as one
``{"source": …}`` edit (``editRequested``), so every gesture is one undo step.
"""

from __future__ import annotations

import random
import statistics

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMenu, QPushButton, QScrollArea, QSpinBox, QTableWidget,
                             QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from ...core.beats import level_signal, quantize_marks
from ...core.joins import OVERLAP_LABELS, OVERLAP_MODES
from ...core.model import (Clip, GeneratorSource, Mark, PATTERN_ANCHORS, PATTERN_FITS,
                           PATTERN_MODES, PatternSource, PoolEntry)
from ...core.pattern import (ANCHOR_LABELS, FIT_LABELS, MODE_LABELS, next_key, resolve,
                             sequence_errors)
from ...core.pulse import FS
from .. import theme
from ..song.items import key_color
from .curves import Y_TICKS

#: grid choices in beats
GRIDS = (("1 beat", 1.0), ("1/2", 0.5), ("1/3", 1 / 3), ("1/4", 0.25), ("1/6", 1 / 6),
         ("1/8", 0.125), ("1/16", 0.0625))
HIT_PX = 6
MIN_ENTRY_S = 0.05


class MarksPlot(pg.PlotWidget):
    """The marks of the page's working copy, editable with the mouse."""

    committed = pyqtSignal(str)                  # undo text
    selectionChanged = pyqtSignal()
    contextRequested = pyqtSignal(object)        # global position

    def __init__(self, page, parent=None):
        super().__init__(parent=parent, background=theme.BG)
        self.page = page
        self.selected: set[str] = set()
        self._drag = None
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setMouseEnabled(x=True, y=False)
        self.setYRange(0.0, 1.05, padding=0)
        self.showGrid(x=True, y=True, alpha=0.12)
        self.getAxis("left").setTicks([Y_TICKS])
        self.getAxis("left").setWidth(32)
        for side in ("left", "bottom"):
            self.getAxis(side).setPen(pg.mkPen(theme.BORDER))
            self.getAxis(side).setTextPen(pg.mkPen(theme.TEXT_DIM))
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.preview = self.plot(pen=pg.mkPen(QColor(theme.TEXT_DIM), width=1.0),
                                 fillLevel=0.0, brush=pg.mkBrush(128, 128, 128, 40))
        dim = pg.mkBrush(0, 0, 0, 90)
        self.outside = [pg.LinearRegionItem(movable=False, brush=dim, pen=pg.mkPen(None))
                        for _ in range(2)]
        for r in self.outside:
            r.setZValue(-5)
            self.addItem(r)
        self.groups: dict[str, pg.PlotDataItem] = {}
        self.sel_item = self.plot(pen=pg.mkPen("#ffffff", width=3.0), connect="pairs")
        self.sel_item.setZValue(5)
        self.labels: list[pg.TextItem] = []

    # ── drawing ─────────────────────────────────────────────────────────────
    def redraw(self) -> None:
        page = self.page
        by_group: dict[str, tuple[list, list]] = {}
        sx, sy = [], []
        for mark, key in page.resolved():
            g = (key or "-") if page.is_pattern else "*"
            xs, ys = by_group.setdefault(g, ([], []))
            xs += [mark.t_s, mark.t_s]
            ys += [0.0, mark.velocity]
            if mark.id in self.selected:
                sx += [mark.t_s, mark.t_s]
                sy += [0.0, mark.velocity]
        for g in [g for g in self.groups if g not in by_group]:
            self.removeItem(self.groups.pop(g))
        for g, (xs, ys) in by_group.items():
            item = self.groups.get(g)
            if item is None:
                color = QColor("#ffffff") if g == "*" else key_color(None if g == "-" else g)
                style = Qt.PenStyle.DotLine if g == "-" else Qt.PenStyle.SolidLine
                item = self.groups[g] = self.plot(pen=pg.mkPen(color, width=1.6, style=style),
                                                  connect="pairs")
            item.setData(np.array(xs), np.array(ys))
        self.sel_item.setData(np.array(sx), np.array(sy))
        for lab in self.labels:
            self.removeItem(lab)
        self.labels = []
        if page.is_pattern:
            pairs = [(m, k) for m, k in page.resolved() if k]
            if len(pairs) <= 96:
                for m, k in pairs:
                    lab = pg.TextItem(k, color=key_color(k), anchor=(0.5, 1.0))
                    lab.setPos(m.t_s, min(1.04, m.velocity + 0.06))
                    self.addItem(lab)
                    self.labels.append(lab)

    def set_preview(self, t: np.ndarray, level: np.ndarray, span: tuple[float, float]) -> None:
        self.preview.setData(t, level)
        a, b = span
        far = max(b, max((m.t_s for m in self.page.marks), default=b)) + 3600.0
        self.outside[0].setRegion((-3600.0, a))
        self.outside[1].setRegion((b, far))

    # ── hit testing ─────────────────────────────────────────────────────────
    def _data_pos(self, pos):
        p = self.getViewBox().mapSceneToView(self.mapToScene(pos.toPoint()))
        return float(p.x()), float(p.y())

    def _px(self) -> tuple[float, float]:
        sx, sy = self.getViewBox().viewPixelSize()
        return float(sx), float(sy)

    def hit(self, pos):
        """``(mark, at_top)`` under *pos*, or ``(None, False)``."""
        x, y = self._data_pos(pos)
        sx, sy = self._px()
        best, best_d = None, HIT_PX * sx
        for m in self.page.marks:
            d = abs(m.t_s - x)
            if d <= best_d and y <= m.velocity + HIT_PX * sy:
                best, best_d = m, d
        if best is None:
            return None, False
        return best, abs(y - best.velocity) <= HIT_PX * sy

    # ── mouse / keys ────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
        self.setFocus()
        mark, top = self.hit(ev.position())
        ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if mark is None:
            if not ctrl and self.selected:
                self.selected.clear()
                self.redraw()
                self.selectionChanged.emit()
            super().mousePressEvent(ev)
            return
        if ctrl:
            self.selected ^= {mark.id}
        elif mark.id not in self.selected:
            self.selected = {mark.id}
        self.redraw()
        self.selectionChanged.emit()
        if mark.id in self.selected:
            vel = top or bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            x, y = self._data_pos(ev.position())
            self._drag = {"mode": "vel" if vel else "time", "x": x, "y": y, "moved": False,
                          "lead": mark.id,
                          "orig": {m.id: (m.t_s, m.velocity) for m in self.page.marks
                                   if m.id in self.selected}}
        ev.accept()

    def mouseMoveEvent(self, ev):
        d = self._drag
        if d is None:
            super().mouseMoveEvent(ev)
            return
        x, y = self._data_pos(ev.position())
        d["moved"] = True
        if d["mode"] == "time":
            lead_t = d["orig"][d["lead"]][0]
            shift = self.page.snap(lead_t + x - d["x"]) - lead_t
            shift = max(shift, -min(t for t, _v in d["orig"].values()))
            for m in self.page.marks:
                if m.id in d["orig"]:
                    m.t_s = d["orig"][m.id][0] + shift
        else:
            dv = y - d["y"]
            for m in self.page.marks:
                if m.id in d["orig"]:
                    m.velocity = float(np.clip(d["orig"][m.id][1] + dv, 0.0, 1.0))
        self.redraw()
        self.page.selection_values()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        d, self._drag = self._drag, None
        if d is None:
            super().mouseReleaseEvent(ev)
            return
        if d["moved"]:
            self.page.marks.sort(key=lambda m: m.t_s)
            self.committed.emit("Move marks" if d["mode"] == "time" else "Mark velocity")
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        mark, _top = self.hit(ev.position())
        if mark is not None:
            return
        x, y = self._data_pos(ev.position())
        new = Mark(max(0.0, self.page.snap(x)), float(np.clip(y, 0.05, 1.0)))
        self.page.marks.append(new)
        self.page.marks.sort(key=lambda m: m.t_s)
        self.selected = {new.id}
        self.committed.emit("Add mark")
        ev.accept()

    def event(self, ev):
        # Del / Ctrl+A are window shortcuts (delete clips, select all): while
        # the plot has focus they edit the marks instead
        if ev.type() == QEvent.Type.ShortcutOverride and (
                ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
                or (ev.key() == Qt.Key.Key_A
                    and ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
            ev.accept()
            return True
        return super().event(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected:
            self.delete_selected()
        elif (ev.key() == Qt.Key.Key_A
              and ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.selected = {m.id for m in self.page.marks}
            self.redraw()
            self.selectionChanged.emit()
        else:
            super().keyPressEvent(ev)

    def delete_selected(self) -> None:
        self.page.marks[:] = [m for m in self.page.marks if m.id not in self.selected]
        self.selected.clear()
        self.committed.emit("Delete marks")

    def contextMenuEvent(self, ev):
        mark, _top = self.hit(ev.position() if hasattr(ev, "position") else ev.pos())
        if mark is not None and mark.id not in self.selected:
            self.selected = {mark.id}
            self.redraw()
            self.selectionChanged.emit()
        self.contextRequested.emit(ev.globalPos())


class _PoolTable(QTableWidget):
    """Pool entries; generators and presets can be dropped from the browser."""

    dropped = pyqtSignal(str, str)       # mime kind ("generator" / "preset"), payload

    def __init__(self):
        super().__init__(0, 4)
        self.setHorizontalHeaderLabels(["Key", "Clip", "Length", "Weight"])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(False)

    def _mime(self, mime):
        from ..song.view import GENERATOR_MIME, PRESET_MIME
        for kind, fmt in (("generator", GENERATOR_MIME), ("preset", PRESET_MIME)):
            if mime.hasFormat(fmt):
                return kind, bytes(mime.data(fmt)).decode("utf-8")
        return None

    def dragEnterEvent(self, ev):
        if self._mime(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):
        if self._mime(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):
        found = self._mime(ev.mimeData())
        if found:
            self.dropped.emit(*found)
            ev.acceptProposedAction()
        else:
            super().dropEvent(ev)


def _combo(items) -> QComboBox:
    c = QComboBox()
    for data, label in items:
        c.addItem(label, data)
    return c


def _dspin(lo, hi, step, decimals, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class PatternPage(QWidget):
    #: (attribute changes, undo text, merge key) for the clip
    editRequested = pyqtSignal(dict, str, object)
    convertRequested = pyqtSignal()
    openEntryRequested = pyqtSignal(str)          # pool key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip: Clip | None = None
        self.src = None              # working copy of the clip source
        self.domain = "pulse"
        self.bpm = 120.0
        self.offset_s = 0.0          # timeline time of source time 0
        self._resolved = None
        self._loading = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        left = QVBoxLayout()
        bar = QHBoxLayout()
        self.info = QLabel()
        self.info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        bar.addWidget(self.info, 1)
        self.snap_on = QCheckBox("Snap")
        self.snap_on.setChecked(True)
        self.grid = _combo((v, f"Grid {lab}") for lab, v in GRIDS)
        self.grid.setCurrentIndex(3)
        self.q_strength = QSpinBox()
        self.q_strength.setRange(0, 100)
        self.q_strength.setValue(100)
        self.q_strength.setSuffix(" %")
        self.q_strength.setToolTip("Quantize strength")
        self.q_btn = QPushButton("Quantize")
        self.q_btn.setToolTip("Pull the selected marks (all when none is selected) to the grid")
        self.q_btn.clicked.connect(self._quantize)
        for w in (self.snap_on, self.grid, self.q_strength, self.q_btn):
            bar.addWidget(w)
        left.addLayout(bar)
        self.plot = MarksPlot(self)
        left.addWidget(self.plot, 1)
        hint = QLabel("Double-click: add · drag: move · drag tick top / Shift: velocity · "
                      "Del: delete · right-click: pin · wheel: zoom")
        hint.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10px;")
        left.addWidget(hint)
        lay.addLayout(left, 1)

        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setFixedWidth(330)
        lay.addWidget(scroll)

        self.convert_btn = QPushButton("Convert to pattern")
        self.convert_btn.setToolTip("Turn these beat marks into a pattern that plays clips. "
                                    "Tip: select clips on the timeline together with the "
                                    "marks block and use its context menu to fill the pool.")
        self.convert_btn.clicked.connect(self.convertRequested.emit)
        side_lay.addWidget(self.convert_btn)

        sel = QGroupBox("Selected marks")
        sf = QFormLayout(sel)
        self.sel_vel = _dspin(0.0, 1.0, 0.05, 2)
        self.sel_vel.valueChanged.connect(self._set_velocity)
        sf.addRow("Velocity", self.sel_vel)
        self.sel_pin = QComboBox()
        self.sel_pin.activated.connect(self._pin_from_combo)
        self.pin_label = QLabel("Pin")
        sf.addRow(self.pin_label, self.sel_pin)
        self.sel_box = sel
        side_lay.addWidget(sel)

        pat = QGroupBox("Pattern")
        pf = QFormLayout(pat)
        self.mode = _combo((m, MODE_LABELS[m]) for m in PATTERN_MODES)
        self.mode.activated.connect(lambda _i: self._set("mode", self.mode.currentData(),
                                                         "Pattern mode"))
        pf.addRow("Pick", self.mode)
        self.sequence = QLineEdit()
        self.sequence.setPlaceholderText("e.g. A B A C  (- = rest)")
        self.sequence.editingFinished.connect(self._sequence_edited)
        self.sequence.textChanged.connect(self._check_sequence)
        self.seq_label = QLabel("Sequence")
        pf.addRow(self.seq_label, self.sequence)
        seed_row = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        self.seed.setKeyboardTracking(False)
        self.seed.valueChanged.connect(lambda v: self._set("seed", int(v), "Pattern seed",
                                                           "seed"))
        self.reseed = QToolButton()
        self.reseed.setText("🎲")
        self.reseed.setToolTip("New random seed")
        self.reseed.clicked.connect(lambda: self.seed.setValue(random.randrange(2 ** 31)))
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(self.reseed)
        self.seed_label = QLabel("Seed")
        pf.addRow(self.seed_label, seed_row)
        self.anchor = _combo((a, ANCHOR_LABELS[a]) for a in PATTERN_ANCHORS)
        self.anchor.setToolTip("Which point of a clip lands on its mark: the start (the hit "
                               "begins there), the middle, or the end (the clip leads up "
                               "to the mark)")
        self.anchor.activated.connect(lambda _i: self._set("anchor", self.anchor.currentData(),
                                                           "Pattern anchor"))
        pf.addRow("Anchor", self.anchor)
        self.fit = _combo((f, FIT_LABELS[f]) for f in PATTERN_FITS)
        self.fit.setToolTip("How long a clip plays. Every mark owns the time around it "
                            "(where it lies depends on the anchor): cut trims a clip to it, "
                            "loop / stretch fill it, own length ignores it")
        self.fit.activated.connect(lambda _i: self._set("fit", self.fit.currentData(),
                                                        "Pattern fit"))
        pf.addRow("Length", self.fit)
        self.overlap = _combo((m, OVERLAP_LABELS[m]) for m in OVERLAP_MODES)
        self.overlap.activated.connect(lambda _i: self._set("overlap",
                                                            self.overlap.currentData(),
                                                            "Pattern overlap"))
        pf.addRow("Overlap", self.overlap)
        self.join = _dspin(0.0, 2.0, 0.01, 3, " s")
        self.join.setToolTip("Fade at a cut and crossfade ramps between the clips")
        self.join.valueChanged.connect(lambda v: self._set("join_s", float(v), "Pattern fade",
                                                           "join"))
        pf.addRow("Cut fade", self.join)
        self.depth = _dspin(0.0, 1.0, 0.05, 2)
        self.depth.setToolTip("How much mark velocity scales the clip (0: not at all)")
        self.depth.valueChanged.connect(lambda v: self._set("velocity_depth", float(v),
                                                            "Velocity depth", "depth"))
        pf.addRow("Velocity depth", self.depth)
        self.pattern_box = pat
        side_lay.addWidget(pat)

        pool = QGroupBox("Pool")
        pl = QVBoxLayout(pool)
        self.table = _PoolTable()
        self.table.setMinimumHeight(120)
        self.table.dropped.connect(self._dropped)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_row(r))
        pl.addWidget(self.table)
        btns = QHBoxLayout()
        self.add_btn = QToolButton()
        self.add_btn.setText("+ Generator")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(self.add_btn)
        self.add_menu.aboutToShow.connect(self._fill_add_menu)
        self.add_btn.setMenu(self.add_menu)
        self.edit_btn = QPushButton("Edit ▸")
        self.edit_btn.setToolTip("Edit the clip of this entry (generator, curves, effects)")
        self.edit_btn.clicked.connect(lambda: self._open_row(self.table.currentRow()))
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_entry)
        for w in (self.add_btn, self.edit_btn, self.remove_btn):
            btns.addWidget(w)
        pl.addLayout(btns)
        self.pool_box = pool
        side_lay.addWidget(pool)
        side_lay.addStretch(1)

        self.plot.committed.connect(self.commit)
        self.plot.selectionChanged.connect(self.selection_values)
        self.plot.contextRequested.connect(self._context_menu)

    # ── data ────────────────────────────────────────────────────────────────
    @property
    def is_pattern(self) -> bool:
        return isinstance(self.src, PatternSource)

    @property
    def marks(self) -> list[Mark]:
        return self.src.marks if self.src is not None else []

    def resolved(self):
        """``(mark, pool key or None)`` for the working copy (marks by time)."""
        if self.is_pattern:
            return [(m, e.key if e else None) for m, e in resolve(self.src)]
        return [(m, None) for m in sorted(self.marks, key=lambda m: m.t_s)]

    def show_clip(self, clip: Clip, rendered: np.ndarray, domain: str = "pulse",
                  bpm: float = 120.0) -> None:
        new_clip = self.clip is None or self.clip.id != clip.id
        self.clip, self.domain, self.bpm = clip, domain, float(bpm)
        self.src = clip.source.clone()
        self.offset_s = clip.start_s - clip.trim_s
        ids = {m.id for m in self.src.marks}
        self.plot.selected &= ids
        if new_clip:
            self.plot.selected.clear()
        t = clip.trim_s + np.arange(len(rendered)) / FS
        self.plot.set_preview(t, level_signal(rendered, domain),
                              (clip.trim_s, clip.trim_s + clip.length_s))
        if new_clip:
            self.plot.setXRange(clip.trim_s, clip.trim_s + max(clip.length_s, 0.1), padding=0.02)
        self._load_widgets()
        self.plot.redraw()

    def _load_widgets(self) -> None:
        self._loading = True
        pattern = self.is_pattern
        for w in (self.pattern_box, self.pool_box, self.sel_pin, self.pin_label):
            w.setVisible(pattern)
        self.convert_btn.setVisible(not pattern)
        n_in = sum(1 for m in self.marks
                   if self.clip.trim_s <= m.t_s < self.clip.trim_s + self.clip.length_s)
        text = f"{len(self.marks)} marks ({n_in} in the clip)"
        if pattern:
            src = self.src
            text += f" · pool {len(src.pool)}"
            self.mode.setCurrentIndex(self.mode.findData(src.mode))
            if not self.sequence.hasFocus():
                self.sequence.setText(src.sequence)
            seq = src.mode == "sequence"
            self.sequence.setVisible(seq)
            self.seq_label.setVisible(seq)
            for w in (self.seed, self.reseed, self.seed_label):
                w.setVisible(not seq)
            self.seed.setValue(int(src.seed))
            self.anchor.setCurrentIndex(self.anchor.findData(src.anchor))
            self.fit.setCurrentIndex(self.fit.findData(src.fit))
            self.overlap.setCurrentIndex(self.overlap.findData(src.overlap))
            self.join.setValue(src.join_s)
            self.depth.setValue(src.velocity_depth)
            self._fill_table()
            self._check_sequence()
        self.info.setText(text)
        self._loading = False
        self.selection_values()

    def _fill_table(self) -> None:
        src = self.src
        row = self.table.currentRow()
        self.table.setRowCount(len(src.pool))
        counts: dict[str, int] = {}
        for _m, key in self.resolved():
            if key:
                counts[key] = counts.get(key, 0) + 1
        for r, e in enumerate(src.pool):
            key_item = QTableWidgetItem(e.key)
            key_item.setForeground(key_color(e.key))
            name = e.clip.name or getattr(e.clip.source, "type_id", "pulses")
            name_item = QTableWidgetItem(f"{name}  ({counts.get(e.key, 0)}×)")
            name_item.setToolTip(f"{counts.get(e.key, 0)} marks play this clip")
            self.table.setItem(r, 0, key_item)
            self.table.setItem(r, 1, name_item)
            self.table.setItem(r, 2, QTableWidgetItem(theme.format_time(e.clip.length_s)))
            w = self.table.cellWidget(r, 3)
            if not isinstance(w, QDoubleSpinBox):
                w = _dspin(0.0, 100.0, 0.5, 2)
                w.valueChanged.connect(self._weight_changed)
                self.table.setCellWidget(r, 3, w)
            w.blockSignals(True)
            w.setValue(e.weight)
            w.setProperty("key", e.key)
            w.setEnabled(src.mode == "random")
            w.blockSignals(False)
        if 0 <= row < len(src.pool):
            self.table.selectRow(row)

    def selection_values(self) -> None:
        sel = [m for m in self.marks if m.id in self.plot.selected]
        self.sel_box.setEnabled(bool(sel))
        self.sel_box.setTitle(f"Selected marks ({len(sel)})" if sel else "Selected marks")
        self.sel_vel.blockSignals(True)
        self.sel_vel.setValue(sel[0].velocity if sel else 1.0)
        self.sel_vel.blockSignals(False)
        if self.is_pattern:
            self.sel_pin.blockSignals(True)
            self.sel_pin.clear()
            self.sel_pin.addItem("— not pinned —", None)
            for e in self.src.pool:
                self.sel_pin.addItem(f"Always {e.key}", e.key)
            pins = {m.pin for m in sel}
            if len(pins) == 1:
                self.sel_pin.setCurrentIndex(max(0, self.sel_pin.findData(pins.pop())))
            self.sel_pin.blockSignals(False)

    # ── commits ─────────────────────────────────────────────────────────────
    def commit(self, text: str, merge_key=None) -> None:
        if self.src is None or self._loading:
            return
        src = self.src.clone()
        src.marks.sort(key=lambda m: m.t_s)
        self.editRequested.emit({"source": src}, text, merge_key)

    def _set(self, attr: str, value, text: str, merge_key=None) -> None:
        if self._loading or not self.is_pattern or getattr(self.src, attr) == value:
            return
        setattr(self.src, attr, value)
        self.commit(text, ("pattern", merge_key) if merge_key else None)

    def snap(self, t: float) -> float:
        if not self.snap_on.isChecked():
            return round(t * FS) / FS
        g = self.grid.currentData() * 60.0 / max(1.0, self.bpm)
        return round((t + self.offset_s) / g) * g - self.offset_s

    def _set_velocity(self, v: float) -> None:
        sel = [m for m in self.marks if m.id in self.plot.selected]
        if not sel:
            return
        for m in sel:
            m.velocity = float(v)
        self.commit("Mark velocity", ("pattern", "velocity"))

    def set_pin(self, key: str | None) -> None:
        sel = [m for m in self.marks if m.id in self.plot.selected]
        if not sel or not self.is_pattern:
            return
        for m in sel:
            m.pin = key
        self.commit("Pin marks" if key else "Unpin marks")

    def _pin_from_combo(self, _index: int) -> None:
        self.set_pin(self.sel_pin.currentData())

    def _quantize(self) -> None:
        if not self.marks:
            return
        g = self.grid.currentData() * 60.0 / max(1.0, self.bpm)
        sel = self.plot.selected
        chosen = [m for m in self.marks if m.id in sel] if sel else list(self.marks)
        rest = [m for m in self.marks if m not in chosen]
        done = quantize_marks(chosen, g, self.q_strength.value() / 100.0, self.offset_s)
        self.src.marks = sorted(rest + done, key=lambda m: m.t_s)
        self.plot.selected &= {m.id for m in self.src.marks}
        self.commit("Quantize marks")

    def _sequence_edited(self) -> None:
        self._set("sequence", self.sequence.text().strip(), "Pattern sequence")

    def _check_sequence(self, *_):
        bad = sequence_errors(self.sequence.text(), self.src.pool) if self.is_pattern else []
        self.sequence.setStyleSheet(f"color: {theme.DANGER};" if bad else "")
        self.sequence.setToolTip(f"Unknown keys: {' '.join(bad)}" if bad else
                                 "Pool keys separated by spaces; - is a rest. Repeats.")

    def _weight_changed(self, v: float) -> None:
        key = self.sender().property("key")
        entry = self.src.entry(key) if self.is_pattern else None
        if entry is not None and entry.weight != v:
            entry.weight = float(v)
            self.commit("Pool weight", ("weight", key))

    # ── pool ────────────────────────────────────────────────────────────────
    def default_entry_length(self) -> float:
        ts = sorted(m.t_s for m in self.marks)
        gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        return max(MIN_ENTRY_S, round(statistics.median(gaps), 3)) if gaps else 0.5

    def add_entry(self, clip: Clip) -> str | None:
        if not self.is_pattern:
            return None
        key = next_key(self.src.pool)
        clip = clip.clone(fresh_id=True)
        clip.start_s = 0.0
        self.src.pool.append(PoolEntry(key, clip))
        if self.src.mode == "sequence":
            self.src.sequence = (self.src.sequence + " " + key).strip()
        self.commit(f"Add {key} to pattern")
        return key

    def add_generator(self, type_id: str) -> str | None:
        from ...generators import get_generator
        gen = get_generator(type_id)
        if gen.domain != self.domain:
            return None
        return self.add_entry(Clip(GeneratorSource(type_id, gen.default_params()),
                                   length_s=self.default_entry_length(), name=gen.label))

    def _fill_add_menu(self) -> None:
        from ...generators import list_generators
        self.add_menu.clear()
        for g in list_generators():
            if g.domain == self.domain:
                a = QAction(g.label, self.add_menu)
                a.triggered.connect(lambda _c=False, t=g.type_id: self.add_generator(t))
                self.add_menu.addAction(a)

    def _dropped(self, kind: str, payload: str) -> None:
        if kind == "generator":
            self.add_generator(payload)
            return
        from ...io.presets import load_preset
        try:
            clip = load_preset(payload)
        except Exception:  # noqa: BLE001 — a broken preset just does not drop
            return
        type_id = getattr(clip.source, "type_id", None)
        if type_id is not None:
            from ...generators import get_generator
            if get_generator(type_id).domain != self.domain:
                return
        self.add_entry(clip)

    def _remove_entry(self) -> None:
        r = self.table.currentRow()
        if not self.is_pattern or not 0 <= r < len(self.src.pool):
            return
        key = self.src.pool.pop(r).key
        for m in self.src.marks:
            if m.pin == key:
                m.pin = None
        self.commit(f"Remove {key} from pattern")

    def _open_row(self, row: int) -> None:
        if self.is_pattern and 0 <= row < len(self.src.pool):
            self.openEntryRequested.emit(self.src.pool[row].key)

    def _context_menu(self, global_pos) -> None:
        menu = QMenu(self)
        sel = bool(self.plot.selected)
        if self.is_pattern and sel:
            for e in self.src.pool:
                a = QAction(f"Always play {e.key}  ({e.clip.name})", menu)
                a.triggered.connect(lambda _c=False, k=e.key: self.set_pin(k))
                menu.addAction(a)
            a = QAction("Unpin", menu)
            a.triggered.connect(lambda: self.set_pin(None))
            menu.addAction(a)
            menu.addSeparator()
        a = QAction("Delete marks", menu)
        a.setEnabled(sel)
        a.triggered.connect(self.plot.delete_selected)
        menu.addAction(a)
        a = QAction("Quantize", menu)
        a.triggered.connect(self._quantize)
        menu.addAction(a)
        menu.exec(global_pos)
