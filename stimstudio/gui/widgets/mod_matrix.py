"""Modulation matrix: route audio-modulator signals to lanes or effect parameters.

One row per route: on/off, source (modulator track + signal), target track,
target (a lane, or an automatable effect parameter of that track), mode,
depth, smoothing and invert. Edits replace ``project.mod_routes`` through one
undoable command; consecutive depth / smoothing edits merge.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QLabel,
                             QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget)

from ...audio.analysis import SIGNAL_LABELS, SIGNALS
from ...core.model import MOD_MODES, ModRoute
from ...core.pulse import LANE_KEYS
from ...effects import get_effect
from .. import theme


def _spin(lo, hi, step, decimals=2, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def target_choices(track) -> list[tuple[str, str]]:
    """(label, target) pairs a route may aim at on *track*."""
    if track.type == "triphase":
        from ...core.triphase import TRI_LANE_NAMES
        names = TRI_LANE_NAMES
    else:
        names = theme.LANE_NAMES
    out = [(f"Lane {names[i]}", f"lane:{k}") for i, k in enumerate(LANE_KEYS)]
    for i, eff in enumerate(track.effects):
        try:
            d = get_effect(eff.get("type"))
        except KeyError:
            continue
        for key in d.automatable:
            out.append((f"FX{i + 1} {d.label}: {d.spec(key).label}", f"fx:{i}:{key}"))
    return out


class RouteRow(QFrame):
    def __init__(self, panel: "ModMatrixPanel", route_id: str):
        super().__init__()
        self.panel = panel
        self.route_id = route_id
        self.setObjectName("fxCard")
        g = QGridLayout(self)
        g.setContentsMargins(6, 4, 6, 6)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(3)
        self.enabled = QCheckBox()
        self.enabled.setToolTip("Enable this route")
        self.source = QComboBox()
        self.target_track = QComboBox()
        self.target = QComboBox()
        self.mode = QComboBox()
        self.mode.addItems(MOD_MODES)
        self.depth = _spin(-1.0, 1.0, 0.05)
        self.smooth = _spin(0.0, 5.0, 0.05, 2, " s")
        self.invert = QCheckBox("invert")
        rm = QToolButton()
        rm.setText("✕")
        rm.setToolTip("Delete route")
        rm.clicked.connect(lambda: panel.remove(route_id))
        for w in (self.source, self.target_track, self.target, self.mode):
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        g.addWidget(self.enabled, 0, 0)
        g.addWidget(self.source, 0, 1, 1, 3)
        g.addWidget(rm, 0, 4)
        g.addWidget(QLabel("→"), 1, 0)
        g.addWidget(self.target_track, 1, 1)
        g.addWidget(self.target, 1, 2, 1, 3)
        g.addWidget(self.mode, 2, 1)
        g.addWidget(QLabel("depth"), 2, 2, Qt.AlignmentFlag.AlignRight)
        g.addWidget(self.depth, 2, 3, 1, 2)
        g.addWidget(self.invert, 3, 1)
        g.addWidget(QLabel("smooth"), 3, 2, Qt.AlignmentFlag.AlignRight)
        g.addWidget(self.smooth, 3, 3, 1, 2)
        g.setColumnStretch(1, 1)
        g.setColumnStretch(2, 1)

        self.enabled.toggled.connect(lambda on: panel.edit(route_id, enabled=on))
        self.source.activated.connect(self._source_changed)
        self.target_track.activated.connect(self._target_track_changed)
        self.target.activated.connect(
            lambda _i: panel.edit(route_id, target=self.target.currentData()))
        self.mode.activated.connect(lambda _i: panel.edit(route_id, mode=self.mode.currentText()))
        self.depth.valueChanged.connect(
            lambda v: panel.edit(route_id, merge=("depth", route_id), depth=v))
        self.smooth.valueChanged.connect(
            lambda v: panel.edit(route_id, merge=("smooth", route_id), smooth_s=v))
        self.invert.toggled.connect(lambda on: panel.edit(route_id, invert=on))

    def sync(self, route: ModRoute, project) -> None:
        widgets = (self.enabled, self.source, self.target_track, self.target, self.mode,
                   self.depth, self.smooth, self.invert)
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(route.enabled)
        self.source.clear()
        for t in project.tracks:
            if t.type == "audio_mod":
                for sig in SIGNALS:
                    self.source.addItem(f"♪ {t.name}: {SIGNAL_LABELS[sig]}", f"{t.id}|{sig}")
        self.source.setCurrentIndex(
            max(0, self.source.findData(f"{route.source_track}|{route.signal}")))
        self.target_track.clear()
        target_track = None
        for t in project.tracks:
            if t.type in ("standard", "triphase"):
                self.target_track.addItem(t.name, t.id)
                if t.id == route.target_track:
                    target_track = t
        self.target_track.setCurrentIndex(max(0, self.target_track.findData(route.target_track)))
        self.target.clear()
        if target_track is not None:
            for label, data in target_choices(target_track):
                self.target.addItem(label, data)
            idx = self.target.findData(route.target)
            if idx < 0:
                self.target.addItem(f"(missing) {route.target}", route.target)
                idx = self.target.count() - 1
            self.target.setCurrentIndex(idx)
        self.mode.setCurrentText(route.mode)
        if not self.depth.hasFocus():
            self.depth.setValue(route.depth)
        if not self.smooth.hasFocus():
            self.smooth.setValue(route.smooth_s)
        self.invert.setChecked(route.invert)
        for w in widgets:
            w.blockSignals(False)

    def _source_changed(self, _i):
        data = self.source.currentData()   # "track id|signal" (QComboBox can't find tuples)
        if data:
            track_id, signal = data.split("|", 1)
            self.panel.edit(self.route_id, source_track=track_id, signal=signal)

    def _target_track_changed(self, _i):
        tid = self.target_track.currentData()
        if tid:
            self.panel.edit(self.route_id, target_track=tid, target="lane:amp_l")


class ModMatrixPanel(QScrollArea):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.rows: dict[str, RouteRow] = {}
        self.current_track_id: str | None = None
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        self.box = QVBoxLayout(body)
        self.box.setContentsMargins(6, 6, 6, 6)
        self.box.setSpacing(6)
        self.add_btn = QPushButton("+ Add route")
        self.add_btn.clicked.connect(self.add)
        self.box.addWidget(self.add_btn)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(6)
        self.box.addLayout(self.rows_box)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {theme.TEXT_DIM}; padding: 6px 2px;")
        self.box.addWidget(self.hint)
        self.box.addStretch(1)
        doc.modelChanged.connect(self.refresh)
        self.refresh()

    def set_current_track(self, track_id: str) -> None:
        self.current_track_id = track_id or None

    # ── edits ───────────────────────────────────────────────────────────────
    def _routes(self) -> list[ModRoute]:
        return [r.clone() for r in self.doc.project.mod_routes]

    def add(self) -> None:
        p = self.doc.project
        sources = [t for t in p.tracks if t.type == "audio_mod"]
        targets = [t for t in p.tracks if t.type in ("standard", "triphase")]
        if not sources or not targets:
            return
        target = next((t for t in targets if t.id == self.current_track_id), targets[0])
        routes = self._routes()
        routes.append(ModRoute(sources[0].id, "envelope", target.id, "lane:amp_l"))
        self.doc.set_project_attrs(text="Add modulation route", mod_routes=routes)

    def remove(self, route_id: str) -> None:
        routes = [r for r in self._routes() if r.id != route_id]
        self.doc.set_project_attrs(text="Delete modulation route", mod_routes=routes)

    def edit(self, route_id: str, merge=None, **changes) -> None:
        routes = self._routes()
        for r in routes:
            if r.id == route_id:
                for k, v in changes.items():
                    setattr(r, k, v)
        self.doc.set_project_attrs(merge_key=merge, text="Edit modulation", mod_routes=routes)

    # ── view ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        p = self.doc.project
        has_source = any(t.type == "audio_mod" for t in p.tracks)
        has_target = any(t.type in ("standard", "triphase") for t in p.tracks)
        self.add_btn.setEnabled(has_source and has_target)
        ids = [r.id for r in p.mod_routes]
        for rid in [r for r in self.rows if r not in ids]:
            self.rows.pop(rid).deleteLater()
        for route in p.mod_routes:
            row = self.rows.get(route.id)
            if row is None:
                row = self.rows[route.id] = RouteRow(self, route.id)
                self.rows_box.addWidget(row)
            row.sync(route, p)
        if not has_source:
            self.hint.setText("Import an audio file as a modulator (File → Import audio…) to "
                              "drive lanes and effect parameters from its loudness, beats and "
                              "frequency bands.")
        elif not p.mod_routes:
            self.hint.setText("Add a route: pick a modulator signal and what it drives.")
        else:
            self.hint.setText("multiply: x·(1−d+d·s) · add: x+d·s · replace: blend toward s")
