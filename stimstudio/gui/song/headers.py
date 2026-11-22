"""Track header column: name, mute/solo, gain, blend mode and lane headers."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (QColorDialog, QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QMenu, QSlider, QToolButton, QVBoxLayout, QWidget)

from ...core.model import BLEND_MODES, ENVELOPE_INTERPS, ENVELOPE_MODES
from ...core.pulse import LANE_KEYS
from .. import theme
from .layout import CLIP_ROW_H, LANE_H

HEADER_W = 230
TYPE_BADGES = {"standard": "", "triphase": "3φ", "audio_mod": "♪", "video": "🎞"}
MODE_LABELS = {"multiply": "×", "offset": "±", "replace": "="}


def _lane_mute_tip(key: str, name: str) -> str:
    if key.startswith("amp"):
        return f"Mute {name}: this track's amplitude leaves the mix (its frequency still counts)"
    return (f"Mute {name}: this track stops steering the frequency; "
            "the other tracks' frequency is used")


def _tool(text: str, name: str, tip: str, checkable=True) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setObjectName(name)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return b


class TrackHeader(QWidget):
    """Header of one track plus, when expanded, its four lane headers."""

    selected = pyqtSignal(str)

    def __init__(self, editor, track_id: str, parent=None, track_type: str = "standard"):
        super().__init__(parent)
        self.editor = editor
        self.track_id = track_id
        self.track_type = track_type
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── main header ──
        self.main = QFrame()
        self.main.setObjectName("trackHeader")
        self.main.setFixedHeight(CLIP_ROW_H)
        grid = QVBoxLayout(self.main)
        grid.setContentsMargins(0, 3, 6, 3)
        grid.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(4)
        self.strip = QFrame()
        self.strip.setFixedWidth(4)
        self.expand = _tool("▶", "expand", "Show / hide the four sub-lanes", checkable=False)
        self.expand.clicked.connect(lambda: editor.toggle_expanded(track_id))
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._name_edited)
        self.badge = QLabel()
        self.badge.setStyleSheet(f"color: {theme.TEXT_DIM};")
        top.addWidget(self.strip)
        top.addWidget(self.expand)
        top.addWidget(self.name, 1)
        top.addWidget(self.badge)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(26, 0, 0, 0)
        bottom.setSpacing(4)
        self.mute = _tool("M", "mute", "Mute track")
        self.solo = _tool("S", "solo", "Solo track")
        self.mute.toggled.connect(lambda on: editor.doc.set_track_attrs(track_id, muted=on))
        self.solo.toggled.connect(lambda on: editor.doc.set_track_attrs(track_id, solo=on))
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 150)
        self.gain.setToolTip("Track gain (%)")
        self.gain.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.gain.valueChanged.connect(
            lambda v: editor.doc.set_track_attrs(track_id, merge_key="gain", gain=v / 100.0))
        self.blend = QComboBox()
        self.blend.addItems(BLEND_MODES)
        self.blend.setToolTip("How this track mixes into the master")
        self.blend.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.blend.currentTextChanged.connect(
            lambda text: editor.doc.set_track_attrs(track_id, blend=text))
        bottom.addWidget(self.mute)
        bottom.addWidget(self.solo)
        bottom.addWidget(self.gain, 1)
        bottom.addWidget(self.blend)
        grid.addLayout(top)
        grid.addLayout(bottom)
        outer.addWidget(self.main)

        # ── lane headers ──
        self.outer = outer
        self.lane_frames: list[QFrame] = []
        self.lane_mutes: list[QToolButton] = []
        self.env_buttons: list[QToolButton] = []
        self.env_modes: list[QComboBox] = []
        for i, key in enumerate(LANE_KEYS):
            f = QFrame()
            f.setObjectName("laneHeader")
            f.setFixedHeight(LANE_H)
            h = QHBoxLayout(f)
            h.setContentsMargins(30, 0, 6, 0)
            h.setSpacing(3)
            label = QLabel(theme.LANE_NAMES[i])
            label.setStyleSheet(f"color: {theme.LANE_COLORS[i]};")
            h.addWidget(label, 1)
            mode = QComboBox()
            mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            for m in ENVELOPE_MODES:
                mode.addItem(MODE_LABELS[m], m)
            mode.setToolTip("Envelope mode: × multiply · ± offset · = replace")
            mode.setFixedWidth(44)
            mode.currentIndexChanged.connect(
                lambda _i, k=key, w=mode: editor.set_lane_envelope_mode(track_id, k, w.currentData()))
            h.addWidget(mode)
            env = _tool("~", "env", "Automation envelope on this lane "
                        "(click the lane to add points, right-click a point to delete)")
            env.toggled.connect(lambda on, k=key: editor.toggle_lane_envelope(track_id, k, on))
            h.addWidget(env)
            mute = _tool("M", "mute", _lane_mute_tip(key, theme.LANE_NAMES[i]))
            mute.toggled.connect(lambda on, k=key: editor.set_lane_muted(track_id, k, on))
            policy = mute.sizePolicy()
            policy.setRetainSizeWhenHidden(True)    # keep the lane headers aligned
            mute.setSizePolicy(policy)
            h.addWidget(mute)
            f.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            f.customContextMenuRequested.connect(
                lambda pos, k=key, w=f: self._lane_menu(k, w.mapToGlobal(pos)))
            self.lane_mutes.append(mute)
            self.env_buttons.append(env)
            self.env_modes.append(mode)
            outer.addWidget(f)
            self.lane_frames.append(f)
        self.auto_frames: list[QFrame] = []
        self._auto_sig: tuple = ()
        self.signal_frames: list[QFrame] = []
        if track_type == "triphase":
            from ...core.triphase import TRI_LANE_NAMES
            from ..pattern.triphase_page import TRI_COLORS
            for f, name, color, key, mute in zip(self.lane_frames, TRI_LANE_NAMES, TRI_COLORS,
                                                 LANE_KEYS, self.lane_mutes):
                label = f.findChild(QLabel)
                label.setText(name)
                label.setStyleSheet(f"color: {color};")
                mute.setToolTip(_lane_mute_tip(key, name))
            for i in (1, 2):                # position X / Y can't be muted
                self.lane_mutes[i].hide()
            self.blend.setToolTip("How this track's A/B result mixes into the master")
        if track_type == "audio_mod":
            from ...audio.analysis import SIGNAL_LABELS, SIGNALS
            from .items import SIGNAL_COLORS
            for f in self.lane_frames:
                f.hide()
            self.lane_frames = []
            self.blend.hide()
            self.gain.setToolTip("Scales all modulation signals of this track (%)")
            for name, color in zip(SIGNALS, SIGNAL_COLORS):
                f = QFrame()
                f.setObjectName("laneHeader")
                f.setFixedHeight(LANE_H)
                h = QHBoxLayout(f)
                h.setContentsMargins(30, 0, 6, 0)
                label = QLabel(SIGNAL_LABELS[name])
                label.setStyleSheet(f"color: {color};")
                h.addWidget(label, 1)
                outer.addWidget(f)
                self.signal_frames.append(f)
        self.video_frames: list[QFrame] = []
        if track_type == "video":
            for f in self.lane_frames:
                f.hide()
            self.lane_frames = []
            for w in (self.solo, self.gain, self.blend):
                w.hide()
            self.thumbs_btn = _tool("▦", "frames", "Show / hide the frames on the video clip")
            self.thumbs_btn.toggled.connect(editor.set_video_thumbs)
            self.audio_btn = _tool("♫", "audio", "Show / hide the video's sound lane")
            self.audio_btn.toggled.connect(lambda on: editor.set_video_audio(track_id, on))
            bottom.insertWidget(1, self.thumbs_btn)
            bottom.insertWidget(2, self.audio_btn)
            bottom.addStretch(1)
            f = QFrame()
            f.setObjectName("laneHeader")
            f.setFixedHeight(LANE_H)
            h = QHBoxLayout(f)
            h.setContentsMargins(30, 0, 6, 0)
            from .items import SIGNAL_COLORS
            label = QLabel("Sound")
            label.setStyleSheet(f"color: {SIGNAL_COLORS[0]};")
            h.addWidget(label, 1)
            outer.addWidget(f)
            self.video_frames.append(f)
            policy = self.expand.sizePolicy()
            policy.setRetainSizeWhenHidden(True)        # keep the name aligned
            self.expand.setSizePolicy(policy)
            self.expand.hide()
            self.mute.setToolTip("Hide the video and silence its sound")

    # ── sync from model ─────────────────────────────────────────────────────
    def sync(self, track, expanded: bool, is_current: bool, auto_lanes=()) -> None:
        widgets = (self.name, self.mute, self.solo, self.gain, self.blend,
                   *self.lane_mutes, *self.env_buttons, *self.env_modes)
        for w in widgets:
            w.blockSignals(True)
        if not self.name.hasFocus():
            self.name.setText(track.name)
        self.mute.setChecked(track.muted)
        self.solo.setChecked(track.solo)
        if not self.gain.isSliderDown():
            self.gain.setValue(round(track.gain * 100))
        self.blend.setCurrentText(track.blend)
        for key, mute in zip(LANE_KEYS, self.lane_mutes):
            mute.setChecked(track.lanes[key].muted)
        for key, env_btn, mode in zip(LANE_KEYS, self.env_buttons, self.env_modes):
            env = track.lanes[key].envelope
            env_btn.setChecked(env is not None and env.enabled)
            mode.setVisible(env is not None)
            if env is not None:
                mode.setCurrentIndex(max(0, mode.findData(env.mode)))
        for w in widgets:
            w.blockSignals(False)
        self._sync_auto(auto_lanes, expanded and track.type in ("standard", "triphase"))

        self.strip.setStyleSheet(f"background: {track.color}; border-radius: 1px;")
        self.badge.setText(TYPE_BADGES.get(track.type, "?"))
        if track.type in ("standard", "triphase"):
            from ...core.joins import OVERLAP_LABELS
            joins = f"smoothed over {track.join_s:g} s" if track.smooth_joins else "not smoothed"
            mode = OVERLAP_LABELS.get(track.overlap, track.overlap)
            self.badge.setToolTip(f"Clip overlaps: {mode} · butt joints {joins}\n"
                                  "(right-click the header to change)")
        has_lanes = track.type in ("standard", "audio_mod", "triphase")
        self.expand.setText("▼" if expanded else "▶")
        self.expand.setEnabled(has_lanes)
        for f in self.lane_frames:
            f.setVisible(expanded and track.type in ("standard", "triphase"))
        for f in self.signal_frames:
            f.setVisible(expanded)
        for f in self.video_frames:
            f.setVisible(expanded)
        if track.type == "video":
            for btn, on in ((self.thumbs_btn, self.editor.video_thumbs),
                            (self.audio_btn, expanded)):
                btn.blockSignals(True)
                btn.setChecked(on)
                btn.blockSignals(False)
        self.main.setProperty("selected", "true" if is_current else "false")
        self.main.style().unpolish(self.main)
        self.main.style().polish(self.main)

    def _sync_auto(self, auto_lanes, visible: bool) -> None:
        sig = tuple((a.fx_index, a.key, a.label) for a in auto_lanes)
        if sig != self._auto_sig:
            for f in self.auto_frames:
                f.deleteLater()
            self.auto_frames = []
            for a in auto_lanes:
                f = QFrame()
                f.setObjectName("laneHeader")
                f.setFixedHeight(LANE_H)
                h = QHBoxLayout(f)
                h.setContentsMargins(30, 0, 6, 0)
                label = QLabel(a.label)
                label.setStyleSheet("color: #c586c0;")
                label.setToolTip(f"Automation of effect #{a.fx_index + 1}: {a.label}")
                h.addWidget(label, 1)
                rm = _tool("✕", "remove", "Remove this automation", checkable=False)
                rm.clicked.connect(lambda _c=False, i=a.fx_index, k=a.key:
                                   self.editor.remove_fx_automation(self.track_id, i, k))
                h.addWidget(rm)
                self.outer.addWidget(f)
                self.auto_frames.append(f)
            self._auto_sig = sig
        for f in self.auto_frames:
            f.setVisible(visible)

    def _lane_menu(self, key: str, global_pos) -> None:
        track = self.editor.project.track_by_id(self.track_id)
        env = track.lanes[key].envelope
        menu = QMenu(self)
        if env is None:
            a = QAction("Add envelope", menu)
            a.triggered.connect(lambda: self.editor.toggle_lane_envelope(self.track_id, key, True))
            menu.addAction(a)
        else:
            curve = menu.addMenu("Curve")
            for interp in ENVELOPE_INTERPS:
                a = QAction(interp, curve)
                a.setCheckable(True)
                a.setChecked(env.interp == interp)
                a.triggered.connect(lambda _c=False, i=interp:
                                    self.editor.set_lane_envelope_interp(self.track_id, key, i))
                curve.addAction(a)
            a = QAction("Delete envelope", menu)
            a.triggered.connect(lambda: self.editor.clear_lane_envelope(self.track_id, key))
            menu.addAction(a)
        menu.exec(global_pos)

    def _name_edited(self):
        text = self.name.text().strip()
        track = self.editor.project.track_by_id(self.track_id)
        if text and text != track.name:
            self.editor.doc.set_track_attrs(self.track_id, name=text)

    # ── mouse ───────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.editor.set_current_track(self.track_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        ed = self.editor
        tid = self.track_id
        self.editor.set_current_track(tid)
        menu = QMenu(self)

        def act(text, fn):
            a = QAction(text, menu)
            a.triggered.connect(fn)
            menu.addAction(a)

        act("Rename", lambda: (self.name.setFocus(), self.name.selectAll()))
        act("Color…", self._pick_color)
        act("Duplicate track", lambda: ed.doc.duplicate_track(tid))
        menu.addSeparator()
        if self.track_type in ("standard", "triphase"):
            self._join_menus(menu)
            menu.addSeparator()
        act("Move up", lambda: ed.move_track(tid, -1))
        act("Move down", lambda: ed.move_track(tid, +1))
        menu.addSeparator()
        act("Delete track", lambda: ed.doc.remove_track(tid))
        menu.exec(event.globalPos())

    def _join_menus(self, menu: QMenu) -> None:
        """"Clip overlaps" and "Smooth joins" submenus (see core.joins)."""
        from ...core.joins import JOIN_CHOICES_S, OVERLAP_LABELS, OVERLAP_MODES
        track = self.editor.project.track_by_id(self.track_id)
        doc, tid = self.editor.doc, self.track_id
        sub = menu.addMenu("Clip overlaps")
        sub.setToolTipsVisible(True)
        tips = {"crossfade": "The later clip lies on top; overlaps become crossfades",
                "mix": "Overlapping clips add up (clamped at full level)",
                "max": "The louder clip wins"}
        for mode in OVERLAP_MODES:
            a = sub.addAction(OVERLAP_LABELS[mode])
            a.setCheckable(True)
            a.setChecked(track.overlap == mode)
            a.setToolTip(tips[mode])
            a.triggered.connect(lambda _c=False, m=mode: doc.set_track_attrs(
                tid, text="Clip overlaps", overlap=m))
        sub = menu.addMenu("Smooth joins")
        on = sub.addAction("Smooth butt joints")
        on.setCheckable(True)
        on.setChecked(track.smooth_joins)
        on.triggered.connect(lambda checked: doc.set_track_attrs(
            tid, text="Smooth joins", smooth_joins=checked))
        sub.addSeparator()
        label = sub.addAction("Join length (also nested-clip ramps):")
        label.setEnabled(False)
        for secs in JOIN_CHOICES_S:
            a = sub.addAction(f"{secs:g} s")
            a.setCheckable(True)
            a.setChecked(abs(track.join_s - secs) < 1e-6)
            a.triggered.connect(lambda _c=False, v=secs: doc.set_track_attrs(
                tid, text="Join length", join_s=v))

    def _pick_color(self):
        track = self.editor.project.track_by_id(self.track_id)
        color = QColorDialog.getColor(QColor(track.color), self, "Track color")
        if color.isValid():
            self.editor.doc.set_track_attrs(self.track_id, color=color.name())


class HeaderPanel(QWidget):
    """Absolutely positioned headers that follow the song view's vertical scroll."""

    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.offset = 0.0
        self.headers: dict[str, TrackHeader] = {}
        self.setFixedWidth(HEADER_W)
        self.setAutoFillBackground(True)

    def set_offset(self, y: float) -> None:
        self.offset = y
        self._place()

    def sync(self, project, layout, expanded: set[str], current: str | None) -> None:
        ids = {t.id for t in project.tracks}
        for tid in list(self.headers):
            if tid not in ids:
                self.headers.pop(tid).deleteLater()
        for track in project.tracks:
            h = self.headers.get(track.id)
            if h is None:
                h = self.headers[track.id] = TrackHeader(self.editor, track.id, self, track.type)
                h.show()
            row = layout.row(track.id)
            h.sync(track, track.id in expanded, track.id == current,
                   row.auto_lanes if row else ())
        self._layout = layout
        self._place()

    def _place(self) -> None:
        layout = getattr(self, "_layout", None)
        if layout is None:
            return
        for row in layout.rows:
            h = self.headers.get(row.track_id)
            if h is not None:
                h.setGeometry(0, int(row.y - self.offset), self.width(), int(row.height))
