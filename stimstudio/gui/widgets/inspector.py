"""Clip inspector: timing, fades, gain and the clip source's parameters."""

from __future__ import annotations

import html
import random

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
                             QVBoxLayout, QWidget)

from ...core.model import (AudioFileSource, BeatMarksSource, GeneratorSource, PatternSource,
                           VideoFileSource)
from ...core.pulse import FS
from ...generators import get_generator, list_generators
from .. import theme
from .param_form import ParamForm


def _spin(lo, hi, step=0.1, decimals=3, suffix=" s") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


class ClipInspector(QScrollArea):
    videoRequested = pyqtSignal()        # choose / locate the video file

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.clip_id: str | None = None
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)

        self.placeholder = QLabel("Select a clip to edit it.\n\nDouble-click an empty spot on a "
                                  "track, or drag a generator from the browser, to add one.")
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.placeholder)

        # ── clip ──
        self.clip_box = QGroupBox("Clip")
        f = self.clip_form = QFormLayout(self.clip_box)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._name_edited)
        self.start = _spin(0, 36000, 0.25)
        self.length = _spin(0.1, 36000, 0.25)
        self.trim = _spin(0, 36000, 0.25)
        self.fade_in = _spin(0, 3600, 0.1)
        self.fade_out = _spin(0, 3600, 0.1)
        self.gain = _spin(0, 4, 0.05, 2, "")
        self.muted = QCheckBox("Muted")
        for label, w in (("Name", self.name), ("Start", self.start), ("Length", self.length),
                         ("Trim start", self.trim), ("Fade in", self.fade_in),
                         ("Fade out", self.fade_out), ("Gain", self.gain), ("", self.muted)):
            f.addRow(label, w)
        for w, attr in ((self.start, "start_s"), (self.length, "length_s"), (self.trim, "trim_s"),
                        (self.fade_in, "fade_in_s"), (self.fade_out, "fade_out_s"),
                        (self.gain, "gain")):
            w.valueChanged.connect(lambda v, a=attr: self._set(a, v))
        self.muted.toggled.connect(lambda on: self._set("muted", on, merge=False))
        lay.addWidget(self.clip_box)

        # ── source ──
        self.source_box = QGroupBox("Generator")
        sv = QVBoxLayout(self.source_box)
        top = QFormLayout()
        self.gen_combo = QComboBox()
        for g in list_generators():
            self.gen_combo.addItem(g.label, g.type_id)
        self.gen_combo.activated.connect(self._generator_changed)
        top.addRow("Type", self.gen_combo)
        seed_row = QHBoxLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setKeyboardTracking(False)
        self.seed.valueChanged.connect(lambda v: self._set_source(seed=v))
        self.reseed = QPushButton("New seed")
        self.reseed.setObjectName("secondary")
        self.reseed.clicked.connect(lambda: self._set_source(seed=random.randrange(2**31)))
        seed_row.addWidget(self.seed, 1)
        seed_row.addWidget(self.reseed)
        self.seed_label = QLabel("Seed")
        top.addRow(self.seed_label, seed_row)
        sv.addLayout(top)
        self.params = ParamForm()
        self.params.paramChanged.connect(self._param_changed)
        sv.addWidget(self.params)
        self.no_params = QLabel("Draw the curves in the Pattern editor (bottom panel; "
                                "double-click the clip to open it).")
        self.no_params.setWordWrap(True)
        self.no_params.setStyleSheet(f"color: {theme.TEXT_DIM}; font-style: italic;")
        sv.addWidget(self.no_params)
        lay.addWidget(self.source_box)

        self.pulse_box = QGroupBox("Pulse data")
        pv = QFormLayout(self.pulse_box)
        self.origin = QLabel()
        self.origin.setWordWrap(True)
        self.data_len = QLabel()
        self.loop = QCheckBox("Loop to fill the clip")
        self.loop.toggled.connect(self._loop_toggled)
        pv.addRow("Origin", self.origin)
        pv.addRow("Length", self.data_len)
        pv.addRow("", self.loop)
        lay.addWidget(self.pulse_box)

        self.audio_box = QGroupBox("Audio modulator")
        av = QFormLayout(self.audio_box)
        self.audio_path = QLabel()
        self.audio_path.setWordWrap(True)
        self.audio_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.audio_info = QLabel("Drives other tracks through the Modulation panel. "
                                 "Gain scales its signals.")
        self.audio_info.setWordWrap(True)
        self.audio_info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        av.addRow("File", self.audio_path)
        av.addRow("", self.audio_info)
        lay.addWidget(self.audio_box)

        self.video_box = QGroupBox("Video")
        vv = QFormLayout(self.video_box)
        self.video_path = QLabel()
        self.video_path.setWordWrap(True)
        self.video_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.video_len = QLabel()
        self.video_btn = QPushButton("Replace / locate file…")
        self.video_btn.setObjectName("secondary")
        self.video_btn.clicked.connect(self.videoRequested.emit)
        info = QLabel("Shown in the Video window in sync with the timeline. The clip can "
                      "only be moved; its sound is set in the Video window.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        vv.addRow("File", self.video_path)
        vv.addRow("Duration", self.video_len)
        vv.addRow("", self.video_btn)
        vv.addRow("", info)
        lay.addWidget(self.video_box)

        self.rhythm_box = QGroupBox("Pattern")
        rv = QFormLayout(self.rhythm_box)
        self.rhythm_info = QLabel()
        self.rhythm_info.setWordWrap(True)
        self.rhythm_hint = QLabel()
        self.rhythm_hint.setWordWrap(True)
        self.rhythm_hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
        rv.addRow("", self.rhythm_info)
        rv.addRow("", self.rhythm_hint)
        lay.addWidget(self.rhythm_box)
        lay.addStretch(1)

        doc.modelChanged.connect(self.refresh)
        self.refresh()

    # ── selection ───────────────────────────────────────────────────────────
    def set_selection(self, clip_ids: list[str]) -> None:
        self.clip_id = clip_ids[0] if len(clip_ids) == 1 else None
        if len(clip_ids) > 1:
            self.placeholder.setText(f"{len(clip_ids)} clips selected.")
        else:
            self.placeholder.setText("Select a clip to edit it.\n\nDouble-click an empty spot on "
                                     "a track, or drag a generator from the browser, to add one.")
        self.refresh()

    def _clip(self):
        found = self.doc.project.find_clip(self.clip_id) if self.clip_id else None
        return found[1] if found else None

    def refresh(self) -> None:
        clip = self._clip()
        has = clip is not None
        self.placeholder.setVisible(not has)
        self.clip_box.setVisible(has)
        if not has:
            self.source_box.setVisible(False)
            self.pulse_box.setVisible(False)
            self.audio_box.setVisible(False)
            self.video_box.setVisible(False)
            self.rhythm_box.setVisible(False)
            return
        widgets = (self.name, self.start, self.length, self.trim, self.fade_in, self.fade_out,
                   self.gain, self.muted, self.gen_combo, self.seed, self.loop)
        for w in widgets:
            w.blockSignals(True)
        if not self.name.hasFocus():
            self.name.setText(clip.name)
        for w, v in ((self.start, clip.start_s), (self.length, clip.length_s),
                     (self.trim, clip.trim_s), (self.fade_in, clip.fade_in_s),
                     (self.fade_out, clip.fade_out_s), (self.gain, clip.gain)):
            if not w.hasFocus():
                w.setValue(v)
        self.muted.setChecked(clip.muted)

        src = clip.source
        is_gen = isinstance(src, GeneratorSource)
        is_audio = isinstance(src, AudioFileSource)
        is_video = isinstance(src, VideoFileSource)
        is_rhythm = isinstance(src, (PatternSource, BeatMarksSource))
        self.source_box.setVisible(is_gen)
        self.pulse_box.setVisible(not (is_gen or is_audio or is_video or is_rhythm))
        self.audio_box.setVisible(is_audio)
        self.video_box.setVisible(is_video)
        self.rhythm_box.setVisible(is_rhythm)
        for w in (self.length, self.trim, self.fade_in, self.fade_out, self.gain, self.muted):
            self.clip_form.setRowVisible(w, not is_video)     # a video clip only moves
        # a pattern pool entry: its position comes from the marks
        self.clip_form.setRowVisible(self.start, "/" not in (self.clip_id or ""))
        self.clip_box.setTitle("Pool entry clip" if "/" in (self.clip_id or "") else "Clip")
        if is_rhythm:
            self._show_rhythm(clip)
        if is_video:
            import os
            missing = "" if os.path.exists(src.path) else "  (file not found)"
            self.video_path.setText(src.path + missing)
            self.video_len.setText(theme.format_time(src.duration_s or clip.length_s))
        if is_audio:
            import os
            missing = "" if os.path.exists(src.path) else "  (file not found)"
            self.audio_path.setText(src.path + missing)
        if is_gen:
            self._fill_generators(get_generator(src.type_id).domain)
            idx = self.gen_combo.findData(src.type_id)
            self.gen_combo.setCurrentIndex(idx)
            gen = get_generator(src.type_id)
            self.seed.setValue(int(src.seed))
            for w in (self.seed_label, self.seed, self.reseed):
                w.setVisible(gen.stochastic)
            specs = gen.param_specs()
            self.params.set_specs(specs, gen.merged_params(src.params))
            self.params.setVisible(bool(specs))
            self.no_params.setVisible(not specs)
        elif not (is_audio or is_video or is_rhythm):
            self.origin.setText(src.origin or "—")
            self.data_len.setText(theme.format_time(len(src.data) / FS))
            self.loop.setChecked(src.loop)
        for w in widgets:
            w.blockSignals(False)

    def _show_rhythm(self, clip) -> None:
        src = clip.source
        n_in = sum(1 for m in src.marks if clip.trim_s <= m.t_s < clip.trim_s + clip.length_s)
        if isinstance(src, PatternSource):
            self.rhythm_box.setTitle("Pattern")
            keys = ", ".join(f"{e.key}: {html.escape(e.clip.name)}" for e in src.pool) or "empty"
            self.rhythm_info.setText(f"{len(src.marks)} marks ({n_in} in the clip)<br>"
                                     f"Pool: {keys}")
            self.rhythm_hint.setText("Edit marks, pick rules and the pool in the Pattern "
                                     "editor (double-click the clip).")
        else:
            self.rhythm_box.setTitle("Beat marks")
            self.rhythm_info.setText(f"{len(src.marks)} marks ({n_in} in the clip)")
            self.rhythm_hint.setText("Silent. Select it together with clips and choose "
                                     "Convert to pattern in its context menu to play those "
                                     "clips on the marks.")

    def _fill_generators(self, domain: str) -> None:
        if getattr(self, "_gen_domain", None) == domain:
            return
        self._gen_domain = domain
        self.gen_combo.clear()
        for g in list_generators():
            if g.domain == domain:
                self.gen_combo.addItem(g.label, g.type_id)

    # ── edits → undoable commands ───────────────────────────────────────────
    def _set(self, attr: str, value, merge: bool = True) -> None:
        if self._clip() is None:
            return
        self.doc.set_clip_attrs(self.clip_id, merge_key=attr if merge else None, **{attr: value})

    def _name_edited(self) -> None:
        clip = self._clip()
        if clip is not None and self.name.text() != clip.name:
            self.doc.set_clip_attrs(self.clip_id, text="Rename clip", name=self.name.text())

    def _set_source(self, merge_key=None, **changes) -> None:
        clip = self._clip()
        if clip is None or not isinstance(clip.source, GeneratorSource):
            return
        src = clip.source.clone()
        for k, v in changes.items():
            setattr(src, k, v)
        self.doc.set_clip_attrs(self.clip_id, merge_key=merge_key, text="Edit generator",
                                source=src)

    def _param_changed(self, key: str, value) -> None:
        clip = self._clip()
        if clip is None:
            return
        params = dict(clip.source.params)
        params[key] = value
        self._set_source(merge_key=("param", key), params=params)

    def _generator_changed(self, index: int) -> None:
        clip = self._clip()
        type_id = self.gen_combo.itemData(index)
        if clip is None or type_id == clip.source.type_id:
            return
        gen = get_generator(type_id)
        changes = {"source": GeneratorSource(type_id, gen.default_params(), clip.source.seed)}
        old_label = get_generator(clip.source.type_id).label
        if clip.name in ("", old_label):
            changes["name"] = gen.label
        self.doc.set_clip_attrs(self.clip_id, text="Change generator", **changes)

    def _loop_toggled(self, on: bool) -> None:
        clip = self._clip()
        if clip is None or isinstance(clip.source, GeneratorSource):
            return
        src = clip.source.clone()
        src.loop = on
        self.doc.set_clip_attrs(self.clip_id, text="Loop pulse data", source=src)
