"""PatternEditor: detailed editing of the selected clip (bottom dock).

* ``draw`` clips → free-hand curve editor;
* ``steps`` clips → step editor;
* other generators and pulse data → rendered preview, with stage navigation
  for Howl activities;
* beat-marks and pattern clips → the marks / pool editor. A pool entry opens
  in the page of its own kind ("Pattern › A"); its address is
  ``"<pattern clip id>/<key>"`` (see ``Project.find_clip``).

Edits go to the document as undoable commands; the editor reloads from the
model on every change (except mid-stroke).
"""

from __future__ import annotations

import json

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from ...core.model import (AudioFileSource, BeatMarksSource, GeneratorSource, PatternSource,
                           entry_address)
from ...core.render import render_clip
from ...generators import get_generator
from .. import theme
from .curve_editors import DrawEditor, StepEditor
from .generator_page import GeneratorPage
from .pattern_page import PatternPage
from .triphase_page import TriphasePage


def _clip_signature(clip, bpm: float) -> str:
    """Changes whenever the clip's rendered output may change."""
    if isinstance(clip.source, PatternSource):
        from ...core.pattern import pattern_key
        d = {**{k: v for k, v in clip.to_dict().items() if k != "source"},
             "pattern": pattern_key(clip.source, "", bpm),
             "marks": [m.to_dict() for m in clip.source.marks]}
        return json.dumps([d, bpm], sort_keys=True, default=str)
    d = clip.to_dict() if isinstance(clip.source, (GeneratorSource, BeatMarksSource)) else {
        **{k: v for k, v in clip.to_dict().items() if k != "source"},
        "data": id(clip.source.data), "loop": clip.source.loop}
    return json.dumps([d, bpm], sort_keys=True, default=str)


class PatternEditor(QWidget):
    #: the clip (or pool entry address) the editor shows changed by itself
    focusChanged = pyqtSignal(list)
    EMPTY_TEXT = ("Select one clip to edit it here.\n\nDraw and Step Sequencer clips are "
                  "drawn directly; Howl activities show their stages.")

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.clip_id: str | None = None
        self._sig = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self.back_btn = QPushButton("◀ Pattern")
        self.back_btn.setToolTip("Back to the pattern this pool entry belongs to")
        self.back_btn.clicked.connect(self.close_entry)
        self.back_btn.setVisible(False)
        head.addWidget(self.back_btn)
        self.title = QLabel()
        self.title.setStyleSheet(f"padding: 3px 8px; color: {theme.TEXT_DIM};")
        head.addWidget(self.title, 1)
        lay.addLayout(head)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)

        self.empty = QLabel(self.EMPTY_TEXT)
        self.empty.setWordWrap(True)
        self.empty.setStyleSheet(f"color: {theme.TEXT_DIM}; padding: 16px;")
        self.generator_page = GeneratorPage()
        self.draw_editor = DrawEditor()
        self.step_editor = StepEditor()
        self.triphase_page = TriphasePage()
        self.pattern_page = PatternPage()
        for w in (self.empty, self.generator_page, self.draw_editor, self.step_editor,
                  self.triphase_page, self.pattern_page):
            self.stack.addWidget(w)

        self.generator_page.editRequested.connect(self._apply_edit)
        self.triphase_page.editRequested.connect(self._apply_edit)
        self.pattern_page.editRequested.connect(self._apply_merged_edit)
        self.pattern_page.convertRequested.connect(
            lambda: self.doc.convert_to_pattern(self.clip_id))
        self.pattern_page.openEntryRequested.connect(self.open_entry)
        self.draw_editor.committed.connect(self._commit_curves)
        self.step_editor.committed.connect(self._commit_curves)
        doc.modelChanged.connect(self.refresh)
        self.refresh()

    def set_selection(self, clip_ids: list[str]) -> None:
        new = clip_ids[0] if len(clip_ids) == 1 else None
        if new != self.clip_id:
            self.clip_id = new
            self._sig = None
        self.refresh()

    @property
    def in_entry(self) -> bool:
        return bool(self.clip_id) and "/" in self.clip_id

    def open_entry(self, key: str) -> None:
        """Edit the clip of a pool entry of the shown pattern."""
        if self.clip_id is None or self.in_entry:
            return
        self.clip_id = entry_address(self.clip_id, key)
        self._sig = None
        self.refresh()
        self.focusChanged.emit([self.clip_id])

    def close_entry(self) -> None:
        if self.in_entry:
            self.clip_id = self.clip_id.split("/", 1)[0]
            self._sig = None
            self.refresh()
            self.focusChanged.emit([self.clip_id])

    def _clip(self):
        found = self.doc.project.find_clip(self.clip_id) if self.clip_id else None
        return found[1] if found else None

    def _track(self):
        found = self.doc.project.find_clip(self.clip_id) if self.clip_id else None
        return found[0] if found else None

    def set_time(self, seconds: float) -> None:
        """Cursor / playhead position (the triphase pad follows it)."""
        if self.stack.currentWidget() is self.triphase_page:
            self.triphase_page.set_time(seconds)

    def refresh(self) -> None:
        clip = self._clip()
        self.back_btn.setVisible(self.in_entry and clip is not None)
        if clip is None:
            self.title.setText("No clip selected")
            self.empty.setText(self.EMPTY_TEXT)
            self.stack.setCurrentWidget(self.empty)
            self._sig = None
            return
        src = clip.source
        if src.kind == "video":
            self.title.setText(f"{clip.name or '(unnamed)'}  ·  video")
            self.empty.setText("Video clip — it plays in the Video window. Drag it on the "
                               "timeline to line the video up with the stim tracks.")
            self.stack.setCurrentWidget(self.empty)
            self._sig = None
            return
        if isinstance(src, AudioFileSource):
            self.title.setText(f"{clip.name or '(unnamed)'}  ·  audio modulator")
            self.empty.setText("Audio modulator clip — expand its track to see the envelope, "
                               "beats and band signals; route them in the Modulation panel.")
            self.stack.setCurrentWidget(self.empty)
            self._sig = None
            return
        self.empty.setText(self.EMPTY_TEXT)
        if isinstance(src, GeneratorSource):
            kind = get_generator(src.type_id).label
        elif isinstance(src, PatternSource):
            kind = "pattern"
        elif isinstance(src, BeatMarksSource):
            kind = "beat marks"
        else:
            kind = f"pulse data{' (loop)' if src.loop else ''}"
        crumb = ""
        if self.in_entry:
            parent_id, key = self.clip_id.split("/", 1)
            parent = self.doc.project.find_clip(parent_id)[1]
            crumb = f"{parent.name or 'Pattern'} › {key}:  "
        self.title.setText(f"{crumb}{clip.name or '(unnamed)'}  ·  {kind}  ·  "
                           f"{theme.format_time(clip.length_s)}"
                           f"{f'  ·  trim {theme.format_time(clip.trim_s)}' if clip.trim_s else ''}")

        if self.draw_editor.drawing or self.step_editor.drawing:
            return
        bpm = self.doc.project.bpm
        track = self._track()
        triphase = track is not None and track.type == "triphase"
        sig = _clip_signature(clip, bpm) + json.dumps(self.doc.project.triphase_cal,
                                                      sort_keys=True)
        if sig == self._sig:
            return
        self._sig = sig
        if isinstance(src, (PatternSource, BeatMarksSource)):
            domain = "triphase" if triphase else "pulse"
            rendered = render_clip(clip, self.doc.cache, bpm, domain=domain)
            self.pattern_page.show_clip(clip, rendered, domain, bpm)
            self.stack.setCurrentWidget(self.pattern_page)
            return
        if triphase:
            from ...core.triphase import Calibration
            rendered = render_clip(clip, self.doc.cache, bpm, domain="triphase")
            self.triphase_page.show_clip(clip, rendered,
                                         Calibration.from_dict(self.doc.project.triphase_cal))
            self.triphase_page.set_time(self.doc.cursor_s)
            self.stack.setCurrentWidget(self.triphase_page)
            return
        rendered = render_clip(clip, self.doc.cache, bpm)

        type_id = src.type_id if isinstance(src, GeneratorSource) else None
        if type_id in ("draw", "steps"):
            ed = self.draw_editor if type_id == "draw" else self.step_editor
            ed.load(src.params)
            ed.set_span(clip.trim_s, clip.length_s)
            ed.set_rendered(rendered)
            self.stack.setCurrentWidget(ed)
        else:
            self.generator_page.show_clip(clip, rendered)
            self.stack.setCurrentWidget(self.generator_page)

    # ── edits ───────────────────────────────────────────────────────────────
    def _apply_edit(self, changes: dict, text: str) -> None:
        if self._clip() is not None:
            self.doc.set_clip_attrs(self.clip_id, text=text, **changes)

    def _apply_merged_edit(self, changes: dict, text: str, merge_key) -> None:
        if self._clip() is not None:
            self.doc.set_clip_attrs(self.clip_id, merge_key=merge_key, text=text, **changes)

    def _commit_curves(self, values: dict) -> None:
        clip = self._clip()
        if clip is None or not isinstance(clip.source, GeneratorSource):
            return
        src = clip.source.clone()
        src.params = {**src.params, **values}
        self.doc.set_clip_attrs(self.clip_id, text="Draw curves", source=src)
