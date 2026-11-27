"""Main window: song editor in the centre, browser / inspector / master preview docks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialog, QDockWidget, QDoubleSpinBox,
                             QAbstractScrollArea, QFileDialog, QFrame, QInputDialog, QLabel,
                             QMainWindow, QMessageBox, QScrollArea, QTabBar, QToolBar,
                             QToolButton, QWidget)

from ..core.model import AudioFileSource, Clip, PulseDataSource, Track
from ..core.pulse import FS, n_pulses
from ..core.render import render_source
from ..io.hwl import read_hwl, write_hwl
from ..io.parthwl import clip_to_part, load_parthwl, part_to_clip, save_parthwl
from ..io.project_file import EXTENSION
from . import commands as cmd
from . import metrics, theme
from .document import ProjectDocument
from .pattern.editor import PatternEditor
from .playback import PlaybackController
from .settings import app_settings
from .song.editor import SongEditor
from .song.layout import SNAP_CHOICES, Snap
from .widgets.browser import Browser
from .widgets.device_panel import DevicePanel
from .widgets.fx_chain import EffectsPanel
from .widgets.mod_matrix import ModMatrixPanel
from .widgets.inspector import ClipInspector
from .widgets.master_preview import MasterPreview

PROJECT_FILTER = f"StimStudio project (*{EXTENSION})"


class MainWindow(QMainWindow):
    def __init__(self, doc: ProjectDocument | None = None, user_preset_dir=None,
                 autosave_dir=None):
        super().__init__()
        self.doc = doc or ProjectDocument()
        self.settings = app_settings()
        metrics.fit_to_screen(self, 1500, 900, frac=0.95)
        self._shown_once = False
        self.clipboard: list[tuple[str, Clip]] = []
        self._tool_windows: dict = {}               # Tools menu windows by tool id

        self.editor = SongEditor(self.doc)
        self.setCentralWidget(self.editor)
        self.browser = Browser(user_preset_dir)
        self.pattern = PatternEditor(self.doc)
        self.effects = EffectsPanel(self.doc)
        self.modulation = ModMatrixPanel(self.doc)
        self.inspector = ClipInspector(self.doc)
        self.preview = MasterPreview()
        self.editor.generator_provider = self.browser.current_type
        self.playback = PlaybackController(self.doc, self)
        from .autosave import Autosave
        self.autosave = Autosave(self.doc, autosave_dir, parent=self)
        self.device_panel = DevicePanel(self.playback)

        browser_dock = self._dock("Browser", self.browser, Qt.DockWidgetArea.LeftDockWidgetArea,
                                  "browserDock")
        inspector_dock = self._dock("Inspector", self.inspector,
                                    Qt.DockWidgetArea.RightDockWidgetArea, "inspectorDock")
        preview_dock = self._dock("Master output", self.preview,
                                  Qt.DockWidgetArea.BottomDockWidgetArea, "previewDock")
        output_dock = self._dock("Output", self.device_panel,
                                 Qt.DockWidgetArea.RightDockWidgetArea, "outputDock")
        self.splitDockWidget(inspector_dock, output_dock, Qt.Orientation.Vertical)
        self.effects_dock = self._dock("Effects", self.effects,
                                       Qt.DockWidgetArea.RightDockWidgetArea, "effectsDock")
        self.mod_dock = self._dock("Modulation", self.modulation,
                                   Qt.DockWidgetArea.RightDockWidgetArea, "modulationDock")
        self.pattern_dock = self._dock("Pattern editor", self.pattern,
                                       Qt.DockWidgetArea.BottomDockWidgetArea, "patternDock")
        from .video.dock import VideoPanel
        self.video = VideoPanel(self.doc, self.playback)
        self.video_dock = self._dock("Video", self.video, Qt.DockWidgetArea.RightDockWidgetArea,
                                     "videoDock")
        from .widgets.record_panel import RecordPanel
        self.record = RecordPanel(self.doc, self.playback, self.editor)
        self.record_dock = self._dock("Record", self.record,
                                      Qt.DockWidgetArea.RightDockWidgetArea, "recordDock")
        from .safety import SafetyMonitor, SafetyPanel
        self.safety = SafetyMonitor(self.doc, self)
        self.safety_panel = SafetyPanel(self.safety)
        self.safety_dock = self._dock("Safety", self.safety_panel,
                                      Qt.DockWidgetArea.RightDockWidgetArea, "safetyDock")

        self._build_actions()
        self._build_toolbar()
        self._build_menus()
        self.render_label = QLabel()
        self.statusBar().addPermanentWidget(self.render_label)

        # wiring
        self.editor.selectionChanged.connect(self.inspector.set_selection)
        self.editor.selectionChanged.connect(self.pattern.set_selection)
        self.editor.selectionChanged.connect(self.effects.set_clip_selection)
        self.editor.currentTrackChanged.connect(self.effects.set_track)
        self.editor.currentTrackChanged.connect(self.modulation.set_current_track)
        self.editor.funscriptDropped.connect(
            lambda paths, t, tid: self.import_funscript(paths, start_s=t, track_id=tid or None))
        self.editor.audioDropped.connect(
            lambda path, t, tid: self.import_audio(path, start_s=t, track_id=tid or None))
        self.editor.beatsDropped.connect(lambda paths, t: self.import_beats(paths, start_s=t))
        self.editor.peakMarksRequested.connect(lambda cid: self.marks_from_peaks(cid))
        self.pattern.focusChanged.connect(self.inspector.set_selection)
        self.pattern.focusChanged.connect(self.effects.set_clip_selection)
        self.effects.track.automationAdded.connect(self._show_track_lanes)
        self.editor.clipActivated.connect(self._clip_activated)
        self.editor.videoDropped.connect(lambda path: self.open_video(path))
        self.editor.videoRequested.connect(lambda: self.open_video())
        self.editor.videoAnalysisRequested.connect(self.analyse_video)
        self.inspector.videoRequested.connect(lambda: self.open_video())
        self.a_panic.triggered.connect(lambda: self.video.set_fullscreen(False))
        shuttle = self.video.shuttle
        shuttle.play_fn = self._resume_after_shuttle
        shuttle.positionChanged.connect(self._on_shuttle_position)
        self.a_stop.triggered.connect(shuttle.cancel)
        self.video.openRequested.connect(lambda: self.open_video())
        self.editor.savePresetRequested.connect(self.save_clip_preset)
        self.browser.presets.presetActivated.connect(self._add_preset_at_cursor)
        self.editor.selectionChanged.connect(self._update_actions)
        self.editor.visibleRangeChanged.connect(self.preview.set_view_range)
        self.editor.statusMessage.connect(lambda m: self.statusBar().showMessage(m, 3000))
        self.browser.generators.generatorActivated.connect(self._add_generator_at_cursor)
        self.doc.renderStarted.connect(lambda: self.render_label.setText("rendering…"))
        self.doc.renderReady.connect(self._on_render)
        self.doc.renderFailed.connect(self._on_render_failed)
        self.doc.cursorChanged.connect(self._on_cursor)
        self.doc.modelChanged.connect(self._on_model)
        self.doc.dirtyChanged.connect(lambda _d: self._update_title())
        self.doc.pathChanged.connect(lambda _p: self._update_title())
        pb = self.playback
        pb.playheadChanged.connect(self._on_playhead)
        pb.playingChanged.connect(self._on_playing)
        QApplication.instance().installEventFilter(self)   # Esc, see eventFilter
        pb.message.connect(lambda m: self.statusBar().showMessage(m, 6000))
        pb.loopChanged.connect(lambda: self.editor.set_loop(pb.loop_enabled, pb.loop_region))
        self.editor.loopRegionEdited.connect(self._on_loop_region_edited)
        self.safety_panel.issueActivated.connect(self._show_issue)
        self.record.recordingChanged.connect(self.a_record.setChecked)
        self.safety.reportChanged.connect(self.editor.set_safety_report)

        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        state = self.settings.value("windowState")
        self._pending_dock_sizes = None
        if state is not None:
            self.restoreState(state)
        else:
            # First run: tab groups and sizes are applied once the window is shown.
            # Tabifying before show() and then resizing leaves a stale, visible tab
            # bar behind (QMainWindowLayout quirk); restoreState() above is fine.
            self._pending_dock_sizes = (browser_dock, inspector_dock, preview_dock,
                                        output_dock)
        self.effects.set_track(self.editor.current_track_id or "")
        self._on_model()
        self._on_cursor(0.0)
        self._update_title()
        self._update_actions()
        self.doc.schedule_render()

    # ── construction ────────────────────────────────────────────────────────
    def _dock(self, title, widget, area, name) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        if not isinstance(widget, QAbstractScrollArea):
            # a dock as small as the screen needs: its contents scroll instead of
            # pushing the window past the screen (1920×1080 at 150 % is 1280×720)
            scroll = QScrollArea()
            scroll.setObjectName("dockScroll")
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            widget = scroll
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(area, dock)
        dock.dockLocationChanged.connect(lambda _a: QTimer.singleShot(0, self._unelide_dock_tabs))
        return dock

    def _unelide_dock_tabs(self):
        """Whole names on the tabs of tabbed docks ("Insp…", "S…" in a narrow
        dock say nothing): scroll arrows when they do not fit. The main window
        makes these tab bars itself, anew when docks are regrouped."""
        for bar in self.findChildren(QTabBar):
            if bar.parent() is self:
                bar.setElideMode(Qt.TextElideMode.ElideNone)
                bar.setUsesScrollButtons(True)

    def _action(self, text, slot, shortcut=None, tip=None) -> QAction:
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if tip:
            a.setStatusTip(tip)
        self.addAction(a)
        return a

    def _build_actions(self):
        ed = self.editor
        A = self._action
        self.a_new = A("&New", self.file_new, QKeySequence.StandardKey.New)
        self.a_open = A("&Open…", self.file_open, QKeySequence.StandardKey.Open)
        self.a_save = A("&Save", self.file_save, QKeySequence.StandardKey.Save)
        self.a_save_as = A("Save &As…", self.file_save_as, "Ctrl+Shift+S")
        self.a_media = A("Project &media…", lambda: self.show_media_dialog(), None,
                         "Files used by the project: find missing ones, collect them "
                         "next to the project")
        self.a_import_hwl = A("Import .hwl as clip…", self.import_hwl)
        self.a_import_part = A("Import .parthwl clip…", self.import_parthwl)
        self.a_import_audio = A("Import &audio…", lambda: self.import_audio(), "Ctrl+I",
                                "Convert audio to stim lanes or add it as a modulator")
        self.a_export_hwl = A("&Export .hwl…", self.export_hwl, "Ctrl+E")
        self.a_export_part = A("Export clip as .parthwl…", self.export_parthwl)
        self.a_save_preset = A("Save clip as &preset…", lambda: self.save_clip_preset(),
                               "Ctrl+Shift+P")
        self.a_quit = A("&Quit", self.close, QKeySequence.StandardKey.Quit)

        self.a_undo = self.doc.undo.createUndoAction(self, "&Undo")
        self.a_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.a_redo = self.doc.undo.createRedoAction(self, "&Redo")
        self.a_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.addAction(self.a_undo)
        self.addAction(self.a_redo)
        self.a_cut = A("Cu&t", self.edit_cut, QKeySequence.StandardKey.Cut)
        self.a_copy = A("&Copy", self.edit_copy, QKeySequence.StandardKey.Copy)
        self.a_paste = A("&Paste at cursor", self.edit_paste, QKeySequence.StandardKey.Paste)
        self.a_delete = A("&Delete", ed.delete_selected, QKeySequence.StandardKey.Delete)
        self.a_duplicate = A("D&uplicate", ed.duplicate_selected, "Ctrl+D")
        self.a_split = A("&Split at cursor", lambda: ed.split_selected(), "S")
        self.a_glue = A("&Glue clips", ed.glue_selected, "Ctrl+J",
                        "Render the selected clips of a track into one clip")
        self.a_mute_clip = A("&Mute / unmute clip", ed.toggle_mute_selected, "M")
        self.a_select_all = A("Select &all", ed.select_all, QKeySequence.StandardKey.SelectAll)

        self.a_add_track = A("&Add track", self.add_track, "Ctrl+T")
        self.a_add_mod_track = A("Add &modulator track (audio)", self.add_mod_track)
        self.a_add_tri_track = A("Add t&riphase track", self.add_triphase_track)
        self.a_tri_cal = A("Triphase &calibration…", self.show_triphase_calibration)
        self.a_dup_track = A("D&uplicate track", self.duplicate_track)
        self.a_del_track = A("&Delete track", self.delete_track)

        pb = self.playback
        self.a_play = A("&Play", self.toggle_play, "Space", "Play from the cursor / stop")
        self.a_stop = A("&Stop", pb.stop, None, "Stop with a short fade")
        self.a_panic = A("PANIC — instant stop", pb.panic, "Esc", "Silence the output immediately")
        self.a_panic.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.a_loop = A("&Loop", pb.set_loop_enabled, "L", "Loop the loop region (or everything)")
        self.a_loop.setCheckable(True)
        self.a_follow = A("&Follow playhead", lambda _on: None, "F")
        self.a_follow.setCheckable(True)
        self.a_follow.setChecked(True)
        self.a_loop_sel = A("Loop &selected clips", self.loop_selection, "Ctrl+L")
        self.a_loop_clear = A("&Clear loop region", lambda: pb.set_loop_region(None))
        self.a_record = A("&Record take", self.toggle_record, "R",
                          "Play and perform a take live on the selected track (Record dock)")
        self.a_record.setCheckable(True)
        self.a_add_marker = A("Add &marker at cursor", self.add_marker_at_cursor, "Shift+M",
                              "Drop a named marker at the edit cursor")
        self.a_prev_marker = A("Previous marker", lambda: ed.goto_marker(-1), "Ctrl+Left")
        self.a_next_marker = A("Next marker", lambda: ed.goto_marker(+1), "Ctrl+Right")
        self.a_loop_markers = A("Loop between &markers", self.loop_between_markers,
                                "Ctrl+Shift+L", "Loop the stretch between the markers "
                                "around the cursor")
        self.a_clear_markers = A("Clear all markers",
                                 lambda: self.doc.remove_markers(
                                     [m.id for m in self.doc.project.markers],
                                     "Clear markers"))
        self.a_export_audio = A("Export &audio (WAV / FLAC)…", self.export_audio)
        self.a_export_funscript = A("Export &funscript…", self.export_funscript, None,
                                    "Amplitude and frequency of channels A and B as funscript")
        self.a_export = A("E&xport…", self.show_export_dialog, "Ctrl+Shift+E",
                          "Export .hwl or audio: whole project, loop region or selection; "
                          "mix or per-track stems")
        self.a_import_funscript = A("Import &funscript…", lambda: self.import_funscript())
        self.a_import_beats = A("Import &beat timestamps (CSV)…", lambda: self.import_beats(),
                                None, "BeatGen beat timestamps → tracks of beat-marks blocks")
        self.a_open_video = A("Open &video…", lambda: self.open_video(), "Ctrl+Shift+O",
                              "Show a video in sync with the timeline (adds the video track)")
        self.a_video_full = A("Video &full screen", lambda: self.video.toggle_fullscreen(),
                              "F11")
        self.a_session = A("&Session builder…", lambda: self.build_session(), None,
                           "Generate a whole session: warm-up, build-ups, peak, cool-down")
        self.a_video_motion = A("Video motion → stim…", lambda: self.analyse_video(True, False),
                                None, "Turn the movement in the video into a stim clip")
        self.a_video_cuts = A("Detect scene cuts → markers…",
                              lambda: self.analyse_video(False, True), None,
                              "Put a marker at every hard cut of the video")
        from .tools import TOOLS
        self.tool_actions = {t.id: A(t.label, lambda _c=False, tid=t.id: self.show_tool(tid),
                                     None, t.tooltip) for t in TOOLS}
        self.a_shortcuts = A("&Keyboard shortcuts", self.show_shortcuts, "F1")
        self.a_support = A("&Support development…", self.show_support)
        self.a_about = A("&About StimStudio", self.show_about)

        self.a_zoom_in = A("Zoom &in", lambda: ed.zoom(1.5), "Ctrl+=")
        self.a_zoom_out = A("Zoom &out", lambda: ed.zoom(1 / 1.5), "Ctrl+-")
        self.a_zoom_fit = A("Zoom to &fit", ed.zoom_fit, "Ctrl+0")
        self.a_overview = A("Project &overview bar", ed.set_overview_visible, None,
                            "Whole-project intensity map above the timeline")
        self.a_overview.setCheckable(True)
        self.a_overview.setChecked(ed.overview.isVisibleTo(ed))

    def _build_toolbar(self):
        tb = QToolBar("Transport")
        tb.setObjectName("transportBar")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.time_label = QLabel("00:00.000")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setToolTip("Edit cursor")
        tb.addWidget(self.time_label)
        for a in (self.a_play, self.a_stop, self.a_loop, self.a_follow):
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" BPM "))
        self.bpm = QDoubleSpinBox()
        self.bpm.setRange(20, 300)
        self.bpm.setDecimals(1)
        self.bpm.setKeyboardTracking(False)
        self.bpm.valueChanged.connect(lambda v: self.doc.set_project_attrs(merge_key="bpm", bpm=v))
        tb.addWidget(self.bpm)
        tb.addWidget(QLabel("  Snap "))
        self.snap_combo = QComboBox()
        for label, kind, value in SNAP_CHOICES:
            self.snap_combo.addItem(label, (kind, value))
        self.snap_combo.setCurrentIndex(1)
        self.snap_combo.currentIndexChanged.connect(self._snap_changed)
        tb.addWidget(self.snap_combo)
        tb.addSeparator()
        for a in (self.a_add_track, self.a_zoom_out, self.a_zoom_in, self.a_zoom_fit):
            tb.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding,
                             spacer.sizePolicy().verticalPolicy())
        tb.addWidget(spacer)
        tb.addAction(self.a_export_hwl)
        # PANIC on a bar of its own: a narrow window (a small screen, a large
        # display scale) folds the transport bar's last buttons into its » menu,
        # and PANIC must stay in sight
        tb = QToolBar("Panic")
        tb.setObjectName("panicBar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.toggleViewAction().setEnabled(False)       # cannot be hidden either
        self.addToolBar(tb)
        self.panic_button = panic = QToolButton()
        panic.setDefaultAction(self.a_panic)
        panic.setText("■ PANIC  Esc")
        panic.setStyleSheet(f"QToolButton {{ background: {theme.DANGER}; color: white; "
                            "font-weight: bold; padding: 4px 12px; border-radius: 3px; }"
                            "QToolButton:hover { background: #ff6b6b; }")
        tb.addWidget(panic)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.a_new, self.a_open, "recent", self.a_save, self.a_save_as,
                  self.a_media, None,
                  self.a_open_video, self.a_import_audio, self.a_import_funscript,
                  self.a_import_beats, self.a_import_hwl,
                  self.a_import_part, None, self.a_export, self.a_export_hwl,
                  self.a_export_audio, self.a_export_funscript, self.a_export_part, None,
                  self.a_quit):
            if a == "recent":
                self.recent_menu = m.addMenu("Open &recent")
                self.recent_menu.aboutToShow.connect(self._fill_recent)
            elif a is None:
                m.addSeparator()
            else:
                m.addAction(a)
        m = mb.addMenu("&Edit")
        for a in (self.a_undo, self.a_redo, None, self.a_cut, self.a_copy, self.a_paste,
                  self.a_delete, None, self.a_duplicate, self.a_split, self.a_glue, self.a_mute_clip,
                  self.a_save_preset, None, self.a_select_all, None, self.a_clear_markers):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("T&ransport")
        for a in (self.a_play, self.a_stop, self.a_panic, None, self.a_loop, self.a_loop_sel,
                  self.a_loop_markers, self.a_loop_clear, None, self.a_record, None,
                  self.a_add_marker,
                  self.a_prev_marker, self.a_next_marker, None, self.a_follow):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&Track")
        for a in (self.a_add_track, self.a_add_tri_track, self.a_add_mod_track, None,
                  self.a_dup_track, self.a_del_track, None, self.a_session, None,
                  self.a_video_motion,
                  self.a_video_cuts, None, self.a_tri_cal):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("T&ools")
        for a in (*self.tool_actions.values(), None, self.a_video_motion, self.a_video_cuts):
            m.addSeparator() if a is None else m.addAction(a)
        m = mb.addMenu("&View")
        for a in (self.a_zoom_in, self.a_zoom_out, self.a_zoom_fit, None, self.a_overview,
                  self.a_video_full):
            m.addSeparator() if a is None else m.addAction(a)
        m.addSeparator()
        for dock in self.findChildren(QDockWidget):
            m.addAction(dock.toggleViewAction())
        m.addSeparator()
        self._build_ui_scale_menu(m.addMenu("Interface &size"))
        m = mb.addMenu("&Help")
        m.addAction(self.a_shortcuts)
        m.addAction(self.a_support)
        m.addAction(self.a_about)

    def _build_ui_scale_menu(self, menu):
        """View → Interface size: everything larger or smaller than the Windows
        display scale; applied at the next start (Qt reads it before any window)."""
        menu.setToolTipsVisible(True)
        saved = metrics.parse_scale(self.settings.value(metrics.UI_SCALE_KEY, 100)) or 1.0
        saved = round(saved * 100)
        group = QActionGroup(self)
        group.setExclusive(True)
        for pct in metrics.UI_SCALES:
            a = QAction(f"{pct} %" + ("  (as Windows)" if pct == 100 else ""), self)
            a.setCheckable(True)
            a.setChecked(pct == saved)
            if pct < 100:
                a.setToolTip("Smaller text; the rest keeps the Windows display scale "
                             "(Qt cannot draw below it without display faults)")
            a.triggered.connect(lambda _on=False, p=pct: self._set_ui_scale(p))
            group.addAction(a)
            menu.addAction(a)
        menu.addSeparator()
        now = QAction(f"Now: {metrics.ui_scale_percent()} % — changes apply after a restart",
                      self)
        now.setEnabled(False)
        menu.addAction(now)

    def _set_ui_scale(self, pct: int):
        self.settings.setValue(metrics.UI_SCALE_KEY, int(pct))
        if pct == metrics.ui_scale_percent():
            self.statusBar().showMessage(f"Interface size {pct} %.", 5000)
        else:
            self.statusBar().showMessage(
                f"Interface size {pct} % — takes effect when StimStudio is restarted.", 10000)

    # ── state updates ───────────────────────────────────────────────────────
    def _on_model(self):
        self.bpm.blockSignals(True)
        self.bpm.setValue(self.doc.project.bpm)
        self.bpm.blockSignals(False)
        self._update_actions()

    def _on_render(self, render, seconds):
        self.preview.set_pulses(render.master)
        self.render_label.setText(f"{theme.format_time(render.n / FS, 1)} · rendered in "
                                  f"{seconds * 1000:.0f} ms")

    def _on_render_failed(self, message):
        self.render_label.setText("render failed")
        self.statusBar().showMessage(message.strip().splitlines()[-1], 10000)

    def _on_cursor(self, t):
        if not self.playback.playing:
            self.time_label.setText(theme.format_time(t))
            self.pattern.set_time(t)
        self.preview.set_cursor(t)

    # ── transport ───────────────────────────────────────────────────────────
    def _on_playhead(self, t):
        self.editor.set_playhead(t)
        self.preview.set_playhead(t)
        if t >= 0:
            self.time_label.setText(theme.format_time(t))
            self.pattern.set_time(t)
            if self.a_follow.isChecked():
                self.editor.ensure_visible(t)
        else:
            self.time_label.setText(theme.format_time(self.doc.cursor_s))

    def _on_shuttle_position(self, t):
        if self.a_follow.isChecked():
            self.editor.ensure_visible(t)

    def _resume_after_shuttle(self):
        """Fast-forward ended where the output had been playing: start it again."""
        if self.safety_ok("play"):
            self.playback.play()

    def _on_playing(self, playing):
        self.a_play.setText("&Pause" if playing else "&Play")
        self.time_label.setStyleSheet(f"color: {theme.DANGER};" if playing else "")
        # A native file dialog keeps its keys from Qt, and so from eventFilter. It can't
        # outlast this: while one is open, nothing can start the output
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, playing)

    def eventFilter(self, obj, event):
        # The PANIC shortcut alone isn't enough: a modal dialog shadows it, an open menu
        # takes Esc for itself, a progress dialog has its own Esc shortcut. So while
        # output runs Esc overrides every shortcut and arrives here as a key press.
        # With nothing playing it stays with the dialog or menu, to close it.
        if (event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.key() == Qt.Key.Key_Escape and self.playback.playing):
            if event.type() == QEvent.Type.KeyPress:
                self.a_panic.trigger()
            event.accept()
            return True
        return False

    def _on_loop_region_edited(self, region):
        self.playback.set_loop_region(region)
        if region is not None and not self.a_loop.isChecked():
            self.a_loop.setChecked(True)     # drawing a region implies looping it
            self.playback.set_loop_enabled(True)

    def loop_selection(self):
        span = self.editor.selection_span()
        if span is None:
            self.statusBar().showMessage("Select clips to loop.", 3000)
            return
        self._on_loop_region_edited(span)

    def analyse_video(self, motion: bool = True, cuts: bool = False, result=None):
        """Scene cuts → markers and / or motion → a pulse clip on a new track.

        *result* (the dialog's ``result_data()``) skips the dialog — for tests.
        """
        from ..io.video import have_pyav
        vt = self.doc.project.video_track()
        clip = vt.clips[0] if vt is not None and vt.clips else None
        if clip is None:
            self.statusBar().showMessage("Open a video first (File → Open video…).", 4000)
            return None
        path = clip.source.path
        if result is None:
            if not have_pyav():
                QMessageBox.information(self, "Video analysis",
                                        "Video analysis needs PyAV:  pip install av")
                return None
            if not Path(path).is_file():
                QMessageBox.warning(self, "Video analysis", "Video file not found:\n" + path)
                return None
            from .video_analysis import VideoAnalysisDialog
            dlg = VideoAnalysisDialog(path, clip.source.duration_s, self, motion, cuts)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            result = dlg.result_data()
        pulses, cut_times = result.get("pulses"), result.get("cuts")
        if pulses is None and not cut_times:
            if cut_times is not None:
                self.statusBar().showMessage("No scene cuts found.", 4000)
            return None
        new_clip = None
        self.doc.undo.beginMacro("Video analysis")
        try:
            if pulses is not None and len(pulses):
                tri = bool(result.get("triphase"))
                index = self.doc.project.tracks.index(vt) + 1
                track = self.doc.add_track("Motion (3φ)" if tri else "Motion", index,
                                           "triphase" if tri else "standard")
                new_clip = Clip(PulseDataSource(np.asarray(pulses, np.float32),
                                                origin="video motion: " + Path(path).name),
                                start_s=clip.start_s, length_s=len(pulses) / FS,
                                name="Motion")
                self.doc.add_clip(track.id, new_clip)
            if cut_times:
                self.doc.add_markers([(clip.start_s + t, f"Cut {i + 1}")
                                      for i, t in enumerate(cut_times)], "Add scene cuts")
        finally:
            self.doc.undo.endMacro()
        parts = []
        if new_clip is not None:
            parts.append("motion clip added")
            self.editor.select_clips([new_clip.id])
        if cut_times:
            parts.append(f"{len(cut_times)} scene cut markers")
        self.statusBar().showMessage("Video analysis: " + ", ".join(parts) + ".", 5000)
        return new_clip

    def show_tool(self, tool_id: str):
        """Open (or bring up) a Tools window; one window per tool."""
        from .tools import tool_info
        win = self._tool_windows.get(tool_id)
        if win is None:
            win = self._tool_windows[tool_id] = tool_info(tool_id).factory(self)
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def add_color_rhythm(self, result: dict) -> list[Clip]:
        """Clips from the Color rhythm tool (``ColorRhythmWindow.result_data()``):
        one new standard track per clip, below the video track and lined up
        with the project video (another file: at the edit cursor), one undo step."""
        items = [c for c in result.get("clips", []) if c.get("pulses") is not None
                 and len(c["pulses"])]
        if not items:
            return []
        project = self.doc.project
        vt = project.video_track()
        vclip = vt.clips[0] if vt is not None and vt.clips else None
        path = result.get("path") or ""
        if (vclip is not None and result.get("from_project", True)
                and (not path or Path(vclip.source.path) == Path(path))):
            start, index = vclip.start_s, project.tracks.index(vt) + 1
        else:
            start, index = self.doc.cursor_s, None
        origin = "video colours: " + (Path(path).name if path else "?")
        clips = []
        self.doc.undo.beginMacro("Color rhythm")
        try:
            for i, item in enumerate(items):
                track = Track(name=item.get("name") or "Colour", type="standard",
                              color=item.get("color") or theme.TRACK_COLORS[0])
                self.doc.push(cmd.InsertTrack(self.doc, track,
                                              None if index is None else index + i))
                pulses = np.asarray(item["pulses"], np.float32)
                clip = Clip(PulseDataSource(pulses, origin=origin), start_s=start,
                            length_s=len(pulses) / FS, name=track.name)
                self.doc.add_clip(track.id, clip)
                clips.append(clip)
        finally:
            self.doc.undo.endMacro()
        self.editor.select_clips([c.id for c in clips])
        self.statusBar().showMessage(
            f"Color rhythm: {len(clips)} clip{'s' if len(clips) != 1 else ''} added — "
            "right-click → Beat marks from peaks… for beats", 6000)
        return clips

    def add_triphase_audio(self, result: dict) -> Clip | None:
        """The clip from Tools → Triphase audio (``{"path", "pulses", "triphase",
        "start_s"}``): a new standard or triphase track with the clip at *start_s*
        (None: the edit cursor), one undo step."""
        pulses = result.get("pulses")
        if pulses is None or not len(pulses):
            return None
        path = result.get("path") or ""
        name = Path(path).stem if path else "3φ audio"
        tri = bool(result.get("triphase"))
        pulses = np.asarray(pulses, np.float32)
        start = result.get("start_s")
        clip = Clip(PulseDataSource(pulses, origin="3φ audio: " + Path(path).name),
                    start_s=self.doc.cursor_s if start is None else float(start),
                    length_s=len(pulses) / FS, name=name)
        self.doc.add_tracks_with_clips([(name + (" (3φ)" if tri else ""), [clip])],
                                       "triphase" if tri else "standard", "Triphase audio")
        self.editor.select_clips([clip.id])
        self.statusBar().showMessage(
            f"Triphase audio: {'triphase' if tri else 'A/B'} clip added "
            f"({theme.format_time(len(pulses) / FS)}).", 5000)
        return clip

    # ── safety ──────────────────────────────────────────────────────────────
    def safety_ok(self, what: str = "play") -> bool:
        """Run the safety prompt; False if the user chose not to go on."""
        from .safety import prompt_before
        answer = prompt_before(self, self.safety, what)
        if answer == "show":
            self.safety_dock.show()
            self.safety_dock.raise_()
        return answer in ("go", "limit")

    def toggle_record(self, _checked=None):
        if not self.record.recording:
            self.record_dock.show()
            self.record_dock.raise_()
        self.record.toggle_record()
        self.a_record.setChecked(self.record.recording)

    def toggle_play(self):
        if self.playback.playing:
            self.playback.stop()
        elif self.safety_ok("play"):
            self.playback.play()

    def _show_issue(self, issue):
        if issue is None:
            return
        self.doc.set_cursor(issue.t0)
        self.editor.ensure_visible(issue.t0)
        if issue.track_id:
            try:
                track = self.doc.project.track_by_id(issue.track_id)
            except KeyError:
                return
            self.editor.set_current_track(track.id)
            hit = [c.id for c in track.clips if c.start_s - 0.05 <= issue.t0 <= c.end_s]
            if hit:
                self.editor.select_clips(hit[:1])

    def build_session(self, plan=None, start_s: float | None = None):
        """Session builder: two new tracks + phase markers, one undo step.

        *plan* (a ``core.session.SessionPlan``) skips the dialog — for tests.
        """
        from ..core.session import plan_tracks
        vt = self.doc.project.video_track()
        vclip = vt.clips[0] if vt is not None and vt.clips else None
        if plan is None:
            from .session_dialog import SessionDialog
            default = vclip.length_s if vclip is not None else 1200.0
            dlg = SessionDialog(self, self.doc.project.bpm, default,
                                vclip.start_s if vclip is not None else 0.0,
                                [m.t_s for m in self.doc.project.markers])
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return None
            plan, start_s = dlg.plan, dlg.start.value()
        start_s = 0.0 if start_s is None else float(start_s)
        tracks = plan_tracks(plan, offset_s=start_s)
        self.doc.undo.beginMacro("Session builder")
        try:
            for track in tracks:
                self.doc.push(cmd.InsertTrack(self.doc, track, None, "Add session track"))
            existing = {round(m.t_s, 2) for m in self.doc.project.markers}
            self.doc.add_markers([(start_s + t, name) for t, name in plan.markers
                                  if round(start_s + t, 2) not in existing], "Phase markers")
        finally:
            self.doc.undo.endMacro()
        self.editor.set_current_track(tracks[0].id)
        self.editor.zoom_fit()
        self.statusBar().showMessage(f"Session created: {len(plan.clips)} clips on two tracks.",
                                     5000)
        return tracks

    def add_marker_at_cursor(self):
        t = self.doc.cursor_s                 # where Shift+M was pressed, not after the dialog
        m = self.editor.create_marker(t)
        if m is not None:
            self.statusBar().showMessage(f"{m.name} at {theme.format_time(m.t_s)} — "
                                         "double-click its flag to edit", 3000)

    def loop_between_markers(self):
        span = self.editor.span_between_markers(self.doc.cursor_s)
        if span is None:
            self.statusBar().showMessage("Add markers first (Shift+M).", 3000)
            return
        self._on_loop_region_edited(span)

    def _snap_changed(self, index):
        kind, value = self.snap_combo.itemData(index)
        self.editor.snap = Snap(kind, value)
        self.editor.view.viewport().update()

    def _update_title(self):
        name = Path(self.doc.path).name if self.doc.path else "Untitled"
        self.setWindowTitle(f"{name}{' *' if self.doc.dirty else ''} — StimStudio")

    def _single_exportable_clip(self, clip_id: str | None = None):
        """The one selected (or given) clip if it can become a .parthwl, else None."""
        ids = [clip_id] if clip_id else self.editor.selected_clip_ids()
        if len(ids) != 1:
            return None
        found = self.doc.project.find_clip(ids[0])
        if found is None or found[1].source.kind in ("audio", "video", "marks"):
            return None
        return found[1]

    def _update_actions(self, *_):
        has_sel = bool(self.editor.selected_clip_ids())
        for a in (self.a_cut, self.a_copy, self.a_delete, self.a_duplicate, self.a_mute_clip):
            a.setEnabled(has_sel)
        single = self._single_exportable_clip() is not None
        self.a_export_part.setEnabled(single)
        self.a_save_preset.setEnabled(single)
        self.a_paste.setEnabled(bool(self.clipboard))
        has_track = self.editor.current_track_id is not None
        self.a_dup_track.setEnabled(has_track)
        self.a_del_track.setEnabled(has_track)

    # ── files ───────────────────────────────────────────────────────────────
    def _last_dir(self) -> str:
        return str(self.settings.value("lastDir", str(Path.home())))

    def _remember_dir(self, path: str) -> None:
        self.settings.setValue("lastDir", str(Path(path).parent))

    def maybe_save(self) -> bool:
        if not self.doc.dirty:
            return True
        r = QMessageBox.question(self, "StimStudio", "Save changes to the current project?",
                                 QMessageBox.StandardButton.Save
                                 | QMessageBox.StandardButton.Discard
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.file_save()
        return r == QMessageBox.StandardButton.Discard

    def file_new(self):
        if self.maybe_save():
            from .document import new_project
            self.doc.set_project(new_project())

    def file_open(self, path: str | None = None):
        if not self.maybe_save():
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self._last_dir(),
                                                  PROJECT_FILTER)
        if not path:
            return
        try:
            self.doc.open(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open project", f"Could not open {path}:\n{e}")
            return
        self._remember_dir(path)
        self._add_recent(path)
        self.editor.zoom_fit()
        self.check_missing_media()

    def show_media_dialog(self):
        from .media_dialog import ProjectMediaDialog
        dlg = ProjectMediaDialog(self.doc, self, save=self.file_save)
        dlg.exec()
        dlg.deleteLater()

    def check_missing_media(self, ask=None) -> int:
        """After opening: offer to find the files the project cannot reach.
        *ask* (tests) gets the missing list and returns True to open the dialog."""
        from ..core.media import missing_files
        missing = missing_files(self.doc.project)
        if not missing:
            return 0
        if ask is None:
            names = "\n".join(Path(m.path).name for m in missing[:6])
            more = f"\n… and {len(missing) - 6} more" if len(missing) > 6 else ""
            r = QMessageBox.warning(
                self, "Missing media",
                f"{len(missing)} file(s) used by the project were not found:\n\n{names}{more}"
                "\n\nFind them now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Ignore)
            go = r == QMessageBox.StandardButton.Yes
        else:
            go = ask(missing)
        if go:
            self.show_media_dialog()
        return len(missing)

    def file_save(self) -> bool:
        if not self.doc.path:
            return self.file_save_as()
        self.doc.save()
        self._add_recent(self.doc.path)
        self.statusBar().showMessage(f"Saved {self.doc.path}", 3000)
        return True

    def file_save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self._last_dir(),
                                              PROJECT_FILTER)
        if not path:
            return False
        if not path.endswith(EXTENSION):
            path += EXTENSION
        self.doc.save(path)
        self._remember_dir(path)
        self._add_recent(path)
        self.statusBar().showMessage(f"Saved {path}", 3000)
        return True

    # ── recent files ────────────────────────────────────────────────────────
    MAX_RECENT = 8

    def recent_files(self) -> list[str]:
        value = self.settings.value("recentFiles", [])
        if isinstance(value, str):
            value = [value]
        return [p for p in (value or []) if p]

    def _add_recent(self, path: str) -> None:
        path = str(Path(path))
        files = [path] + [p for p in self.recent_files() if Path(p) != Path(path)]
        self.settings.setValue("recentFiles", files[:self.MAX_RECENT])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        files = self.recent_files()
        for p in files:
            a = self.recent_menu.addAction(p)
            a.setEnabled(Path(p).exists())
            a.triggered.connect(lambda _c=False, path=p: self.file_open(path))
        if files:
            self.recent_menu.addSeparator()
            clear = self.recent_menu.addAction("Clear list")
            clear.triggered.connect(lambda: self.settings.setValue("recentFiles", []))
        else:
            self.recent_menu.addAction("(no recent projects)").setEnabled(False)

    # ── recovery ────────────────────────────────────────────────────────────
    def offer_recovery(self, ask=None) -> bool:
        """Offer the newest autosave left by an unclean exit. *ask* (tests) returns
        "recover" / "discard" / "later"."""
        from ..io.project_file import load_project
        from .autosave import find_recoverable, remove_autosave
        found = [(p, m) for p, m in find_recoverable(self.autosave.folder)
                 if p != self.autosave.project_path]
        if not found:
            return False
        path, meta = found[0]
        original = meta.get("original") or "an untitled project"
        if ask is None:
            import datetime
            when = datetime.datetime.fromtimestamp(meta.get("saved", 0)).strftime("%Y-%m-%d %H:%M")
            r = QMessageBox.question(
                self, "Recover project",
                f"StimStudio did not close normally.\n\nRecover the autosave of {original} "
                f"from {when}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            choice = {QMessageBox.StandardButton.Yes: "recover",
                      QMessageBox.StandardButton.Discard: "discard"}.get(r, "later")
        else:
            choice = ask(path, meta)
        if choice == "recover":
            try:
                base = Path(meta["original"]).parent if meta.get("original") else None
                project = load_project(path, base_dir=base)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Recover project", str(e))
                return False
            self.doc.set_project(project, meta.get("original"))
            self.doc.undo.resetClean()          # recovered work is unsaved
            remove_autosave(path)
            self.editor.zoom_fit()
            return True
        if choice == "discard":
            for p, _m in found:
                remove_autosave(p)
        return False

    def _target_track(self) -> str | None:
        tid = self.editor.current_track_id
        if tid is not None and self.doc.project.track_by_id(tid).type == "standard":
            return tid
        for t in self.doc.project.tracks:
            if t.type == "standard":
                return t.id
        return self.doc.add_track().id

    def import_hwl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import .hwl", self._last_dir(),
                                              "Howl files (*.hwl)")
        if not path:
            return
        try:
            data = read_hwl(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Import .hwl", str(e))
            return
        self._remember_dir(path)
        clip = Clip(PulseDataSource(data, origin=Path(path).name), start_s=self.doc.cursor_s,
                    length_s=len(data) / FS, name=Path(path).stem)
        self.doc.add_clip(self._target_track(), clip)
        self.editor.select_clips([clip.id])

    def import_parthwl(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import .parthwl", self._last_dir(),
                                              "Part files (*.parthwl)")
        if not path:
            return
        try:
            clip = part_to_clip(load_parthwl(path), start_s=self.doc.cursor_s)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Import .parthwl", str(e))
            return
        self._remember_dir(path)
        clip.name = clip.name or Path(path).stem
        self.doc.add_clip(self._target_track(), clip)
        self.editor.select_clips([clip.id])

    def add_triphase_track(self):
        n = sum(t.type == "triphase" for t in self.doc.project.tracks) + 1
        track = self.doc.add_track(f"Triphase {n}", track_type="triphase")
        self.editor.set_current_track(track.id)
        return track

    def show_triphase_calibration(self):
        from .widgets.triphase_cal import TriphaseCalibrationDialog
        if getattr(self, "_tri_cal_dialog", None) is None:
            self._tri_cal_dialog = TriphaseCalibrationDialog(self.doc, self)
        self._tri_cal_dialog.show()
        self._tri_cal_dialog.raise_()
        return self._tri_cal_dialog

    def add_mod_track(self):
        n = sum(t.type == "audio_mod" for t in self.doc.project.tracks) + 1
        track = self.doc.add_track(f"Modulator {n}", track_type="audio_mod")
        self.editor.set_current_track(track.id)
        return track

    def _clip_activated(self, clip_id: str) -> None:
        found = self.doc.project.find_clip(clip_id)
        if found is not None and found[0].type == "video":
            self.video_dock.show()
            self.video_dock.raise_()
        else:
            self.pattern_dock.raise_()

    def open_video(self, path: str | None = None, ask_funscripts=None):
        """Put a video on the video track (replacing the current one, keeping its
        position) and offer to import funscripts found next to it.

        *ask_funscripts(paths) -> bool* replaces the question (tests).
        Returns the video clip, or None.
        """
        from ..io.video import VIDEO_EXTENSIONS, find_sibling_funscripts
        from .video.media import probe_duration
        if path is None:
            exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
            path, _ = QFileDialog.getOpenFileName(self, "Open video", self._last_dir(),
                                                  f"Video ({exts});;All files (*)")
            if not path:
                return None
        self._remember_dir(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            duration = probe_duration(path)
        finally:
            QApplication.restoreOverrideCursor()
        if duration <= 0:
            QMessageBox.warning(self, "Open video", f"Could not read the video:\n{path}")
            return None
        clip = self.doc.set_video(path, duration)
        self.video_dock.show()
        self.video_dock.raise_()
        self.statusBar().showMessage(f"Video: {Path(path).name} "
                                     f"({theme.format_time(duration)})", 5000)
        scripts = find_sibling_funscripts(path)
        if scripts:
            names = "\n".join(Path(s).name for s in scripts)
            if ask_funscripts is not None:
                ok = ask_funscripts(scripts)
            else:
                ok = QMessageBox.question(
                    self, "Funscript found",
                    f"Found next to the video:\n{names}\n\nImport at the video's position "
                    f"({theme.format_time(clip.start_s)})?") == QMessageBox.StandardButton.Yes
            if ok:
                self.import_funscript(scripts, start_s=clip.start_s)
        return clip

    def import_audio(self, path: str | None = None, start_s: float | None = None,
                     track_id: str | None = None, mode: str | None = None,
                     method: str = "rolloff"):
        """Import an audio file (dialogs fill in whatever is not given).

        *mode* "convert" → pulse clip on a standard track; "analyze" → audio
        clip on a modulator track; "triphase" → Tools → Triphase audio opens with
        the file (its clip lands at *start_s*). Without a mode, a look at three
        excerpts says three-phase or two-channel and the import dialog asks
        (a file dropped on a modulator track becomes a modulator directly).
        Returns the new clip, or None.
        """
        from .audio_import import AudioImportDialog, AudioJob, run_job
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import audio", self._last_dir(),
                "Audio (*.wav *.flac *.ogg *.mp3 *.aiff *.aif);;All files (*)")
            if not path:
                return None
        self._remember_dir(path)
        target = None
        if track_id:
            try:
                target = self.doc.project.track_by_id(track_id)
            except KeyError:
                target = None
        name = Path(path).stem
        if mode is None and target is not None and target.type == "audio_mod":
            mode = "analyze"
        if mode is None:
            try:
                probe = run_job(self, AudioJob(path, "probe"), f"Looking at {name}")
            except RuntimeError as e:
                QMessageBox.critical(self, "Import audio", f"Could not read {path}:\n{e}")
                return None
            if probe is None:
                self.statusBar().showMessage("Import cancelled.", 3000)
                return None
            dlg = AudioImportDialog(path, probe, self)
            if dlg.exec() != dlg.DialogCode.Accepted or dlg.mode is None:
                return None
            mode, method = dlg.mode, dlg.method.currentData()
        start = self.doc.cursor_s if start_s is None else start_s
        if mode == "triphase":
            self.show_tool("triphase_audio").open_file(path, start_s=start)
            return None
        try:
            result = run_job(self, AudioJob(path, mode, method),
                             f"{'Converting' if mode == 'convert' else 'Analysing'} {name}")
        except RuntimeError as e:
            QMessageBox.critical(self, "Import audio", f"Could not import {path}:\n{e}")
            return None
        if result is None:
            self.statusBar().showMessage("Import cancelled.", 3000)
            return None
        if mode == "convert":
            clip = Clip(PulseDataSource(result, origin=Path(path).name), start_s=start,
                        length_s=len(result) / FS, name=name)
            tid = target.id if target is not None and target.type == "standard" \
                else self._target_track()
            self.doc.add_clip(tid, clip)
        else:
            length = len(result["envelope"]) / FS
            clip = Clip(AudioFileSource(str(path)), start_s=start, length_s=length, name=name)
            self.doc.undo.beginMacro("Import modulator audio")
            if target is None or target.type != "audio_mod":
                n = len(self.doc.project.tracks)
                target = Track(name=name, type="audio_mod",
                               color=theme.TRACK_COLORS[n % len(theme.TRACK_COLORS)])
                from . import commands as cmd
                self.doc.push(cmd.InsertTrack(self.doc, target))
            self.doc.add_clip(target.id, clip)
            self.doc.undo.endMacro()
            self.mod_dock.raise_()
        self.editor.select_clips([clip.id])
        self.statusBar().showMessage(f"Imported {Path(path).name} "
                                     f"({theme.format_time(clip.length_s)})", 5000)
        return clip

    def export_hwl(self):
        if not self.safety_ok("export"):
            return
        default = str(Path(self._last_dir()) / ((Path(self.doc.path).stem if self.doc.path
                                                 else "untitled") + ".hwl"))
        path, _ = QFileDialog.getSaveFileName(self, "Export .hwl", default, "Howl files (*.hwl)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            master = self.doc.render_now().master
            size = write_hwl(path, master)
        finally:
            QApplication.restoreOverrideCursor()
        self._remember_dir(path)
        self.statusBar().showMessage(
            f"Exported {theme.format_time(len(master) / FS)} ({size} bytes) → {path}", 5000)

    def export_audio(self):
        """The export dialog set to WAV: the Signal box shows what will be written,
        Export / Cancel decide. The dialog stops the output before writing."""
        return self.show_export_dialog("wav")

    def export_funscript(self):
        return self.show_export_dialog("funscript")

    def show_export_dialog(self, fmt: str | None = None):
        from .export_dialog import ExportDialog
        if not self.safety_ok("export"):
            return None
        dlg = ExportDialog(self, fmt=fmt)
        dlg.exec()
        return dlg

    def import_funscript(self, paths=None, start_s: float | None = None,
                         track_id: str | None = None, mapping=None):
        """Import funscript file(s) as a pulse clip (*mapping* skips the dialog)."""
        from .funscript_import import FunscriptImportDialog, load_axes
        from ..io.funscript import INTERP_LINEAR, funscript_to_pulses
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "Import funscript", self._last_dir(),
                                                    "Funscript (*.funscript);;All files (*)")
            if not paths:
                return None
        self._remember_dir(paths[0])
        tracks = self.doc.project.tracks
        target = next((t for t in tracks if t.id == (track_id or self.editor.current_track_id)
                       and t.type in ("standard", "triphase")), None)
        if target is None:
            target = self.doc.project.track_by_id(self._target_track())
        if mapping is None:
            dlg = FunscriptImportDialog(paths, target.type, self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return None
            pulses = dlg.pulses()
        else:
            axes, errors = load_axes(paths)
            if not axes:
                QMessageBox.critical(self, "Import funscript", "\n".join(errors) or "No axes")
                return None
            pulses = funscript_to_pulses(axes, mapping, FS, INTERP_LINEAR)
        name = Path(paths[0]).stem.split(".")[0]
        clip = Clip(PulseDataSource(pulses, origin=", ".join(Path(p).name for p in paths)),
                    start_s=self.doc.cursor_s if start_s is None else start_s,
                    length_s=len(pulses) / FS, name=name)
        self.doc.add_clip(target.id, clip)
        self.editor.select_clips([clip.id])
        return clip

    def import_beats(self, paths=None, start_s: float | None = None, options=None):
        """Import beat timestamp CSV file(s) as tracks of beat-marks blocks.
        *options* (``BeatImportDialog.options()``) skips the dialog."""
        from .beat_import import BeatImportDialog, beat_track_items, load_beat_files
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "Import beat timestamps",
                                                    self._last_dir(),
                                                    "Beat timestamps (*.csv);;All files (*)")
            if not paths:
                return []
        self._remember_dir(paths[0])
        offset = 0.0 if start_s is None else start_s
        if options is None:
            dlg = BeatImportDialog(paths, offset, self)
            if not dlg.tracks or dlg.exec() != dlg.DialogCode.Accepted:
                return []
            tracks, options = dlg.tracks, dlg.options()
        else:
            tracks, errors = load_beat_files(paths)
            if errors:
                QMessageBox.critical(self, "Import beat timestamps", "\n".join(errors))
                return []
        items = beat_track_items(tracks, options.get("offset_s", offset), options.get("tracks"))
        if not items:
            return []
        added = self.doc.add_tracks_with_clips(items, options.get("track_type", "standard"),
                                               "Import beat timestamps")
        self.editor.set_current_track(added[0].id)
        n = sum(len(clips) for _name, clips in items)
        self.statusBar().showMessage(f"Imported {n} beat blocks on {len(added)} tracks", 4000)
        return added

    def marks_from_peaks(self, clip_id: str, options=None):
        """Beat marks from a clip's peaks (dialog unless *options* is given,
        see ``peak_marks.DEFAULTS``). Returns the new / changed clip."""
        from ..core.beats import clip_level_signal
        from .peak_marks import PeakMarksDialog, peak_marks
        found = self.doc.project.find_clip(clip_id)
        if found is None or found[1].source.kind not in ("generator", "pulses", "pattern"):
            return None
        track, clip = found
        domain = self.doc.track_domain(track)
        signal = clip_level_signal(clip, self.doc.cache, self.doc.project.bpm, domain)
        if options is None:
            dlg = PeakMarksDialog(signal, f"{clip.name or 'Clip'} · "
                                          f"{theme.format_time(clip.length_s)}", self)
            if dlg.exec() != dlg.DialogCode.Accepted:
                return None
            options, marks = dlg.options(), dlg.marks()
        else:
            marks = peak_marks(signal, options)
        if not marks:
            self.statusBar().showMessage("No peaks above that level", 4000)
            return None
        new = self.doc.add_marks_from_peaks(clip_id, marks, bool(options.get("replace")))
        if new is not None:
            self.editor.select_clips([new.id])
        return new

    def show_shortcuts(self):
        from .help import ShortcutsDialog
        dlg = ShortcutsDialog(self)
        dlg.show()
        return dlg

    def show_support(self):
        from .donate_dialog import show_donate_dialog
        show_donate_dialog(self)

    def show_about(self):
        from .help import about_text
        QMessageBox.about(self, "About StimStudio", about_text())

    def export_parthwl(self):
        clip = self._single_exportable_clip()
        if clip is None:
            return
        default = str(Path(self._last_dir()) / f"{clip.name or 'clip'}.parthwl")
        path, _ = QFileDialog.getSaveFileName(self, "Export clip", default,
                                              "Part files (*.parthwl)")
        if not path:
            return
        found = self.doc.project.find_clip(clip.id)
        domain = self.doc.track_domain(found[0]) if found else "pulse"
        rendered = render_source(clip.source, n_pulses(clip.length_s), n_pulses(clip.trim_s),
                                 self.doc.cache, self.doc.project.bpm, domain)
        save_parthwl(path, clip_to_part(clip, rendered))
        self._remember_dir(path)
        self.statusBar().showMessage(f"Exported clip → {path}", 5000)

    # ── edit / track actions ────────────────────────────────────────────────
    def edit_copy(self):
        self.clipboard = []
        for cid in self.editor.selected_clip_ids():
            track, clip = self.doc.project.find_clip(cid)
            self.clipboard.append((track.id, clip.clone()))
        self._update_actions()

    def edit_cut(self):
        self.edit_copy()
        self.editor.delete_selected()

    def edit_paste(self):
        clips = self.doc.paste_clips(self.clipboard, self.doc.cursor_s)
        self.editor.select_clips([c.id for c in clips])

    def add_track(self):
        track = self.doc.add_track()
        self.editor.set_current_track(track.id)

    def duplicate_track(self):
        if self.editor.current_track_id:
            track = self.doc.duplicate_track(self.editor.current_track_id)
            self.editor.set_current_track(track.id)

    def delete_track(self):
        if self.editor.current_track_id:
            self.doc.remove_track(self.editor.current_track_id)

    def _show_track_lanes(self, track_id: str):
        if track_id not in self.editor.expanded:
            self.editor.toggle_expanded(track_id)

    def _add_preset_at_cursor(self, path: str):
        from ..io.presets import load_preset
        try:
            clip = load_preset(path, self.doc.cursor_s)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Preset", f"Could not load {path}:\n{e}")
            return
        self.doc.add_clip(self._target_track(), clip)
        self.editor.select_clips([clip.id])

    def save_clip_preset(self, clip_id: str | None = None):
        from ..io.presets import safe_filename, save_preset
        clip = self._single_exportable_clip(clip_id)
        if clip is None:
            self.statusBar().showMessage("Select one generator or pulse clip to save.", 4000)
            return
        name, ok = QInputDialog.getText(self, "Save as preset", "Preset name:",
                                        text=clip.name or "My preset")
        if not ok or not name.strip():
            return
        folder = self.browser.presets.user_dir
        path = folder / f"{safe_filename(name.strip())}.parthwl"
        if path.exists() and QMessageBox.question(
                self, "Save as preset", f"Replace the existing preset “{path.stem}”?") \
                != QMessageBox.StandardButton.Yes:
            return
        preset = clip.clone()
        preset.name = name.strip()
        found = self.doc.project.find_clip(clip.id)
        domain = self.doc.track_domain(found[0]) if found else "pulse"
        save_preset(preset, path, self.doc.cache, self.doc.project.bpm, domain)
        self.browser.presets.refresh()
        self.browser.setCurrentWidget(self.browser.presets)
        self.statusBar().showMessage(f"Saved preset → {path}", 5000)

    def _add_generator_at_cursor(self, type_id: str):
        from ..generators import get_generator
        if get_generator(type_id).domain == "triphase":
            cur = self.editor.current_track_id
            tracks = self.doc.project.tracks
            target = next((t for t in tracks if t.id == cur and t.type == "triphase"), None) \
                or next((t for t in tracks if t.type == "triphase"), None) \
                or self.add_triphase_track()
            tid = target.id
        else:
            tid = self._target_track()
        clip = self.doc.add_generator_clip(tid, type_id, self.doc.cursor_s,
                                           self.editor.default_clip_length())
        self.editor.select_clips([clip.id])

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending_dock_sizes:
            QTimer.singleShot(0, self._apply_first_run_docks)
        elif not self._shown_once and not (self.isMaximized() or self.isFullScreen()):
            # a size saved on a bigger monitor, or at a smaller display scale
            QTimer.singleShot(0, lambda: metrics.keep_on_screen(self))
        QTimer.singleShot(0, self._unelide_dock_tabs)
        self._shown_once = True

    def _apply_first_run_docks(self):
        if not self._pending_dock_sizes:
            return
        browser, inspector, preview, output = self._pending_dock_sizes
        self._first_run_docks = self._pending_dock_sizes
        self._pending_dock_sizes = None
        self.tabifyDockWidget(preview, self.pattern_dock)
        self.tabifyDockWidget(inspector, self.effects_dock)
        self.tabifyDockWidget(inspector, self.mod_dock)
        self.tabifyDockWidget(output, self.video_dock)
        self.tabifyDockWidget(inspector, self.safety_dock)
        self.tabifyDockWidget(output, self.record_dock)
        output.raise_()
        preview.raise_()
        inspector.raise_()
        # the separate docks made the window grow while they were stacked: back to
        # the screen, then dock sizes as shares of it (chosen on 1500×900)
        avail = metrics.available(self)
        if avail.width() < 1400 or avail.height() < 800:
            self.showMaximized()
        else:
            metrics.fit_to_screen(self, 1500, 900, frac=0.95)
            metrics.keep_on_screen(self)
        QTimer.singleShot(0, self._apply_first_run_dock_sizes)
        QTimer.singleShot(0, self._unelide_dock_tabs)

    def _apply_first_run_dock_sizes(self):
        browser, inspector, preview, output = self._first_run_docks
        w, h = self.width(), self.height()
        # the right column holds the Output panel: not under ~320 px on small screens
        right = max(round(w * 0.22), min(320, round(w * 0.3)))
        self.resizeDocks([browser, inspector], [round(w * 0.14), right],
                         Qt.Orientation.Horizontal)
        self.resizeDocks([preview], [round(h * 0.3)], Qt.Orientation.Vertical)
        self.resizeDocks([inspector, output], [round(h * 0.42), round(h * 0.47)],
                         Qt.Orientation.Vertical)

    # ── shutdown ────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if not self.maybe_save():
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.video.shutdown()
        self.playback.shutdown()
        self.autosave.stop()
        self.doc.shutdown()
        event.accept()
