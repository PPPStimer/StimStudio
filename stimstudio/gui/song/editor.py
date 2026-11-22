"""SongEditor: ruler + track headers + OpenGL song view, kept in sync with the document.

The editor never edits the model directly while a gesture is in progress: a
drag only moves graphics items, and the release turns into one undoable
command. Items are reconciled with the model by id on every ``modelChanged``,
so selection and drag state survive edits made elsewhere (inspector, undo).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QGraphicsScene, QGridLayout, QLabel, QMenu, QWidget

from ...core.joins import track_layers
from ...core.model import Clip, Envelope, Lane
from ...core.pulse import LANE_KEYS
from ...core.pulse import FS
from .. import commands as cmd
from .. import theme
from .headers import HEADER_W, HeaderPanel
from .envelope import EnvelopeItem
from ..pattern.triphase_page import TRI_COLORS
from .items import SIGNAL_COLORS, ClipItem, LaneCurveItem
from ...audio.analysis import SIGNALS
from .layout import N_LANES, N_SIGNAL_LANES, Snap, SongLayout, nearest_magnet
from .ruler import RULER_H, Ruler
from .view import SongView
from .overview import OVERVIEW_H, Overview

MAGNET_PX = 8        # markers attract snapped times within this many pixels

MIN_PPS, MAX_PPS = 0.5, 1200.0
MIN_CLIP_S = 0.1
TAIL_S = 120.0          # empty timeline kept after the last clip


@dataclass
class _Drag:
    zone: str
    primary: ClipItem
    origin: QPointF
    items: list[ClipItem]
    start: dict = field(default_factory=dict)    # clip_id -> dict of original values
    result: dict = field(default_factory=dict)   # clip_id -> dict of new values
    moved: bool = False


class SongEditor(QWidget):
    selectionChanged = pyqtSignal(list)          # selected clip ids
    visibleRangeChanged = pyqtSignal(float, float)
    currentTrackChanged = pyqtSignal(str)
    statusMessage = pyqtSignal(str)
    loopRegionEdited = pyqtSignal(object)        # (start_s, end_s) or None
    clipActivated = pyqtSignal(str)              # double-clicked clip id
    audioDropped = pyqtSignal(str, float, str)   # path, start (s), track id ("" = none)
    funscriptDropped = pyqtSignal(list, float, str)  # paths, start (s), track id
    savePresetRequested = pyqtSignal(str)        # clip id
    videoDropped = pyqtSignal(str)               # video file path
    videoRequested = pyqtSignal()                # Replace video…
    videoAnalysisRequested = pyqtSignal(bool, bool)  # motion → stim, scene cuts
    peakMarksRequested = pyqtSignal(str)         # clip id: Beat marks from peaks…
    beatsDropped = pyqtSignal(list, float)       # beat timestamp CSV paths, start (s)

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.pps = 20.0
        self.snap = Snap("beats", 1.0)
        self.expanded: set[str] = set()
        self.current_track_id: str | None = None
        self.layout: SongLayout | None = None
        self.clip_items: dict[str, ClipItem] = {}
        self.lane_items: dict[tuple[str, int], LaneCurveItem] = {}
        self.env_items: dict[tuple[str, tuple], EnvelopeItem] = {}
        self.drop_marker: tuple[float, float] | None = None
        self.playhead_s = -1.0                       # running output position; -1 = none
        self.loop_region: tuple[float, float] | None = None
        self.loop_enabled = False
        #: returns the generator type id chosen in the browser
        self.generator_provider = lambda: "waveform"
        self._drag: _Drag | None = None
        self.safety_report = None                    # core.safety.SafetyReport
        self.heat_img = None                         # QImage, one pixel per bin
        self.heat_bin_s = 0.25

        self.scene = QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._on_selection)
        self.view = SongView(self, self.scene)
        self.ruler = Ruler(self)
        self.headers = HeaderPanel(self)
        self.overview = Overview(self)
        self.overview_corner = QLabel("  Overview")
        self.overview_corner.setFixedSize(HEADER_W, OVERVIEW_H)
        self.overview_corner.setStyleSheet(f"background: {theme.PANEL_BG}; "
                                           f"color: {theme.TEXT_DIM}; font-size: 10px;")
        corner = QWidget()
        corner.setFixedSize(HEADER_W, RULER_H)
        corner.setStyleSheet(f"background: {theme.PANEL_BG};")

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        grid.addWidget(self.overview_corner, 0, 0)
        grid.addWidget(self.overview, 0, 1)
        grid.addWidget(corner, 1, 0)
        grid.addWidget(self.ruler, 1, 1)
        grid.addWidget(self.headers, 2, 0)
        grid.addWidget(self.view, 2, 1)
        self.overview.scrollRequested.connect(self.center_on)

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild_visible)

        self.view.horizontalScrollBar().valueChanged.connect(self._on_hscroll)
        self.view.verticalScrollBar().valueChanged.connect(
            lambda v: self.headers.set_offset(float(v)))
        self.view.zoomRequested.connect(self.zoom)
        doc.modelChanged.connect(self.refresh)
        doc.renderReady.connect(self._on_render)
        doc.cursorChanged.connect(self._on_cursor)
        from ..settings import app_settings
        from ..video.media import thumbnail_cache, waveform_cache
        self._view_settings = app_settings()
        self.video_thumbs = self._view_settings.value("view/video_thumbs", True, bool)
        self.set_overview_visible(self._view_settings.value("view/overview", True, bool),
                                  remember=False)
        self._seen_video: set[str] = set()
        thumbnail_cache().updated.connect(self._on_thumbnails)
        waveform_cache().updated.connect(self._on_thumbnails)
        self.refresh()

    # ── video track view (filmstrip / sound lane) ───────────────────────────
    def set_video_thumbs(self, on: bool) -> None:
        self.video_thumbs = bool(on)
        self._view_settings.setValue("view/video_thumbs", self.video_thumbs)
        self.refresh()

    def set_video_audio(self, track_id: str, on: bool) -> None:
        """Show / hide the sound lane of the video track (remembered for new videos)."""
        if on:
            self.expanded.add(track_id)
        else:
            self.expanded.discard(track_id)
        self._view_settings.setValue("view/video_audio", bool(on))
        self.refresh()

    # ── intensity map / safety ──────────────────────────────────────────────
    def set_safety_report(self, report) -> None:
        from .overview import heat_image
        self.safety_report = report
        self.heat_bin_s = report.profile_bin_s
        self.heat_img = heat_image(report.profile)
        self.ruler.update()
        self.overview.viewport().update()

    def set_overview_visible(self, on: bool, remember: bool = True) -> None:
        self.overview.setVisible(on)
        self.overview_corner.setVisible(on)
        if remember:
            self._view_settings.setValue("view/overview", bool(on))

    def center_on(self, seconds: float) -> None:
        t0, t1 = self.visible_range()
        bar = self.view.horizontalScrollBar()
        bar.setValue(round(max(0.0, seconds - (t1 - t0) / 2) * self.pps))

    def _on_thumbnails(self, path: str) -> None:
        import os
        for item in self.clip_items.values():
            if item.video and os.path.abspath(item.video_path) == path:
                item.update()

    @property
    def project(self):
        return self.doc.project

    def sizeHint(self) -> QSize:
        return QSize(1000, 520)

    # ── reconcile with the model ────────────────────────────────────────────
    def refresh(self) -> None:
        project = self.project
        ids = [t.id for t in project.tracks]
        self.expanded &= set(ids)
        for t in project.tracks:            # a new video track opens with its sound lane
            if t.type == "video" and t.id not in self._seen_video:
                self._seen_video.add(t.id)
                if self._view_settings.value("view/video_audio", True, bool):
                    self.expanded.add(t.id)
        if self.current_track_id not in ids:
            self.current_track_id = ids[0] if ids else None
            self.currentTrackChanged.emit(self.current_track_id or "")
        self.layout = SongLayout(project, self.expanded)
        self.headers.sync(project, self.layout, self.expanded, self.current_track_id)

        seen = set()
        for track in project.tracks:
            row = self.layout.row(track.id)
            layers = track_layers(track)
            for clip in track.clips:
                item = self.clip_items.get(clip.id)
                if item is None:
                    item = self.clip_items[clip.id] = ClipItem(self, clip.id)
                    self.scene.addItem(item)
                item.sync(clip, track, row.y, self.pps)
                item.set_layer(layers.get(clip.id))
                if item.video:
                    item.set_video_view(self.video_thumbs, row.n_lanes > 0)
                seen.add(clip.id)
        for cid in [c for c in self.clip_items if c not in seen]:
            self.scene.removeItem(self.clip_items.pop(cid))

        wanted = set()
        for track in project.tracks:
            row = self.layout.row(track.id)
            if not row.n_lanes:
                continue
            audio = row.kind == "audio_mod"
            for lane in range(N_SIGNAL_LANES if audio else N_LANES):
                key = (track.id, lane)
                wanted.add(key)
                item = self.lane_items.get(key)
                if item is None:
                    if audio:
                        item = LaneCurveItem(track.id, lane, SIGNAL_COLORS[lane], fill=True,
                                             mask_inactive=False)
                    elif row.kind == "triphase":
                        item = LaneCurveItem(track.id, lane, TRI_COLORS[lane], fill=lane == 0,
                                             mask_inactive=lane > 0)
                    else:
                        item = LaneCurveItem(track.id, lane)
                    item.signal = SIGNALS[lane] if audio else None
                    self.lane_items[key] = item
                    self.scene.addItem(item)
                    self._fill_lane(item)
                item.setPos(0, row.lane_y(lane))
        for key in [k for k in self.lane_items if k not in wanted]:
            self.scene.removeItem(self.lane_items.pop(key))
        self._sync_envelopes(project)

        self._update_scene_rect()
        self.view.viewport().update()
        self.ruler.update()
        self.overview.viewport().update()
        self._rebuild_timer.start()

    def _sync_envelopes(self, project) -> None:
        """One EnvelopeItem per lane envelope and per automated effect parameter."""
        wanted = {}
        for track in project.tracks:
            row = self.layout.row(track.id)
            if not row.n_lanes:
                continue
            for i, key in enumerate(LANE_KEYS):
                env = track.lanes[key].envelope
                if env is not None:
                    wanted[(track.id, ("lane", key))] = (
                        row.lane_y(i), env.points, env.interp, env.enabled, theme.LANE_COLORS[i])
            for j, auto in enumerate(row.auto_lanes):
                env = track.effects[auto.fx_index]["automation"][auto.key]
                wanted[(track.id, ("fx", auto.fx_index, auto.key))] = (
                    row.lane_y(4 + j), env.get("points", []), env.get("interp", "linear"),
                    env.get("enabled", True), "#c586c0")
        for key in [k for k in self.env_items if k not in wanted]:
            self.scene.removeItem(self.env_items.pop(key))
        for key, (y, points, interp, enabled, color) in wanted.items():
            item = self.env_items.get(key)
            if item is None:
                item = self.env_items[key] = EnvelopeItem(self, key[0], key[1], color)
                self.scene.addItem(item)
            item.setPos(0, y)
            item.set_envelope(points, interp, enabled)

    # ── envelope commands ───────────────────────────────────────────────────
    def commit_envelope(self, track_id: str, ref: tuple, points) -> None:
        points = [(round(t, 4), round(v, 4)) for t, v in points]
        track = self.project.track_by_id(track_id)
        if ref[0] == "lane":
            lane = track.lanes[ref[1]].clone()
            lane.envelope.points = points
            self.doc.push(cmd.SetLane(self.doc, track_id, ref[1], lane, "Edit envelope"))
        else:
            _kind, index, key = ref
            effects = copy.deepcopy(track.effects)
            effects[index]["automation"][key]["points"] = [list(p) for p in points]
            self.doc.set_track_attrs(track_id, text="Edit automation", effects=effects)

    def _edit_lane(self, track_id: str, key: str, fn, text: str) -> None:
        lane = self.project.track_by_id(track_id).lanes[key].clone()
        fn(lane)
        self.doc.push(cmd.SetLane(self.doc, track_id, key, lane, text))

    def toggle_lane_envelope(self, track_id: str, key: str, on: bool) -> None:
        """Create the envelope on first use (flat: no change), else enable/disable it."""
        def fn(lane):
            if lane.envelope is None:
                lane.envelope = Envelope(points=[(0.0, 1.0)], mode="multiply")
            lane.envelope.enabled = on
        self._edit_lane(track_id, key, fn, "Envelope on" if on else "Envelope off")

    def set_lane_envelope_mode(self, track_id: str, key: str, mode: str) -> None:
        def fn(lane):
            if lane.envelope is not None:
                lane.envelope.mode = mode
        self._edit_lane(track_id, key, fn, "Envelope mode")

    def set_lane_envelope_interp(self, track_id: str, key: str, interp: str) -> None:
        def fn(lane):
            if lane.envelope is not None:
                lane.envelope.interp = interp
        self._edit_lane(track_id, key, fn, "Envelope curve")

    def clear_lane_envelope(self, track_id: str, key: str) -> None:
        def fn(lane):
            lane.envelope = None
        self._edit_lane(track_id, key, fn, "Delete envelope")

    def remove_fx_automation(self, track_id: str, fx_index: int, key: str) -> None:
        track = self.project.track_by_id(track_id)
        effects = copy.deepcopy(track.effects)
        effects[fx_index].get("automation", {}).pop(key, None)
        self.doc.set_track_attrs(track_id, text="Remove automation", effects=effects)

    def _fill_lane(self, item: LaneCurveItem) -> None:
        render = self.doc.last_render
        tr = render.tracks.get(item.track_id) if render else None
        if tr is None:
            item.set_data(None, None)
        elif getattr(item, "signal", None):
            item.set_data(tr.signals[item.signal] if tr.signals else None, tr.active)
        elif tr.triphase is not None:
            item.set_data(tr.triphase[:, item.lane], tr.active)
        else:
            item.set_data(tr.pulses[:, item.lane], tr.active)

    def _on_render(self, render, _seconds: float) -> None:
        for item in self.clip_items.values():
            tr = render.tracks.get(item.track_id)
            if tr is None or item.dragging:
                continue
            i0 = round(item.start_s * FS)
            i1 = i0 + round(item.length_s * FS)
            lo, hi = max(0, i0), max(0, i1)
            if tr.signals is not None:   # audio clip: envelope + onsets
                prev = np.zeros((hi - lo, 4), np.float32)
                prev[:, 0] = tr.signals["envelope"][lo:hi]
                prev[:, 1] = tr.signals["onsets"][lo:hi]
                item.set_preview(prev)
            else:
                item.set_preview(tr.pulses[lo:hi])
        for item in self.lane_items.values():
            self._fill_lane(item)
        self._update_scene_rect()
        self._rebuild_timer.start()

    def _update_scene_rect(self) -> None:
        vp = self.view.viewport()
        width = max((self.project.duration_s() + TAIL_S) * self.pps, vp.width())
        height = max((self.layout.height if self.layout else 0) + 120, vp.height())
        self.scene.setSceneRect(QRectF(0, 0, width, height))
        for item in self.lane_items.values():
            item.set_width(width)
        for item in self.env_items.values():
            item.set_width(width)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scene_rect()
        self._rebuild_timer.start()

    # ── visible window ──────────────────────────────────────────────────────
    def visible_range(self) -> tuple[float, float]:
        r = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        return max(0.0, r.left() / self.pps), max(0.0, r.right() / self.pps)

    def _on_hscroll(self, value: int) -> None:
        self.ruler.set_offset(float(value))
        self.overview.viewport().update()
        self._rebuild_timer.start()

    def _rebuild_visible(self) -> None:
        t0, t1 = self.visible_range()
        margin = (t1 - t0) * 0.5
        a, b = max(0.0, t0 - margin), t1 + margin
        for item in self.clip_items.values():
            item.rebuild_preview(a, b)
        for item in self.lane_items.values():
            item.rebuild(a, b, self.pps)
        for item in self.env_items.values():
            item.rebuild(a, b, self.pps)
        self.visibleRangeChanged.emit(t0, t1)

    # ── zoom ────────────────────────────────────────────────────────────────
    def zoom(self, factor: float, anchor_t: float | None = None) -> None:
        bar = self.view.horizontalScrollBar()
        if anchor_t is None:
            t0, t1 = self.visible_range()
            anchor_t = (t0 + t1) / 2
        screen_x = anchor_t * self.pps - bar.value()
        new_pps = min(MAX_PPS, max(MIN_PPS, self.pps * factor))
        if new_pps == self.pps:
            return
        self.pps = new_pps
        self.refresh()
        bar.setValue(round(anchor_t * self.pps - screen_x))
        self._rebuild_timer.start()

    def zoom_fit(self) -> None:
        duration = max(10.0, self.project.duration_s())
        width = max(100, self.view.viewport().width() - 20)
        self.zoom(width / duration / self.pps, 0.0)
        self.view.horizontalScrollBar().setValue(0)

    # ── selection / current track ───────────────────────────────────────────
    def selected_clip_ids(self) -> list[str]:
        return [it.clip_id for it in self.scene.selectedItems() if isinstance(it, ClipItem)]

    def select_clips(self, clip_ids) -> None:
        self.scene.blockSignals(True)
        self.scene.clearSelection()
        for cid in clip_ids:
            item = self.clip_items.get(cid)
            if item is not None:
                item.setSelected(True)
        self.scene.blockSignals(False)
        self._on_selection()

    def select_all(self) -> None:
        self.select_clips(list(self.clip_items))

    def _on_selection(self) -> None:
        self.selectionChanged.emit(self.selected_clip_ids())

    def set_current_track(self, track_id: str | None) -> None:
        if track_id != self.current_track_id:
            self.current_track_id = track_id
            self.headers.sync(self.project, self.layout, self.expanded, track_id)
            self.view.viewport().update()
            self.currentTrackChanged.emit(track_id or "")

    def toggle_expanded(self, track_id: str) -> None:
        self.expanded ^= {track_id}
        self.refresh()

    def move_track(self, track_id: str, delta: int) -> None:
        idx = [t.id for t in self.project.tracks].index(track_id)
        if 0 <= idx + delta < len(self.project.tracks):
            self.doc.push(cmd.MoveTrack(self.doc, track_id, idx + delta))

    def set_lane_muted(self, track_id: str, key: str, muted: bool) -> None:
        lane: Lane = self.project.track_by_id(track_id).lanes[key].clone()
        lane.muted = muted
        self.doc.push(cmd.SetLane(self.doc, track_id, key, lane,
                                  "Mute lane" if muted else "Unmute lane"))

    # ── clip dragging ───────────────────────────────────────────────────────
    def begin_drag(self, item: ClipItem, zone: str, scene_pos: QPointF) -> None:
        items = ([it for it in self.scene.selectedItems() if isinstance(it, ClipItem)]
                 if zone == "move" else [item])
        if item not in items:
            items.append(item)
        drag = _Drag(zone, item, scene_pos, items)
        for it in items:
            track, clip = self.project.find_clip(it.clip_id)
            drag.start[it.clip_id] = {"start": clip.start_s, "length": clip.length_s,
                                      "trim": clip.trim_s, "fade_in": clip.fade_in_s,
                                      "fade_out": clip.fade_out_s, "track": track.id,
                                      "row": self.layout.row(track.id).index}
            it.dragging = True
        self._drag = drag

    def update_drag(self, scene_pos: QPointF, modifiers) -> None:
        d = self._drag
        if d is None:
            return
        snap_on = not modifiers & Qt.KeyboardModifier.AltModifier
        bpm = self.project.bpm
        dt = (scene_pos.x() - d.origin.x()) / self.pps
        if abs(scene_pos.x() - d.origin.x()) + abs(scene_pos.y() - d.origin.y()) > 2:
            d.moved = True
        p0 = d.start[d.primary.clip_id]

        def snap(t):
            return self.snap_time(t, snap_on)

        if d.zone == "move":
            raw = p0["start"] + dt
            new_start = snap(raw)
            if snap_on:                     # the clip's end may stick to a marker too
                end_m = self.magnet(raw + p0["length"])
                start_m = self.magnet(raw)
                if end_m is not None and (start_m is None or abs(end_m - raw - p0["length"])
                                          < abs(start_m - raw)):
                    new_start = end_m - p0["length"]
            delta = max(new_start - p0["start"], -min(s["start"] for s in d.start.values()))
            rows = self.layout.rows
            target = self.layout.row_at(scene_pos.y(), clamp=True)
            dr = (target.index - p0["row"]) if target else 0
            lo = -min(s["row"] for s in d.start.values())
            hi = len(rows) - 1 - max(s["row"] for s in d.start.values())
            dr = max(lo, min(hi, dr))
            if any(rows[s["row"] + dr].kind != rows[s["row"]].kind for s in d.start.values()):
                dr = 0  # clips stay on tracks of their own kind
            for it in d.items:
                s = d.start[it.clip_id]
                row = rows[s["row"] + dr]
                it.set_geometry(s["start"] + delta, s["length"], row.y, self.pps)
                d.result[it.clip_id] = {"start_s": s["start"] + delta, "track": row.track_id}
            self.statusMessage.emit(f"Move: {theme.format_time(p0['start'] + delta)}")
        elif d.zone == "left":
            end = p0["start"] + p0["length"]
            new_start = snap(p0["start"] + dt)
            new_start = max(new_start, p0["start"] - p0["trim"], 0.0)
            new_start = min(new_start, end - MIN_CLIP_S)
            length = end - new_start
            d.primary.set_geometry(new_start, length, self.layout.row(p0["track"]).y, self.pps)
            d.result[d.primary.clip_id] = {"start_s": new_start, "length_s": length,
                                           "trim_s": p0["trim"] + (new_start - p0["start"])}
            self.statusMessage.emit(f"Start {theme.format_time(new_start)}, "
                                    f"length {theme.format_time(length)}")
        elif d.zone == "right":
            new_end = snap(p0["start"] + p0["length"] + dt)
            length = max(MIN_CLIP_S, new_end - p0["start"])
            d.primary.set_geometry(p0["start"], length, self.layout.row(p0["track"]).y, self.pps)
            d.result[d.primary.clip_id] = {"length_s": length}
            self.statusMessage.emit(f"Length {theme.format_time(length)}")
        elif d.zone in ("fade_in", "fade_out"):
            it = d.primary
            if d.zone == "fade_in":
                v = (scene_pos.x() - it.x()) / self.pps
                v = min(max(0.0, v), p0["length"] - it.fade_out_s)
                it.fade_in_s = round(v * FS) / FS
                d.result[it.clip_id] = {"fade_in_s": it.fade_in_s}
            else:
                v = (it.x() + it.width - scene_pos.x()) / self.pps
                v = min(max(0.0, v), p0["length"] - it.fade_in_s)
                it.fade_out_s = round(v * FS) / FS
                d.result[it.clip_id] = {"fade_out_s": it.fade_out_s}
            it.update()
            self.statusMessage.emit(f"Fade {theme.format_time(v)}")
        self.view.viewport().update()

    def end_drag(self, modifiers) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        for it in d.items:
            it.dragging = False
        if not d.moved or not d.result:
            self.refresh()
            return
        if d.zone == "move":
            copyable = {cid for cid in d.result if not self.clip_items[cid].video}
            copy = bool(modifiers & Qt.KeyboardModifier.ControlModifier) and bool(copyable)
            if copy:
                items = []
                for cid, r in d.result.items():
                    if cid not in copyable:
                        continue                     # the video is never copied
                    _track, clip = self.project.find_clip(cid)
                    dup = clip.clone(fresh_id=True)
                    dup.start_s = r["start_s"]
                    items.append((r["track"], dup))
                self.doc.push(cmd.InsertClips(self.doc, items, "Copy clips"))
                self.refresh()                       # a dragged video snaps back
                self.select_clips([c.id for _, c in items])
            else:
                moves = [(cid, r["track"], r["start_s"]) for cid, r in d.result.items()]
                unchanged = all(r["track"] == d.start[cid]["track"]
                                and abs(r["start_s"] - d.start[cid]["start"]) < 1e-9
                                for cid, r in d.result.items())
                if unchanged:
                    self.refresh()
                else:
                    self.doc.move_clips(moves)
        else:
            cid, changes = next(iter(d.result.items()))
            _track, clip = self.project.find_clip(cid)
            if d.zone in ("left", "right"):
                length = changes["length_s"]
                # keep fades inside the new length
                changes["fade_in_s"] = min(clip.fade_in_s, length)
                changes["fade_out_s"] = min(clip.fade_out_s, length - changes["fade_in_s"])
                text = "Resize clip"
            else:
                text = "Change fade"
            self.doc.set_clip_attrs(cid, text=text, **changes)
        self.statusMessage.emit("")

    # ── empty-space gestures ────────────────────────────────────────────────
    def _time_at(self, scene_pos: QPointF, snap: bool = True) -> float:
        t = max(0.0, scene_pos.x() / self.pps)
        return self.snap_time(t) if snap else t

    # ── snapping and markers ────────────────────────────────────────────────
    def marker_times(self, exclude: str | None = None) -> list[float]:
        return [m.t_s for m in self.project.markers if m.id != exclude]

    def magnet_times(self, exclude_marker: str | None = None) -> list[float]:
        """Times that attract a dragged edge: markers and the edges of every clip
        (except the clips being dragged)."""
        times = self.marker_times(exclude_marker)
        dragged = self._drag.start.keys() if self._drag is not None else ()
        for track in self.project.tracks:
            for c in track.clips:
                if c.id not in dragged:
                    times += (c.start_s, c.end_s)
        return times

    def magnet(self, t: float, exclude_marker: str | None = None) -> float | None:
        """The marker or clip edge *t* sticks to (within MAGNET_PX on screen), if any."""
        return nearest_magnet(t, self.magnet_times(exclude_marker), MAGNET_PX / self.pps)

    def snap_time(self, t: float, snap_on: bool = True,
                  exclude_marker: str | None = None) -> float:
        """Grid snap with magnets (markers, clip edges); off (*snap_on* False)
        lands on a pulse."""
        if not snap_on:
            return round(t * FS) / FS
        return self.snap.apply(t, self.project.bpm, self.magnet_times(exclude_marker),
                               MAGNET_PX / self.pps)

    def marker_near(self, t: float, direction: int):
        """The first marker strictly before (-1) / after (+1) time *t*."""
        eps = 0.5 / FS
        ms = self.project.markers
        if direction > 0:
            return next((m for m in ms if m.t_s > t + eps), None)
        return next((m for m in reversed(ms) if m.t_s < t - eps), None)

    def goto_marker(self, direction: int) -> None:
        m = self.marker_near(self.doc.cursor_s, direction)
        if m is None:
            self.statusMessage.emit("No marker " + ("after" if direction > 0 else "before")
                                    + " the cursor.")
            return
        self.doc.set_cursor(m.t_s)
        self.ensure_visible(m.t_s)

    def span_between_markers(self, t: float) -> tuple[float, float] | None:
        """The stretch between the markers around *t* (project ends count)."""
        ms = self.project.markers
        if not ms:
            return None
        before = [m.t_s for m in ms if m.t_s <= t + 0.5 / FS]
        after = [m.t_s for m in ms if m.t_s > t + 0.5 / FS]
        a = before[-1] if before else 0.0
        b = after[0] if after else max(self.project.duration_s(), a)
        return (a, b) if b - a > 1.0 / FS else None

    def rename_marker(self, marker_id: str) -> None:
        from PyQt6.QtWidgets import QInputDialog
        m = self.doc.find_marker(marker_id)
        if m is None:
            return
        name, ok = QInputDialog.getText(self, "Rename marker", "Name:", text=m.name)
        if ok and name != m.name:
            self.doc.set_marker_attrs(marker_id, text="Rename marker", name=name)

    def recolor_marker(self, marker_id: str) -> None:
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        m = self.doc.find_marker(marker_id)
        if m is None:
            return
        color = QColorDialog.getColor(QColor(m.color), self, "Marker colour")
        if color.isValid():
            self.doc.set_marker_attrs(marker_id, text="Marker colour", color=color.name())

    def marker_menu(self, marker_id: str, global_pos) -> None:
        m = self.doc.find_marker(marker_id)
        if m is None:
            return
        menu = QMenu(self)
        menu.addAction("Go to marker", lambda: self.doc.set_cursor(m.t_s))
        menu.addAction("Loop to next marker",
                       lambda: self.loopRegionEdited.emit(self.span_between_markers(m.t_s)))
        menu.addSeparator()
        menu.addAction("Rename…", lambda: self.rename_marker(marker_id))
        menu.addAction("Colour…", lambda: self.recolor_marker(marker_id))
        menu.addSeparator()
        menu.addAction("Delete marker", lambda: self.doc.remove_markers([marker_id]))
        menu.exec(global_pos)

    def default_clip_length(self) -> float:
        if self.snap.kind == "beats":
            return 16 * 60.0 / self.project.bpm   # 4 bars
        return 8.0

    def empty_clicked(self, scene_pos: QPointF, modifiers) -> None:
        snap = not modifiers & Qt.KeyboardModifier.AltModifier
        self.doc.set_cursor(self._time_at(scene_pos, snap))
        row = self.layout.row_at(scene_pos.y()) if self.layout else None
        if row is not None:
            self.set_current_track(row.track_id)

    def empty_double_clicked(self, scene_pos: QPointF) -> None:
        self._add_clip_at(self.generator_provider(), scene_pos)

    def generator_for_row(self, type_id: str, row) -> str | None:
        """*type_id* if it fits the row's track kind, else that kind's default."""
        from ...generators import get_generator
        domain = get_generator(type_id).domain
        if row.kind == "triphase":
            return type_id if domain == "triphase" else "tri_circle"
        if row.kind == "standard":
            return type_id if domain == "pulse" else None
        return None

    def _add_clip_at(self, type_id: str, scene_pos: QPointF) -> Clip | None:
        row = self.layout.row_at(scene_pos.y()) if self.layout else None
        if row is None or not row.takes_clips:
            return None
        fitted = self.generator_for_row(type_id, row)
        if fitted is None:
            self.statusMessage.emit("Triphase generators go on triphase tracks.")
            return None
        type_id = fitted
        clip = self.doc.add_generator_clip(row.track_id, type_id, self._time_at(scene_pos),
                                           self.default_clip_length())
        self.set_current_track(row.track_id)
        self.select_clips([clip.id])
        return clip

    def drop_hover(self, scene_pos: QPointF | None) -> None:
        if scene_pos is None or self.layout is None:
            self.drop_marker = None
        else:
            row = self.layout.row_at(scene_pos.y())
            self.drop_marker = ((self._time_at(scene_pos) * self.pps, row.y)
                                if row is not None and row.takes_clips else None)
        self.view.viewport().update()

    def drop_generator(self, type_id: str, scene_pos: QPointF) -> None:
        self.drop_marker = None
        self._add_clip_at(type_id, scene_pos)

    def drop_audio(self, path: str, scene_pos: QPointF) -> None:
        row = self.layout.row_at(scene_pos.y()) if self.layout else None
        self.audioDropped.emit(path, self._time_at(scene_pos), row.track_id if row else "")

    def drop_beats(self, paths: list[str], scene_pos: QPointF) -> None:
        self.beatsDropped.emit(list(paths), self._time_at(scene_pos))

    def drop_funscripts(self, paths: list[str], scene_pos: QPointF) -> None:
        row = self.layout.row_at(scene_pos.y()) if self.layout else None
        self.funscriptDropped.emit(list(paths), self._time_at(scene_pos),
                                   row.track_id if row else "")

    def drop_preset(self, path: str, scene_pos: QPointF) -> Clip | None:
        from ...io.presets import load_preset
        self.drop_marker = None
        self.view.viewport().update()
        row = self.layout.row_at(scene_pos.y()) if self.layout else None
        if row is None or not row.takes_clips:
            return None
        try:
            clip = load_preset(path, self._time_at(scene_pos))
        except Exception as exc:  # noqa: BLE001
            self.statusMessage.emit(f"Could not load preset: {exc}")
            return None
        src_type = getattr(clip.source, "type_id", None)
        if src_type is not None and self.generator_for_row(src_type, row) != src_type:
            self.statusMessage.emit("This preset does not fit this kind of track.")
            return None
        self.doc.add_clip(row.track_id, clip)
        self.set_current_track(row.track_id)
        self.select_clips([clip.id])
        return clip

    def clip_double_clicked(self, clip_id: str) -> None:
        self.select_clips([clip_id])
        self.clipActivated.emit(clip_id)

    def _on_cursor(self, _t: float) -> None:
        self.view.viewport().update()
        self.ruler.update()
        self.overview.viewport().update()

    # ── transport display ───────────────────────────────────────────────────
    def set_playhead(self, seconds: float) -> None:
        if seconds == self.playhead_s:
            return
        self.playhead_s = seconds
        self.view.viewport().update()
        self.ruler.update()
        self.overview.viewport().update()

    def set_loop(self, enabled: bool, region: tuple[float, float] | None) -> None:
        self.loop_enabled, self.loop_region = enabled, region
        self.view.viewport().update()
        self.ruler.update()
        self.overview.viewport().update()

    def ensure_visible(self, seconds: float) -> None:
        """Page the view so *seconds* stays on screen (follow playhead)."""
        t0, t1 = self.visible_range()
        if t0 <= seconds <= t0 + (t1 - t0) * 0.92:
            return
        bar = self.view.horizontalScrollBar()
        bar.setValue(round(max(0.0, seconds - (t1 - t0) * 0.08) * self.pps))

    def selection_span(self) -> tuple[float, float] | None:
        found = [self.project.find_clip(cid) for cid in self.selected_clip_ids()]
        found = [c for _t, c in filter(None, found)]
        if not found:
            return None
        return min(c.start_s for c in found), max(c.end_s for c in found)

    # ── edit actions (also used by the main window) ─────────────────────────
    def delete_selected(self) -> None:
        self.doc.remove_clips(self.selected_clip_ids())

    def duplicate_selected(self) -> None:
        clips = self.doc.duplicate_clips(self.selected_clip_ids())
        self.select_clips([c.id for c in clips])

    def split_selected(self, at_s: float | None = None) -> None:
        at = self.doc.cursor_s if at_s is None else at_s
        ids = self.selected_clip_ids()
        if not ids:  # nothing selected: split whatever is under the cursor
            ids = [c.id for t in self.project.tracks for c in t.clips if c.start_s < at < c.end_s]
        self.doc.split_clips(ids, at)

    def glue_selected(self) -> None:
        clips = self.doc.glue_clips(self.selected_clip_ids())
        if clips:
            self.select_clips([c.id for c in clips])
        else:
            self.statusMessage.emit("Glue: select two or more clips on the same track")

    def toggle_mute_selected(self) -> None:
        ids = [cid for cid in self.selected_clip_ids()
               if cid in self.clip_items and not self.clip_items[cid].video]
        if not ids:
            return
        self.doc.undo.beginMacro("Mute clips")
        for cid in ids:
            _t, clip = self.project.find_clip(cid)
            self.doc.set_clip_attrs(cid, muted=not clip.muted)
        self.doc.undo.endMacro()

    def _rhythm_actions(self, menu, act, item) -> None:
        """Beat marks / pattern entries of a clip's context menu."""
        cid = item.clip_id
        others = [c for c in self.selected_clip_ids() if c != cid]
        menu.addSeparator()
        if item.kind in ("generator", "pulses"):
            act("Beat marks from peaks…", lambda: self.peakMarksRequested.emit(cid))
        elif item.kind == "marks":
            label = (f"Convert to pattern (pool: {len(others)} selected clips)" if others
                     else "Convert to pattern")
            act(label, lambda: self.convert_to_pattern(cid, others))
        elif item.kind == "pattern":
            if others:
                act(f"Add {len(others)} selected clips to pattern",
                    lambda: self._add_to_pool(cid, others))
            act("New random seed", lambda: self.reseed_pattern(cid))
            act("Back to beat marks", lambda: self.doc.pattern_to_marks(cid))
            act("Bake pattern to pulses", lambda: self.doc.bake_pattern(cid))
        menu.addSeparator()

    def convert_to_pattern(self, clip_id: str, pool_clip_ids=()) -> None:
        self.doc.convert_to_pattern(clip_id, pool_clip_ids)
        self.select_clips([clip_id])
        self.clipActivated.emit(clip_id)

    def _add_to_pool(self, clip_id: str, clip_ids) -> None:
        n = self.doc.add_to_pool(clip_id, clip_ids)
        if n < len(clip_ids):
            self.statusMessage.emit("Only generator and pulse clips of the same track kind "
                                    "go into a pattern.")
        self.select_clips([clip_id])

    def reseed_pattern(self, clip_id: str) -> None:
        import random
        found = self.project.find_clip(clip_id)
        if found is None or found[1].source.kind != "pattern":
            return
        src = found[1].source.clone()
        src.seed = random.randrange(2 ** 31)
        self.doc.set_clip_attrs(clip_id, text="New pattern seed", source=src)

    def context_menu(self, scene_pos: QPointF, item, global_pos) -> None:
        menu = QMenu(self)

        def act(text, fn, enabled=True):
            a = QAction(text, menu)
            a.triggered.connect(fn)
            a.setEnabled(enabled)
            menu.addAction(a)

        at = self._time_at(scene_pos)
        if isinstance(item, ClipItem):
            if not item.isSelected():
                self.select_clips([item.clip_id])
            if item.video:
                act("Replace video…", self.videoRequested.emit)
                act("Show video window", lambda: self.clipActivated.emit(item.clip_id))
                menu.addSeparator()
                act("Motion → stim…", lambda: self.videoAnalysisRequested.emit(True, False))
                act("Detect scene cuts → markers…",
                    lambda: self.videoAnalysisRequested.emit(False, True))
                menu.addSeparator()
                act("Remove video", lambda: self.doc.remove_clips([item.clip_id]))
                menu.exec(global_pos)
                return
            act("Split here", lambda: self.split_selected(at))
            act("Glue selected clips", self.glue_selected, len(self.selected_clip_ids()) > 1)
            self._rhythm_actions(menu, act, item)
            act("Duplicate", self.duplicate_selected)
            act("Mute / unmute", self.toggle_mute_selected)
            act("Save as preset…", lambda: self.savePresetRequested.emit(item.clip_id))
            menu.addSeparator()
            act("Delete", self.delete_selected)
        else:
            self.empty_clicked(scene_pos, Qt.KeyboardModifier.NoModifier)
            row = self.layout.row_at(scene_pos.y()) if self.layout else None
            ok = row is not None and row.takes_clips
            act(f"Add clip ({self.generator_provider()})",
                lambda: self._add_clip_at(self.generator_provider(), scene_pos), ok)
            act("Split at cursor", lambda: self.split_selected(at))
        menu.exec(global_pos)
