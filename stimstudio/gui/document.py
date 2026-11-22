"""ProjectDocument: the live project, its undo stack and background rendering.

All model edits go through :meth:`ProjectDocument.push` (QUndoCommands from
``gui.commands``) or the convenience methods below. Every edit ends in
:meth:`notify`, which emits ``modelChanged`` and schedules a render.

Rendering runs on a worker thread against a :meth:`Project.clone` snapshot, so
the GUI may keep editing while it runs. Renders are coalesced: at most one runs
at a time, and a result that is older than the latest edit is dropped.
"""

from __future__ import annotations

import atexit
import time
import traceback
import weakref
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QUndoStack

from ..core.model import (BeatMarksSource, Clip, GeneratorSource, Marker, PatternSource, PoolEntry,
                          Project, Track, VideoFileSource)
from ..core.pulse import FS
from ..core.render import ProjectRender, RenderCache, render_project
from ..io.project_file import load_project, save_project
from . import commands as cmd
from .theme import TRACK_COLORS

#: Delay between an edit and the render it triggers (coalesces bursts).
RENDER_DEBOUNCE_MS = 30


class _RenderSignals(QObject):
    done = pyqtSignal(int, object, float)
    failed = pyqtSignal(int, str)


class _RenderTask(QRunnable):
    def __init__(self, generation: int, snapshot: Project, cache: RenderCache):
        super().__init__()
        self.setAutoDelete(True)
        self.generation = generation
        self.snapshot = snapshot
        self.cache = cache
        self.signals = _RenderSignals()

    def run(self):
        t = time.perf_counter()
        try:
            result = render_project(self.snapshot, self.cache)
        except Exception:  # noqa: BLE001 — report any render failure to the GUI
            self.signals.failed.emit(self.generation, traceback.format_exc())
            return
        self.signals.done.emit(self.generation, result, time.perf_counter() - t)


#: Live documents, shut down before interpreter teardown (see ``shutdown``).
_LIVE_DOCS: "weakref.WeakSet[ProjectDocument]" = weakref.WeakSet()


@atexit.register
def _shutdown_all() -> None:
    for doc in list(_LIVE_DOCS):
        doc.shutdown()


def new_project() -> Project:
    return Project(name="Untitled", tracks=[Track(name="Track 1", color=TRACK_COLORS[0])])


class ProjectDocument(QObject):
    modelChanged = pyqtSignal()
    renderStarted = pyqtSignal()
    renderReady = pyqtSignal(object, float)   # ProjectRender, seconds taken
    renderFailed = pyqtSignal(str)
    pathChanged = pyqtSignal(str)
    dirtyChanged = pyqtSignal(bool)
    cursorChanged = pyqtSignal(float)

    def __init__(self, project: Project | None = None, parent=None):
        super().__init__(parent)
        self.project = project or new_project()
        self.path: str | None = None
        self.cursor_s = 0.0
        self.cache = RenderCache()
        self.last_render: ProjectRender | None = None

        self.undo = QUndoStack(self)
        self.undo.cleanChanged.connect(lambda clean: self.dirtyChanged.emit(not clean))

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(RENDER_DEBOUNCE_MS)
        self._timer.timeout.connect(self._start_render)
        self._generation = 0       # bumped by every edit
        self._rendered_gen = -1    # generation of last_render
        self._running = False
        self._tasks: set[_RenderTask] = set()
        _LIVE_DOCS.add(self)

    def shutdown(self) -> None:
        """Stop rendering and drop the undo history.

        Must run before interpreter teardown: Qt otherwise deletes the Python
        QUndoCommands after their wrappers are gone and the process crashes on
        exit. Called by the main window on close and by an atexit hook.
        """
        self._timer.stop()
        self._pool.waitForDone()
        self.undo.clear()

    # ── model change plumbing ───────────────────────────────────────────────
    def notify(self) -> None:
        """Called after every model mutation."""
        self._generation += 1
        self.modelChanged.emit()
        self._timer.start()

    def push(self, command) -> None:
        self.undo.push(command)

    @property
    def dirty(self) -> bool:
        return not self.undo.isClean()

    @property
    def render_is_current(self) -> bool:
        return self._rendered_gen == self._generation

    # ── rendering ───────────────────────────────────────────────────────────
    def schedule_render(self) -> None:
        self._timer.start()

    def _start_render(self) -> None:
        if self._running:
            return  # _on_done starts the next one
        self._running = True
        task = _RenderTask(self._generation, self.project.clone(), self.cache)
        task.signals.done.connect(self._on_done)
        task.signals.failed.connect(self._on_failed)
        self._tasks.add(task)
        self.renderStarted.emit()
        self._pool.start(task)

    def _finish_task(self) -> None:
        self._running = False
        self._tasks.clear()  # only one task runs at a time; its signals are delivered
        if not self.render_is_current and not self._timer.isActive():
            self._start_render()

    def _on_done(self, generation: int, result: ProjectRender, seconds: float) -> None:
        if generation >= self._rendered_gen:
            self._rendered_gen = generation
            self.last_render = result
            if generation == self._generation:
                self.renderReady.emit(result, seconds)
        self._finish_task()

    def _on_failed(self, generation: int, message: str) -> None:
        self._rendered_gen = max(self._rendered_gen, generation)
        self.renderFailed.emit(message)
        self._finish_task()

    def render_now(self) -> ProjectRender:
        """Synchronous render of the live project (for export)."""
        return render_project(self.project, self.cache)

    def wait_for_render(self, timeout_ms: int = 10000) -> bool:
        """Process events until the render matches the model (tests, export)."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not self.render_is_current and time.monotonic() < deadline:
            if not self._running and not self._timer.isActive():
                self._start_render()
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.002)
        return self.render_is_current

    # ── files ───────────────────────────────────────────────────────────────
    def set_project(self, project: Project, path: str | None = None) -> None:
        self.project = project
        self.path = path
        self.cursor_s = 0.0
        self.undo.clear()
        self.cache.clear()
        self.last_render = None
        self.pathChanged.emit(path or "")
        self.cursorChanged.emit(0.0)
        self.notify()

    def open(self, path: str) -> None:
        self.set_project(load_project(path), path)

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("no file name")
        save_project(self.project, path)
        self.undo.setClean()
        if path != self.path:
            self.path = path
            self.pathChanged.emit(path)

    # ── cursor ──────────────────────────────────────────────────────────────
    def set_cursor(self, seconds: float) -> None:
        seconds = max(0.0, round(float(seconds) * FS) / FS)
        if seconds != self.cursor_s:
            self.cursor_s = seconds
            self.cursorChanged.emit(seconds)

    # ── convenience edits ───────────────────────────────────────────────────
    def add_track(self, name: str | None = None, index: int | None = None,
                  track_type: str = "standard") -> Track:
        n = len(self.project.tracks)
        track = Track(name=name or f"Track {n + 1}", type=track_type,
                      color=TRACK_COLORS[n % len(TRACK_COLORS)])
        self.push(cmd.InsertTrack(self, track, index))
        return track

    def remove_track(self, track_id: str) -> None:
        self.push(cmd.RemoveTrack(self, track_id))

    def duplicate_track(self, track_id: str) -> Track:
        src = self.project.track_by_id(track_id)
        copy_ = src.clone(new_ids=True)
        copy_.name = f"{src.name} copy"
        self.push(cmd.InsertTrack(self, copy_, self.project.tracks.index(src) + 1,
                                  "Duplicate track"))
        return copy_

    def set_track_attrs(self, track_id: str, merge_key=None, text="Edit track",
                        **changes) -> None:
        self.push(cmd.SetTrackAttrs(self, track_id, changes, text, merge_key))

    def set_clip_attrs(self, clip_id: str, merge_key=None, text="Edit clip", **changes) -> None:
        self.push(cmd.SetClipAttrs(self, clip_id, changes, text, merge_key))

    def set_project_attrs(self, merge_key=None, text="Edit project", **changes) -> None:
        self.push(cmd.SetProjectAttrs(self, None, changes, text, merge_key))

    def add_clip(self, track_id: str, clip: Clip) -> Clip:
        self.push(cmd.InsertClips(self, [(track_id, clip)]))
        return clip

    def add_generator_clip(self, track_id: str, type_id: str, start_s: float,
                           length_s: float = 8.0) -> Clip:
        from ..generators import get_generator
        gen = get_generator(type_id)
        clip = Clip(GeneratorSource(type_id, gen.default_params(), seed=0),
                    start_s=start_s, length_s=length_s, name=gen.label)
        return self.add_clip(track_id, clip)

    # ── markers ─────────────────────────────────────────────────────────────
    def set_markers(self, markers, text="Edit markers", merge_key=None) -> None:
        markers = sorted((m.clone() for m in markers), key=lambda m: m.t_s)
        self.set_project_attrs(merge_key=merge_key, text=text, markers=markers)

    def add_marker(self, t_s: float, name: str = "", color: str | None = None) -> Marker:
        t_s = max(0.0, round(float(t_s) * FS) / FS)
        marker = Marker(t_s, name or f"Marker {len(self.project.markers) + 1}")
        if color:
            marker.color = color
        self.set_markers([*self.project.markers, marker], "Add marker")
        return marker

    def add_markers(self, items, text="Add markers") -> list[Marker]:
        """Add several ``(t_s, name)`` markers as one undo step."""
        new = [Marker(max(0.0, round(float(t) * FS) / FS), name) for t, name in items]
        if new:
            self.set_markers([*self.project.markers, *new], text)
        return new

    def set_marker_attrs(self, marker_id: str, merge_key=None, text="Edit marker",
                         **changes) -> None:
        if "t_s" in changes:
            changes["t_s"] = max(0.0, round(float(changes["t_s"]) * FS) / FS)
        markers = []
        for m in self.project.markers:
            m = m.clone()
            if m.id == marker_id:
                for k, v in changes.items():
                    setattr(m, k, v)
            markers.append(m)
        self.set_markers(markers, text, merge_key)

    def remove_markers(self, marker_ids, text="Delete marker") -> None:
        ids = set(marker_ids)
        kept = [m for m in self.project.markers if m.id not in ids]
        if len(kept) != len(self.project.markers):
            self.set_markers(kept, text)

    def find_marker(self, marker_id: str) -> Marker | None:
        return next((m for m in self.project.markers if m.id == marker_id), None)

    def set_video(self, path: str, duration_s: float) -> Clip:
        """Put *path* on the video track (created as the top row when missing).

        A project has one video: a new file replaces it and keeps its position.
        """
        source = VideoFileSource(str(path), float(duration_s))
        name = Path(path).stem
        track = self.project.video_track()
        self.undo.beginMacro("Set video")
        if track is None:
            track = Track(name="Video", type="video", color="#808080")
            self.push(cmd.InsertTrack(self, track, 0, "Add video track"))
        if track.clips:
            clip = track.clips[0]
            self.push(cmd.SetClipAttrs(self, clip.id, {"source": source,
                                                       "length_s": float(duration_s),
                                                       "name": name}, "Replace video"))
            extra = [c.id for c in track.clips[1:]]
            if extra:
                self.push(cmd.RemoveClips(self, extra))
        else:
            clip = Clip(source, start_s=0.0, length_s=float(duration_s), name=name)
            self.push(cmd.InsertClips(self, [(track.id, clip)], "Add video"))
        self.undo.endMacro()
        return clip

    def relink_media(self, mapping: dict[str, str], text="Relink media") -> int:
        """Point every clip that uses a file in *mapping* (old → new path) at the
        new file, as one undo step. Returns the number of clips changed."""
        from ..core.media import relinked_sources
        changes = relinked_sources(self.project, mapping)
        if not changes:
            return 0
        self.undo.beginMacro(text)
        for clip_id, source in changes.items():
            self.push(cmd.SetClipAttrs(self, clip_id, {"source": source}, text))
        self.undo.endMacro()
        return len(changes)

    def remove_clips(self, clip_ids) -> None:
        if clip_ids:
            self.push(cmd.RemoveClips(self, list(clip_ids)))

    def move_clips(self, moves) -> None:
        if moves:
            self.push(cmd.MoveClips(self, moves))

    def duplicate_clips(self, clip_ids) -> list[Clip]:
        """Copies placed right after the selection (keeping relative layout)."""
        found = [self.project.find_clip(cid) for cid in clip_ids]
        found = [f for f in found if f and f[0].type != "video"]   # one video only
        if not found:
            return []
        lo = min(c.start_s for _, c in found)
        hi = max(c.end_s for _, c in found)
        items = []
        for track, clip in found:
            dup = clip.clone(fresh_id=True)
            dup.start_s = clip.start_s + (hi - lo)
            items.append((track.id, dup))
        self.push(cmd.InsertClips(self, items, "Duplicate clips"))
        return [c for _, c in items]

    def paste_clips(self, clips_with_tracks, at_s: float) -> list[Clip]:
        """Paste ``[(track_id, clip)]`` (from the clipboard) starting at *at_s*."""
        if not clips_with_tracks:
            return []
        lo = min(c.start_s for _, c in clips_with_tracks)
        valid = {t.id for t in self.project.tracks if t.type != "video"}
        items = []
        for tid, clip in clips_with_tracks:
            if tid not in valid:
                continue
            dup = clip.clone(fresh_id=True)
            dup.start_s = at_s + (clip.start_s - lo)
            items.append((tid, dup))
        if items:
            self.push(cmd.InsertClips(self, items, "Paste clips"))
        return [c for _, c in items]

    def glue_clips(self, clip_ids) -> list[Clip]:
        """Render the given clips of each track into one pulse clip that sounds
        the same (fades, effects, crossfades and joins baked in) and replace them,
        as one undo step. Muted clips and tracks with fewer than two of the clips
        are left alone. Returns the new clips."""
        from ..core.model import PulseDataSource
        from ..core.pulse import n_pulses, pulse_index
        from ..core.render import mix_clips
        by_track: dict[str, list[Clip]] = {}
        for cid in clip_ids:
            found = self.project.find_clip(cid)
            if found is None:
                continue
            track, clip = found
            if (track.type in ("standard", "triphase") and not clip.muted
                    and clip.source.kind in ("generator", "pulses", "pattern")
                    and clip.length_s > 0):
                by_track.setdefault(track.id, []).append(clip)
        jobs = [(tid, clips) for tid, clips in by_track.items() if len(clips) >= 2]
        if not jobs:
            return []
        self.undo.beginMacro("Glue clips")
        glued = []
        for tid, clips in jobs:
            track = self.project.track_by_id(tid)
            clips = sorted(clips, key=lambda c: c.start_s)
            s0 = min(pulse_index(c.start_s) for c in clips)
            e0 = max(pulse_index(c.start_s) + n_pulses(c.length_s) for c in clips)
            domain = "triphase" if track.type == "triphase" else "pulse"
            data, _active = mix_clips(clips, e0 - s0, self.cache, self.project.bpm, domain,
                                      track.overlap, track.join_s, track.smooth_joins,
                                      offset=s0)
            names = [c.name or getattr(c.source, "type_id", "pulses") for c in clips]
            origin = "glued: " + ", ".join(dict.fromkeys(names))
            clip = Clip(PulseDataSource(np.ascontiguousarray(data, np.float32), origin=origin),
                        start_s=s0 / FS, length_s=(e0 - s0) / FS,
                        name=f"{names[0]} +{len(clips) - 1}")
            self.push(cmd.RemoveClips(self, [c.id for c in clips]))
            self.push(cmd.InsertClips(self, [(tid, clip)], "Glue clips"))
            glued.append(clip)
        self.undo.endMacro()
        return glued

    # ── beat marks / patterns ───────────────────────────────────────────────
    @staticmethod
    def track_domain(track) -> str:
        return "triphase" if track.type == "triphase" else "pulse"

    def poolable(self, clip_id: str, domain: str) -> Clip | None:
        """The clip if it can go into a pattern pool of *domain*, else None."""
        found = self.project.find_clip(clip_id)
        if found is None:
            return None
        track, clip = found
        if (clip.source.kind not in ("generator", "pulses")
                or self.track_domain(track) != domain or clip.length_s <= 0):
            return None
        return clip

    @staticmethod
    def pool_entry(clip: Clip, key: str) -> PoolEntry:
        copy_ = clip.clone(fresh_id=True)
        copy_.start_s = 0.0
        if not copy_.name:
            copy_.name = getattr(clip.source, "type_id", "pulses")
        return PoolEntry(key, copy_)

    def convert_to_pattern(self, clip_id: str, pool_clip_ids=()) -> Clip | None:
        """Turn a beat-marks clip into a pattern; the given clips (fitting the
        track) are copied into its pool."""
        from ..core.pattern import next_key
        found = self.project.find_clip(clip_id)
        if found is None or not isinstance(found[1].source, BeatMarksSource):
            return None
        track, clip = found
        domain = self.track_domain(track)
        src = PatternSource([m.clone() for m in clip.source.marks])
        for cid in pool_clip_ids:
            other = self.poolable(cid, domain) if cid != clip_id else None
            if other is not None:
                src.pool.append(self.pool_entry(other, next_key(src.pool)))
        if src.pool:
            src.sequence = " ".join(e.key for e in src.pool)
        self.set_clip_attrs(clip_id, text="Convert to pattern", source=src)
        return clip

    def add_to_pool(self, clip_id: str, pool_clip_ids, text="Add clips to pattern") -> int:
        """Copy clips into a pattern's pool; returns how many were added."""
        from ..core.pattern import next_key
        found = self.project.find_clip(clip_id)
        if found is None or not isinstance(found[1].source, PatternSource):
            return 0
        track, clip = found
        src = clip.source.clone()
        n = len(src.pool)
        for cid in pool_clip_ids:
            other = self.poolable(cid, self.track_domain(track)) if cid != clip_id else None
            if other is not None:
                src.pool.append(self.pool_entry(other, next_key(src.pool)))
        if len(src.pool) > n:
            self.set_clip_attrs(clip_id, text=text, source=src)
        return len(src.pool) - n

    def pattern_to_marks(self, clip_id: str) -> None:
        found = self.project.find_clip(clip_id)
        if found is not None and isinstance(found[1].source, PatternSource):
            marks = [m.clone() for m in found[1].source.marks]
            for m in marks:
                m.pin = None
            self.set_clip_attrs(clip_id, text="Pattern to marks", source=BeatMarksSource(marks))

    def bake_pattern(self, clip_id: str) -> None:
        """Replace a pattern by the pulses it renders (clip gain, fades and
        effects stay on the clip)."""
        from ..core.model import PulseDataSource
        from ..core.pulse import n_pulses
        from ..core.render import render_source
        found = self.project.find_clip(clip_id)
        if found is None or not isinstance(found[1].source, PatternSource):
            return
        track, clip = found
        data = render_source(clip.source, n_pulses(clip.length_s), n_pulses(clip.trim_s),
                             self.cache, self.project.bpm, self.track_domain(track))
        src = PulseDataSource(np.array(data, np.float32), origin=f"pattern: {clip.name}")
        self.set_clip_attrs(clip_id, text="Bake pattern", source=src, trim_s=0.0)

    def add_marks_from_peaks(self, clip_id: str, marks, replace: bool = False) -> Clip | None:
        """Beat marks found in a clip: replace its content, or put a marks clip
        on a new track right below (same kind)."""
        found = self.project.find_clip(clip_id)
        if found is None:
            return None
        track, clip = found
        if replace:
            self.set_clip_attrs(clip_id, text="Clip to beat marks",
                                source=BeatMarksSource(list(marks)), trim_s=0.0)
            return clip
        new = Clip(BeatMarksSource(list(marks)), start_s=clip.start_s, length_s=clip.length_s,
                   name=f"{clip.name or 'Clip'} beats")
        self.undo.beginMacro("Beat marks from peaks")
        n = len(self.project.tracks)
        beats = Track(name=f"{track.name} beats", type=track.type,
                      color=TRACK_COLORS[n % len(TRACK_COLORS)])
        self.push(cmd.InsertTrack(self, beats, self.project.tracks.index(track) + 1))
        self.push(cmd.InsertClips(self, [(beats.id, new)], "Beat marks from peaks"))
        self.undo.endMacro()
        return new

    def add_tracks_with_clips(self, items, track_type: str = "standard",
                              text: str = "Import beats") -> list[Track]:
        """New tracks filled with clips, as one undo step; *items* is
        ``[(track name, [clips])]``."""
        if not items:
            return []
        self.undo.beginMacro(text)
        tracks = []
        for name, clips in items:
            n = len(self.project.tracks)
            track = Track(name=name, type=track_type, color=TRACK_COLORS[n % len(TRACK_COLORS)])
            self.push(cmd.InsertTrack(self, track, None, text))
            if clips:
                self.push(cmd.InsertClips(self, [(track.id, c) for c in clips], text))
            tracks.append(track)
        self.undo.endMacro()
        return tracks

    def split_clips(self, clip_ids, at_s: float) -> list[Clip]:
        """Split every given clip that spans *at_s*; returns the new right halves."""
        targets = []
        for cid in clip_ids:
            found = self.project.find_clip(cid)
            if found and found[0].type != "video" and found[1].start_s < at_s < found[1].end_s:
                targets.append(found)
        if not targets:
            return []
        self.undo.beginMacro("Split clips")
        right_halves = []
        for track, clip in targets:
            left_len = at_s - clip.start_s
            right = clip.clone(fresh_id=True)
            right.start_s = at_s
            right.length_s = clip.length_s - left_len
            right.trim_s = clip.trim_s + left_len
            right.fade_in_s = 0.0
            self.push(cmd.SetClipAttrs(self, clip.id, {"length_s": left_len, "fade_out_s": 0.0},
                                       "Split clip"))
            self.push(cmd.InsertClips(self, [(track.id, right)], "Split clip"))
            right_halves.append(right)
        self.undo.endMacro()
        return right_halves
