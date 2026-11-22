"""Undoable edits of the project model (QUndoCommand subclasses).

Commands address tracks and clips by id, never by object position, so they stay
valid however the stack is replayed. Each ``redo``/``undo`` mutates the model
and then calls ``doc.notify()``, which refreshes the views and re-renders.
"""

from __future__ import annotations

import copy

from PyQt6.QtGui import QUndoCommand

from ..core.model import Clip, Track

_MERGE_CLIP, _MERGE_TRACK, _MERGE_PROJECT = 1001, 1002, 1003


def _snap(value):
    """Copy a value so later edits of the model cannot change the stored state."""
    if hasattr(value, "clone"):
        return value.clone()
    return copy.deepcopy(value)


class DocCommand(QUndoCommand):
    def __init__(self, doc, text: str):
        super().__init__(text)
        self.doc = doc

    @property
    def project(self):
        return self.doc.project


# ── attribute edits (mergeable, for spin boxes and sliders) ─────────────────
class _SetAttrs(DocCommand):
    merge_id = -1

    def __init__(self, doc, target_id, changes: dict, text: str, merge_key=None):
        super().__init__(doc, text)
        self.target_id = target_id
        self.after = {k: _snap(v) for k, v in changes.items()}
        target = self._target()
        self.before = {k: _snap(getattr(target, k)) for k in changes}
        self.merge_key = merge_key

    def _target(self):
        raise NotImplementedError

    def _apply(self, values: dict) -> None:
        target = self._target()
        for k, v in values.items():
            setattr(target, k, _snap(v))
        self.doc.notify()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)

    def id(self) -> int:
        return self.merge_id if self.merge_key is not None else -1

    def mergeWith(self, other) -> bool:
        if (type(other) is not type(self) or other.target_id != self.target_id
                or other.merge_key != self.merge_key or other.after.keys() != self.after.keys()):
            return False
        self.after = other.after
        return True


class SetClipAttrs(_SetAttrs):
    merge_id = _MERGE_CLIP

    def _target(self):
        found = self.project.find_clip(self.target_id)
        if found is None:
            raise KeyError(f"clip {self.target_id} not found")
        return found[1]


class SetTrackAttrs(_SetAttrs):
    merge_id = _MERGE_TRACK

    def _target(self):
        return self.project.track_by_id(self.target_id)


class SetProjectAttrs(_SetAttrs):
    merge_id = _MERGE_PROJECT

    def _target(self):
        return self.project


class SetLane(DocCommand):
    def __init__(self, doc, track_id: str, lane_key: str, lane, text="Edit lane"):
        super().__init__(doc, text)
        self.track_id, self.lane_key = track_id, lane_key
        self.after = lane.clone()
        self.before = self.project.track_by_id(track_id).lanes[lane_key].clone()

    def _apply(self, lane):
        self.project.track_by_id(self.track_id).lanes[self.lane_key] = lane.clone()
        self.doc.notify()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


# ── clips ───────────────────────────────────────────────────────────────────
class InsertClips(DocCommand):
    """Insert clips; *items* is ``[(track_id, clip)]``."""

    def __init__(self, doc, items: list[tuple[str, Clip]], text="Add clip"):
        super().__init__(doc, text)
        self.items = [(tid, clip) for tid, clip in items]

    def redo(self):
        for tid, clip in self.items:
            self.project.track_by_id(tid).clips.append(clip)
        self.doc.notify()

    def undo(self):
        ids = {clip.id for _, clip in self.items}
        for tid, _ in self.items:
            track = self.project.track_by_id(tid)
            track.clips = [c for c in track.clips if c.id not in ids]
        self.doc.notify()


class RemoveClips(DocCommand):
    def __init__(self, doc, clip_ids, text="Delete clips"):
        super().__init__(doc, text)
        self.removed: list[tuple[str, int, Clip]] = []
        for t in self.project.tracks:
            for i, c in enumerate(t.clips):
                if c.id in set(clip_ids):
                    self.removed.append((t.id, i, c))

    def redo(self):
        ids = {c.id for _, _, c in self.removed}
        for t in self.project.tracks:
            t.clips = [c for c in t.clips if c.id not in ids]
        self.doc.notify()

    def undo(self):
        for tid, index, clip in self.removed:
            self.project.track_by_id(tid).clips.insert(index, clip)
        self.doc.notify()


class MoveClips(DocCommand):
    """Move clips in time and/or to other tracks. *moves*: ``[(clip_id, track_id, start_s)]``."""

    def __init__(self, doc, moves, text="Move clips"):
        super().__init__(doc, text)
        self.after = list(moves)
        self.before = []
        for clip_id, _tid, _start in moves:
            track, clip = self.project.find_clip(clip_id)
            self.before.append((clip_id, track.id, clip.start_s))

    def _apply(self, moves):
        for clip_id, tid, start in moves:
            track, clip = self.project.find_clip(clip_id)
            clip.start_s = float(start)
            if track.id != tid:
                track.clips.remove(clip)
                self.project.track_by_id(tid).clips.append(clip)
        self.doc.notify()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


# ── tracks ──────────────────────────────────────────────────────────────────
class InsertTrack(DocCommand):
    def __init__(self, doc, track: Track, index: int | None = None, text="Add track"):
        super().__init__(doc, text)
        self.track = track
        self.index = len(self.project.tracks) if index is None else index

    def redo(self):
        self.project.tracks.insert(self.index, self.track)
        self.doc.notify()

    def undo(self):
        self.project.tracks.remove(self.track)
        self.doc.notify()


class RemoveTrack(DocCommand):
    def __init__(self, doc, track_id: str, text="Delete track"):
        super().__init__(doc, text)
        self.track = self.project.track_by_id(track_id)
        self.index = self.project.tracks.index(self.track)

    def redo(self):
        self.project.tracks.remove(self.track)
        self.doc.notify()

    def undo(self):
        self.project.tracks.insert(self.index, self.track)
        self.doc.notify()


class MoveTrack(DocCommand):
    def __init__(self, doc, track_id: str, new_index: int, text="Move track"):
        super().__init__(doc, text)
        self.track_id = track_id
        self.old_index = [t.id for t in self.project.tracks].index(track_id)
        self.new_index = max(0, min(len(self.project.tracks) - 1, new_index))

    def _move(self, to_index):
        track = self.project.track_by_id(self.track_id)
        self.project.tracks.remove(track)
        self.project.tracks.insert(to_index, track)
        self.doc.notify()

    def redo(self):
        self._move(self.new_index)

    def undo(self):
        self._move(self.old_index)
