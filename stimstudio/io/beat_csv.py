"""Beat timestamps CSV (BeatGen "Export Beat Timestamps"), see
``docs/BEAT_TIMESTAMPS_FORMAT.md``.

Each CSV track is a column pair (time, mark). Its beats are cut into blocks:
every ``pattern start`` … ``pattern end`` group is one block, and a run of
standalone beats between groups is one block. Each block becomes a beat-marks
clip.
"""

from __future__ import annotations

import csv
import os
import statistics
from dataclasses import dataclass, field

from ..core.model import BeatMarksSource, Clip, Mark
from ..core.pulse import FS

START, END = "pattern start", "pattern end"
#: block tail when the block has a single beat (s)
DEFAULT_TAIL_S = 0.5
MIN_TAIL_S = 0.25


@dataclass
class BeatTrack:
    name: str
    beats: list[tuple[float, list[str]]] = field(default_factory=list)

    def blocks(self) -> list[list[float]]:
        return split_blocks(self.beats)


def read_beat_timestamps(path) -> list[BeatTrack]:
    """The CSV tracks of *path*, column pairs taken by position."""
    name = os.path.basename(str(path))
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: not a UTF-8 text file") from exc
    if not rows:
        return []
    header, data = rows[0], rows[1:]
    if len(header) < 2:
        raise ValueError(f"{name}: no beat columns in the header")
    tracks = []
    for c in range(0, len(header), 2):
        beats = []
        for r, row in enumerate(data, start=2):
            time = row[c].strip() if c < len(row) else ""
            if time == "":
                break
            try:
                t = float(time)
            except ValueError:
                raise ValueError(f"{name}: row {r}: {time!r} is not a time") from None
            mark = row[c + 1].strip() if c + 1 < len(row) else ""
            beats.append((t, [m.strip().lower() for m in mark.split(";") if m.strip()]))
        beats.sort(key=lambda b: b[0])
        tracks.append(BeatTrack(header[c].strip() or f"Track {c // 2 + 1}", beats))
    return tracks


def split_blocks(beats) -> list[list[float]]:
    """Cut ``[(time, marks)]`` into blocks of times (see the module doc).

    A beat marked both start and end inside a group is shared: it ends that
    group and starts the next one, so it lies in both blocks. Outside a group
    it is a one-beat group. A group left open ends at its last beat.
    """
    blocks: list[list[float]] = []
    loose: list[float] = []
    group: list[float] | None = None

    def flush_loose():
        if loose:
            blocks.append(list(loose))
            loose.clear()

    for t, marks in beats:
        start, end = START in marks, END in marks
        if group is not None:
            group.append(t)
            if end:
                blocks.append(group)
                group = [t] if start else None
            continue
        if start:
            flush_loose()
            if end:
                blocks.append([t])
            else:
                group = [t]
        else:
            loose.append(t)      # a stray "pattern end" counts as a standalone beat
    if group:
        blocks.append(group)
    flush_loose()
    blocks.sort(key=lambda b: b[0])
    return blocks


def block_tail(times: list[float]) -> float:
    if len(times) < 2:
        return DEFAULT_TAIL_S
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return max(MIN_TAIL_S, statistics.median(gaps)) if gaps else DEFAULT_TAIL_S


def blocks_to_clips(blocks: list[list[float]], name: str = "Beats",
                    offset_s: float = 0.0) -> list[Clip]:
    """One beat-marks clip per block. A block ends one tail after its last
    beat, but never past the start of the next block (a shared beat then
    sounds in the next block only)."""
    clips = []
    starts = [b[0] for b in blocks]
    for k, times in enumerate(blocks):
        first, last = times[0], times[-1]
        end = last + block_tail(times)
        if k + 1 < len(blocks) and starts[k + 1] >= first:
            end = min(end, max(starts[k + 1], first + 1.0 / FS))
        marks = [Mark(t - first) for t in times]
        clips.append(Clip(BeatMarksSource(marks), start_s=first + offset_s,
                          length_s=max(1.0 / FS, end - first),
                          name=f"{name} {k + 1}" if len(blocks) > 1 else name))
    return clips
