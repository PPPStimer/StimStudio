"""Live recording: timestamped input samples → a 40 Hz pulse take.

The recorder receives ``(timeline_s, row)`` samples while the output plays
(``row`` is a clip row: amp A, amp B, freq A, freq B — or volume, x, y, freq
on a triphase track). Each sample holds until the next one; when the timeline
jumps backwards (a loop wrapped, the user clicked) the later pass overwrites
the earlier one, like a punch-in. Qt-free.
"""

from __future__ import annotations

import numpy as np

from .pulse import FS

MAX_HOLD_S = 0.5          # a gap longer than this between samples is not bridged
CHANNEL_MODES = ("AB", "A", "B")


def pad_row(amp: float, freq: float, channels: str = "AB") -> tuple[float, float, float, float]:
    """A standard-track row from the pad's level / frequency."""
    a = amp if channels in ("AB", "A") else 0.0
    b = amp if channels in ("AB", "B") else 0.0
    return (a, b, freq, freq)


class TakeRecorder:
    def __init__(self, start_s: float = 0.0, end_s: float | None = None,
                 smooth_ms: float = 40.0):
        #: only samples inside [start_s, end_s) are kept (pre-roll / punch range)
        self.start_s = float(start_s)
        self.end_s = None if end_s is None else float(end_s)
        self.smooth_ms = float(smooth_ms)
        self.samples: list[tuple[float, tuple]] = []

    def add(self, t_s: float, row) -> None:
        self.samples.append((float(t_s), tuple(float(v) for v in row)))

    def __len__(self) -> int:
        return len(self.samples)

    def _inside(self, t: float) -> bool:
        return t >= self.start_s - 1e-9 and (self.end_s is None or t < self.end_s)

    def result(self) -> tuple[float, np.ndarray] | None:
        """``(start_s, (N, 4) float32)`` of the take, or None if nothing was kept."""
        pts = [(t, r) for t, r in self.samples if self._inside(t)]
        if not pts:
            return None
        t_lo = min(t for t, _ in pts)
        t_hi = max(t for t, _ in pts)
        k0 = int(np.floor(t_lo * FS + 1e-6))
        n = int(np.floor(t_hi * FS + 1e-6)) - k0 + 1
        data = np.zeros((n, 4), np.float32)
        filled = np.zeros(n, bool)
        max_hold = int(round(MAX_HOLD_S * FS))
        for i, (t, row) in enumerate(pts):
            a = int(np.floor(t * FS + 1e-6)) - k0
            if i + 1 < len(pts):
                t_next = pts[i + 1][0]
                b = int(np.floor(t_next * FS + 1e-6)) - k0
                if b <= a or b - a > max_hold:   # jumped back / long gap: this pulse only
                    b = a + 1
            else:
                b = a + 1
            data[a:b] = row
            filled[a:b] = True
        # bridge holes (dropped frames) by holding the previous value
        if not filled.all():
            idx = np.where(filled, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            data = data[idx]
        win = int(round(self.smooth_ms / 1000.0 * FS))
        if win > 1 and n > 1:
            kernel = np.ones(win) / win
            pad = win // 2
            for c in range(4):
                col = np.pad(data[:, c].astype(np.float64), (pad, win - 1 - pad), mode="edge")
                data[:, c] = np.convolve(col, kernel, mode="valid")
        return k0 / FS, np.clip(data, 0.0, 1.0).astype(np.float32)


def overlapping_takes(track, start_s: float, end_s: float) -> list:
    """Earlier recorded takes on *track* that overlap ``[start_s, end_s)``."""
    out = []
    for clip in track.clips:
        origin = getattr(clip.source, "origin", "")
        if (clip.source.kind == "pulses" and str(origin).startswith("take")
                and not clip.muted and clip.start_s < end_s and clip.end_s > start_s):
            out.append(clip)
    return out


def next_take_number(project) -> int:
    n = 0
    for track in project.tracks:
        for clip in track.clips:
            origin = str(getattr(clip.source, "origin", ""))
            if origin.startswith("take "):
                try:
                    n = max(n, int(origin.split()[1]))
                except (IndexError, ValueError):
                    pass
    return n + 1
