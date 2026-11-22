"""Pulse data containers.

A pulse is one update slot of 1/40 s (Howl / Coyote 3). Each pulse holds four
normalized values in [0, 1]: left/right amplitude and left/right frequency.

Two equivalent representations are used:

* ``PulseArray`` — one ``(N, 4)`` float32 array, columns in on-disk order
  (``L_AMP, R_AMP, L_FREQ, R_FREQ``). The timeline, mixer and effects use it.
* :class:`PulseTrack` — four separate arrays; the generator API (ported from
  WaveHowl) produces these. Bridge with :meth:`PulseTrack.to_array` /
  :meth:`PulseTrack.from_array`.
"""

from __future__ import annotations

import numpy as np

#: Pulses per second. Fixed to the Coyote 3 / Howl update rate.
FS = 40

#: Number of bytes per pulse on disk: 4 x float32.
PULSE_BYTES = 16

#: Column indices of a PulseArray.
L_AMP, R_AMP, L_FREQ, R_FREQ = 0, 1, 2, 3
N_COLS = 4
AMP_COLS = (L_AMP, R_AMP)      # indexed by side (A, B)
FREQ_COLS = (L_FREQ, R_FREQ)

#: Lane keys in column order (used by the project model).
LANE_KEYS = ("amp_l", "amp_r", "freq_l", "freq_r")

PulseArray = np.ndarray  # shape (N, 4), float32, values in 0..1

_CHANNELS = ("l_amp", "r_amp", "l_freq", "r_freq")


def n_pulses(seconds: float) -> int:
    """Pulse count for a duration (never negative)."""
    return max(0, round(seconds * FS))


def pulse_index(seconds: float) -> int:
    """Index of the pulse at a timeline position."""
    return round(seconds * FS)


class PulseTrack:
    """Four equal-length float32 arrays: l_amp, r_amp, l_freq, r_freq."""

    __slots__ = _CHANNELS

    def __init__(self, l_amp, r_amp, l_freq, r_freq):
        arrs = [np.ascontiguousarray(a, dtype=np.float32) for a in (l_amp, r_amp, l_freq, r_freq)]
        n = arrs[0].shape[0]
        for a in arrs:
            if a.ndim != 1 or a.shape[0] != n:
                raise ValueError("all channel arrays must be 1-D and of equal length")
        self.l_amp, self.r_amp, self.l_freq, self.r_freq = arrs

    # -- construction -----------------------------------------------------
    @classmethod
    def zeros(cls, n: int) -> "PulseTrack":
        n = max(0, int(n))
        return cls(*[np.zeros(n, dtype=np.float32) for _ in range(4)])

    @classmethod
    def from_duration(cls, duration_s: float) -> "PulseTrack":
        return cls.zeros(round(duration_s * FS))

    @classmethod
    def from_array(cls, arr) -> "PulseTrack":
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != N_COLS:
            raise ValueError("expected an (N, 4) pulse array")
        return cls(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])

    # -- properties -------------------------------------------------------
    @property
    def n(self) -> int:
        return self.l_amp.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n / FS

    @property
    def channels(self):
        return (self.l_amp, self.r_amp, self.l_freq, self.r_freq)

    # -- ops --------------------------------------------------------------
    def clamp(self) -> "PulseTrack":
        """Clip every channel into [0, 1] in place and return self."""
        for name in _CHANNELS:
            arr = getattr(self, name)
            np.clip(arr, 0.0, 1.0, out=arr)
        return self

    def to_array(self) -> PulseArray:
        """(N, 4) float32 array in column order l_amp, r_amp, l_freq, r_freq."""
        return np.stack(self.channels, axis=1).astype(np.float32, copy=False)

    def copy(self) -> "PulseTrack":
        return PulseTrack(*[a.copy() for a in self.channels])

    def slice(self, start: int, stop: int) -> "PulseTrack":
        """Return a new PulseTrack with pulses [start:stop]."""
        return PulseTrack(*[a[start:stop].copy() for a in self.channels])

    # -- serialization ----------------------------------------------------
    def to_bytes(self) -> bytes:
        """Interleave to the on-disk layout: per pulse l_amp, r_amp, l_freq, r_freq."""
        return np.ascontiguousarray(self.to_array(), dtype="<f4").tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "PulseTrack":
        return cls.from_array(array_from_bytes(data))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"PulseTrack(n={self.n}, {self.duration_s:.2f}s)"


def array_to_bytes(pulses: PulseArray) -> bytes:
    """Little-endian f32 interleaved bytes (the .hwl body / parthwl cache layout)."""
    return np.ascontiguousarray(pulses, dtype="<f4").tobytes()


def array_from_bytes(data: bytes) -> PulseArray:
    flat = np.frombuffer(data, dtype="<f4")
    if flat.size % N_COLS != 0:
        raise ValueError("pulse data length must be a multiple of 4 float32 values")
    return flat.reshape(-1, N_COLS).astype(np.float32)  # writable copy


def as_array(pulses) -> PulseArray:
    """Accept a PulseTrack or an (N, 4) array; return an (N, 4) float32 array."""
    if isinstance(pulses, PulseTrack):
        return pulses.to_array()
    arr = np.asarray(pulses, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_COLS:
        raise ValueError("expected an (N, 4) pulse array")
    return arr
