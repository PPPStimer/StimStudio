"""Howl's native binary ``.hwl`` format.

Layout: 8-byte ASCII magic ``YEAHBOI!`` followed by N pulses, each 4 × float32
little-endian (l_amp, r_amp, l_freq, r_freq) in [0, 1], one pulse per 1/40 s.
No other header: total size = 8 + N × 16 bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.pulse import PULSE_BYTES, array_from_bytes, array_to_bytes, as_array

HWL_MAGIC = b"YEAHBOI!"


def encode_hwl(pulses) -> bytes:
    """Bytes of a .hwl file. Values are clamped to [0, 1]."""
    arr = np.clip(as_array(pulses), 0.0, 1.0)
    return HWL_MAGIC + array_to_bytes(arr)


def decode_hwl(data: bytes) -> np.ndarray:
    if data[:len(HWL_MAGIC)] != HWL_MAGIC:
        raise ValueError("not a valid .hwl file: missing 'YEAHBOI!' header")
    body = data[len(HWL_MAGIC):]
    if len(body) % PULSE_BYTES != 0:
        raise ValueError(f"corrupt .hwl: body length {len(body)} is not a multiple of {PULSE_BYTES}")
    return array_from_bytes(body)


def write_hwl(path, pulses) -> int:
    """Write *pulses* (an (N, 4) array or PulseTrack) to *path*. Returns bytes written."""
    data = encode_hwl(pulses)
    Path(path).write_bytes(data)
    return len(data)


def read_hwl(path) -> np.ndarray:
    """Read a .hwl file into an (N, 4) float32 array."""
    return decode_hwl(Path(path).read_bytes())
