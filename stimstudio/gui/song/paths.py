"""Fast curve paths for lanes and clip previews.

Only the visible time window is turned into a path, and when there are more
samples than pixels the data is reduced to one min/max pair per pixel column
(peak decimation), so a path never has many more vertices than the view is
wide, at any zoom level. ``pyqtgraph.arrayToQPath`` builds the QPainterPath
straight from numpy arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPainterPath

from ...core.pulse import FS


def curve_path(values: np.ndarray, t_start: float, pps: float, t0: float, t1: float,
               height: float, fs: int = FS, fill: bool = False) -> QPainterPath:
    """Path of a 0..1 curve in item-local coordinates.

    *values[i]* sits at time ``t_start + i / fs``; x = (t − t_start) · pps and
    y = height · (1 − v). Only samples within [t0, t1] (plus one on each side)
    are used. With *fill*, the path is closed down to the baseline.
    """
    n = len(values)
    if n == 0 or t1 <= t0 or pps <= 0:
        return QPainterPath()
    i0 = max(0, int(math.floor((t0 - t_start) * fs)) - 1)
    i1 = min(n, int(math.ceil((t1 - t_start) * fs)) + 2)
    if i1 - i0 < 1:
        return QPainterPath()
    seg = np.asarray(values[i0:i1], dtype=np.float64)

    samples_per_px = fs / pps
    if samples_per_px > 2.0:
        b = int(samples_per_px)
        m = (len(seg) // b) * b
        if m >= b:
            blocks = seg[:m].reshape(-1, b)
            mn, mx = blocks.min(axis=1), blocks.max(axis=1)
            idx = i0 + np.arange(len(mn)) * b
            x = np.repeat((idx / fs) * pps, 2)
            y = np.empty(2 * len(mn))
            y[0::2], y[1::2] = mn, mx
        else:
            x = ((i0 + np.arange(len(seg))) / fs) * pps
            y = seg
    else:
        x = ((i0 + np.arange(len(seg))) / fs) * pps
        y = seg
    y = height * (1.0 - np.clip(y, 0.0, 1.0))
    if fill:
        x = np.concatenate(([x[0]], x, [x[-1]]))
        y = np.concatenate(([height], y, [height]))
    return pg.arrayToQPath(x, y, connect="all")
