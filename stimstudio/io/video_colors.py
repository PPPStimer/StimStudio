"""Colour rhythm: how much of a target zone something fills, frame by frame.

Many videos carry their rhythm in shapes: something flies into a target (a
ring, a box) and vanishes, a flash fills it... One decoding pass (PyAV)
measures, per sampled frame, the **measure zone** -- the inner part of the
target the user dragged out (ellipse or rectangle, ``inner`` of its size, so
the target's own outline stays out) -- two ways:

* colour shares: its pixels sorted into a fixed palette of ``N_BINS`` HSV bins;
* difference from the surroundings: the background colour is the median of a
  band just outside the target (``BAND`` x its size), and each zone pixel's
  RGB distance to it is kept as a histogram. Whatever flies in counts, of any
  colour or shape, and a background that changes from scene to scene does not.

Everything after that -- the table of top colours, the fill signals, the
pulses -- is cheap numpy on those per-sample arrays.

Palette: greys (low saturation or very dark) in 8 brightness steps; colours in
24 hues of 15 degrees (bin 0 centred on pure red) x 2 saturations x 3
brightnesses.

Qt-free; tested on synthetic frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.pulse import FS
from .video_analysis import Cancelled, _smooth, iter_frames

__all__ = ["Cancelled", "ColorAnalysis", "ColorStat", "N_BINS", "Zone",
           "analyse_color_frames", "analyse_video_colors", "bin_label", "bin_mask", "bin_rgb",
           "color_bins", "color_stats", "difference_mask", "difference_signal",
           "dominant_bin", "fill_signal", "fill_to_pulses", "shape_signal", "zone_geometry"]

GREY_BINS = 8
HUES = 24
SATS = 2
VALS = 3
N_BINS = GREY_BINS + HUES * SATS * VALS            # 152

GREY_SAT = 0.20           # below: a grey
DARK_VAL = 0.12           # below: black, whatever the hue
SAT_SPLIT = 0.55          # pale | vivid
VAL_SPLITS = (0.45, 0.75)  # dark | normal | bright

SHAPES = ("ellipse", "rect")
BAND = (1.15, 1.5)        # background band: this far outside the target (x its half size)
DIST_STEP = 8             # RGB distance histogram: bins of this width ...
DIST_BINS = 56            # ... up to 448 (> sqrt(3) * 255)
DEFAULT_DIFFERENCE = 48   # distance from the background that counts as "something there"

MIN_ROI_PX = 128          # decode wide enough that the measure zone is at least this wide
MAX_WIDTH = 1280
BASE_WIDTH = 320

_HUE_NAMES = ("red", "orange", "yellow", "chartreuse", "green", "spring green", "cyan",
              "azure", "blue", "violet", "magenta", "rose")
_GREY_NAMES = ("black", "near black", "dark grey", "dark grey", "grey", "light grey",
               "light grey", "white")


# ── palette ─────────────────────────────────────────────────────────────────
def color_bins(rgb: np.ndarray) -> np.ndarray:
    """Palette bin (0..N_BINS-1, int16) of every pixel of an ``(..., 3)`` uint8 image."""
    a = np.asarray(rgb, dtype=np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    v = mx / 255.0
    s = c / np.maximum(mx, 1.0)
    safe = np.maximum(c, 1e-6)
    h = np.where(mx == r, np.mod((g - b) / safe, 6.0),
                 np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    hue = np.floor((h + 7.5) / 15.0).astype(np.int16) % HUES
    sat = (s >= SAT_SPLIT).astype(np.int16)
    val = np.searchsorted(np.asarray(VAL_SPLITS, np.float32), v, side="right").astype(np.int16)
    chroma = GREY_BINS + (hue * SATS + sat) * VALS + val
    grey = np.minimum(GREY_BINS - 1, (v * GREY_BINS).astype(np.int16))
    return np.where((s < GREY_SAT) | (v < DARK_VAL), grey, chroma).astype(np.int16)


def _split(bin_: int) -> tuple[int, int, int]:
    k = int(bin_) - GREY_BINS
    hue, rest = divmod(k, SATS * VALS)
    sat, val = divmod(rest, VALS)
    return hue, sat, val


def bin_rgb(bin_: int) -> tuple[int, int, int]:
    """A colour in the middle of the bin (for swatches before any analysis)."""
    import colorsys
    if bin_ < GREY_BINS:
        g = int(round((bin_ + 0.5) / GREY_BINS * 255))
        return g, g, g
    hue, sat, val = _split(bin_)
    s = (0.38, 0.8)[sat]
    v = (0.3, 0.6, 0.9)[val]
    rgb = colorsys.hsv_to_rgb(hue * 15.0 / 360.0, s, v)
    return tuple(int(round(x * 255)) for x in rgb)


def bin_label(bin_: int) -> str:
    """Human name of a bin, e.g. "bright vivid orange", "dark grey"."""
    if bin_ < GREY_BINS:
        return _GREY_NAMES[int(bin_)]
    hue, sat, val = _split(bin_)
    base = _HUE_NAMES[hue // 2] if hue % 2 == 0 else \
        f"{_HUE_NAMES[hue // 2]}-{_HUE_NAMES[(hue // 2 + 1) % len(_HUE_NAMES)]}"
    words = [("dark", "", "bright")[val], ("pale", "")[sat], base]
    return " ".join(w for w in words if w)


def bin_mask(rgb: np.ndarray, bins) -> np.ndarray:
    """``(h, w)`` bool: pixels of *rgb* that fall into any of *bins*."""
    return np.isin(color_bins(rgb), np.asarray(list(bins), dtype=np.int16))


def dominant_bin(rgb: np.ndarray, x: int, y: int, radius: int = 2) -> int:
    """The most common bin in a small square around pixel (x, y) — an eyedropper
    that ignores single compression-noise pixels."""
    h, w = rgb.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    patch = rgb[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    return int(np.bincount(color_bins(patch).ravel(), minlength=N_BINS).argmax())


# ── the target zone ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Zone:
    """Where to measure: *roi* (fractions x, y, w, h; None = whole frame) is the
    target's outline; the measure zone is its *inner* part (same *shape*,
    scaled around the centre); the background comes from the band ``BAND``
    outside it."""

    roi: tuple[float, float, float, float] | None = None
    shape: str = "rect"
    inner: float = 1.0


def zone_geometry(img_shape, zone: Zone):
    """``(ys, xs, measure, band)``: the crop around the target and its band, and
    the boolean masks of the measure zone and the background band inside that
    crop (the band may be empty, e.g. for the whole frame)."""
    h, w = img_shape[:2]
    if zone.roi is None:
        cx, cy, rx, ry = w / 2, h / 2, w / 2, h / 2
    else:
        x, y, rw, rh = zone.roi
        rx, ry = max(0.5, rw * w / 2), max(0.5, rh * h / 2)
        cx, cy = x * w + rx, y * h + ry
    out = BAND[1]
    x0 = int(np.clip(np.floor(cx - rx * out), 0, w - 1))
    x1 = int(np.clip(np.ceil(cx + rx * out), x0 + 1, w))
    y0 = int(np.clip(np.floor(cy - ry * out), 0, h - 1))
    y1 = int(np.clip(np.ceil(cy + ry * out), y0 + 1, h))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = (xx + 0.5 - cx) / rx
    dy = (yy + 0.5 - cy) / ry
    r = np.hypot(dx, dy) if zone.shape == "ellipse" else np.maximum(np.abs(dx), np.abs(dy))
    measure = r <= max(0.05, float(zone.inner)) + 1e-9
    if not measure.any():
        measure[np.unravel_index(int(np.argmin(r)), r.shape)] = True
    band = (r >= BAND[0]) & (r <= BAND[1])
    return slice(y0, y1), slice(x0, x1), measure, band


def _background(crop: np.ndarray, measure: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Background colour: median of the band (or of the zone if the band is
    too small, e.g. the target fills the frame)."""
    px = crop[band] if np.count_nonzero(band) >= 8 else crop[measure]
    return np.median(px.reshape(-1, 3), axis=0).astype(np.float32)


def difference_mask(rgb: np.ndarray, zone: Zone, threshold: float = DEFAULT_DIFFERENCE):
    """``(h, w)`` bool: zone pixels of *rgb* at least *threshold* away from the
    background colour (what the difference measure counts)."""
    ys, xs, measure, band = zone_geometry(rgb.shape, zone)
    crop = rgb[ys, xs]
    bg = _background(crop, measure, band)
    d = np.sqrt(((crop.astype(np.float32) - bg) ** 2).sum(axis=-1))
    out = np.zeros(rgb.shape[:2], bool)
    out[ys, xs] = measure & (d >= threshold)
    return out


# ── the decoding pass ───────────────────────────────────────────────────────
@dataclass
class ColorAnalysis:
    rate: float
    frac: np.ndarray              # (n, N_BINS) float32: share of the zone per colour bin
    mean_rgb: np.ndarray          # (N_BINS, 3) uint8: average real colour of each bin
    duration_s: float
    roi: tuple[float, float, float, float] | None = None
    roi_size: tuple[int, int] = (0, 0)       # measure zone in pixels (w, h)
    dist: np.ndarray | None = None           # (n, DIST_BINS) float32: share per distance bin
    bg_rgb: np.ndarray | None = None         # (n, 3) uint8: background colour per sample
    shape: str = "rect"
    inner: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frac)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n) / self.rate

    @property
    def zone(self) -> Zone:
        return Zone(self.roi, self.shape, self.inner)


def decode_width(roi, inner: float = 1.0) -> int:
    """Decoding width that keeps the measure zone at least ``MIN_ROI_PX`` wide."""
    rw = (roi[2] if roi else 1.0) * max(0.05, inner)
    return int(np.clip(MIN_ROI_PX / max(rw, 1e-3), BASE_WIDTH, MAX_WIDTH))


def analyse_color_frames(frames, rate: float, roi=None, duration_s: float | None = None,
                         shape: str = "rect", inner: float = 1.0) -> ColorAnalysis:
    """Measure an iterable of ``(t_s, rgb frame)`` (see module docstring)."""
    zone = Zone(tuple(roi) if roi else None, shape, float(inner))
    times, rows, dists, bgs = [], [], [], []
    sums = np.zeros((N_BINS, 3), np.float64)
    totals = np.zeros(N_BINS, np.float64)
    geo = None
    size = (0, 0)
    for t, img in frames:
        if geo is None:
            geo = zone_geometry(img.shape, zone)
            size = (int(np.any(geo[2], axis=0).sum()), int(np.any(geo[2], axis=1).sum()))
        ys, xs, measure, band = geo
        crop = img[ys, xs]
        px = crop[measure]
        bins = color_bins(px)
        counts = np.bincount(bins, minlength=N_BINS).astype(np.float64)
        for k in range(3):
            sums[:, k] += np.bincount(bins, weights=px[:, k], minlength=N_BINS)
        totals += counts
        bg = _background(crop, measure, band)
        d = np.sqrt(((px.astype(np.float32) - bg) ** 2).sum(axis=1))
        dh = np.bincount(np.minimum(d // DIST_STEP, DIST_BINS - 1).astype(np.int64),
                         minlength=DIST_BINS)
        times.append(float(t))
        rows.append((counts / max(1, len(bins))).astype(np.float32))
        dists.append((dh / max(1, len(px))).astype(np.float32))
        bgs.append(np.round(bg).astype(np.uint8))
    if not times:
        raise ValueError("no frames could be decoded")
    times_a = np.asarray(times)
    end = duration_s if duration_s and duration_s > times_a[-1] else times_a[-1] + 1.0 / rate
    n = max(1, int(round(end * rate)))
    grid = np.arange(n) / rate
    # sample-and-hold: every grid step shows the last frame at or before it
    idx = np.clip(np.searchsorted(times_a, grid + 1e-6, side="right") - 1, 0, len(rows) - 1)
    mean = np.array([bin_rgb(b) for b in range(N_BINS)], np.float64)
    seen = totals > 0
    mean[seen] = sums[seen] / totals[seen, None]
    return ColorAnalysis(rate=float(rate), frac=np.stack(rows)[idx],
                         mean_rgb=np.clip(np.round(mean), 0, 255).astype(np.uint8),
                         duration_s=float(end), roi=zone.roi, roi_size=size,
                         dist=np.stack(dists)[idx], bg_rgb=np.stack(bgs)[idx],
                         shape=zone.shape, inner=zone.inner)


def analyse_video_colors(path, rate: float = 30.0, roi=None, progress=None, cancel=None,
                         shape: str = "rect", inner: float = 1.0) -> ColorAnalysis:
    from .video import probe_duration_av
    duration = probe_duration_av(path)
    frames = iter_frames(path, rate, decode_width(roi, inner), "rgb24", cancel=cancel,
                         progress=progress)
    result = analyse_color_frames(frames, rate, roi, duration or None, shape, inner)
    if progress:
        progress(1.0)
    return result


# ── statistics ──────────────────────────────────────────────────────────────
@dataclass
class ColorStat:
    bin: int
    rgb: tuple[int, int, int]
    screen_pct: float        # average share of the region, %
    frames_pct: float        # share of the time the colour is present, %
    appearances: int         # times it showed up (absent → present)
    peak_pct: float          # largest share of the region, %

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)

    @property
    def label(self) -> str:
        return bin_label(self.bin)


def appearance_counts(frac: np.ndarray, presence: float) -> np.ndarray:
    """Per column of *frac*: how often it rose to *presence* after having fallen
    below ``presence / 2`` (hysteresis against flicker at the threshold)."""
    n, cols = frac.shape
    if n == 0:
        return np.zeros(cols, np.int64)
    on = frac >= presence
    off = frac < presence * 0.5
    event = np.where(on, 1, np.where(off, 0, -1)).astype(np.int8)
    event = np.vstack([np.zeros((1, cols), np.int8), event])     # starts absent
    rows = np.where(event >= 0, np.arange(n + 1)[:, None], 0)
    last = np.maximum.accumulate(rows, axis=0)
    state = np.take_along_axis(event, last, axis=0)
    return np.count_nonzero(np.diff(state, axis=0) > 0, axis=0)


def color_stats(analysis: ColorAnalysis, presence: float = 0.005) -> list[ColorStat]:
    """Every colour seen in the region (present ≥ *presence* of it at least
    once), most screen time first."""
    frac = analysis.frac
    screen = frac.mean(axis=0) * 100.0
    present = (frac >= presence).mean(axis=0) * 100.0
    peak = frac.max(axis=0) * 100.0
    appear = appearance_counts(frac, presence)
    out = [ColorStat(int(b), tuple(int(c) for c in analysis.mean_rgb[b]), float(screen[b]),
                     float(present[b]), int(appear[b]), float(peak[b]))
           for b in range(N_BINS) if present[b] > 0]
    out.sort(key=lambda s: -s.screen_pct)
    return out


# ── fill → pulses ───────────────────────────────────────────────────────────
def fill_signal(analysis: ColorAnalysis, bins) -> np.ndarray:
    """Share of the region (0..1) covered by any of *bins*, per sample."""
    bins = sorted(set(int(b) for b in bins))
    if not bins:
        return np.zeros(analysis.n, np.float64)
    return np.clip(analysis.frac[:, bins].astype(np.float64).sum(axis=1), 0.0, 1.0)


def difference_signal(analysis: ColorAnalysis,
                      threshold: float = DEFAULT_DIFFERENCE) -> np.ndarray:
    """Share of the zone (0..1) at least *threshold* (RGB distance, rounded to
    ``DIST_STEP``) away from the background colour, per sample."""
    k = int(np.clip(round(float(threshold) / DIST_STEP), 0, DIST_BINS - 1))
    return np.clip(analysis.dist[:, k:].astype(np.float64).sum(axis=1), 0.0, 1.0)


SIGNAL_MODES = ("fill", "rises", "drops")


def shape_signal(fill: np.ndarray, mode: str = "fill") -> np.ndarray:
    """What the amplitude follows: the *fill* itself, its rises (``rises``: a
    shape arriving) or its drops (``drops``: a shape vanishing — one sharp
    spike per hit, even when the next shape is already on its way)."""
    x = np.asarray(fill, dtype=np.float64)
    if mode == "fill" or len(x) < 2:
        return x
    d = np.diff(x, prepend=x[0])
    if mode == "rises":
        return np.maximum(d, 0.0)
    if mode == "drops":
        return np.maximum(-d, 0.0)
    raise ValueError(f"unknown signal mode: {mode!r}")


def resample_peaks(x: np.ndarray, rate: float, n_out: int, fs: int = FS) -> np.ndarray:
    """*x* (at *rate*) at *fs* samples per second. Slower sources are
    interpolated; faster ones keep the maximum of every output step, so a
    one-frame spike (a hit at 60 fps) is never lost between pulses."""
    x = np.asarray(x, dtype=np.float64)
    out = np.interp(np.arange(n_out) / fs, np.arange(len(x)) / rate, x)
    if rate > fs and len(x):
        slots = np.floor(np.arange(len(x)) * fs / rate + 1e-9).astype(np.int64)
        keep = slots < n_out
        peak = np.full(n_out, -np.inf)
        np.maximum.at(peak, slots[keep], x[keep])
        out = np.where(np.isfinite(peak), peak, out)
    return out.astype(np.float32)


def fill_to_pulses(signal: np.ndarray, rate: float, duration_s: float, smooth_s: float = 0.0,
                   normalize: bool = True, floor: float = 0.0, ceiling: float = 1.0,
                   freq: float = 0.5, mode: str = "fill") -> np.ndarray:
    """``(N, 4)`` pulses at 40 Hz: amplitude (L = R) follows the fill signal
    (or its rises / drops, see ``shape_signal``) mapped into [*floor*,
    *ceiling*] (after optional smoothing and scaling the maximum to 1),
    frequency is constant. Peaks survive the resampling (``resample_peaks``)."""
    s = np.asarray(signal, dtype=np.float64)
    s = _smooth(s, max(1, int(round(smooth_s * rate))))
    s = shape_signal(s, mode)
    if normalize:
        top = float(s.max()) if len(s) else 0.0
        s = s / top if top > 1e-9 else np.zeros_like(s)
    amp = floor + (ceiling - floor) * np.clip(s, 0.0, 1.0)
    n_out = max(1, int(round(duration_s * FS)))
    out = np.zeros((n_out, 4), np.float32)
    out[:, 0] = out[:, 1] = resample_peaks(amp, rate, n_out)
    out[:, 2] = out[:, 3] = float(freq)
    return np.clip(out, 0.0, 1.0)
