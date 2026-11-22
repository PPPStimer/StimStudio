"""Video analysis for scripting: scene cuts and motion → stim signals.

One decoding pass (PyAV) samples the video at a fixed rate, shrinks each frame
to a small grey image and measures, per sample:

* ``activity``  — mean absolute frame difference inside the region of interest
  (how much is moving there), 0..1;
* ``cut_score`` — spikes at hard cuts, 0..1: the geometric mean of the
  (soft-binned) grey-histogram chi² distance and the structural change of the
  whole frame (mean absolute difference after removing each frame's mean
  brightness). A brightness drift changes only the histogram, fast motion
  mostly only the structure; a cut changes both;
* ``motion_y``  — vertical motion inside the ROI in ROI heights per second
  (positive = downwards). Phase correlation in numpy by default; OpenCV's
  Farneback optical flow when ``flow=True`` and ``cv2`` is installed.

Qt-free; everything above the decoder is plain numpy and tested on synthetic
clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.pulse import FS

ANALYSIS_WIDTH = 160      # frames are shrunk to this width before measuring
HIST_BINS = 32


class Cancelled(Exception):
    pass


@dataclass
class VideoAnalysis:
    rate: float
    activity: np.ndarray
    cut_score: np.ndarray
    motion_y: np.ndarray
    duration_s: float
    roi: tuple[float, float, float, float] | None = None
    method: str = "phase"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.activity)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n) / self.rate


def have_cv2() -> bool:
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


# ── per-frame measures ──────────────────────────────────────────────────────
def roi_slice(shape, roi) -> tuple[slice, slice]:
    """Pixel slices of *roi* (fractions x, y, w, h) inside an image of *shape*."""
    h, w = shape[:2]
    if roi is None:
        return slice(0, h), slice(0, w)
    x, y, rw, rh = roi
    x0 = int(np.clip(round(x * w), 0, w - 1))
    y0 = int(np.clip(round(y * h), 0, h - 1))
    x1 = int(np.clip(round((x + rw) * w), x0 + 1, w))
    y1 = int(np.clip(round((y + rh) * h), y0 + 1, h))
    return slice(y0, y1), slice(x0, x1)


def grey_histogram(img: np.ndarray) -> np.ndarray:
    """Normalised grey histogram with linear (soft) binning: a slow brightness
    drift moves mass gradually between neighbouring bins instead of flipping
    a flat area from one bin to the next (which would look like a cut)."""
    pos = img.astype(np.float32).ravel() * (HIST_BINS / 256.0) - 0.5
    lo = np.floor(pos)
    frac = pos - lo
    lo = lo.astype(np.int64)
    hist = (np.bincount(np.clip(lo, 0, HIST_BINS - 1), weights=1.0 - frac,
                        minlength=HIST_BINS)
            + np.bincount(np.clip(lo + 1, 0, HIST_BINS - 1), weights=frac,
                          minlength=HIST_BINS))
    return hist / max(1e-9, hist.sum())


def chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """Symmetric chi² distance of two normalised histograms, 0..1."""
    s = h1 + h2
    nz = s > 0
    return float(0.5 * np.sum((h1[nz] - h2[nz]) ** 2 / s[nz]))


STRUCT_FULL = 0.12          # structural change treated as "completely different"


def structural_change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |Δ| of two grey frames after removing their mean brightness, 0..1."""
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    return float(np.abs((fa - fa.mean()) - (fb - fb.mean())).mean()) / 255.0


def cut_measure(prev_img, img, prev_hist, hist) -> float:
    c = chi2(prev_hist, hist)
    st = min(1.0, structural_change(prev_img, img) / STRUCT_FULL)
    return float(np.sqrt(c * st))


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (dy, dx) in pixels that moves *a* onto *b* (phase correlation)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def sub(c_m, c_0, c_p):             # parabolic sub-pixel refinement
        d = c_m - 2 * c_0 + c_p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_m - c_p) / d

    dy = py + sub(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = px + sub(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def flow_y(a: np.ndarray, b: np.ndarray) -> float:
    """Mean vertical optical flow (pixels), weighted by flow magnitude (OpenCV)."""
    import cv2
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    wsum = float(mag.sum())
    if wsum < 1e-6:
        return 0.0
    return float((flow[..., 1] * mag).sum() / wsum)


# ── the decoding pass ───────────────────────────────────────────────────────
def iter_frames(path, rate: float, width: int = ANALYSIS_WIDTH, fmt: str = "gray",
                cancel=None, progress=None):
    """Yield ``(t_s, image uint8)`` sampled at about *rate* frames per second,
    scaled to *width* (never wider than the source). *fmt* is a PyAV pixel
    format: ``"gray"`` gives ``(h, w)``, ``"rgb24"`` gives ``(h, w, 3)``."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("the file has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        src_w = stream.codec_context.width or width
        src_h = stream.codec_context.height or width
        width = max(8, min(int(width), src_w) // 2 * 2)
        height = max(8, int(round(width * src_h / max(1, src_w) / 2)) * 2)
        step = 1.0 / rate
        next_t = 0.0
        last_report = -1.0
        for frame in container.decode(stream):
            if cancel and cancel():
                raise Cancelled()
            t = frame.time if frame.time is not None else next_t
            if t + 1e-6 < next_t:
                continue
            next_t = (np.floor(t / step + 1e-6) + 1) * step
            img = frame.reformat(width=width, height=height, format=fmt).to_ndarray()
            if progress and duration > 0 and t - last_report >= 0.5:
                last_report = t
                progress(min(1.0, t / duration))
            yield float(t), img


def iter_grey_frames(path, rate: float, width: int = ANALYSIS_WIDTH, cancel=None,
                     progress=None):
    """Yield ``(t_s, grey uint8 (h, w))`` sampled at about *rate* frames per second."""
    return iter_frames(path, rate, width, "gray", cancel, progress)


def analyse_frames(frames, rate: float, roi=None, flow: bool = False,
                   duration_s: float | None = None) -> VideoAnalysis:
    """Measure an iterable of ``(t_s, grey frame)`` (see module docstring)."""
    use_flow = flow and have_cv2()
    times, act, cuts, mot = [], [], [], []
    prev = prev_roi = prev_hist = None
    ys = xs = None
    for t, img in frames:
        if ys is None:
            ys, xs = roi_slice(img.shape, roi)
        cur_roi = img[ys, xs]
        hist = grey_histogram(img)
        if prev is None:
            a = c = m = 0.0
        else:
            a = float(np.abs(cur_roi.astype(np.int16) - prev_roi.astype(np.int16)).mean()) / 255
            c = cut_measure(prev, img, prev_hist, hist)
            dt = max(1e-3, t - times[-1])
            if c > 0.5:                   # across a cut motion is meaningless
                m = 0.0
            elif use_flow:
                m = flow_y(prev_roi, cur_roi) / cur_roi.shape[0] / dt
            else:
                m = phase_shift(prev_roi, cur_roi)[0] / cur_roi.shape[0] / dt
        times.append(t)
        act.append(a)
        cuts.append(c)
        mot.append(m)
        prev, prev_roi, prev_hist = img, cur_roi, hist
    if not times:
        raise ValueError("no frames could be decoded")
    times_a = np.asarray(times)
    end = duration_s if duration_s and duration_s > times_a[-1] else times_a[-1] + 1.0 / rate
    n = max(1, int(round(end * rate)))
    grid = np.arange(n) / rate
    cut_grid = np.zeros(n, np.float32)
    np.maximum.at(cut_grid, np.clip(np.round(times_a * rate).astype(int), 0, n - 1),
                  np.asarray(cuts, np.float32))
    return VideoAnalysis(
        rate=float(rate),
        activity=np.interp(grid, times_a, act).astype(np.float32),
        cut_score=cut_grid,
        motion_y=np.interp(grid, times_a, mot).astype(np.float32),
        duration_s=float(end), roi=tuple(roi) if roi else None,
        method="flow" if use_flow else "phase")


def analyse_video(path, rate: float = 20.0, roi=None, flow: bool = False,
                  progress=None, cancel=None) -> VideoAnalysis:
    from .video import probe_duration_av
    duration = probe_duration_av(path)
    result = analyse_frames(iter_grey_frames(path, rate, cancel=cancel, progress=progress),
                            rate, roi, flow, duration or None)
    if progress:
        progress(1.0)
    return result


# ── scene cuts ──────────────────────────────────────────────────────────────
def detect_cuts(analysis: VideoAnalysis, sensitivity: float = 0.5,
                min_gap_s: float = 1.0) -> list[float]:
    """Times (s, relative to the video start) of hard cuts.

    A sample is a cut when its histogram distance stands out from its
    neighbourhood (median + k·MAD over ±2 s) and exceeds an absolute floor;
    *sensitivity* 0..1 lowers both. Peaks closer than *min_gap_s* keep the
    strongest.
    """
    score = analysis.cut_score.astype(np.float64)
    n = len(score)
    if n < 3:
        return []
    s = float(np.clip(sensitivity, 0.0, 1.0))
    k = 12.0 - 9.0 * s                     # MADs above the local median
    floor = 0.50 - 0.34 * s                # absolute floor (0.33 at the default 0.5)
    half = max(2, int(round(2.0 * analysis.rate)))
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(score, half, mode="edge")
    windows = sliding_window_view(padded, 2 * half + 1)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1)
    thresh = np.maximum(med + k * np.maximum(mad, 0.004), floor)
    cand = np.flatnonzero(score > thresh)
    cuts: list[int] = []
    gap = max(1, int(round(min_gap_s * analysis.rate)))
    for i in cand:
        if cuts and i - cuts[-1] < gap:
            if score[i] > score[cuts[-1]]:
                cuts[-1] = int(i)
            continue
        cuts.append(int(i))
    return [i / analysis.rate for i in cuts if i > 0]


# ── motion → pulses ─────────────────────────────────────────────────────────
FREQ_MODES = ("constant", "speed", "direction")


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < 2:
        return x
    n = min(n, len(x))
    kernel = np.hanning(n + 2)[1:-1]
    kernel /= kernel.sum()
    pad = n // 2
    return np.convolve(np.pad(x, (pad, n - 1 - pad), mode="edge"), kernel, mode="valid")


def _normalise(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    top = float(np.percentile(np.abs(x), pct)) if len(x) else 0.0
    if top < 1e-9:
        return np.zeros_like(x)
    return np.clip(x / top, -1.0, 1.0)


def _to_pulse_rate(x: np.ndarray, rate: float, n_out: int) -> np.ndarray:
    src_t = np.arange(len(x)) / rate
    return np.interp(np.arange(n_out) / FS, src_t, x).astype(np.float32)


def motion_to_pulses(analysis: VideoAnalysis, triphase: bool = False, smooth_s: float = 0.3,
                     gamma: float = 1.0, floor: float = 0.0, ceiling: float = 1.0,
                     freq_mode: str = "constant", freq: float = 0.5,
                     stroke_depth: float = 0.8) -> np.ndarray:
    """Turn the analysis into an ``(N, 4)`` pulse array at 40 Hz.

    Standard: ``amp_l = amp_r`` = normalised activity mapped into
    [*floor*, *ceiling*] (with *gamma*); frequency is constant (*freq*), follows
    the speed of motion, or the vertical direction (``freq_mode``).

    Triphase: (volume, x, y, freq) — volume from activity, the position runs
    along the vertical axis (x = 0.5) with the integrated vertical motion
    (a stroke), high-passed so it never drifts off.
    """
    rate = analysis.rate
    n_out = max(1, int(round(analysis.duration_s * FS)))
    win = max(1, int(round(smooth_s * rate)))
    act = _smooth(_normalise(analysis.activity.astype(np.float64)), win)
    act = np.clip(act, 0.0, 1.0) ** max(0.05, gamma)
    amp = floor + (ceiling - floor) * act
    motion = _smooth(analysis.motion_y.astype(np.float64), win)
    if freq_mode == "speed":
        fr = 0.15 + 0.8 * np.clip(_normalise(np.abs(motion)), 0, 1)
    elif freq_mode == "direction":
        fr = 0.5 - 0.45 * _normalise(motion)          # up → higher
    else:
        fr = np.full_like(amp, float(freq))
    out = np.zeros((n_out, 4), np.float32)
    if not triphase:
        a = _to_pulse_rate(amp, rate, n_out)
        f = _to_pulse_rate(np.clip(fr, 0, 1), rate, n_out)
        out[:, 0] = out[:, 1] = a
        out[:, 2] = out[:, 3] = f
        return np.clip(out, 0.0, 1.0)
    pos = np.cumsum(motion) / rate                    # ROI heights travelled
    hp = max(3, int(round(4.0 * rate)))               # 4 s high-pass (drift removal)
    pos = pos - _smooth(pos, hp)
    pos = _normalise(pos, 98.0) * stroke_depth
    # triphase columns are volume, x, y, freq with x / y in 0..1 (0.5 = centre);
    # moving down the screen moves towards the bottom electrode (N)
    out[:, 0] = _to_pulse_rate(amp, rate, n_out)
    out[:, 1] = 0.5
    out[:, 2] = _to_pulse_rate(0.5 - 0.5 * pos, rate, n_out)
    out[:, 3] = _to_pulse_rate(np.clip(fr, 0, 1), rate, n_out)
    return np.clip(out, 0.0, 1.0)
