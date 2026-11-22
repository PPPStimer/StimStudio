"""Video files (no Qt): recognition, duration probe and thumbnails via PyAV.

PyAV (``pip install av``) is optional. Without it the GUI probes the duration
with Qt Multimedia and the clip shows no filmstrip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mkv", ".webm", ".avi", ".mov", ".wmv", ".mpg", ".mpeg")


def is_video_file(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def have_pyav() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def find_sibling_funscripts(video_path) -> list[str]:
    """``name.funscript`` and multi-axis ``name.<axis>.funscript`` next to the video."""
    video = Path(video_path)
    stem = video.stem.lower()
    found = []
    try:
        entries = sorted(video.parent.iterdir())
    except OSError:
        return []
    for f in entries:
        name = f.name.lower()
        if not name.endswith(".funscript") or not f.is_file():
            continue
        base = name[:-len(".funscript")]
        if base == stem or (base.startswith(stem + ".") and "." not in base[len(stem) + 1:]):
            found.append(str(f))
    return found


def probe_duration_av(path) -> float:
    """Duration in seconds with PyAV (raises ImportError without it, OSError if unreadable)."""
    import av
    try:
        with av.open(str(path)) as container:
            if container.duration:
                return container.duration / 1_000_000.0          # AV_TIME_BASE
            stream = container.streams.video[0]
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
    except av.error.FFmpegError as exc:
        raise OSError(str(exc)) from exc
    return 0.0


def probe_fps(path) -> float:
    """Average frame rate of the first video stream (0.0 if unknown)."""
    import av
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return 0.0
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            return float(rate) if rate else 0.0
    except av.error.FFmpegError:
        return 0.0


def thumbnail_times(duration_s: float, every_s: float = 2.0, max_count: int = 400) -> np.ndarray:
    """Evenly spaced sample times (cell centres) covering the video."""
    if duration_s <= 0:
        return np.zeros(0)
    count = int(np.clip(round(duration_s / every_s), 1, max_count))
    return (np.arange(count) + 0.5) * duration_s / count


PEAK_RATE = 100          # waveform bins per second
PEAK_SR = 8000           # decode rate for the waveform (mono)


def has_audio(path) -> bool:
    import av
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except av.error.FFmpegError:
        return False


def iter_audio_peaks(path, rate: int = PEAK_RATE, cancel=None,
                     chunk_s: float = 20.0) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_bin, peaks)``: the absolute peak (0..1) of each 1/rate s of the
    video's first audio stream, mixed to mono, in chunks of about *chunk_s* seconds."""
    import av
    per_bin = PEAK_SR // rate
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="flt", layout="mono", rate=PEAK_SR)
        pending = np.zeros(0, np.float32)
        parts: list[np.ndarray] = []
        done_bins = 0
        chunk = int(chunk_s * PEAK_SR)

        def flush(final=False):
            nonlocal pending, done_bins
            data = np.concatenate([pending, *parts]) if parts else pending
            parts.clear()
            usable = len(data) if final else len(data) // per_bin * per_bin
            block = data[:usable]
            pending = data[usable:]
            if not len(block):
                return None
            if final and len(block) % per_bin:
                block = np.pad(block, (0, per_bin - len(block) % per_bin))
            peaks = np.abs(block.reshape(-1, per_bin)).max(axis=1)
            first = done_bins
            done_bins += len(peaks)
            return first, np.minimum(peaks, 1.0).astype(np.float32)

        size = 0
        for frame in container.decode(stream):
            if cancel and cancel():
                return
            for out in resampler.resample(frame):
                arr = out.to_ndarray().reshape(-1)
                parts.append(arr)
                size += len(arr)
            if size >= chunk:
                size = 0
                got = flush()
                if got is not None:
                    yield got
        for out in resampler.resample(None):
            parts.append(out.to_ndarray().reshape(-1))
        got = flush(final=True)
        if got is not None:
            yield got


def iter_thumbnails(path, times, height: int = 36,
                    cancel=None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, rgb uint8 (h, w, 3))`` for each time (nearest earlier key frame)."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        w0, h0 = stream.codec_context.width or 16, stream.codec_context.height or 9
        width = max(2, round(height * w0 / max(h0, 1)) // 2 * 2)
        tb = float(stream.time_base) if stream.time_base else 1e-6
        for i, t in enumerate(times):
            if cancel and cancel():
                return
            try:
                container.seek(int(t / tb), stream=stream, any_frame=False, backward=True)
                frame = next(container.decode(stream), None)
            except (av.error.FFmpegError, StopIteration):
                frame = None
            if frame is None:
                continue
            yield i, frame.reformat(width=width, height=height, format="rgb24").to_ndarray()


def grab_frame(path, t: float, height: int = 360) -> np.ndarray | None:
    """The frame shown at *t* seconds (decoded from the key frame before it),
    as ``rgb uint8 (h, w, 3)``; None if nothing could be decoded."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        w0, h0 = stream.codec_context.width or 16, stream.codec_context.height or 9
        height = min(int(height), h0) // 2 * 2 or 2
        width = max(2, round(height * w0 / max(h0, 1)) // 2 * 2)
        tb = float(stream.time_base) if stream.time_base else 1e-6
        best = None
        try:
            container.seek(int(max(0.0, t) / tb), stream=stream, any_frame=False, backward=True)
            for frame in container.decode(stream):
                if best is not None and frame.time is not None and frame.time > t + 1e-3:
                    break
                best = frame
                if frame.time is None:
                    break
        except av.error.FFmpegError:
            pass
        if best is None:
            return None
        return best.reformat(width=width, height=height, format="rgb24").to_ndarray()
