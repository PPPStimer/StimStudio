"""Video duration probing and the filmstrip thumbnail cache (background, PyAV)."""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QImage

from ...io import video as vio

THUMB_HEIGHT = 36


def probe_duration(path: str, timeout_ms: int = 5000) -> float:
    """Video duration in seconds: PyAV if installed, else Qt Multimedia (0 if unknown)."""
    if vio.have_pyav():
        try:
            d = vio.probe_duration_av(path)
            if d > 0:
                return d
        except OSError:
            pass
    from PyQt6.QtMultimedia import QMediaPlayer
    player = QMediaPlayer()
    loop = QEventLoop()
    player.durationChanged.connect(lambda _v: loop.quit())
    player.errorOccurred.connect(lambda *_a: loop.quit())
    QTimer.singleShot(timeout_ms, loop.quit)
    player.setSource(QUrl.fromLocalFile(path))
    if player.duration() <= 0:
        loop.exec()
    duration = player.duration() / 1000.0
    player.setSource(QUrl())
    player.deleteLater()
    return max(0.0, duration)


def _to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Filmstrip:
    """Thumbnails of one video: ``times`` (s) and ``images`` (None until decoded)."""

    def __init__(self, times: np.ndarray):
        self.times = times
        self.images: list[QImage | None] = [None] * len(times)

    def ready(self) -> int:
        return sum(img is not None for img in self.images)

    def image_at(self, t: float) -> QImage | None:
        """The decoded thumbnail nearest to video time *t*."""
        if not len(self.times):
            return None
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        if i > 0 and abs(self.times[i - 1] - t) < abs(self.times[i] - t):
            i -= 1
        if self.images[i] is not None:
            return self.images[i]
        for d in range(1, len(self.images)):       # nearest decoded neighbour
            for j in (i - d, i + d):
                if 0 <= j < len(self.images) and self.images[j] is not None:
                    return self.images[j]
        return None


class _JobSignals(QObject):
    frame = pyqtSignal(object, int, object)      # key, index, rgb array


class _ThumbJob(QRunnable):
    def __init__(self, key, path: str, times, height: int):
        super().__init__()
        self.key, self.path, self.times, self.height = key, path, times, height
        self.signals = _JobSignals()
        self.cancelled = False

    def run(self):
        try:
            for i, rgb in vio.iter_thumbnails(self.path, self.times, self.height,
                                              cancel=lambda: self.cancelled):
                self.signals.frame.emit(self.key, i, rgb)
        except Exception:  # noqa: BLE001 — unreadable video: no filmstrip
            pass


class ThumbnailCache(QObject):
    """Filmstrips keyed on (path, size, mtime); decoding runs on a worker thread."""

    updated = pyqtSignal(str)                    # path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: dict[tuple, Filmstrip] = {}
        self._jobs: dict[tuple, _ThumbJob] = {}
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pending: set[str] = set()
        self._flush = QTimer(self, singleShot=True, interval=150)
        self._flush.timeout.connect(self._emit_updates)

    @staticmethod
    def _key(path: str):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (os.path.abspath(path), st.st_size, int(st.st_mtime_ns))

    def strip(self, path: str, duration_s: float) -> Filmstrip | None:
        """The filmstrip of *path* (starts decoding on first request); None without PyAV."""
        if not vio.have_pyav() or duration_s <= 0:
            return None
        key = self._key(path)
        if key is None:
            return None
        strip = self._strips.get(key)
        if strip is None:
            strip = self._strips[key] = Filmstrip(vio.thumbnail_times(duration_s))
            job = self._jobs[key] = _ThumbJob(key, path, strip.times, THUMB_HEIGHT)
            job.signals.frame.connect(self._on_frame)
            self._pool.start(job)
        return strip

    def _on_frame(self, key, index: int, rgb) -> None:
        strip = self._strips.get(key)
        if strip is None:
            return
        strip.images[index] = _to_qimage(rgb)
        self._pending.add(key[0])
        if not self._flush.isActive():
            self._flush.start()

    def _emit_updates(self) -> None:
        for path in self._pending:
            self.updated.emit(path)
        self._pending.clear()

    def wait(self, timeout_ms: int = 10000) -> bool:
        """Block until decoding finished (tests)."""
        ok = self._pool.waitForDone(timeout_ms)
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        self._emit_updates()
        return ok

    def shutdown(self) -> None:
        for job in self._jobs.values():
            job.cancelled = True
        self._pool.waitForDone(3000)


class Waveform:
    """Peak envelope of a video's sound: ``peaks[i]`` = |max| over bin i (1/rate s)."""

    def __init__(self, duration_s: float, rate: int = vio.PEAK_RATE):
        self.rate = rate
        self.peaks = np.zeros(int(np.ceil(duration_s * rate)) + 1, np.float32)
        self.ready_bins = 0
        self.finished = False
        self.has_audio = True          # False once decoding found no audio stream
        self.top = 0.0                 # loudest peak so far (display normalisation)

    def add(self, first: int, values: np.ndarray) -> None:
        end = min(len(self.peaks), first + len(values))
        if end > first:
            self.peaks[first:end] = values[:end - first]
            self.ready_bins = max(self.ready_bins, end)
            self.top = max(self.top, float(values.max()))

    def column_peaks(self, t0: float, seconds_per_px: float, n_px: int) -> np.ndarray:
        """Max peak under each of *n_px* pixel columns starting at video time *t0*."""
        n = len(self.peaks)
        if n_px <= 0 or n == 0:
            return np.zeros(max(0, n_px), np.float32)
        edges = np.floor((t0 + np.arange(n_px + 1) * seconds_per_px) * self.rate).astype(int)
        edges = np.clip(edges, 0, n)
        lo, hi = edges[:-1], np.maximum(edges[1:], edges[:-1] + 1)   # at least one bin
        lo, hi = np.minimum(lo, n - 1), np.minimum(hi, n)
        padded = np.append(self.peaks, 0.0)                           # hi may equal n
        pairs = np.column_stack([lo, hi]).ravel()
        return np.maximum.reduceat(padded, pairs)[::2].astype(np.float32)


class _WaveSignals(QObject):
    chunk = pyqtSignal(object, int, object)      # key, first bin, peaks
    done = pyqtSignal(object, bool)              # key, found audio


class _WaveJob(QRunnable):
    def __init__(self, key, path: str):
        super().__init__()
        self.key, self.path = key, path
        self.signals = _WaveSignals()
        self.cancelled = False

    def run(self):
        found = False
        try:
            for first, peaks in vio.iter_audio_peaks(self.path, cancel=lambda: self.cancelled):
                found = True
                self.signals.chunk.emit(self.key, first, peaks)
        except Exception:  # noqa: BLE001 — unreadable audio: no waveform
            pass
        self.signals.done.emit(self.key, found)


class WaveformCache(QObject):
    """Waveforms of video files keyed on (path, size, mtime), decoded on a worker thread."""

    updated = pyqtSignal(str)                    # path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._waves: dict[tuple, Waveform] = {}
        self._jobs: dict[tuple, _WaveJob] = {}
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pending: set[str] = set()
        self._flush = QTimer(self)
        self._flush.setSingleShot(True)
        self._flush.setInterval(200)
        self._flush.timeout.connect(self._emit_updates)

    def waveform(self, path: str, duration_s: float) -> Waveform | None:
        if not vio.have_pyav() or duration_s <= 0:
            return None
        key = ThumbnailCache._key(path)
        if key is None:
            return None
        wave = self._waves.get(key)
        if wave is None:
            wave = self._waves[key] = Waveform(duration_s)
            job = self._jobs[key] = _WaveJob(key, path)
            job.signals.chunk.connect(self._on_chunk)
            job.signals.done.connect(self._on_done)
            self._pool.start(job)
        return wave

    def _on_chunk(self, key, first: int, peaks) -> None:
        wave = self._waves.get(key)
        if wave is not None:
            wave.add(first, peaks)
            self._touch(key)

    def _on_done(self, key, found: bool) -> None:
        wave = self._waves.get(key)
        if wave is not None:
            wave.finished = True
            wave.has_audio = found
            self._touch(key)

    def _touch(self, key) -> None:
        self._pending.add(key[0])
        if not self._flush.isActive():
            self._flush.start()

    def _emit_updates(self) -> None:
        for path in self._pending:
            self.updated.emit(path)
        self._pending.clear()

    def wait(self, timeout_ms: int = 20000) -> bool:
        ok = self._pool.waitForDone(timeout_ms)
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        self._emit_updates()
        return ok


_cache: ThumbnailCache | None = None
_wave_cache: WaveformCache | None = None


def thumbnail_cache() -> ThumbnailCache:
    global _cache
    if _cache is None:
        _cache = ThumbnailCache()
    return _cache


def waveform_cache() -> WaveformCache:
    global _wave_cache
    if _wave_cache is None:
        _wave_cache = WaveformCache()
    return _wave_cache
