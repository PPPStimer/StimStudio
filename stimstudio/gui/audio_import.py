"""Audio import: choose how a file is used, then run the work in the background.

A quick look at three excerpts (``audio.triphase_probe``) says whether the
file is three-phase or two-channel; the dialog shows why and offers both:

* **Three-phase** — Tools → Triphase audio opens with the file (phase kept).
* **Two-channel** — loudness → amplitude, pitch → frequency, per channel
  (``audio.convert``); the result is a pulse-data clip.
* **Modulator** — the file goes on an ``audio_mod`` track and is analysed into
  envelope / onsets / band signals that can drive other tracks (Modulation).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QProgressDialog,
                             QPushButton, QSizePolicy, QToolButton, QVBoxLayout)

from ..audio.convert import METHODS, Cancelled
from . import metrics, theme

METHOD_LABELS = {"rolloff": "Spectral rolloff (bright vs. dark — default)",
                 "centroid": "Spectral centroid (smoother)",
                 "yin": "Pitch (YIN — melodic material)"}


class _JobSignals(QObject):
    progress = pyqtSignal(str, float)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()


class AudioJob(QRunnable):
    """``mode`` "convert" → (N, 4) pulses; "analyze" → signal dict;
    "triphase" → ``TriAudioAnalysis`` (Tools → Triphase audio); "probe" →
    ``ProbeResult`` (three-phase or two-channel?)."""

    def __init__(self, path: str, mode: str, method: str = "rolloff"):
        super().__init__()
        self.setAutoDelete(False)
        self.path, self.mode, self.method = path, mode, method
        self.signals = _JobSignals()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.mode == "convert":
                from ..audio.convert import audio_to_pulses
                result = audio_to_pulses(self.path, self.method,
                                         lambda stage, f: self.signals.progress.emit(stage, f),
                                         lambda: self._cancel)
            elif self.mode == "probe":
                from ..audio.triphase_probe import probe_file
                result = probe_file(self.path,
                                    lambda stage, f: self.signals.progress.emit(stage, f),
                                    lambda: self._cancel)
            elif self.mode == "triphase":
                from ..audio.triphase_decode import analyse_file as analyse_tri
                result = analyse_tri(self.path,
                                     lambda stage, f: self.signals.progress.emit(stage, f),
                                     lambda: self._cancel)
            else:
                from ..audio.analysis import analyze_file
                result = analyze_file(self.path,
                                      lambda f: self.signals.progress.emit("Analysing audio", f),
                                      lambda: self._cancel)
        except Cancelled:
            self.signals.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit(result)


def run_job(parent, job, title: str, window_title: str = "Audio import"):
    """Run *job* on the thread pool behind a modal progress dialog.

    Returns the result, or None if it was cancelled; raises RuntimeError on failure.
    """
    dlg = QProgressDialog(title, "Cancel", 0, 1000, parent)
    dlg.setWindowTitle(window_title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(300)
    outcome: dict = {}
    loop = QEventLoop()

    def on_progress(stage, frac):
        dlg.setLabelText(f"{title}\n{stage}…")
        dlg.setValue(int(frac * 1000))

    job.signals.progress.connect(on_progress)
    job.signals.done.connect(lambda r: (outcome.setdefault("result", r), loop.quit()))
    job.signals.failed.connect(lambda m: (outcome.setdefault("error", m), loop.quit()))
    job.signals.cancelled.connect(loop.quit)
    dlg.canceled.connect(job.cancel)
    QThreadPool.globalInstance().start(job)
    loop.exec()
    dlg.reset()
    dlg.deleteLater()
    if "error" in outcome:
        raise RuntimeError(outcome["error"])
    return outcome.get("result")


class AudioImportDialog(QDialog):
    """The import choice after a quick look at the file (``audio.triphase_probe``):
    a verdict with its reasons, two equal buttons — three-phase or two-channel —
    the recommended one as the default, and a small "Add as modulator…"."""

    def __init__(self, path: str, probe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import audio")
        self.choice: str | None = None            # "triphase" | "convert" | "analyze"
        self.probe = probe
        lay = QVBoxLayout(self)
        name = QLabel(f"<b>{escape(Path(path).name)}</b>")
        name.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(name)
        if len(probe.segments) > 1:
            where = ", ".join(theme.format_time(s, 0) for s, _ in probe.segments)
            looked = (f"{len(probe.segments)} excerpts of {probe.segments[0][1]:.0f} s "
                      f"(at {where})")
        else:
            looked = "the whole file"
        info = QLabel(f"{theme.format_time(probe.duration_s, 0)} · {probe.sr} Hz · "
                      f"{'stereo' if probe.channels == 2 else f'{probe.channels} ch'} · "
                      f"looked at {looked}")
        info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(info)

        tri = probe.is_triphase
        if probe.silent:
            head = "Could not tell: the excerpts are silent"
        else:
            head = ("The file is most likely <b>three-phase</b>" if tri else
                    "The file is most likely <b>two-channel</b> (ordinary stereo)")
        self.verdict = QLabel(head)
        self.verdict.setTextFormat(Qt.TextFormat.RichText)
        self.verdict.setStyleSheet(f"font-size: 14pt; color: {theme.TEXT};")
        lay.addSpacing(6)
        lay.addWidget(self.verdict)
        self.reasons = QLabel("<ul style='margin-left: -24px'>" + "".join(
            f"<li>{escape(r)}</li>" for r in probe.reasons) + "</ul>")
        self.reasons.setTextFormat(Qt.TextFormat.RichText)
        self.reasons.setWordWrap(True)
        lay.addWidget(self.reasons)

        row = QHBoxLayout()
        row.addWidget(QLabel("Two-channel frequency from:"))
        self.method = QComboBox()
        for m in METHODS:
            self.method.addItem(METHOD_LABELS[m], m)
        row.addWidget(self.method, 1)
        lay.addLayout(row)

        buttons = QHBoxLayout()
        self.tri_btn = QPushButton("Import as three-phase audio")
        self.tri_btn.setToolTip("Opens Tools → Triphase audio with this file: choose the "
                                "output (A/B or a triphase track) and Create clip")
        self.two_btn = QPushButton("Import as two-channel audio")
        self.two_btn.setToolTip("Loudness → amplitude, pitch → frequency, per channel: "
                                "a pulse clip")
        for b, key in ((self.tri_btn, "triphase"), (self.two_btn, "convert")):
            b.setMinimumHeight(40)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(lambda _c=False, k=key: self._choose(k))
            buttons.addWidget(b, 1)
        self.tri_btn.setEnabled(probe.channels >= 2)
        recommended = self.tri_btn if tri else self.two_btn
        recommended.setDefault(True)
        recommended.setAutoDefault(True)
        recommended.setFocus()
        other = self.two_btn if tri else self.tri_btn
        other.setAutoDefault(False)
        other.setObjectName("secondary")                 # same size, grey: not recommended
        lay.addSpacing(6)
        lay.addLayout(buttons)

        bottom = QHBoxLayout()
        self.mod_btn = QToolButton()                     # small, link-like
        self.mod_btn.setText("Add as modulator…")
        self.mod_btn.setToolTip("Envelope, beats and bands drive other tracks "
                                "(Modulation panel); the audio itself is not converted")
        self.mod_btn.clicked.connect(lambda: self._choose("analyze"))
        bottom.addWidget(self.mod_btn)
        bottom.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("secondary")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        bottom.addWidget(cancel)
        lay.addLayout(bottom)
        self.setMinimumWidth(metrics.cap_to_screen(560, parent))

    def _choose(self, key: str) -> None:
        self.choice = key
        self.accept()

    @property
    def mode(self) -> str | None:
        return self.choice
