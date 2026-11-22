"""Live recording: take resampling, punch-in, take muting, live monitoring."""

import queue

import numpy as np
import pytest

from stimstudio.core import record as rec
from stimstudio.core.model import Clip, Project, PulseDataSource, Track
from stimstudio.core.pulse import FS
from stimstudio.outputs import Controls, LiveInput, PulseSource
from stimstudio.outputs.dryrun import DryRunSink

from test_outputs import audio_sink, render


def test_pad_row_channels():
    assert rec.pad_row(0.7, 0.3, "AB") == (0.7, 0.7, 0.3, 0.3)
    assert rec.pad_row(0.7, 0.3, "A") == (0.7, 0.0, 0.3, 0.3)
    assert rec.pad_row(0.7, 0.3, "B") == (0.0, 0.7, 0.3, 0.3)


def test_samples_hold_until_the_next():
    r = rec.TakeRecorder(start_s=1.0, smooth_ms=0)
    r.add(0.5, (1, 1, 1, 1))                      # pre-roll: dropped
    r.add(1.0, (0.2, 0.2, 0.5, 0.5))
    r.add(1.5, (0.8, 0.8, 0.5, 0.5))
    r.add(2.0, (0.0, 0.0, 0.5, 0.5))
    start, data = r.result()
    assert start == 1.0 and len(data) == FS + 1
    assert np.allclose(data[:FS // 2, 0], 0.2)
    assert np.allclose(data[FS // 2:FS, 0], 0.8)
    assert data[-1, 0] == 0.0


def test_jitter_holes_and_long_gaps():
    r = rec.TakeRecorder(smooth_ms=0)
    for k in range(0, 40):
        if k in (10, 11, 12):                     # dropped timer ticks
            continue
        r.add(k / FS + 0.003, (k / 40, 0, 0.5, 0.5))
    _s, data = r.result()
    assert len(data) == 40
    assert data[11, 0] == pytest.approx(9 / 40)   # held across the hole
    r2 = rec.TakeRecorder(smooth_ms=0)
    r2.add(0.0, (0.5, 0.5, 0.5, 0.5))
    r2.add(2.0, (0.9, 0.9, 0.5, 0.5))             # 2 s gap is not bridged by holding
    _s, d2 = r2.result()
    assert d2[0, 0] == 0.5 and d2[FS, 0] == 0.5   # filled by the hole bridge, not a hold
    assert d2[-1, 0] == 0.9


def test_loop_pass_overwrites_and_punch_range():
    r = rec.TakeRecorder(start_s=1.0, end_s=2.0, smooth_ms=0)
    for t in np.arange(0.5, 2.5, 1 / FS):          # first pass, partly outside
        r.add(t, (0.3, 0.3, 0.5, 0.5))
    for t in np.arange(1.0, 1.5, 1 / FS):          # the loop wrapped: second pass
        r.add(t, (0.9, 0.9, 0.5, 0.5))
    start, data = r.result()
    assert start == pytest.approx(1.0) and len(data) == FS
    assert np.allclose(data[:FS // 2 - 1, 0], 0.9)
    assert np.allclose(data[FS // 2 + 1:, 0], 0.3)


def test_smoothing():
    r = rec.TakeRecorder(smooth_ms=200)
    for k in range(80):
        r.add(k / FS, (1.0 if k >= 40 else 0.0, 0, 0.5, 0.5))
    _s, data = r.result()
    step = data[37:44, 0]
    assert 0.0 < step[0] < step[-1] < 1.0 + 1e-6 and np.all(np.diff(step) >= 0)


def test_takes_helpers():
    t = Track(clips=[Clip(PulseDataSource(np.zeros((40, 4), np.float32), origin="take 3"), 1, 1),
                     Clip(PulseDataSource(np.zeros((40, 4), np.float32), origin="x.hwl"), 1, 1),
                     Clip(PulseDataSource(np.zeros((40, 4), np.float32), origin="take 1"), 5, 1)])
    assert [c.source.origin for c in rec.overlapping_takes(t, 0.5, 2.0)] == ["take 3"]
    assert rec.next_take_number(Project(tracks=[t])) == 4
    assert rec.next_take_number(Project()) == 1


def test_live_input_replaces_the_window_in_clocked_sinks():
    src = PulseSource()
    src.set_window(np.zeros((400, 4), np.float32), loop=True)
    src.set_controls(power_a=100, power_b=100)
    sink = DryRunSink(src, queue.Queue())
    frame = sink._collect(src.window, src.controls, 1.0)
    assert frame[0].amp_a == 0.0
    src.set_live(LiveInput(0.7, 0.2, 0.5, 0.5))
    frame = sink._collect(src.window, src.controls, 1.0)
    assert len(frame) == sink.pulse_batch_size
    assert frame[0].amp_a == pytest.approx(0.7) and frame[0].amp_b == pytest.approx(0.2)
    muted = sink._collect(src.window, Controls(power_a=100, muted=True), 1.0)
    assert muted[0].amp_a == 0.0                 # mute still wins
    src.set_live(None)
    assert sink._collect(src.window, src.controls, 1.0)[0].amp_a == 0.0


def test_live_input_in_the_audio_callback():
    from stimstudio.outputs.audio import ContinuousSink
    src = PulseSource()
    src.set_window(np.zeros((400, 4), np.float32), loop=True)
    src.set_controls(power_a=200, power_b=200, soft_start=0.0)
    sink = audio_sink(ContinuousSink, src)
    sink.start()
    assert np.abs(render(sink, 2)).max() == 0.0
    src.set_live(LiveInput(0.8, 0.0, 0.5, 0.5))
    out = render(sink, 4)
    assert np.abs(out[:, 0]).max() == pytest.approx(0.8, abs=0.05)
    assert np.abs(out[:, 1]).max() < 1e-6
    sink.stop(panic=True)


def test_howl_cannot_monitor():
    from stimstudio.outputs.howl_remote import HowlRemoteSink
    assert HowlRemoteSink.supports_live is False and DryRunSink.supports_live is True


# ── GUI ─────────────────────────────────────────────────────────────────────
def _std_track(window):
    tr = next(t for t in window.doc.project.tracks if t.type == "standard")
    window.editor.set_current_track(tr.id)
    return tr


def test_record_a_take(window, qapp):
    panel, pb, doc = window.record, window.playback, window.doc
    track = _std_track(window)
    panel.refresh()
    assert panel.stack.currentWidget() is panel.pad
    panel.preroll.setCurrentIndex(0)
    doc.set_cursor(4.0)
    n_clips = len(track.clips)
    window.a_record.trigger()
    assert panel.recording and pb.playing and window.a_record.isChecked()
    # perform: three samples driven by the timer, values from the pad
    panel.pad.set_value(amp=0.6, freq=0.25)
    for _ in range(3):
        panel._tick()
    assert pb.source.live is not None and pb.source.live.amp_a == pytest.approx(0.6)
    # fake a longer take straight into the recorder
    for k in range(40):
        panel.recorder.add(4.0 + k / FS, (0.6, 0.6, 0.25, 0.25))
    window.a_record.trigger()
    assert not panel.recording and pb.source.live is None
    assert not window.a_record.isChecked()
    assert len(track.clips) == n_clips + 1
    take = next(c for c in track.clips if c.name == "Take 1")
    assert take.start_s == pytest.approx(4.0, abs=0.05)
    assert take.source.origin == "take 1"
    assert np.allclose(take.source.data[5:30, 0], 0.6, atol=1e-3)
    assert doc.cursor_s == pytest.approx(4.0, abs=0.05)
    # a second take over it mutes the first
    window.a_record.trigger()
    for k in range(40):
        panel.recorder.add(4.2 + k / FS, (0.3, 0.3, 0.25, 0.25))
    clip_id = panel.stop()
    assert take.muted and doc.project.find_clip(clip_id)[1].name == "Take 2"
    doc.undo.undo()
    assert not take.muted and doc.project.find_clip(clip_id) is None


def test_panic_ends_recording_and_monitor(window, qapp):
    panel, pb = window.record, window.playback
    _std_track(window)
    panel.preroll.setCurrentIndex(0)
    assert panel.start()
    panel.pad.set_value(amp=0.9)
    panel._tick()
    assert pb.source.live is not None
    window.a_panic.trigger()
    assert not panel.recording and pb.source.live is None and not pb.playing


def test_record_needs_target_and_punch_region(window, qapp):
    panel, pb = window.record, window.playback
    vt = window.doc.project.video_track()
    mod = next((t for t in window.doc.project.tracks if t.type == "audio_mod"), None)
    other = vt or mod
    if other is not None:
        window.editor.set_current_track(other.id)
        panel.refresh()
        assert not panel.rec_btn.isEnabled()
    _std_track(window)
    panel.punch.setChecked(True)
    pb.set_loop_region(None)
    assert not panel.start() and "loop region" in panel.status.text()
    panel.punch.setChecked(False)


def test_triphase_target(window, qapp):
    doc, panel = window.doc, window.record
    window.add_triphase_track()
    panel.refresh()
    assert panel.stack.currentWidget() is panel.tri_pad
    panel.tri_pad.point = (0.0, -0.8)                  # towards N (bottom)
    panel.tri_pad.set_volume(0.7)
    vol, x, y, f = panel.current_row()
    assert vol == pytest.approx(0.7) and x == pytest.approx(0.5) and y < 0.2
    live = panel.live_input()
    assert 0.0 <= live.amp_a <= 1.0 and 0.0 <= live.amp_b <= 1.0
    del doc
