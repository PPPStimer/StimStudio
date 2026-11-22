"""Transport and output integration (dry-run sink: no sound card, no network)."""

import time

import numpy as np
import pytest

from stimstudio.core.pulse import FS


def spin(qapp, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)


def wait_until(qapp, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def pb(window):
    window.device_panel.soft_start.setValue(0.0)
    window.device_panel.set_power(200, 200)
    return window.playback


def test_play_from_cursor_and_panic(window, pb, qapp):
    assert pb.output_key == "dryrun"
    window.doc.set_cursor(10.0)
    assert pb.play()
    assert wait_until(qapp, lambda: window.editor.playhead_s >= 10.0)
    assert pb.position_s() >= 10.0 and window.a_play.text() == "&Pause"
    assert wait_until(qapp, lambda: len(pb.sink.frames) >= 2)
    pulse = pb.sink.frames[-1][0]
    assert 0.0 <= pulse.amp_a <= 1.0 and 10.0 <= pulse.freq_a_hz <= 100.0
    window.a_panic.trigger()
    assert not pb.playing and window.editor.playhead_s == -1.0
    assert window.a_play.text() == "&Play"


def test_frames_follow_the_master_render(window, pb, qapp):
    master = window.doc.last_render.master
    window.doc.set_cursor(30.0)
    pb.play()
    assert wait_until(qapp, lambda: len(pb.sink.frames) >= 3)
    pb.panic()
    frames = pb.sink.frames
    assert all(p.amp_a == 0.0 for p in frames[-1])     # panic sends one silent frame
    # every other amplitude is a value taken from the master (gain is 1, power full)
    amps = {round(float(p.amp_a), 5) for frame in frames[:-1] for p in frame}
    window_vals = {round(float(v), 5) for v in master[30 * FS:40 * FS, 0]}
    assert amps <= window_vals


def test_space_toggles_and_play_needs_content(window, pb, qapp):
    window.a_play.trigger()
    assert pb.playing
    window.a_play.trigger()           # graceful stop (0.3 s fade)
    assert wait_until(qapp, lambda: not pb.playing, 2.0)
    window.doc.set_project(__import__("stimstudio.gui.document", fromlist=["x"]).new_project())
    assert window.doc.wait_for_render()
    assert not pb.play()


def test_cursor_click_seeks_running_output(window, pb, qapp):
    pb.play()
    window.doc.set_cursor(40.0)
    assert wait_until(qapp, lambda: pb.position_s() >= 40.0)
    pb.panic()


def test_loop_region_window(window, pb, qapp):
    window._on_loop_region_edited((5.0, 7.0))
    assert window.a_loop.isChecked() and pb.loop_enabled
    assert window.editor.loop_region == (5.0, 7.0)
    w = pb.source.window
    assert w.loop and w.n == 2 * FS and w.offset == 5 * FS
    np.testing.assert_array_equal(w.pulses, window.doc.last_render.master[200:280])
    window.doc.set_cursor(0.0)        # outside the region: starts at its beginning
    pb.play()
    spin(qapp, 0.3)
    assert 5.0 <= pb.position_s() < 7.0
    pb.panic()
    window.editor.loopRegionEdited.emit(None)
    assert pb.loop_region is None and pb.source.window.n == window.doc.last_render.n


def test_loop_selection(window, pb):
    clip = window.doc.project.tracks[1].clips[0]
    window.editor.select_clips([clip.id])
    window.loop_selection()
    assert pb.loop_region == (clip.start_s, clip.end_s)


def test_edits_reach_the_running_output(window, pb, qapp):
    pb.play()
    before = pb.source.window.pulses.copy()
    clip = window.doc.project.tracks[0].clips[0]
    window.doc.set_clip_attrs(clip.id, gain=0.25)
    assert window.doc.wait_for_render()
    qapp.processEvents()
    after = pb.source.window.pulses
    assert pb.playing and after[100, 0] < before[100, 0]
    pb.panic()


def test_power_link_and_limits(window, pb):
    panel = window.device_panel
    panel.limits["a"].setValue(80)
    panel.set_power(150, 30)
    c = pb.source.controls
    assert c.power_a == 80 and c.power_b == 30          # the slider cannot pass the limit
    panel.link.setChecked(True)
    panel.spins["b"].setValue(60)
    assert pb.source.controls.power_a == 60
    panel.mute.setChecked(True)
    assert pb.source.controls.muted


def test_device_switch_stops_output(window, pb, qapp):
    pb.play()
    panel = window.device_panel
    panel.device.setCurrentIndex(panel.device.findData("howl"))
    assert not pb.playing and pb.output_key == "howl"
    assert not pb.play()                                # Howl needs an explicit Connect
    panel.remote_host.setText("192.0.2.7")
    panel.remote_host.editingFinished.emit()
    assert panel.settings.value("output/howl_host") == "192.0.2.7"


def test_export_audio(window, tmp_path, monkeypatch):
    import soundfile as sf
    from PyQt6.QtWidgets import QFileDialog
    out = str(tmp_path / "out.wav")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    window.export_audio()
    info = sf.info(out)
    assert info.samplerate == 48000 and info.channels == 2
    assert abs(info.frames / 48000 - window.doc.last_render.n / FS) < 0.01
    data, _ = sf.read(out)
    assert np.abs(data).max() > 0.1


def test_close_silences_output(window, pb, qapp):
    pb.play()
    sink = pb.sink
    window.doc.undo.setClean()
    window.close()
    assert not sink.playing and pb.sink is None
