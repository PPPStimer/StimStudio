"""M7: triphase model, generators, rendering, phase-true audio and the GUI."""

import math

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from stimstudio.core.model import Clip, Envelope, GeneratorSource, Lane, ModRoute, Project, Track
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_clip, render_project, render_track
from stimstudio.core.triphase import (Calibration, currents, electrode_levels, mix_tri,
                                      output_phasors, phasors, to_ab, tri_to_pulses, unit_to_xy,
                                      xy_to_unit)
from stimstudio.effects import apply_effects, get_effect
from stimstudio.generators import get_generator, list_generators
from stimstudio.outputs.audio import render_offline

DIRS = {k: (math.cos(math.radians(a)), math.sin(math.radians(a)))
        for k, a in {"N": -90, "L": 150, "R": 30}.items()}


# ── model ───────────────────────────────────────────────────────────────────
def test_centre_is_balanced_three_phase():
    lv = electrode_levels(1.0, 0.0, 0.0)
    assert lv["N"] == pytest.approx(lv["L"]) == pytest.approx(lv["R"])
    a, b = phasors(*currents(0.0, 0.0))
    assert abs(a) == pytest.approx(abs(b))
    assert math.degrees(np.angle(b) - np.angle(a)) == pytest.approx(60.0)


@pytest.mark.parametrize("k", ["N", "L", "R"])
def test_moving_towards_an_electrode_concentrates_current(k):
    lv = electrode_levels(1.0, *DIRS[k])
    others = [v for e, v in lv.items() if e != k]
    assert lv[k] == pytest.approx(2 * others[0]) and others[0] == pytest.approx(others[1])


def test_special_positions_on_pulse_outputs():
    a, b = to_ab(1.0, *DIRS["L"])
    assert b == pytest.approx(0.0, abs=1e-9) and a > 0.5     # R at N potential
    a, b = to_ab(1.0, *DIRS["N"])
    assert a == pytest.approx(b)                             # L and R together vs N
    a1, b1 = to_ab(1.0, 0.3, 0.2)
    a2, b2 = to_ab(1.0, -0.3, 0.2)                           # mirror image swaps L/R
    assert a1 == pytest.approx(b2) and b1 == pytest.approx(a2)


def test_physics_holds_everywhere():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)
    for cal in (Calibration(), Calibration(reach=1.0), Calibration(gain_l=1.6, center_x=0.2)):
        mn, ml, mr = currents(x, y, cal)
        a, b = phasors(mn, ml, mr)
        np.testing.assert_allclose(np.abs(2 * a - b), ml, atol=1e-9)
        np.testing.assert_allclose(np.abs(2 * b - a), mr, atol=1e-9)
        np.testing.assert_allclose(np.abs(a + b), mn, atol=1e-9)
        oa, ob = output_phasors(1.0, x, y, cal)
        assert np.abs(oa).max() <= 1 + 1e-9 and np.abs(ob).max() <= 1 + 1e-9


def test_calibration_gain_and_centre():
    base = electrode_levels(1.0, 0.0, 0.0)
    boosted = electrode_levels(1.0, 0.0, 0.0, Calibration(gain_l=1.4))
    assert boosted["L"] / boosted["R"] > base["L"] / base["R"] * 1.3
    moved = Calibration(center_x=0.3, center_y=-0.2)
    lv = electrode_levels(1.0, 0.3, -0.2, moved)
    assert lv["N"] == pytest.approx(lv["L"]) == pytest.approx(lv["R"])
    assert Calibration.from_dict(moved.to_dict()) == moved


def test_tri_to_pulses_and_mix():
    tri = np.array([[1.0, 0.5, 0.5, 0.3], [0.0, 0.5, 0.5, 0.3]], np.float32)
    p = tri_to_pulses(tri)
    assert p[0, 0] == pytest.approx(p[0, 1]) and p[1, :2].max() == 0
    assert p[0, 2] == p[0, 3] == pytest.approx(0.3)
    a = np.array([[0.6, 1.0, 0.5, 0.2]], np.float32)
    b = np.array([[0.2, 0.0, 0.5, 0.8]], np.float32)
    m = mix_tri([(a, 1.0), (b, 1.0)])
    assert m[0, 0] == pytest.approx(0.8)
    assert m[0, 1] == pytest.approx(0.75, abs=1e-4)                # volume-weighted x
    assert mix_tri([]) is None


# ── generators ──────────────────────────────────────────────────────────────
TRI_GENS = [g for g in list_generators() if g.domain == "triphase"]


@pytest.mark.parametrize("gen", TRI_GENS, ids=lambda g: g.type_id)
def test_triphase_generators_stay_in_the_disk(gen):
    out = gen().render(400, {}, 3).to_array()
    x, y = unit_to_xy(out[:, 1]), unit_to_xy(out[:, 2])
    assert np.hypot(x, y).max() <= 1 + 1e-6 and out.min() >= 0 and out.max() <= 1


def test_circle_and_electrode_hop():
    c = get_generator("tri_circle").render(160, {"radius": 0.6}, 0).to_array()
    np.testing.assert_allclose(np.hypot(unit_to_xy(c[:, 1]), unit_to_xy(c[:, 2])), 0.6, atol=1e-3)
    h = get_generator("tri_electrodes").render(
        3 * FS, {"step": 1.0, "glide": 0.0, "sequence": "L → R → N"}, 0).to_array()
    for second, k in enumerate("LRN"):
        row = h[second * FS + 10]
        assert unit_to_xy(row[1]) == pytest.approx(DIRS[k][0], abs=1e-3)
        assert unit_to_xy(row[2]) == pytest.approx(DIRS[k][1], abs=1e-3)


# ── rendering ───────────────────────────────────────────────────────────────
def tri_track(**kw):
    return Track(name="tri", type="triphase", clips=[
        Clip(GeneratorSource("tri_circle", {"radius": 0.8, "volume": 0.9}), length_s=4.0, **kw)])


def test_triphase_track_render_and_master_split():
    std = Track(clips=[Clip(GeneratorSource("waveform", {"amp_speed": 0, "amp_min": 0.3,
                                                         "amp_max": 0.3}), length_s=4.0)])
    tri = tri_track()
    p = Project(tracks=[std, tri])
    r = render_project(p)
    tr = r.tracks[tri.id]
    np.testing.assert_allclose(tr.pulses, tri_to_pulses(tr.triphase), atol=1e-6)
    np.testing.assert_allclose(r.master_audio[:, 0], 0.3, atol=1e-6)    # standard only
    assert r.master[:, 0].max() > 0.3                                     # + triphase as A/B
    np.testing.assert_array_equal(r.triphase, tr.triphase)


def test_triphase_fades_and_gain_touch_volume_only():
    clip = tri_track(fade_in_s=1.0, gain=0.5).clips[0]
    out = render_clip(clip, domain="triphase")
    plain = render_clip(tri_track().clips[0], domain="triphase")
    np.testing.assert_array_equal(out[:, 1:], plain[:, 1:])
    assert out[0, 0] == 0.0 and out[-1, 0] == pytest.approx(0.45)


def test_triphase_lanes_effects_and_mute():
    t = tri_track()
    t.lanes["amp_r"] = Lane(muted=True, envelope=Envelope(points=[(0.0, 0.5)], mode="replace"))
    t.lanes["freq_l"] = Lane(envelope=Envelope(points=[(0.0, 0.0)], mode="replace"))
    t.effects = [get_effect("scale_amplitude").make(value=0.5)]
    tr = render_track(t, 160)
    np.testing.assert_allclose(tr.triphase[:, 1], 0.5, atol=1e-6)   # X replaced, mute ignored
    np.testing.assert_allclose(tr.triphase[:, 2], 0.0, atol=1e-6)   # Y → bottom (N)
    np.testing.assert_allclose(tr.triphase[:, 0], 0.45, atol=1e-6)  # volume halved by FX
    np.testing.assert_allclose(tr.pulses[:, 0], tr.pulses[:, 1], atol=1e-6)  # at N: A = B


def test_triphase_frequency_mute():
    std = Track(clips=[Clip(GeneratorSource("waveform", {
        "amp_speed": 0, "amp_min": 0.3, "amp_max": 0.3,
        "freq_speed": 0, "freq_min": 0.7, "freq_max": 0.7}), length_s=4.0)])
    std.lanes["amp_l"] = Lane(muted=True)
    std.lanes["amp_r"] = Lane(muted=True)
    tri = tri_track()
    tri.lanes["freq_r"] = Lane(muted=True)
    r = render_project(Project(tracks=[std, tri]))
    np.testing.assert_allclose(r.master[:, 2:], 0.7, atol=1e-4)     # pulse freq from std
    assert r.master[:, 0].max() > 0                                  # volume from tri
    tri.lanes["amp_l"] = Lane(muted=True)
    tri.lanes["freq_r"] = Lane()
    r = render_project(Project(tracks=[tri]))
    assert r.master[:, :2].max() == 0 and r.triphase[:, 0].max() == 0
    np.testing.assert_allclose(r.triphase[:, 3], r.tracks[tri.id].triphase[:, 3], atol=1e-6)


def test_mod_route_moves_triphase_position(tmp_path):
    import soundfile as sf
    from stimstudio.core.model import AudioFileSource
    sr = 22050
    y = np.sin(2 * np.pi * 220 * np.arange(sr * 4) / sr).astype(np.float32) * 0.5
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr)
    tri = tri_track()
    mod = Track(type="audio_mod", clips=[Clip(AudioFileSource(str(path)), length_s=4.0)])
    route = ModRoute(mod.id, "envelope", tri.id, "lane:amp_r", mode="replace", depth=1.0)
    r = render_project(Project(tracks=[tri, mod], mod_routes=[route]))
    env = r.signals[mod.id]["envelope"]
    np.testing.assert_allclose(r.tracks[tri.id].triphase[:, 1], np.clip(env, 0, 1), atol=1e-5)


def test_triphase_pan_effect():
    p = np.zeros((40, 4), np.float32)
    p[:, :2], p[:, 2:] = 0.8, 0.5
    centre = apply_effects(p, [get_effect("triphase_pan").make(radius=0.0)])
    assert centre[0, 0] == pytest.approx(centre[0, 1])
    to_l = apply_effects(p, [get_effect("triphase_pan").make(angle=150.0, radius=1.0)])
    assert to_l[0, 1] == pytest.approx(0.0, abs=1e-6) and to_l[0, 0] > 0.4
    np.testing.assert_array_equal(to_l[:, 2:], p[:, 2:])


# ── audio with phase ────────────────────────────────────────────────────────
def _corr(sig):
    l, r = sig[:, 0].astype(np.float64), sig[:, 1].astype(np.float64)
    return float(np.mean(l * r) / (np.std(l) * np.std(r)))


@pytest.mark.parametrize("pos,expected", [((0.0, 0.0), 0.5), (DIRS["N"], 1.0)])
def test_continuous_audio_plays_real_phase(pos, expected):
    n = 2 * FS
    tri = np.zeros((n, 4), np.float32)
    tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 3] = 1.0, xy_to_unit(pos[0]), xy_to_unit(pos[1]), 0.5
    silent = np.zeros((n, 4), np.float32)
    sig = render_offline(silent, "continuous", 48000, triphase=tri)
    assert _corr(sig[4800:]) == pytest.approx(expected, abs=0.02)   # cos(phase difference)
    a, b = output_phasors(1.0, pos[0], pos[1])
    assert np.abs(sig[4800:, 0]).max() == pytest.approx(abs(a), abs=0.01)


def test_wavelet_ignores_phase_data():
    tri = np.full((40, 4), 0.5, np.float32)
    pulses = np.zeros((40, 4), np.float32)
    sig = render_offline(pulses, "wavelet", 48000, triphase=tri)
    assert np.all(sig == 0)


# ── GUI ─────────────────────────────────────────────────────────────────────
def test_triphase_track_in_the_editor(window):
    doc, ed = window.doc, window.editor
    track = window.add_triphase_track()
    assert track.type == "triphase" and ed.layout.row(track.id).takes_clips
    window.browser.select("waveform")
    y = ed.layout.row(track.id).y + 10
    ed.empty_double_clicked(QPointF(5.0 * ed.pps, y))
    clip = track.clips[0]
    assert clip.source.type_id == "tri_circle"                # pulse generator replaced
    assert window.inspector.gen_combo.findData("waveform") < 0
    assert window.inspector.gen_combo.findData("tri_spiral") >= 0
    std = doc.project.tracks[0]
    window.browser.select("tri_spiral")
    n = len(std.clips)
    ed.empty_double_clicked(QPointF(80.0 * ed.pps, ed.layout.row(std.id).y + 10))
    assert len(std.clips) == n                                  # triphase gen rejected here
    ed.toggle_expanded(track.id)
    header = ed.headers.headers[track.id]
    assert header.lane_frames[1].findChild(type(header.badge)).text() == "Position X"
    assert doc.wait_for_render()
    ed.refresh()
    lanes = [it for (tid, _), it in ed.lane_items.items() if tid == track.id]
    assert len(lanes) == 4 and all(it.values is not None for it in lanes)


def test_triphase_pattern_page_and_point_drag(window):
    doc, ed = window.doc, window.editor
    track = window.add_triphase_track()
    clip = Clip(GeneratorSource("tri_point", {"x": 0.0, "y": 0.0}), start_s=2.0, length_s=4.0)
    doc.add_clip(track.id, clip)
    ed.select_clips([clip.id])
    page = window.pattern.triphase_page
    assert window.pattern.stack.currentWidget() is page and page.pad.editable
    page._point_dropped(0.4, -0.3)
    assert clip.source.params["x"] == 0.4 and clip.source.params["y"] == -0.3
    doc.undo.undo()
    assert clip.source.params["x"] == 0.0
    circle = Clip(GeneratorSource("tri_circle", {"radius": 0.5}), start_s=10.0, length_s=4.0)
    doc.add_clip(track.id, circle)
    ed.select_clips([circle.id])
    assert not page.pad.editable and page.pad.trajectory is not None
    doc.set_cursor(11.0)
    x, y = page.pad.point
    assert math.hypot(x, y) == pytest.approx(0.5, abs=0.02)


def test_calibration_dialog(window):
    doc = window.doc
    dlg = window.show_triphase_calibration()
    dlg.spins["gain_l"].setValue(1.5)
    assert doc.project.triphase_cal["gain_l"] == 1.5
    dlg.spins["gain_l"].setValue(1.6)
    doc.undo.undo()
    assert doc.project.triphase_cal.get("gain_l", 1.0) == 1.0    # merged into one step
    dlg.spins["reach"].setValue(0.8)
    dlg.reset()
    assert doc.project.triphase_cal == {}
    dlg.close()


def test_phase_capable_sinks_get_triphase_data(window):
    doc, pb = window.doc, window.playback
    track = window.add_triphase_track()
    doc.add_clip(track.id, Clip(GeneratorSource("tri_circle", {}), length_s=10.0))
    assert doc.wait_for_render()
    pb._publish_window()
    assert pb.source.window.triphase is None                 # dry run: A/B mix
    np.testing.assert_array_equal(pb.source.window.pulses, doc.last_render.master)
    pb.sink.supports_triphase = True                          # pretend continuous audio
    pb._publish_window()
    w = pb.source.window
    assert w.triphase is not None
    np.testing.assert_array_equal(w.pulses, doc.last_render.master_audio)
