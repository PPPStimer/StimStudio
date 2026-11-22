import numpy as np
import pytest

from stimstudio.core.automation import evaluate
from stimstudio.core.model import (Clip, Envelope, GeneratorSource, Lane, Project,
                                   PulseDataSource, Track)
from stimstudio.core.render import (RenderCache, render_clip, render_master,
                                    render_project, render_source, render_track)
from stimstudio.demo import build_demo_project
from stimstudio.effects import get_effect
from stimstudio.generators import get_generator

FLAT = GeneratorSource("waveform", {"amp_speed": 0.0, "amp_min": 0.8, "amp_max": 0.8,
                                    "freq_speed": 0.0, "freq_min": 0.4, "freq_max": 0.4})


def flat_clip(start, length, **kw):
    return Clip(FLAT, start_s=start, length_s=length, **kw)


def test_clip_matches_generator_and_trim_is_a_window():
    src = GeneratorSource("licks", {}, seed=3)
    full = get_generator("licks").render(200, {}, 3).clamp().to_array()
    clip = Clip(src, length_s=2.0, trim_s=1.5)
    np.testing.assert_array_equal(render_clip(clip), full[60:140])


def test_clip_fades_and_gain():
    out = render_clip(flat_clip(0, 4, fade_in_s=1.0, fade_out_s=1.0, gain=0.5))
    assert out[0, 0] == 0.0 and out[-1, 0] == 0.0
    assert out[80, 0] == pytest.approx(0.4)
    assert out[80, 2] == pytest.approx(0.4)        # frequency untouched


def test_source_cache():
    cache = RenderCache()
    src = GeneratorSource("chaos", {}, seed=1)
    a = render_source(src, 100, 0, cache)
    b = render_source(src, 100, 0, cache)
    assert a is b and cache.hits == 1
    assert not a.flags.writeable
    render_source(src, 100, 10, cache)               # different trim -> new entry
    assert len(cache) == 2


def test_cached_clip_render_is_not_corrupted_by_effects():
    cache = RenderCache()
    clip = flat_clip(0, 2, effects=[get_effect("scale_amplitude").make(value=0.5)])
    first = render_clip(clip, cache)
    second = render_clip(clip, cache)
    np.testing.assert_array_equal(first, second)


def test_pulse_data_padding_and_loop():
    data = np.tile(np.array([[1.0, 1.0, 0.3, 0.6]], np.float32), (40, 1))
    padded = render_clip(Clip(PulseDataSource(data), length_s=2.0))
    assert padded[39, 0] == 1.0 and padded[40, 0] == 0.0
    assert padded[79, 2] == pytest.approx(0.3)        # frequency held past the end
    looped = render_clip(Clip(PulseDataSource(data, loop=True), length_s=2.0))
    assert looped[79, 0] == 1.0


def test_track_is_silent_outside_clips_even_with_effects():
    track = Track(clips=[flat_clip(1, 1)],
                  effects=[get_effect("random_amplitude_noise").make(amount=0.5)])
    tr = render_track(track, 120)
    assert tr.active[40:80].all() and not tr.active[:40].any()
    assert np.all(tr.pulses[:40, :2] == 0) and np.all(tr.pulses[80:, :2] == 0)


def test_overlapping_clips_add():
    track = Track(overlap="mix", clips=[flat_clip(0, 2, gain=0.5), flat_clip(1, 2, gain=0.5)])
    tr = render_track(track, 120)
    assert tr.pulses[20, 0] == pytest.approx(0.4)
    assert tr.pulses[60, 0] == pytest.approx(0.8)


def test_lane_envelope_and_mute():
    track = Track(clips=[flat_clip(0, 2)])
    track.lanes["amp_l"] = Lane(envelope=Envelope(points=[(0.0, 0.0), (2.0, 1.0)]))
    track.lanes["amp_r"] = Lane(muted=True)
    tr = render_track(track, 80)
    assert tr.pulses[40, 0] == pytest.approx(0.4, abs=0.01)
    assert np.all(tr.pulses[:, 1] == 0)


def wave_clip(start, length, amp, freq):
    return Clip(GeneratorSource("waveform", {"amp_speed": 0.0, "amp_min": amp, "amp_max": amp,
                                             "freq_speed": 0.0, "freq_min": freq,
                                             "freq_max": freq}),
                start_s=start, length_s=length)


def test_amplitude_from_one_track_frequency_from_another():
    amp_src = Track(name="amp", clips=[wave_clip(0, 2, 0.8, 0.2)])
    freq_src = Track(name="freq", clips=[wave_clip(0, 2, 0.5, 0.9)])
    other = Track(name="other", clips=[wave_clip(0, 2, 0.4, 0.3)])
    for key in ("freq_l", "freq_r"):
        amp_src.lanes[key] = Lane(muted=True)
    for key in ("amp_l", "amp_r"):
        freq_src.lanes[key] = Lane(muted=True)
    m = render_master(Project(tracks=[amp_src, freq_src]))
    assert m[10, 0] == pytest.approx(0.8) and m[10, 1] == pytest.approx(0.8)
    assert m[10, 2] == pytest.approx(0.9, abs=1e-4) and m[10, 3] == pytest.approx(0.9, abs=1e-4)
    # a muted amplitude still weighs its frequency by the level it would have had
    m = render_master(Project(tracks=[freq_src, other]))
    assert m[10, 0] == pytest.approx(0.4)
    assert m[10, 2] == pytest.approx((0.5 * 0.9 + 0.4 * 0.3) / 0.9, abs=1e-4)


def test_lane_mute_beats_modulation_and_old_projects_mute_frequency_too():
    track = Track(clips=[flat_clip(0, 2)])
    track.lanes["amp_l"] = Lane(muted=True, envelope=Envelope(points=[(0.0, 1.0)], mode="replace"))
    assert np.all(render_track(track, 80).pulses[:, 0] == 0)
    d = Project(tracks=[track]).to_dict()
    assert Project.from_dict(d).tracks[0].lanes["freq_l"].muted is False
    d["version"] = 2
    lanes = Project.from_dict(d).tracks[0].lanes
    assert lanes["freq_l"].muted and not lanes["freq_r"].muted


@pytest.mark.parametrize("interp", ["linear", "hold", "smooth"])
def test_envelope_interpolation_endpoints(interp):
    env = Envelope(points=[(1.0, 0.2), (3.0, 0.8)], interp=interp)
    v = evaluate(env, np.array([0.0, 1.0, 2.0, 3.0, 5.0]))
    assert v[0] == pytest.approx(0.2) and v[-1] == pytest.approx(0.8)
    assert v[3] == pytest.approx(0.8)
    expected_mid = {"linear": 0.5, "hold": 0.2, "smooth": 0.5}[interp]
    assert v[2] == pytest.approx(expected_mid)


def test_mute_solo_and_gain():
    a = Track(name="a", clips=[flat_clip(0, 1)], gain=0.5)
    b = Track(name="b", clips=[flat_clip(0, 1)])
    p = Project(tracks=[a, b])
    assert render_master(p)[0, 0] == pytest.approx(1.0)   # 0.4 + 0.8 clamped
    b.muted = True
    assert render_master(p)[0, 0] == pytest.approx(0.4)
    b.muted = False
    b.solo = True
    assert render_master(p)[0, 0] == pytest.approx(0.8)


def test_master_effects_and_length():
    p = Project(tracks=[Track(clips=[flat_clip(0, 2)])],
                master_effects=[get_effect("scale_amplitude").make(value=0.5)])
    assert render_master(p).shape == (80, 4)
    assert render_master(p)[10, 0] == pytest.approx(0.4)
    p.length_s = 3.0
    assert render_master(p).shape == (120, 4)


def test_demo_project_is_deterministic():
    a = render_project(build_demo_project(), RenderCache()).master
    b = render_project(build_demo_project()).master
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2000, 4) and a[:, :2].max() > 0.5
