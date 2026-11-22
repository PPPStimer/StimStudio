import numpy as np
import pytest

from stimstudio.core.model import (BeatMarksSource, Clip, GeneratorSource, Mark, PatternSource,
                                   PoolEntry, Project, PulseDataSource, Track)
from stimstudio.core.pattern import (key_name, next_key, parse_sequence, placements, resolve,
                                     sequence_errors)
from stimstudio.core.render import RenderCache, render_clip, render_track, source_key
from stimstudio.io.project_file import load_project, save_project


def block(amp, n, freq=0.5):
    """Pulse clip of *n* pulses at amplitude *amp*."""
    data = np.zeros((n, 4), np.float32)
    data[:, :2] = amp
    data[:, 2:] = freq
    return Clip(PulseDataSource(data), length_s=n / 40)


def pattern(times, pool=None, **kw):
    pool = pool if pool is not None else [PoolEntry("A", block(0.8, 8))]
    return PatternSource([Mark(t) for t in times], pool, **kw)


def pclip(src, length_s=4.0, **kw):
    return Clip(src, length_s=length_s, **kw)


def test_key_names():
    assert [key_name(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]
    pool = [PoolEntry("A", block(1, 1)), PoolEntry("C", block(1, 1))]
    assert next_key(pool) == "B"


def test_sequence_parsing():
    assert parse_sequence(" a, b  - C ") == ["A", "B", "-", "C"]
    pool = [PoolEntry("A", block(1, 1))]
    assert sequence_errors("A B -", pool) == ["B"]


def test_marks_source_is_silent_and_not_mixed():
    clip = pclip(BeatMarksSource([Mark(0.0), Mark(1.0)]))
    assert not render_clip(clip).any()
    t = Track(clips=[clip])
    assert not render_track(t, 160).active.any()


def test_hits_land_on_marks_with_cut_fit():
    out = render_clip(pclip(pattern([0.0, 1.0, 2.0]), 3.0))
    amp = out[:, 0]
    for s in (0, 40, 80):
        assert amp[s] == pytest.approx(0.8)
    assert amp[20] == 0 and amp[60] == 0      # 8-pulse hits, silence between


def test_cut_fit_truncates_at_next_mark():
    long = [PoolEntry("A", block(0.8, 80))]
    out = render_clip(pclip(pattern([0.0, 0.5], long, join_s=0.1), 3.0))
    # first hit stops at the second mark; second runs its full 80 pulses
    assert out[19, 0] < 0.8 and out[20, 0] == pytest.approx(0.8)
    assert out[99, 0] == pytest.approx(0.8)


def test_own_fit_overlaps_by_overlap_mode():
    long = [PoolEntry("A", block(0.4, 80))]
    mix = render_clip(pclip(pattern([0.0, 0.5], long, fit="own", overlap="mix"), 3.0))
    mx = render_clip(pclip(pattern([0.0, 0.5], long, fit="own", overlap="max"), 3.0))
    assert mix[40, 0] == pytest.approx(0.8)
    assert mx[40, 0] == pytest.approx(0.4)


def test_loop_and_stretch_fill_the_gap():
    short = [PoolEntry("A", block(0.6, 4))]
    loop = render_clip(pclip(pattern([0.0, 1.0], short, fit="loop"), 2.0))
    assert (loop[:80, 0] == pytest.approx(0.6))
    ramp = np.zeros((10, 4), np.float32)
    ramp[:, 0] = np.linspace(0, 1, 10)
    stretched = render_clip(pclip(pattern([0.0], [PoolEntry("A", Clip(PulseDataSource(ramp),
                                                                      length_s=0.25))],
                                          fit="stretch"), 1.0))
    assert stretched[0, 0] == pytest.approx(0.0) and stretched[39, 0] == pytest.approx(1.0)
    assert stretched[20, 0] == pytest.approx(20 / 39, abs=0.02)


def test_random_pick_is_seeded_and_weighted():
    pool = [PoolEntry("A", block(0.2, 2)), PoolEntry("B", block(0.4, 2)),
            PoolEntry("C", block(0.6, 2), weight=0.0)]
    pat = pattern(np.arange(60) * 0.1, pool, seed=5)
    keys = [e.key for _m, e in resolve(pat)]
    assert keys == [e.key for _m, e in resolve(pat.clone())]
    assert "C" not in keys and {"A", "B"} <= set(keys)
    pat.seed = 6
    assert [e.key for _m, e in resolve(pat)] != keys


def test_pins_do_not_shift_other_picks():
    pool = [PoolEntry("A", block(0.2, 2)), PoolEntry("B", block(0.4, 2))]
    pat = pattern(np.arange(20) * 0.1, pool, seed=3)
    before = [e.key for _m, e in resolve(pat)]
    pat.marks[4].pin = "B" if before[4] == "A" else "A"
    after = [e.key for _m, e in resolve(pat)]
    assert after[4] != before[4]
    assert after[:4] + after[5:] == before[:4] + before[5:]


def test_sequence_mode_with_rests_and_unknown_keys():
    pool = [PoolEntry("A", block(0.2, 2)), PoolEntry("B", block(0.4, 2))]
    pat = pattern([0, 1, 2, 3, 4], pool, mode="sequence", sequence="A B - X")
    picked = [e.key if e else None for _m, e in resolve(pat)]
    assert picked == ["A", "B", None, None, "A"]
    pat.sequence = ""
    assert all(e is None for _m, e in resolve(pat))


def test_velocity_scales_by_depth():
    pat = pattern([0.0])
    pat.marks[0].velocity = 0.5
    assert render_clip(pclip(pat, 1.0))[0, 0] == pytest.approx(0.4)
    pat.velocity_depth = 0.5
    assert render_clip(pclip(pat, 1.0))[0, 0] == pytest.approx(0.6)


def test_trim_is_a_window_and_clip_end_cuts():
    pat = pattern([0.0, 1.0, 2.0])
    full = render_clip(pclip(pat, 3.0))
    part = render_clip(pclip(pat, 1.0, trim_s=1.0))
    np.testing.assert_array_equal(part, full[40:80])
    assert len(placements(pat, 60)) == 2


def test_generator_pool_entry_and_cache_key():
    gen = Clip(GeneratorSource("waveform", {"amp_speed": 0.0, "amp_min": 0.5, "amp_max": 0.5}),
               length_s=0.2)
    pat = pattern([0.0, 0.5], [PoolEntry("A", gen)])
    cache = RenderCache()
    a = render_clip(pclip(pat, 1.0), cache)
    assert a[0, 0] == pytest.approx(0.5)
    k1 = source_key(pat, 0, 40)
    pat.marks[1].t_s = 0.6
    assert source_key(pat, 0, 40) != k1
    k2 = source_key(pat, 0, 40)
    pat.pool[0].clip.gain = 0.5
    assert source_key(pat, 0, 40) != k2
    assert source_key(pat, 0, 40, domain="triphase") != source_key(pat, 0, 40)
    # placements are never cached, only real sources and the pattern itself
    assert not any(k.startswith("pulses:") for k in cache._items)


def test_triphase_pattern_scales_volume_only():
    data = np.zeros((10, 4), np.float32)
    data[:, 0], data[:, 1], data[:, 2], data[:, 3] = 0.8, 0.3, 0.7, 0.5
    pat = pattern([0.0], [PoolEntry("A", Clip(PulseDataSource(data), length_s=0.25))])
    pat.marks[0].velocity = 0.5
    out = render_clip(pclip(pat, 1.0), domain="triphase")
    np.testing.assert_allclose(out[0], [0.4, 0.3, 0.7, 0.5], atol=1e-6)


def test_patterns_mix_on_a_track():
    t = Track(clips=[pclip(pattern([0.0, 1.0]), 2.0, start_s=1.0)])
    r = render_track(t, 160)
    assert r.pulses[40, 0] == pytest.approx(0.8) and r.pulses[0, 0] == 0
    assert r.active[40]


def test_round_trip(tmp_path):
    pat = pattern([0.0, 0.5], [PoolEntry("A", block(0.8, 8)), PoolEntry("B", block(0.3, 4), 2.0)],
                  mode="sequence", sequence="A B", seed=9, fit="loop", overlap="mix",
                  join_s=0.1, velocity_depth=0.3, anchor="center")
    pat.marks[1].pin, pat.marks[1].velocity = "B", 0.25
    p = Project(tracks=[Track(clips=[pclip(pat, 2.0),
                                     pclip(BeatMarksSource([Mark(0.25, 0.5)]), 1.0, start_s=3)])])
    path = tmp_path / "p.stimproj"
    save_project(p, path)
    q = load_project(path)
    assert q.to_dict() == p.to_dict()
    src = q.tracks[0].clips[0].source
    assert isinstance(src, PatternSource) and src.marks[1].pin == "B"
    assert isinstance(q.tracks[0].clips[1].source, BeatMarksSource)


def ramp_entry(n):
    data = np.zeros((n, 4), np.float32)
    data[:, 0] = np.arange(1, n + 1) / n          # 1/n … 1: the clip's own position
    return [PoolEntry("A", Clip(PulseDataSource(data), length_s=n / 40))]


def amp(pat, length_s):
    return render_clip(pclip(pat, length_s))[:, 0]


@pytest.mark.parametrize("anchor, first", [("start", 40), ("center", 36), ("end", 32)])
def test_anchor_places_clip_point_on_mark(anchor, first):
    a = amp(pattern([1.0], ramp_entry(8), fit="own", anchor=anchor), 2.0)
    hit = np.flatnonzero(a)
    assert hit[0] == first and len(hit) == 8
    assert a[first] == pytest.approx(1 / 8) and a[first + 7] == pytest.approx(1.0)


def test_end_anchor_before_block_start_is_cut_off():
    a = amp(pattern([0.1], ramp_entry(8), fit="own", anchor="end"), 1.0)
    np.testing.assert_allclose(a[:4], np.arange(5, 9) / 8)    # only the clip's tail fits


def test_cut_with_end_anchor_trims_the_head_at_the_previous_mark():
    long = [PoolEntry("A", block(0.8, 80))]
    a = amp(pattern([1.0, 2.0], long, anchor="end", join_s=0.1), 3.0)
    # zones [0, 40) and [40, 80): each clip ends on its mark, nothing overlaps
    assert a[0] == 0.0 and a[39] == pytest.approx(0.8) and a[30] == pytest.approx(0.8)
    assert a[40] == 0.0 and a[79] == pytest.approx(0.8)
    assert not a[80:].any()


def test_cut_with_center_anchor_meets_halfway():
    long = [PoolEntry("A", block(0.4, 80))]
    a = amp(pattern([1.0, 2.0], long, anchor="center", join_s=0.0, overlap="mix"), 3.0)
    assert (a[20:100] == pytest.approx(0.4))          # zones [20, 60) + [60, 100): no doubling
    assert a[19] == 0.0 and a[100] == 0.0


def test_loop_end_anchor_ends_the_clip_on_the_mark():
    a = amp(pattern([1.0, 2.0], ramp_entry(4), fit="loop", anchor="end"), 3.0)
    assert a[79] == pytest.approx(1.0) and a[76] == pytest.approx(0.25)
    assert a[39] == pytest.approx(1.0) and not a[80:].any()


def test_loop_center_anchor_puts_clip_middle_on_the_mark():
    a = amp(pattern([1.0], ramp_entry(4), fit="loop", anchor="center"), 2.0)
    assert a[40] == pytest.approx(3 / 4)          # sample round(0.5 * 4) = 2 → value 3/4


def test_stretch_center_anchor_fills_the_zone():
    a = amp(pattern([1.0, 2.0], ramp_entry(10), fit="stretch", anchor="center"), 3.0)
    assert a[19] == 0.0 and a[20] == pytest.approx(0.1) and a[59] == pytest.approx(1.0)
    assert a[60] == pytest.approx(0.1) and a[99] == pytest.approx(1.0) and a[100] == 0.0


def test_anchor_is_part_of_the_cache_key():
    pat = pattern([0.5])
    k = source_key(pat, 0, 40)
    pat.anchor = "end"
    assert source_key(pat, 0, 40) != k
