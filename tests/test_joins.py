"""Clip overlaps (crossfade / mix / max), smoothed butt joints, glue, clip-edge magnets."""

import json

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QMenu

from stimstudio.core.joins import (Layer, butt_joints, composite_gains, layout_layers,
                                   smooth_joints)
from stimstudio.core.model import Clip, GeneratorSource, Project, Track
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_project, render_track
from stimstudio.core.triphase import FREQ, VOL, X


def flat(start, length, amp=0.8, freq=0.4, **kw):
    src = GeneratorSource("waveform", {"amp_speed": 0.0, "amp_min": amp, "amp_max": amp,
                                       "freq_speed": 0.0, "freq_min": freq, "freq_max": freq})
    return Clip(src, start_s=start, length_s=length, **kw)


def tri_point(start, length, x, vol=0.8):
    return Clip(GeneratorSource("tri_point", {"x": x, "y": 0.5, "volume": vol}),
                start_s=start, length_s=length)


# ── core layout ─────────────────────────────────────────────────────────────
def test_layout_partial_and_nested():
    a, b, c = layout_layers([(0, 200), (120, 200), (40, 20)], join_n=10)
    assert [L.order for L in (a, b, c)] == [0, 2, 1]
    assert (b.ramp_in, b.ramp_out) == (80, 0)          # crossfade over the whole overlap
    assert (c.ramp_in, c.ramp_out) == (10, 10)         # nested: join ramps both ways
    assert a.covered_out and not a.covered_in
    assert b.covered_in and not b.covered_out
    assert c.covered_in and c.covered_out


def test_gains_always_sum_to_one_where_clips_overlap():
    spans = [(0, 400), (100, 200), (150, 300), (380, 40)]
    layers = layout_layers(spans, join_n=12)
    total = np.zeros(500)
    for L, g in zip(layers, composite_gains(layers)):
        total[L.start:L.end] += g
    covered = np.zeros(500, bool)
    for s, n in spans:
        covered[s:s + n] = True
    np.testing.assert_allclose(total[covered], 1.0, atol=1e-9)


def test_short_nested_clip_shares_its_length():
    _a, b = layout_layers([(0, 100), (40, 6)], join_n=10)
    assert b.ramp_in + b.ramp_out == 6


def test_butt_joints():
    assert butt_joints([(0, 40), (40, 40), (100, 20), (121, 10)]) == [40, 121]
    assert butt_joints([(0, 40), (30, 40)]) == []          # overlap: not a joint
    assert butt_joints([(0, 40), (50, 40)]) == []          # gap: not a joint


# ── render ──────────────────────────────────────────────────────────────────
def test_crossfade_has_no_bump_and_glides_the_frequency():
    t = Track(clips=[flat(0, 4, freq=0.2), flat(3, 4, freq=0.8)])      # 1 s overlap
    out = render_track(t, 7 * FS).pulses
    np.testing.assert_allclose(out[:, 0], 0.8, atol=1e-5)               # level never doubles
    xf = out[3 * FS:4 * FS, 2]
    assert xf[0] == pytest.approx(0.2, abs=0.02) and xf[-1] == pytest.approx(0.8, abs=0.02)
    assert np.all(np.diff(xf) >= -1e-6)                                 # smooth glide up


def test_mix_and_max_modes():
    clips = [flat(0, 4, amp=0.5), flat(2, 4, amp=0.3)]
    mix = render_track(Track(overlap="mix", clips=[c.clone() for c in clips]), 6 * FS).pulses
    mx = render_track(Track(overlap="max", clips=[c.clone() for c in clips]), 6 * FS).pulses
    assert mix[3 * FS, 0] == pytest.approx(0.8)
    assert mx[3 * FS, 0] == pytest.approx(0.5)


def test_nested_clip_replaces_the_outer_one():
    t = Track(join_s=0.5, clips=[flat(0, 10, amp=0.4), flat(4, 2, amp=1.0)])
    out = render_track(t, 10 * FS).pulses[:, 0]
    assert out[2 * FS] == pytest.approx(0.4) and out[5 * FS] == pytest.approx(1.0)
    assert out[8 * FS] == pytest.approx(0.4)
    ramp = out[4 * FS:4 * FS + 20]
    assert 0.4 < ramp[10] < 1.0 and np.all(np.diff(ramp) > 0)          # 0.5 s ramp up


def test_manual_fade_on_a_crossfaded_edge_is_ignored():
    t = Track(clips=[flat(0, 4), flat(3, 4, fade_in_s=1.0)])
    out = render_track(t, 7 * FS).pulses[:, 0]
    np.testing.assert_allclose(out, 0.8, atol=1e-5)                    # no dip
    free = render_track(Track(clips=[flat(0, 4, fade_in_s=1.0)]), 4 * FS).pulses[:, 0]
    assert free[0] == 0.0                                               # free edges still fade


def test_muted_clips_do_not_cut_holes():
    t = Track(clips=[flat(0, 6, amp=0.6), flat(2, 2, amp=1.0, muted=True)])
    np.testing.assert_allclose(render_track(t, 6 * FS).pulses[:, 0], 0.6, atol=1e-6)


def test_smooth_joins_round_off_a_butt_joint():
    clips = [flat(0, 4, amp=0.2, freq=0.1), flat(4, 4, amp=0.9, freq=0.7)]
    hard = render_track(Track(clips=[c.clone() for c in clips]), 8 * FS).pulses
    soft = render_track(Track(clips=[c.clone() for c in clips], smooth_joins=True,
                              join_s=1.0), 8 * FS).pulses
    j = 4 * FS
    assert hard[j - 1, 0] == pytest.approx(0.2) and hard[j, 0] == pytest.approx(0.9)
    win = soft[j - 20:j + 20]
    for col in (0, 2):
        assert np.all(np.diff(win[:, col]) >= -1e-6)
        assert np.max(np.abs(np.diff(win[:, col]))) < 0.1                # no step left
    np.testing.assert_allclose(soft[:j - 20], hard[:j - 20])            # untouched elsewhere
    np.testing.assert_allclose(soft[j + 20:], hard[j + 20:])


def test_smooth_joints_holds_neighbours():
    arr = np.zeros((40, 4), np.float32)
    arr[20:] = 1.0
    smooth_joints(arr, [20], 5, range(4))
    assert arr[14, 0] == 0.0 and arr[25, 0] == 1.0 and 0.0 < arr[20, 0] < 1.0


def test_triphase_crossfade_moves_the_position():
    t = Track(type="triphase", clips=[tri_point(0, 4, -0.8), tri_point(3, 4, 0.8)])
    tri = render_track(t, 7 * FS).triphase
    np.testing.assert_allclose(tri[:, VOL], 0.8, atol=1e-4)
    xs = tri[3 * FS:4 * FS, X]
    assert xs[0] < 0.2 and xs[-1] > 0.8 and np.all(np.diff(xs) >= -1e-6)
    assert np.isfinite(tri[:, FREQ]).all()


def test_old_projects_load_as_mix(tmp_path):
    data = Project(tracks=[Track(clips=[flat(0, 2)])]).to_dict()
    for key in ("overlap", "join_s", "smooth_joins"):
        del data["tracks"][0][key]
    p = Project.from_dict(json.loads(json.dumps(data)))
    assert (p.tracks[0].overlap, p.tracks[0].smooth_joins) == ("mix", False)
    assert Track().overlap == "crossfade"
    q = Project.from_dict(Project(tracks=[Track(overlap="max", join_s=1.0,
                                                smooth_joins=True)]).to_dict())
    assert (q.tracks[0].overlap, q.tracks[0].join_s, q.tracks[0].smooth_joins) == \
        ("max", 1.0, True)


# ── GUI: glue, magnets, header menu, crossfade display ──────────────────────
def fresh_track(window, clips, **attrs):
    doc = window.doc
    track = doc.add_track()
    for k, v in attrs.items():
        setattr(track, k, v)
    for c in clips:
        doc.add_clip(track.id, c)
    window.editor.refresh()
    return track


def test_glue_sounds_the_same_and_undoes(window):
    doc = window.doc
    track = fresh_track(window, [flat(100, 4, freq=0.2, fade_in_s=0.5),
                                 flat(103, 4, freq=0.7), flat(108, 2, amp=0.5)],
                        smooth_joins=True)
    before = render_project(doc.project).tracks[track.id].pulses.copy()
    ids = [c.id for c in track.clips]
    (glued,) = doc.glue_clips(ids)
    assert len(track.clips) == 1 and glued.start_s == 100 and glued.length_s == 10
    assert glued.source.kind == "pulses" and glued.source.origin.startswith("glued")
    after = render_project(doc.project).tracks[track.id].pulses
    np.testing.assert_allclose(after, before, atol=1e-5)
    doc.undo.undo()
    assert sorted(c.id for c in track.clips) == sorted(ids)
    assert doc.glue_clips(ids[:1]) == []                       # one clip: nothing to glue


def test_clip_edges_are_magnets(window):
    ed = window.editor
    track = fresh_track(window, [flat(200, 4)])
    near = 204 + 4 / ed.pps                                     # 4 px past the clip end
    assert ed.snap_time(near) == pytest.approx(204)
    far = 204 + 30 / ed.pps
    assert ed.snap_time(far) != pytest.approx(204)
    item = ed.clip_items[track.clips[0].id]
    ed.begin_drag(item, "move", QPointF(item.x(), item.y()))
    assert 204 not in ed.magnet_times()                         # its own edges do not pull
    ed.end_drag(0)


def test_header_menu_sets_overlap_and_joins(window):
    ed = window.editor
    track = fresh_track(window, [flat(0, 2)])
    header = ed.headers.headers[track.id]
    menu = QMenu()
    header._join_menus(menu)
    subs = {a.text(): a.menu() for a in menu.actions() if a.menu()}
    {a.text(): a for a in subs["Clip overlaps"].actions()}["Max (louder wins)"].trigger()
    assert track.overlap == "max"
    joins = {a.text(): a for a in subs["Smooth joins"].actions()}
    joins["Smooth butt joints"].trigger()
    joins["1 s"].trigger()
    assert track.smooth_joins and track.join_s == 1.0
    window.doc.undo.undo()
    assert track.join_s == 0.25


def test_crossfaded_edges_lose_their_fade_handles(window):
    ed = window.editor
    track = fresh_track(window, [flat(300, 4), flat(303, 4)])
    a, b = (ed.clip_items[c.id] for c in track.clips)
    assert isinstance(b.layer, Layer) and b.layer.ramp_in == FS and b.zValue() > a.zValue()
    assert b.covered_in and a.covered_out
    assert b.zone_at(QPointF(0, 2)) != "fade_in"
    assert a.zone_at(QPointF(a.width, 2)) != "fade_out"
    window.doc.set_track_attrs(track.id, overlap="mix")
    assert b.layer is None and b.zone_at(QPointF(0, 2)) == "fade_in"
