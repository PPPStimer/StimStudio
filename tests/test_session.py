"""Session builder: determinism, phases, arc, marker alignment, safety, undo."""

import numpy as np
import pytest

from stimstudio.core import session as ss
from stimstudio.core.model import Project
from stimstudio.core.render import render_project
from stimstudio.core.safety import HIGH, check


def plan(**kw):
    kw.setdefault("duration_s", 600.0)
    markers = kw.pop("markers", ())
    return ss.build_session(ss.SessionSpec(**kw), 120.0, markers)


def test_deterministic_per_seed():
    a, b, c = plan(seed=3), plan(seed=3), plan(seed=4)
    key = lambda p: [(x.type_id, x.start_s, x.length_s, x.seed, tuple(sorted(x.params.items())))
                     for x in p.clips]
    assert key(a) == key(b) and a.arc == b.arc
    assert key(a) != key(c)


def test_phases_and_coverage():
    p = plan(cycles=4)
    names = [ph.name for ph in p.phases]
    assert names == ["Warm-up", "Build 1", "Build 2", "Build 3", "Build 4", "Peak", "Cool-down"]
    assert p.phases[0].t0 == 0.0 and p.phases[-1].t1 == pytest.approx(600.0)
    assert all(a.t1 == b.t0 for a, b in zip(p.phases, p.phases[1:]))
    assert p.clips[0].start_s == 0.0
    assert max(c.start_s + c.length_s for c in p.clips) == pytest.approx(600.0)
    for a, b in zip(p.clips, p.clips[1:]):             # crossfades overlap, lanes alternate
        assert b.start_s == pytest.approx(a.start_s + a.length_s - 4.0, abs=1e-6)
        assert a.lane != b.lane
    assert all(20.0 - 4.1 <= c.length_s for c in p.clips[:-1])


def test_arc_shape():
    p = plan(peak=0.8)
    t = np.linspace(0, 600, 1201)
    arc = p.arc_at(t)
    assert arc[0] == 0.0 and arc[-1] == 0.0 and arc.max() <= 0.8 + 1e-6
    peak_ph = next(ph for ph in p.phases if ph.kind == "peak")
    in_peak = arc[(t > peak_ph.t0 + 12) & (t < peak_ph.t1)]
    assert in_peak.min() > 0.7 * 0.8
    crests = [arc[(t >= ph.t0) & (t < ph.t1)].max() for ph in p.phases if ph.kind == "build"]
    assert crests == sorted(crests)                    # waves grow towards the peak


def test_tease_dips_before_crests():
    p = plan(style="tease", cycles=2)
    t = np.linspace(0, 600, 6001)
    arc = p.arc_at(t)
    ph = next(x for x in p.phases if x.kind == "build")
    seg = arc[(t >= ph.t0) & (t < ph.t1)]
    top = int(np.argmax(seg))
    assert seg[max(0, top - 300):top].min() < 0.6 * seg[top]


def test_align_to_markers():
    base = plan(cycles=2)
    target = base.phases[2].t0 + 7.0                   # near Build 2's start
    aligned = plan(cycles=2, align_to_markers=True, markers=[target])
    assert aligned.phases[2].t0 == pytest.approx(target)
    far = plan(cycles=2, align_to_markers=True, markers=[base.phases[2].t0 + 200])
    assert far.phases[2].t0 == pytest.approx(base.phases[2].t0)


@pytest.mark.parametrize("style", ss.STYLES)
@pytest.mark.parametrize("tri", [False, True])
def test_tracks_render_without_serious_issues(style, tri):
    p = plan(style=style, triphase=tri, seed=1, duration_s=360.0)
    tracks = ss.plan_tracks(p, offset_s=5.0)
    assert len(tracks) == 2 and {t.type for t in tracks} == {"triphase" if tri else "standard"}
    assert all(t.effects[0]["type"] == "slew_limit" for t in tracks)
    assert tracks[0].lanes["amp_l"].envelope.points[0] == (5.0, 0.0)
    assert min(c.start_s for t in tracks for c in t.clips) == 5.0
    proj = Project(tracks=tracks)
    render = render_project(proj)
    assert render.n == pytest.approx(365 * 40, abs=2)
    assert check(render, proj).count(HIGH) == 0


def test_bad_spec():
    with pytest.raises(ValueError):
        plan(style="nope")
    with pytest.raises(ValueError):
        plan(duration_s=10)


def test_gui_builds_in_one_undo(window, qapp):
    doc = window.doc
    n_tracks, n_markers = len(doc.project.tracks), len(doc.project.markers)
    p = plan(duration_s=240.0)
    tracks = window.build_session(p, start_s=2.0)
    assert len(doc.project.tracks) == n_tracks + 2
    assert [m.name for m in doc.project.markers if m.name in ("Warm-up", "Peak")] == \
        ["Warm-up", "Peak"]
    assert doc.project.find_clip(tracks[0].clips[0].id)[1].start_s == 2.0
    doc.undo.undo()
    assert len(doc.project.tracks) == n_tracks and len(doc.project.markers) == n_markers


def test_dialog_preview(qapp):
    from stimstudio.gui.session_dialog import SessionDialog
    dlg = SessionDialog(None, 120.0, 300.0, 0.0, markers=[50.0])
    assert dlg.align.isEnabled() and dlg.align.isChecked()
    first = [c.type_id for c in dlg.plan.clips]
    dlg.seed.setValue(dlg.seed.value() + 1)
    assert [c.type_id for c in dlg.plan.clips] != first or len(first) < 3
    dlg.triphase.setChecked(True)
    assert all(c.type_id.startswith("tri_") for c in dlg.plan.clips)
    dlg.minutes.setValue(2.0)
    assert dlg.plan.spec.duration_s == 120.0
    assert "clips" in dlg.summary.text()
    dlg.deleteLater()
