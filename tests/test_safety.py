"""Safety check: rules on synthetic renders, fixes, the GUI dock and prompt."""

from types import SimpleNamespace

import numpy as np
import pytest

from stimstudio.core import safety as sc
from stimstudio.core.model import Clip, Project, PulseDataSource, Track
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_project


def pulses(amp, freq=0.5) -> np.ndarray:
    amp = np.asarray(amp, np.float32)
    out = np.zeros((len(amp), 4), np.float32)
    out[:, 0] = out[:, 1] = amp
    out[:, 2] = out[:, 3] = freq
    return out


def project_of(*segments, gap_s=0.0) -> Project:
    """One track; *segments* are amplitude arrays placed back to back."""
    data = np.concatenate([np.zeros(int(gap_s * FS), np.float32), *segments])
    clip = Clip(PulseDataSource(pulses(data)), 0.0, len(data) / FS)
    return Project(tracks=[Track(name="T", clips=[clip])])


def kinds(project, **rules):
    rep = sc.check(render_project(project), project, sc.SafetyRules(**rules))
    return rep, [i.kind for i in rep.issues]


def sec(s, v=0.0):
    return np.full(int(s * FS), v, np.float32)


def ramp(s, a, b):
    return np.linspace(a, b, int(s * FS), dtype=np.float32)


def test_jump_after_silence_is_serious():
    rep, ks = kinds(project_of(sec(5), sec(3, 1.0), ramp(3, 1.0, 0.0)))
    assert ks.count("jump") == 1
    jump = next(i for i in rep.issues if i.kind == "jump")
    assert jump.severity == sc.HIGH and jump.t0 == pytest.approx(5.0, abs=0.05)
    assert jump.track_id is not None and rep.worst == sc.HIGH
    assert "cold_start" not in ks                       # not reported twice


def test_ramp_and_steady_rhythm_are_fine():
    beats = np.tile(np.r_[sec(0.25, 0.8), sec(0.25, 0.0)], 20)
    _rep, ks = kinds(project_of(ramp(4, 0.0, 0.8), beats, ramp(4, 0.8, 0.0)))
    assert "jump" not in ks and "cold_start" not in ks


def test_medium_jump_is_a_warning():
    rep, ks = kinds(project_of(ramp(3, 0, 0.3), sec(2, 0.3), sec(2, 0.85), ramp(3, 0.85, 0)))
    jump = next(i for i in rep.issues if i.kind == "jump")
    assert jump.severity == sc.WARN and jump.value == pytest.approx(0.55, abs=0.02)


def test_cold_start():
    _rep, ks = kinds(project_of(sec(5), ramp(1.5, 0, 0.7), sec(3, 0.7), ramp(4, 0.7, 0)))
    assert ks == ["cold_start"]
    _rep, ks = kinds(project_of(sec(5), ramp(5, 0, 0.7), ramp(4, 0.7, 0)))
    assert ks == []


def test_sustained():
    _rep, ks = kinds(project_of(ramp(4, 0, 0.9), sec(100, 0.9), ramp(4, 0.9, 0)))
    assert "sustained" in ks
    _rep, ks = kinds(project_of(ramp(4, 0, 0.9), sec(80, 0.9), ramp(4, 0.9, 0)))
    assert "sustained" not in ks


def test_low_frequency_at_high_level():
    amp = np.r_[ramp(4, 0, 0.9), sec(3, 0.9), ramp(4, 0.9, 0)]
    for freq, expect in ((0.0, True), (0.5, False)):
        clip = Clip(PulseDataSource(pulses(amp, freq)), 0.0, len(amp) / FS)
        p = Project(tracks=[Track(clips=[clip])])
        _rep, ks = kinds(p)
        assert ("low_freq" in ks) is expect


def test_position_jump_and_dose():
    n = 80 * FS
    tri = np.zeros((n, 4), np.float32)
    tri[:, 0] = 0.8
    tri[:, 1] = 0.5
    tri[:, 2] = 0.5
    tri[40 * FS:, 1] = 0.1                              # jump to the left
    render = SimpleNamespace(master=np.zeros((n, 4), np.float32), triphase=tri, tracks={})
    rep = sc.check(render)
    ks = [i.kind for i in rep.issues]
    assert "position_jump" in ks and "dose" in ks
    dose = next(i for i in rep.issues if i.kind == "dose")
    assert dose.severity == sc.INFO and dose.value == pytest.approx(1.0)
    assert len(rep.profile) == n // 10                   # 0.25 s bins


def test_ignore_and_limiter_fix():
    p = project_of(sec(5), sec(3, 1.0), ramp(3, 1.0, 0.0))
    rep = sc.check(render_project(p), p)
    jump = rep.issues[0]
    assert sc.check(render_project(p), p, ignored=[jump.key]).issues == []
    track = p.tracks[0]
    track.effects = sc.apply_limiter(track.effects, jump)
    assert track.effects[-1]["type"] == "slew_limit"
    rep2 = sc.check(render_project(p), p)
    assert all(i.severity != sc.HIGH for i in rep2.issues)
    tightened = sc.apply_limiter([{"type": "slew_limit", "rise": 4.0, "fall": 0.0,
                                   "enabled": True}],
                                 sc.Issue(0, 1, "cold_start", sc.WARN, ""))
    assert len(tightened) == 1 and tightened[0]["rise"] == 0.5


def test_rest_points():
    issue = sc.Issue(10.0, 130.0, "sustained", sc.WARN, "")
    pts = sc.rest_points(issue)
    assert len(pts) == 8 and pts == sorted(pts)
    assert min(v for _t, v in pts) == pytest.approx(0.45)
    merged = sc.merge_points([(0, 1.0), (40, 0.2), (200, 1.0)], pts)
    assert (40, 0.2) not in merged and (0, 1.0) in merged and (200, 1.0) in merged


# ── GUI ─────────────────────────────────────────────────────────────────────
def _load_jumpy(window):
    p = project_of(sec(5), sec(3, 1.0), ramp(3, 1.0, 0.0))
    window.doc.set_project(p)
    assert window.doc.wait_for_render()
    return window.safety.current()


def test_panel_lists_and_fixes(window, qapp):
    rep = _load_jumpy(window)
    panel = window.safety_panel
    assert panel.tree.topLevelItemCount() == len(rep.issues) >= 1
    assert "serious" in panel.summary.text()
    item = panel.tree.topLevelItem(0)
    panel.tree.setCurrentItem(item)
    panel.issueActivated.emit(panel.selected_issue())
    assert window.doc.cursor_s == pytest.approx(5.0, abs=0.05)
    assert panel.fix_btn.isEnabled()
    panel.fix_selected()
    track = window.doc.project.tracks[0]
    assert track.effects and track.effects[-1]["type"] == "slew_limit"
    assert window.doc.wait_for_render()
    rep2 = window.safety.recheck()
    assert rep2.count(sc.HIGH) == 0
    window.doc.undo.undo()
    assert not window.doc.project.tracks[0].effects


def test_ignore_is_stored_in_project(window):
    rep = _load_jumpy(window)
    window.safety_panel.tree.setCurrentItem(window.safety_panel.tree.topLevelItem(0))
    window.safety_panel.ignore_selected()
    assert rep.issues[0].key in window.doc.project.meta["safety_ignored"]
    assert window.doc.wait_for_render()
    assert window.safety.recheck().count(sc.HIGH) == 0
    assert "ignored" in window.safety_panel.summary.text()
    window.safety.reset_ignored()
    assert window.doc.project.meta["safety_ignored"] == []


def test_play_is_gated(window, monkeypatch):
    _load_jumpy(window)
    import stimstudio.gui.safety as gs
    answers = []
    monkeypatch.setattr(gs, "prompt_before", lambda *a, **k: answers.pop(0))
    answers.append("cancel")
    window.a_play.trigger()
    assert not window.playback.playing
    answers.append("show")
    window.a_play.trigger()
    assert not window.playback.playing and window.safety_dock.isVisible()
    answers.append("go")
    window.a_play.trigger()
    assert window.playback.playing
    window.playback.panic()


def test_prompt_dialog_adds_master_limiter(window, qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from stimstudio.gui.safety import prompt_before
    _load_jumpy(window)

    def click():
        box = QApplication.activeModalWidget()
        assert isinstance(box, QMessageBox)
        btn = next(b for b in box.buttons() if b.text() == "Add master limiter")
        btn.click()

    QTimer.singleShot(50, click)
    assert prompt_before(window, window.safety, "play") == "limit"
    assert window.doc.project.master_effects[-1]["type"] == "slew_limit"
    meta = dict(window.doc.project.meta)
    meta["safety_no_prompt"] = True
    window.doc.set_project_attrs(meta=meta)
    assert prompt_before(window, window.safety) == "go"   # no dialog at all


def test_heat_strip_and_overview(window, qapp):
    _load_jumpy(window)
    ed = window.editor
    assert ed.heat_img is not None and ed.heat_img.width() == len(ed.safety_report.profile)
    from stimstudio.gui.song.overview import heat_color
    assert heat_color(1.0).red() > 200 and heat_color(0.0).red() < 60
    ed.ruler.grab()
    ed.overview.grab()
    ed.zoom(4.0)
    ed.overview.scrollRequested.emit(8.0)
    t0, t1 = ed.visible_range()
    assert t0 <= 8.0 <= t1
    window.a_overview.trigger()
    assert not ed.overview.isVisible()
    window.a_overview.trigger()
    assert ed.overview.isVisible()
