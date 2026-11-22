"""M5: effect-parameter automation, effect chains, lane envelopes."""

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, Qt

from stimstudio.core.model import Clip, GeneratorSource, Project, Track
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_master
from stimstudio.effects import apply_effects, automated_keys, get_effect
from stimstudio.io.project_file import load_project, save_project


def const(n=80, amp=0.8, freq=0.5):
    p = np.zeros((n, 4), dtype=np.float32)
    p[:, :2], p[:, 2:] = amp, freq
    return p


def ramp_env(t1=2.0, v0=0.0, v1=1.0, **kw):
    return {"points": [[0.0, v0], [t1, v1]], "interp": "linear", "enabled": True, **kw}


# ── core ────────────────────────────────────────────────────────────────────
def test_automated_parameter_follows_the_envelope():
    eff = get_effect("scale_amplitude").make()
    eff["automation"] = {"value": ramp_env(2.0, 0.0, 0.25)}     # 0..0.25 of 0..4 → 0..1
    out = apply_effects(const(), [eff])
    t = np.arange(80) / FS
    np.testing.assert_allclose(out[:, 0], 0.8 * np.minimum(t / 2.0, 1.0), atol=1e-5)
    np.testing.assert_array_equal(out[:, 2:], const()[:, 2:])


def test_disabled_or_empty_automation_uses_the_static_value():
    eff = get_effect("scale_amplitude").make(value=0.5)
    eff["automation"] = {"value": ramp_env(enabled=False)}
    assert automated_keys(eff) == []
    np.testing.assert_allclose(apply_effects(const(), [eff])[:, 0], 0.4, atol=1e-6)
    eff["automation"] = {"value": {"points": [], "enabled": True}}
    np.testing.assert_allclose(apply_effects(const(), [eff])[:, 0], 0.4, atol=1e-6)


def test_non_automatable_key_is_ignored():
    eff = get_effect("phase_shift").make()
    eff["automation"] = {"us": ramp_env()}
    assert automated_keys(eff) == []


@pytest.mark.parametrize("type_id", ["tremolo", "vibrato", "gate", "freq_range", "slew_limit",
                                     "random_amplitude_noise", "random_frequency_noise",
                                     "amplitude_feel", "frequency_feel", "frequency_adjust",
                                     "scale_frequency"])
def test_every_automatable_effect_accepts_an_envelope(type_id):
    d = get_effect(type_id)
    eff = d.make()
    eff["automation"] = {k: ramp_env() for k in d.automatable}
    p = np.random.default_rng(3).random((120, 4)).astype(np.float32)
    out = apply_effects(p, [eff])
    assert out.shape == p.shape and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_noise_amount_automation_scales_noise():
    eff = get_effect("random_amplitude_noise").make(amount=0.3)
    eff["automation"] = {"amount": {"points": [[0.0, 0.0]], "enabled": True}}
    np.testing.assert_array_equal(apply_effects(const(), [eff]), const())


def test_clip_effect_automation_uses_timeline_time():
    eff = get_effect("scale_amplitude").make()
    eff["automation"] = {"value": {"points": [[0.0, 0.0], [10.0, 0.0], [10.01, 0.25]],
                                   "enabled": True}}
    src = GeneratorSource("waveform", {"amp_speed": 0, "amp_min": 0.8, "amp_max": 0.8})
    p = Project(tracks=[Track(clips=[Clip(src, start_s=10.0, length_s=1.0, effects=[eff])])])
    m = render_master(p)
    assert m[401:440, 0].min() == pytest.approx(0.8, abs=1e-4)   # after t = 10.01 s: value 1


def test_automation_survives_save(tmp_path):
    eff = get_effect("tremolo").make()
    eff["automation"] = {"depth": ramp_env()}
    p = Project(tracks=[Track(effects=[eff])])
    save_project(p, tmp_path / "a.stimproj")
    assert load_project(tmp_path / "a.stimproj").tracks[0].effects[0]["automation"] == \
        eff["automation"]


# ── effect chain editor ─────────────────────────────────────────────────────
@pytest.fixture
def fx(window):
    window.editor.set_current_track(window.doc.project.tracks[0].id)
    return window.effects


def test_add_edit_reorder_remove(window, fx):
    doc, chain = window.doc, fx.track
    track = doc.project.tracks[0]
    chain.add("tremolo")
    chain.add("scale_amplitude")
    assert [e["type"] for e in track.effects] == ["tremolo", "scale_amplitude"]
    assert len(chain.cards) == 2
    chain.move(1, -1)
    assert [e["type"] for e in track.effects] == ["scale_amplitude", "tremolo"]
    card = chain.cards[1]
    n = doc.undo.count()
    card._param_changed("depth", 0.2)
    card._param_changed("depth", 0.3)
    assert track.effects[1]["depth"] == 0.3 and doc.undo.count() == n + 1   # merged
    card.enabled.setChecked(False)
    assert track.effects[1]["enabled"] is False
    card.ch_b.setChecked(False)
    assert track.effects[1]["channels"] == ["A"]
    chain.remove(0)
    assert [e["type"] for e in track.effects] == ["tremolo"]
    doc.undo.undo()
    assert len(track.effects) == 2


def test_effects_change_the_render(window, fx):
    doc = window.doc
    assert doc.wait_for_render()
    before = doc.last_render.master.copy()
    fx.master.add("scale_amplitude")             # after the demo's slew limiter
    fx.master.cards[-1]._param_changed("value", 0.5)
    assert doc.wait_for_render()
    np.testing.assert_allclose(doc.last_render.master[:, 0],
                               np.minimum(before[:, 0], 1.0) * 0.5, atol=2e-3)


def test_clip_tab_follows_selection(window, fx):
    clip = window.doc.project.tracks[1].clips[0]
    assert fx.clip.effects() is None and not fx.clip.add_btn.isEnabled()
    window.editor.select_clips([clip.id])
    fx.clip.add("vibrato")
    assert clip.effects[0]["type"] == "vibrato"
    assert not fx.clip.cards[0].auto_buttons      # automation lanes exist only for tracks


def test_automation_toggle_creates_a_lane(window, fx):
    doc, ed = window.doc, window.editor
    track = doc.project.tracks[0]
    fx.track.add("tremolo")
    card = fx.track.cards[0]
    card.auto_buttons["depth"].setChecked(True)
    env = track.effects[0]["automation"]["depth"]
    assert env["points"] == [[0.0, 0.5]]                    # flat at the current depth
    assert track.id in ed.expanded
    row = ed.layout.row(track.id)
    assert [a.key for a in row.auto_lanes] == ["depth"] and row.n_lanes == 5
    assert (track.id, ("fx", 0, "depth")) in ed.env_items
    assert not card.form._widgets["depth"][1].isEnabled()  # driven by the envelope now
    header = ed.headers.headers[track.id]
    assert len(header.auto_frames) == 1
    ed.remove_fx_automation(track.id, 0, "depth")
    assert "depth" not in track.effects[0].get("automation", {})
    assert ed.layout.row(track.id).n_lanes == 4


# ── envelope gestures ───────────────────────────────────────────────────────
class FakeEvent:
    def __init__(self, x, y, button=Qt.MouseButton.LeftButton,
                 mods=Qt.KeyboardModifier.AltModifier):
        self._pos, self._button, self._mods = QPointF(x, y), button, mods

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def modifiers(self):
        return self._mods

    def accept(self):
        pass


def _lane_env(window, key="amp_l"):
    ed, track = window.editor, window.doc.project.tracks[0]
    if track.id not in ed.expanded:
        ed.toggle_expanded(track.id)
    ed.toggle_lane_envelope(track.id, key, True)
    return ed, track, ed.env_items[(track.id, ("lane", key))]


def test_lane_envelope_create_toggle_mode(window):
    ed, track, item = _lane_env(window)
    env = track.lanes["amp_l"].envelope
    assert env.points == [(0.0, 1.0)] and env.mode == "multiply" and env.enabled
    header = ed.headers.headers[track.id]
    assert header.env_buttons[0].isChecked() and header.env_modes[0].isVisible()
    ed.set_lane_envelope_mode(track.id, "amp_l", "replace")
    assert track.lanes["amp_l"].envelope.mode == "replace"
    ed.toggle_lane_envelope(track.id, "amp_l", False)
    assert track.lanes["amp_l"].envelope.enabled is False
    ed.clear_lane_envelope(track.id, "amp_l")
    assert track.lanes["amp_l"].envelope is None
    assert (track.id, ("lane", "amp_l")) not in ed.env_items


def test_envelope_click_drag_delete(window):
    from stimstudio.gui.song.envelope import PAD
    from stimstudio.gui.song.layout import LANE_H
    ed, track, item = _lane_env(window)
    pps = ed.pps
    y_of = lambda v: PAD + (LANE_H - 2 * PAD) * (1 - v)  # noqa: E731
    n = window.doc.undo.count()
    item.mousePressEvent(FakeEvent(10.0 * pps, y_of(0.0)))      # add a point at 10 s, 0.0
    item.mouseReleaseEvent(FakeEvent(10.0 * pps, y_of(0.0)))
    env = track.lanes["amp_l"].envelope
    assert len(env.points) == 2 and env.points[1][0] == pytest.approx(10.0)
    assert env.points[1][1] == pytest.approx(0.0, abs=0.01)
    assert window.doc.undo.count() == n + 1
    item.mousePressEvent(FakeEvent(10.0 * pps, y_of(0.0)))      # drag it to 12 s, 0.5
    item.mouseMoveEvent(FakeEvent(12.0 * pps, y_of(0.5)))
    item.mouseReleaseEvent(FakeEvent(12.0 * pps, y_of(0.5)))
    env = track.lanes["amp_l"].envelope
    assert env.points[1][0] == pytest.approx(12.0) and env.points[1][1] == pytest.approx(0.5, abs=0.01)
    assert window.doc.wait_for_render()
    from stimstudio.core.render import render_track
    m = window.doc.last_render.tracks[track.id].pulses
    plain = track.clone()
    plain.lanes["amp_l"].envelope = None
    base = render_track(plain, len(m)).pulses
    i = round(15 * FS)                       # after the last point: multiplied by 0.5
    assert m[i, 0] == pytest.approx(base[i, 0] * 0.5, abs=0.01)
    np.testing.assert_array_equal(m[:, 1], base[:, 1])
    item.mousePressEvent(FakeEvent(12.0 * pps, y_of(0.5), Qt.MouseButton.RightButton))
    assert len(track.lanes["amp_l"].envelope.points) == 1
    window.doc.undo.undo()
    assert len(track.lanes["amp_l"].envelope.points) == 2


def test_envelope_points_never_cross(window):
    from stimstudio.gui.song.envelope import PAD
    ed, track, item = _lane_env(window)
    pps = ed.pps
    item.mousePressEvent(FakeEvent(10.0 * pps, PAD))
    item.mouseReleaseEvent(FakeEvent(10.0 * pps, PAD))
    item.mousePressEvent(FakeEvent(10.0 * pps, PAD))
    item.mouseMoveEvent(FakeEvent(-50.0, PAD))                   # before the first point
    item.mouseReleaseEvent(FakeEvent(-50.0, PAD))
    ts = [t for t, _ in track.lanes["amp_l"].envelope.points]
    assert ts == sorted(ts) and ts[0] == 0.0


def test_fx_automation_lane_edit_changes_render(window, fx):
    from stimstudio.gui.song.envelope import PAD
    from stimstudio.gui.song.layout import LANE_H
    doc, ed = window.doc, window.editor
    track = doc.project.tracks[0]
    fx.track.add("scale_amplitude")
    fx.track.cards[0].auto_buttons["value"].setChecked(True)   # flat at 1.0 → unit 0.25
    item = ed.env_items[(track.id, ("fx", 0, "value"))]
    y0 = PAD + (LANE_H - 2 * PAD) * (1 - 0.0)
    item.mousePressEvent(FakeEvent(5.0 * ed.pps, y0))           # value 0 from 5 s on
    item.mouseReleaseEvent(FakeEvent(5.0 * ed.pps, y0))
    assert doc.wait_for_render()
    pulses = doc.last_render.tracks[track.id].pulses
    assert pulses[round(8 * FS):round(18 * FS), :2].max() < 1e-6
    assert pulses[round(1 * FS):round(3 * FS), 0].max() > 0.1
