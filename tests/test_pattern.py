"""M4: rhythm generators, activity stages, presets and the pattern editor."""

import numpy as np
import pytest
from PyQt6.QtCore import QPointF

from stimstudio.core.model import Clip, GeneratorSource, PulseDataSource
from stimstudio.core.pulse import FS
from stimstudio.core.render import (RenderCache, manual_params_at, render_clip, render_source,
                                    segment_stages)
from stimstudio.generators.rhythm import euclid_pattern
from stimstudio.io.presets import (BUNDLED_DIR, list_presets, load_preset, safe_filename,
                                   save_preset)


# ── generators ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hits,steps", [(3, 8), (5, 8), (7, 16), (0, 4), (4, 4)])
def test_euclid_pattern(hits, steps):
    p = euclid_pattern(hits, steps)
    assert len(p) == steps and p.sum() == hits
    if 0 < hits < steps:   # onsets as evenly spread as possible: gaps differ by at most 1
        idx = np.flatnonzero(p)
        gaps = np.diff(np.concatenate([idx, [idx[0] + steps]]))
        assert gaps.max() - gaps.min() <= 1


def test_euclid_follows_project_tempo():
    clip = Clip(GeneratorSource("euclid", {"hits": 1, "steps": 4, "division": "1/4",
                                           "pulse": 0.25, "shape": "square"}), length_s=4.0)
    cache = RenderCache()
    fast = render_clip(clip, cache, bpm=120)    # a hit every 2 s (4 beats of 0.5 s)
    slow = render_clip(clip, cache, bpm=60)     # every 4 s
    assert (np.diff((fast[:, 0] > 0.5).astype(int)) == 1).sum() + (fast[0, 0] > 0.5) == 2
    assert (np.diff((slow[:, 0] > 0.5).astype(int)) == 1).sum() + (slow[0, 0] > 0.5) == 1
    assert len(cache) == 2                       # the BPM is part of the cache key


def test_euclid_channel_b_modes():
    base = {"hits": 3, "steps": 8, "shape": "square", "pulse": 1.0, "accent": 0.0}
    same = render_clip(Clip(GeneratorSource("euclid", {**base, "b_mode": "same"}), length_s=2))
    inv = render_clip(Clip(GeneratorSource("euclid", {**base, "b_mode": "inverse"}), length_s=2))
    np.testing.assert_array_equal(same[:, 0], same[:, 1])
    assert np.all((inv[:, 0] > 0.5) != (inv[:, 1] > 0.5))


def test_ramp_build_and_teases():
    plain = render_clip(Clip(GeneratorSource("ramp", {}), length_s=20))
    assert np.all(np.diff(plain[:, 0]) >= -1e-6)
    assert plain[0, 0] == pytest.approx(0.2) and plain[-1, 0] == pytest.approx(1.0)
    teased = render_clip(Clip(GeneratorSource("ramp", {"tease_every": 5.0, "jitter": 0.0}),
                              length_s=20))
    assert teased[:, 0].min() < plain[:, 0].min() + 0.1 and np.any(np.diff(teased[:, 0]) < -0.01)
    lagged = render_clip(Clip(GeneratorSource("ramp", {"b_lag": 1.0}), length_s=10))
    np.testing.assert_allclose(lagged[FS:, 1], lagged[:-FS, 0], atol=1e-6)


# ── stages ──────────────────────────────────────────────────────────────────
def test_segment_stages_cover_the_horizon():
    stages = segment_stages("milkmaster", {}, 5, 300.0)
    assert len(stages) > 3 and stages[0][0] == 0.0 and stages[-1][1] == 300.0
    assert all(a < b for a, b in stages)
    assert all(stages[i][1] == stages[i + 1][0] for i in range(len(stages) - 1))
    assert segment_stages("waveform", {}, 0, 60.0) == []
    assert manual_params_at("licks", {}, 2, 12.0).get("manual") is True


# ── presets ─────────────────────────────────────────────────────────────────
BUNDLED = list_presets(BUNDLED_DIR)


def test_bundled_presets_exist():
    assert len(BUNDLED) >= 10


@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.stem)
def test_bundled_preset_loads_and_renders(path):
    clip = load_preset(path, start_s=3.0)
    assert clip.start_s == 3.0 and clip.name and clip.length_s > 0
    assert isinstance(clip.source, GeneratorSource)
    out = render_clip(clip)
    assert out.shape[0] == round(clip.length_s * FS) and np.isfinite(out).all()


@pytest.mark.parametrize("source", [
    GeneratorSource("waveform", {}),             # default params must stay a generator
    GeneratorSource("euclid", {"hits": 5}),
    PulseDataSource(np.random.default_rng(0).random((60, 4)).astype(np.float32)),
], ids=["waveform-defaults", "euclid", "pulses"])
def test_preset_roundtrip(tmp_path, source):
    clip = Clip(source, length_s=1.5, fade_out_s=0.5, name="mine")
    path = save_preset(clip, tmp_path / "sub" / "mine")
    assert path.suffix == ".parthwl" and path in list_presets(tmp_path)
    back = load_preset(path)
    assert type(back.source) is type(source) and back.name == "mine"
    np.testing.assert_allclose(render_clip(back), render_clip(clip), atol=1e-6)


def test_safe_filename():
    assert safe_filename('a/b:c*"d') == "a_b_c__d"
    assert safe_filename("   ") == "preset"


# ── pattern editor (GUI) ────────────────────────────────────────────────────
def _stroke(ed, kind, points):
    ed._on_stroke_start(kind)
    for x, y in points:
        ed._on_draw(kind, x, y)
    ed._on_stroke_end(kind)


def _add(window, type_id, params=None, start=60.0, length=8.0, **kw):
    doc = window.doc
    clip = Clip(GeneratorSource(type_id, params or {}, kw.pop("seed", 0)), start_s=start,
                length_s=length, name=type_id, **kw)
    doc.add_clip(doc.project.tracks[1].id, clip)
    window.editor.select_clips([clip.id])
    return clip


def test_draw_editor_stroke_commits_one_undo_step(window):
    clip = _add(window, "draw")
    pe = window.pattern
    assert pe.stack.currentWidget() is pe.draw_editor
    ed = pe.draw_editor
    ed.side_group.button(0).setChecked(True)               # A only
    before = window.doc.undo.count()
    _stroke(ed, "amp", [(0.0, 0.9), (4.0, 0.9), (8.0, 0.1)])
    assert window.doc.undo.count() == before + 1
    a = np.array([float(v) for v in clip.source.params["amp_a"].split(",")])
    assert a[0] == pytest.approx(0.9) and a[-1] == pytest.approx(0.1)
    assert "amp_b" not in clip.source.params or \
        all(float(v) == 0.5 for v in clip.source.params["amp_b"].split(","))
    out = render_clip(clip)
    assert out[0, 0] == pytest.approx(0.9, abs=0.02) and out[-1, 0] == pytest.approx(0.1, abs=0.02)
    window.doc.undo.undo()
    assert "amp_a" not in clip.source.params


def test_draw_editor_both_channels_and_tools(window):
    clip = _add(window, "draw")
    ed = window.pattern.draw_editor
    _stroke(ed, "freq", [(0.0, 0.2), (8.0, 0.8)])           # A+B is the default
    p = clip.source.params
    assert p["freq_a"] == p["freq_b"]
    ed._apply(lambda a, _k: 1.0 - a)                         # invert both
    assert float(clip.source.params["freq_a"].split(",")[0]) == pytest.approx(0.8)


def test_step_editor(window):
    clip = _add(window, "steps", {"steps": 4, "transition": "hard"})
    pe = window.pattern
    assert pe.stack.currentWidget() is pe.step_editor
    ed = pe.step_editor
    assert ed.steps == 4
    _stroke(ed, "amp", [(0.5, 1.0), (7.5, 0.0)])            # first step 1.0, last 0.0
    vals = [float(v) for v in clip.source.params["amp_a_vals"].split(",")]
    assert len(vals) == 4 and vals[0] == pytest.approx(1.0) and vals[-1] == pytest.approx(0.0)
    window.doc.set_clip_attrs(clip.id, source=GeneratorSource(
        "steps", {**clip.source.params, "steps": 6}))
    assert ed.steps == 6


def test_generator_page_stage_navigation(window):
    clip = _add(window, "milkmaster", length=10.0, seed=5)
    page = window.pattern.generator_page
    assert window.pattern.stack.currentWidget() is page
    assert len(page.stages) > 3 and not page.prev_btn.isHidden()
    page._step(+1)                                           # snap to stage 2
    a, b = page.stages[1]
    assert clip.trim_s == pytest.approx(a) and clip.length_s == pytest.approx(b - a)
    page._step(+1)
    assert clip.trim_s == pytest.approx(page.stages[2][0])
    page._from_start()
    assert clip.trim_s == 0.0
    window.doc.undo.undo()
    assert clip.trim_s == pytest.approx(page.stages[2][0])


def test_freeze_activity_as_manual(window):
    clip = _add(window, "licks", length=20.0, seed=2)
    page = window.pattern.generator_page
    assert not page.manual_btn.isHidden() and page.snapshot.get("manual")
    page._freeze_manual()
    assert clip.source.params["manual"] is True and "speed" in clip.source.params
    assert page.manual_btn.isHidden()                        # already manual


def test_pattern_editor_follows_selection(window):
    pe = window.pattern
    window.editor.select_clips([])
    assert pe.stack.currentWidget() is pe.empty
    clip = window.doc.project.tracks[0].clips[1]             # licks in the demo
    window.editor.select_clips([clip.id])
    assert pe.stack.currentWidget() is pe.generator_page and "licks" in pe.title.text().lower() \
        or "Infinite licks" in pe.title.text()


def test_preset_drop_and_activation(window):
    ed, doc = window.editor, window.doc
    path = str(BUNDLED[0])
    track = doc.project.tracks[1]
    y = ed.layout.row(track.id).y + 10
    clip = ed.drop_preset(path, QPointF(52.0 * ed.pps, y))
    assert clip is not None and clip in track.clips
    assert clip.start_s == pytest.approx(ed.snap.apply(52.0, doc.project.bpm))
    doc.set_cursor(70.0)
    window._add_preset_at_cursor(path)
    assert any(c.start_s == 70.0 for t in doc.project.tracks for c in t.clips)


def test_save_clip_as_preset(window, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog
    clip = window.doc.project.tracks[1].clips[0]
    window.editor.select_clips([clip.id])
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("My bed", True))
    window.save_clip_preset()
    folder = window.browser.presets.user_dir
    saved = list_presets(folder)
    assert [p.stem for p in saved] == ["My bed"]
    back = load_preset(saved[0])
    np.testing.assert_allclose(render_clip(back), render_clip(clip.clone()), atol=1e-6)


def test_bpm_change_rerenders_tempo_synced_clip(window):
    doc = window.doc
    clip = _add(window, "euclid", {"hits": 1, "steps": 4, "division": "1/4"}, length=4.0)
    assert doc.wait_for_render()
    before = doc.last_render.master.copy()
    doc.set_project_attrs(bpm=60.0)
    assert doc.wait_for_render()
    s, e = round(clip.start_s * FS), round(clip.end_s * FS)
    assert not np.array_equal(before[s:e], doc.last_render.master[s:e])
