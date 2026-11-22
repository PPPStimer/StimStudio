"""Headless GUI tests: document/undo, song editor gestures, inspector, files.

Runs on Qt's offscreen platform with the raster engine (no GL context exists
there); QSettings go to a temporary ini file (see conftest.py).
"""

import numpy as np
import pytest
from PyQt6.QtCore import QPointF, Qt

from stimstudio.core.model import Clip, GeneratorSource, Project, Track

NO_MODS = Qt.KeyboardModifier.NoModifier


def flat(start, length):
    return Clip(GeneratorSource("waveform", {"amp_speed": 0.0, "amp_min": 0.5, "amp_max": 0.5}),
                start_s=start, length_s=length)


# ── document / undo ─────────────────────────────────────────────────────────
def test_render_reaches_views(window):
    ed = window.editor
    assert window.doc.last_render.n == 2000
    for item in ed.clip_items.values():
        assert item._preview is not None and len(item._preview)


def test_track_add_remove_undo(window):
    doc = window.doc
    n = len(doc.project.tracks)
    t = doc.add_track()
    assert len(doc.project.tracks) == n + 1 and t.id in window.editor.headers.headers
    doc.remove_track(t.id)
    assert t.id not in window.editor.headers.headers
    doc.undo.undo()
    doc.undo.undo()
    assert len(doc.project.tracks) == n


def test_render_results_are_never_stale(window):
    doc = window.doc
    clip = doc.project.tracks[0].clips[0]
    for start in (1.0, 2.0, 3.0):          # a burst of edits
        doc.set_clip_attrs(clip.id, merge_key="start", start_s=start)
    assert doc.wait_for_render()
    from stimstudio.core.render import render_master
    np.testing.assert_array_equal(doc.last_render.master, render_master(doc.project))
    assert doc.undo.count() == 1           # merged into one undo step


def test_split_and_undo(window):
    doc = window.doc
    track = doc.project.tracks[0]
    clip = track.clips[1]                  # licks, 19..50 s
    right = doc.split_clips([clip.id], 30.0)[0]
    assert clip.length_s == pytest.approx(11.0)
    assert right.start_s == 30.0 and right.trim_s == pytest.approx(11.0)
    doc.undo.undo()                        # one macro
    assert len(track.clips) == 2 and clip.length_s == pytest.approx(31.0)


def test_split_keeps_audio_identical(window):
    doc = window.doc
    from stimstudio.core.render import render_master
    clip = doc.project.tracks[0].clips[1]
    clip.fade_in_s = clip.fade_out_s = 0.0     # fades legitimately move on split
    doc.project.master_effects = []
    before = render_master(doc.project)
    doc.split_clips([clip.id], 33.0)
    np.testing.assert_allclose(render_master(doc.project), before, atol=1e-6)


# ── editor gestures ─────────────────────────────────────────────────────────
def _drag(ed, item, zone, dx_s, dy_px=0.0, mods=NO_MODS):
    origin = item.scenePos() + QPointF(5, 5)
    ed.begin_drag(item, zone, origin)
    ed.update_drag(origin + QPointF(dx_s * ed.pps, dy_px), mods)
    ed.end_drag(mods)


def test_drag_move_snaps_to_beats(window):
    ed, doc = window.editor, window.doc
    clip = doc.project.tracks[0].clips[0]
    ed.select_clips([clip.id])
    _drag(ed, ed.clip_items[clip.id], "move", 1.37)
    beat = 60.0 / doc.project.bpm
    assert clip.start_s == pytest.approx(round(1.37 / beat) * beat)
    doc.undo.undo()
    assert clip.start_s == 0.0


def test_drag_move_to_other_track(window):
    ed, doc = window.editor, window.doc
    src, dst = doc.project.tracks
    clip = src.clips[0]
    ed.select_clips([clip.id])
    row_h = ed.layout.row(dst.id).y - ed.layout.row(src.id).y
    _drag(ed, ed.clip_items[clip.id], "move", 0.0, row_h)
    assert clip in dst.clips and clip not in src.clips
    doc.undo.undo()
    assert clip in src.clips


def test_drag_never_lands_on_non_standard_track(window):
    ed, doc = window.editor, window.doc
    tri = doc.add_track("3ph", track_type="triphase")
    clip = doc.project.tracks[1].clips[0]
    ed.select_clips([clip.id])
    dy = ed.layout.row(tri.id).y - ed.layout.row(doc.project.tracks[1].id).y
    _drag(ed, ed.clip_items[clip.id], "move", 0.0, dy)
    assert clip in doc.project.tracks[1].clips


def test_ctrl_drag_copies(window):
    ed, doc = window.editor, window.doc
    track = doc.project.tracks[1]
    clip = track.clips[0]
    ed.select_clips([clip.id])
    _drag(ed, ed.clip_items[clip.id], "move", 6.0, mods=Qt.KeyboardModifier.ControlModifier)
    assert len(track.clips) == 2 and clip.start_s == 10.0


def test_resize_edges_and_fades(window):
    ed, doc = window.editor, window.doc
    ed.snap.kind = "off"
    clip = doc.project.tracks[1].clips[0]           # 10..40 s, fades 4 s
    item = ed.clip_items[clip.id]
    _drag(ed, item, "right", -5.0)
    assert clip.length_s == pytest.approx(25.0)
    _drag(ed, item, "left", 2.0)
    assert clip.start_s == pytest.approx(12.0) and clip.trim_s == pytest.approx(2.0)
    assert clip.length_s == pytest.approx(23.0)
    _drag(ed, item, "left", -10.0)                  # cannot trim before the source start
    assert clip.start_s == pytest.approx(10.0) and clip.trim_s == pytest.approx(0.0)
    origin = item.scenePos() + QPointF(clip.fade_in_s * ed.pps, 2)
    ed.begin_drag(item, "fade_in", origin)
    ed.update_drag(origin + QPointF(2.0 * ed.pps, 0), NO_MODS)
    ed.end_drag(NO_MODS)
    assert clip.fade_in_s == pytest.approx(6.0)


def test_double_click_adds_browser_generator(window, qapp):
    ed, doc = window.editor, window.doc
    window.browser.select("chaos")
    track = doc.project.tracks[1]
    y = ed.layout.row(track.id).y + 10
    ed.empty_double_clicked(QPointF(45.0 * ed.pps, y))
    new = [c for c in track.clips if c.source.type_id == "chaos"]
    assert len(new) == 1 and ed.selected_clip_ids() == [new[0].id]
    assert window.inspector.clip_box.isVisible()


def test_duplicate_copy_paste_delete(window):
    ed, doc = window.editor, window.doc
    track = doc.project.tracks[0]
    ed.select_clips([track.clips[0].id])
    ed.duplicate_selected()
    assert len(track.clips) == 3 and track.clips[-1].start_s == pytest.approx(20.0)
    window.edit_copy()
    doc.set_cursor(60.0)
    window.edit_paste()
    assert track.clips[-1].start_s == pytest.approx(60.0)
    ed.delete_selected()
    assert len(track.clips) == 3


def test_zoom_keeps_anchor(window):
    ed = window.editor
    ed.zoom(4.0, 25.0)
    bar = ed.view.horizontalScrollBar()
    x_before = 25.0 * ed.pps - bar.value()
    ed.zoom(2.0, 25.0)
    assert 25.0 * ed.pps - bar.value() == pytest.approx(x_before, abs=1.5)


def test_lane_expand_and_mute(window):
    ed, doc = window.editor, window.doc
    track = doc.project.tracks[0]
    ed.toggle_expanded(track.id)
    assert len(ed.lane_items) == 4
    ed.set_lane_muted(track.id, "amp_r", True)
    assert doc.wait_for_render()
    assert np.all(doc.last_render.tracks[track.id].pulses[:, 1] == 0)
    doc.undo.undo()
    assert not track.lanes["amp_r"].muted


# ── inspector ───────────────────────────────────────────────────────────────
def test_inspector_param_edit_is_undoable_and_merged(window):
    ed, doc, insp = window.editor, window.doc, window.inspector
    clip = doc.project.tracks[0].clips[0]
    ed.select_clips([clip.id])
    insp._param_changed("amp_speed", 0.5)
    insp._param_changed("amp_speed", 0.75)
    assert clip.source.params["amp_speed"] == 0.75
    doc.undo.undo()
    assert clip.source.params["amp_speed"] == 0.25


def test_inspector_generator_change_renames_default_name(window):
    ed, doc, insp = window.editor, window.doc, window.inspector
    track = doc.project.tracks[1]
    clip = doc.add_generator_clip(track.id, "waveform", 45.0)
    ed.select_clips([clip.id])
    insp._generator_changed(insp.gen_combo.findData("adsr"))
    assert clip.source.type_id == "adsr" and clip.name == "ADSR"


# ── files ───────────────────────────────────────────────────────────────────
def test_save_open_roundtrip(window, tmp_path):
    doc = window.doc
    path = str(tmp_path / "p.stimproj")
    doc.save(path)
    assert not doc.dirty
    before = doc.project.to_dict()
    doc.add_track()
    assert doc.dirty
    doc.open(path)
    assert doc.project.to_dict() == before and not doc.dirty
    assert doc.undo.count() == 0


def test_export_hwl_matches_render(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from stimstudio.io.hwl import read_hwl
    out = str(tmp_path / "out.hwl")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    window.export_hwl()
    np.testing.assert_array_equal(read_hwl(out), np.clip(window.doc.render_now().master, 0, 1))


def test_empty_project_renders(qapp):
    from stimstudio.gui.document import ProjectDocument
    doc = ProjectDocument(Project(tracks=[Track()]))
    doc.schedule_render()
    assert doc.wait_for_render()
    assert doc.last_render.n == 0
    doc.shutdown()


def test_app_icon_renders(qapp):
    from stimstudio.gui.theme import app_icon
    icon = app_icon()
    assert not icon.isNull()
    assert all(not icon.pixmap(s, s).isNull() for s in (16, 32, 256))
