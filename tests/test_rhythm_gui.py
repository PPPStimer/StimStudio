"""Beat marks and pattern clips in the GUI: CSV import, peaks, convert, editing."""

import numpy as np
import pytest

from stimstudio.core.model import BeatMarksSource, Mark, PatternSource, PulseDataSource

CSV = ("Kick,Kick mark,Snare,Snare mark\r\n"
       "0.500,,0.750,\r\n"
       "2.000,pattern start,,\r\n"
       "2.500,,,\r\n"
       "3.500,pattern end,,\r\n"
       "4.000,pattern start; pattern end,,\r\n")


def write_csv(tmp_path):
    path = tmp_path / "beats.csv"
    path.write_bytes(b"\xef\xbb\xbf" + CSV.encode("utf-8"))
    return str(path)


def generator_clip(window):
    return next((t, c) for t in window.doc.project.tracks if t.type == "standard"
                for c in t.clips if c.source.kind == "generator")


def import_kick(window, tmp_path):
    added = window.import_beats([write_csv(tmp_path)],
                                options={"track_type": "standard", "offset_s": 1.0,
                                         "tracks": [0]})
    assert len(added) == 1
    return window.doc.project.track_by_id(added[0].id)


def test_import_beats_makes_tracks_of_blocks_in_one_undo(window, tmp_path):
    n = len(window.doc.project.tracks)
    added = window.import_beats([write_csv(tmp_path)],
                                options={"track_type": "standard", "offset_s": 1.0,
                                         "tracks": [0, 1]})
    assert [t.name for t in added] == ["Kick", "Snare"]
    kick = window.doc.project.track_by_id(added[0].id)
    assert [c.start_s for c in kick.clips] == pytest.approx([1.5, 3.0, 5.0])
    assert all(isinstance(c.source, BeatMarksSource) for c in kick.clips)
    window.doc.undo.undo()
    assert len(window.doc.project.tracks) == n


def test_beat_import_dialog_defaults(qapp, tmp_path):
    from stimstudio.gui.beat_import import BeatImportDialog
    dlg = BeatImportDialog([write_csv(tmp_path)], 2.0)
    o = dlg.options()
    assert o == {"track_type": "standard", "offset_s": 2.0, "tracks": [0, 1]}
    dlg.deleteLater()


def test_convert_to_pattern_plays_pool_on_marks(window, tmp_path, qapp):
    kick = import_kick(window, tmp_path)
    marks_clip = kick.clips[1]                     # the group: 2.0, 2.5, 3.5 (+1 s offset)
    _t, gen = generator_clip(window)
    window.editor.select_clips([marks_clip.id, gen.id])
    window.editor.convert_to_pattern(marks_clip.id, [gen.id])
    src = marks_clip.source
    assert isinstance(src, PatternSource) and [e.key for e in src.pool] == ["A"]
    assert src.pool[0].clip.id != gen.id and src.pool[0].clip.start_s == 0.0
    assert window.doc.wait_for_render()
    qapp.processEvents()
    tr = window.doc.last_render.tracks[kick.id]
    assert tr.pulses[round(3.0 * 40):round(3.1 * 40), :2].max() > 0
    item = window.editor.clip_items[marks_clip.id]
    assert item.kind == "pattern" and [k for _t2, _v, k in item.marks] == ["A", "A", "A"]
    window.doc.undo.undo()
    assert isinstance(marks_clip.source, BeatMarksSource)


def test_pattern_page_edits_are_undoable(window, tmp_path, qapp):
    kick = import_kick(window, tmp_path)
    clip = kick.clips[1]
    _t, gen = generator_clip(window)
    window.doc.convert_to_pattern(clip.id, [gen.id])
    window.editor.select_clips([clip.id])
    qapp.processEvents()
    page = window.pattern.pattern_page
    assert window.pattern.stack.currentWidget() is page and page.is_pattern
    page.marks.append(Mark(0.25, 0.5))
    page.commit("Add mark")
    assert len(clip.source.marks) == 4
    page.plot.selected = {clip.source.marks[1].id}
    page.set_pin("A")
    assert clip.source.marks[1].pin == "A"
    page.add_generator("waveform")
    assert [e.key for e in clip.source.pool] == ["A", "B"]
    for _ in range(3):
        window.doc.undo.undo()
    assert len(clip.source.marks) == 3 and len(clip.source.pool) == 1


def test_pool_entry_opens_in_inspector_and_edits_in_place(window, tmp_path, qapp):
    kick = import_kick(window, tmp_path)
    clip = kick.clips[1]
    _t, gen = generator_clip(window)
    window.doc.convert_to_pattern(clip.id, [gen.id])
    window.editor.select_clips([clip.id])
    qapp.processEvents()
    window.pattern.open_entry("A")
    address = f"{clip.id}/A"
    assert window.pattern.clip_id == address and window.inspector.clip_id == address
    assert window.pattern.back_btn.isVisibleTo(window.pattern)
    window.inspector.gain.setValue(0.5)
    qapp.processEvents()
    assert clip.source.pool[0].clip.gain == pytest.approx(0.5)
    assert window.pattern.clip_id == address        # edits keep the entry open
    window.doc.undo.undo()
    assert clip.source.pool[0].clip.gain == pytest.approx(gen.gain)
    window.pattern.close_entry()
    assert window.pattern.clip_id == clip.id and window.inspector.clip_id == clip.id


def test_marks_from_peaks_new_track_or_replace(window, qapp):
    track, gen = generator_clip(window)
    n = len(window.doc.project.tracks)
    new = window.marks_from_peaks(gen.id, {"level": 0.05, "method": "max", "min_gap_s": 0.5})
    assert new is not None and isinstance(new.source, BeatMarksSource) and new.source.marks
    tracks = window.doc.project.tracks
    assert len(tracks) == n + 1
    assert tracks[tracks.index(track) + 1].clips == [new]
    window.doc.undo.undo()
    assert len(window.doc.project.tracks) == n
    same = window.marks_from_peaks(gen.id, {"level": 0.05, "replace": True})
    assert same is gen and isinstance(gen.source, BeatMarksSource)


def test_peak_dialog_counts_marks(qapp):
    from stimstudio.gui.peak_marks import PeakMarksDialog, peak_marks
    sig = np.zeros(200)
    sig[[20, 60, 100]] = [0.9, 0.3, 0.7]
    dlg = PeakMarksDialog(sig)
    dlg.level.setValue(0.5)
    assert len(dlg.marks()) == 2 and "2" in dlg.count.text()
    dlg.level.setValue(0.2)
    assert len(dlg.marks()) == 3
    assert len(peak_marks(sig, {"level": 0.8})) == 1
    dlg.deleteLater()


def test_bake_and_back_to_marks(window, tmp_path, qapp):
    kick = import_kick(window, tmp_path)
    clip = kick.clips[1]
    _t, gen = generator_clip(window)
    window.doc.convert_to_pattern(clip.id, [gen.id])
    window.doc.pattern_to_marks(clip.id)
    assert isinstance(clip.source, BeatMarksSource)
    window.doc.undo.undo()
    window.doc.bake_pattern(clip.id)
    assert isinstance(clip.source, PulseDataSource) and clip.source.data[:, 0].max() > 0
    window.doc.glue_clips([c.id for c in kick.clips])      # marks never glue
    assert any(isinstance(c.source, BeatMarksSource) for c in kick.clips)


def test_marks_plot_mouse_and_painting(window, tmp_path, qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtTest import QTest
    kick = import_kick(window, tmp_path)
    clip = kick.clips[1]
    window.editor.select_clips([clip.id])
    window.pattern_dock.show()
    window.pattern_dock.raise_()
    qapp.processEvents()
    plot = window.pattern.pattern_page.plot
    plot.resize(800, 300)
    qapp.processEvents()

    def px(t, v):
        p = plot.getViewBox().mapViewToScene(QPointF(t, v))
        return plot.mapFromScene(p)

    page = window.pattern.pattern_page
    page.snap_on.setChecked(False)
    QTest.mouseDClick(plot.viewport(), Qt.MouseButton.LeftButton, pos=px(0.75, 0.5))
    assert len(clip.source.marks) == 4
    added = next(m for m in clip.source.marks if abs(m.t_s - 0.75) < 0.05)
    assert added.velocity == pytest.approx(0.5, abs=0.05)
    # drag the new mark right by ~0.1 s
    start, end = px(added.t_s, 0.2), px(added.t_s + 0.1, 0.2)
    QTest.mousePress(plot.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(plot.viewport(), QPoint((start.x() + end.x()) // 2, start.y()))
    QTest.mouseMove(plot.viewport(), end)
    QTest.mouseRelease(plot.viewport(), Qt.MouseButton.LeftButton, pos=end)
    moved = next(m for m in clip.source.marks if m.id == added.id)
    assert moved.t_s == pytest.approx(added.t_s + 0.1, abs=0.03)
    QTest.keyClick(plot, Qt.Key.Key_Delete)
    assert window.doc.project.find_clip(clip.id) is not None     # not the window's Delete
    assert all(m.id != added.id for m in clip.source.marks)
    # the song editor paints marks and pattern clips without errors
    _t, gen = generator_clip(window)
    window.doc.convert_to_pattern(kick.clips[0].id, [gen.id])
    assert window.doc.wait_for_render()
    qapp.processEvents()
    assert not window.editor.view.viewport().grab().isNull()


def test_anchor_combo_sets_anchor(window, tmp_path, qapp):
    kick = import_kick(window, tmp_path)
    clip = kick.clips[1]
    _t, gen = generator_clip(window)
    window.doc.convert_to_pattern(clip.id, [gen.id])
    window.editor.select_clips([clip.id])
    qapp.processEvents()
    page = window.pattern.pattern_page
    assert page.anchor.currentData() == "start"
    page.anchor.setCurrentIndex(page.anchor.findData("end"))
    page.anchor.activated.emit(page.anchor.currentIndex())
    assert clip.source.anchor == "end"
    window.doc.undo.undo()
    assert clip.source.anchor == "start" and page.anchor.currentData() == "start"
