"""Timeline markers: model, undo, snapping magnet, ruler gestures, navigation."""

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest

from stimstudio.core.model import Marker, Project
from stimstudio.gui.song.layout import Snap, nearest_magnet
from stimstudio.gui.song.ruler import MARK_H
from stimstudio.io.project_file import load_project, save_project


def test_markers_roundtrip_sorted(tmp_path):
    p = Project(markers=[Marker(5.0, "b", "#ff0000"), Marker(1.5, "a")])
    path = tmp_path / "m.stimproj"
    save_project(p, path)
    back = load_project(path)
    assert [(m.t_s, m.name) for m in back.markers] == [(1.5, "a"), (5.0, "b")]
    assert back.markers[1].color == "#ff0000"
    assert back.markers[0].id == p.markers[1].id
    clone = back.clone()
    clone.markers[0].name = "changed"
    assert back.markers[0].name == "a"                 # clone is independent


def test_old_projects_load_without_markers():
    d = Project().to_dict()
    del d["markers"]
    assert Project.from_dict(d).markers == []


def test_marker_commands_undo(window):
    doc = window.doc
    doc.set_markers([], "reset")
    m = doc.add_marker(3.01)
    assert abs(m.t_s - 3.0) < 0.03 and m.name == "Marker 1"
    doc.set_marker_attrs(m.id, merge_key=("drag", 1), t_s=4.0)
    doc.set_marker_attrs(m.id, merge_key=("drag", 1), t_s=5.0)   # one drag = one step
    doc.set_marker_attrs(m.id, name="Intro")
    assert [(x.t_s, x.name) for x in doc.project.markers] == [(5.0, "Intro")]
    doc.undo.undo()
    assert doc.project.markers[0].name == "Marker 1"
    doc.undo.undo()
    assert doc.project.markers[0].t_s == 3.0
    doc.remove_markers([m.id])
    assert doc.project.markers == []
    doc.undo.undo()
    assert len(doc.project.markers) == 1
    added = doc.add_markers([(10, "Cut 1"), (2, "Cut 2")])
    assert len(added) == 2
    assert [x.t_s for x in doc.project.markers] == [2.0, 3.0, 10.0]   # kept sorted
    doc.undo.undo()
    assert len(doc.project.markers) == 1


def test_snap_magnet():
    assert nearest_magnet(5.1, [1.0, 5.0], 0.2) == 5.0
    assert nearest_magnet(5.5, [1.0, 5.0], 0.2) is None
    assert nearest_magnet(5.0, [], 1.0) is None
    snap = Snap("beats", 1.0)                          # 0.5 s grid at 120 BPM
    assert snap.apply(5.1, 120) == 5.0
    assert snap.apply(5.1, 120, [5.13], 0.1) == 5.13
    assert snap.apply(5.4, 120, [5.13], 0.1) == 5.5


def test_clip_drag_sticks_to_marker(window):
    ed, doc = window.editor, window.doc
    doc.set_markers([Marker(7.3, "hit")])
    ed.snap = Snap("beats", 1.0)
    ed.pps = 40.0
    ed.refresh()
    track = next(t for t in doc.project.tracks if t.type == "standard" and t.clips)
    clip = track.clips[0]
    item = ed.clip_items[clip.id]
    start = item.scenePos() + item.boundingRect().center()
    ed.begin_drag(item, "move", start)
    target_dt = 7.3 - clip.start_s + 0.05              # 2 px from the marker
    ed.update_drag(start + type(start)(target_dt * ed.pps, 0), Qt.KeyboardModifier.NoModifier)
    ed.end_drag(Qt.KeyboardModifier.NoModifier)
    assert abs(doc.project.find_clip(clip.id)[1].start_s - 7.3) < 1e-6


def test_ruler_double_click_adds_and_renames(window, monkeypatch):
    ed, doc = window.editor, window.doc
    doc.set_markers([])
    ed.pps = 20.0
    ed.view.horizontalScrollBar().setValue(0)
    ed.ruler.set_offset(0.0)
    QTest.mouseDClick(ed.ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier,
                      QPoint(123, MARK_H + 10))
    assert len(doc.project.markers) == 1
    m = doc.project.markers[0]
    assert abs(m.t_s - 123 / 20.0) < 0.03
    from PyQt6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Peak", True)))
    x = round(m.t_s * ed.pps)
    QTest.mouseDClick(ed.ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                      QPoint(x, 5))
    assert doc.project.markers[0].name == "Peak"
    assert len(doc.project.markers) == 1               # no second marker on the flag


def test_ruler_drag_moves_marker_one_undo(window):
    ed, doc = window.editor, window.doc
    doc.set_markers([Marker(2.0, "m")])
    ed.pps = 20.0
    ed.ruler.set_offset(0.0)
    n_before = doc.undo.count()
    QTest.mousePress(ed.ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(40, 5))
    for x in (60, 80, 100):
        QTest.mouseMove(ed.ruler, QPoint(x, 5))
        # QTest.mouseMove sends no buttons; drive the handler directly
        from PyQt6.QtCore import QPointF
        from PyQt6.QtGui import QMouseEvent
        from PyQt6.QtCore import QEvent
        ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, 5), QPointF(x, 5),
                         Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.AltModifier)
        ed.ruler.mouseMoveEvent(ev)
    QTest.mouseRelease(ed.ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       QPoint(100, 5))
    assert abs(doc.project.markers[0].t_s - 5.0) < 0.03
    assert doc.undo.count() == n_before + 1
    doc.undo.undo()
    assert doc.project.markers[0].t_s == 2.0


def test_click_on_flag_moves_cursor(window):
    ed, doc = window.editor, window.doc
    doc.set_markers([Marker(4.0, "m")])
    doc.set_cursor(0.0)
    ed.pps = 20.0
    ed.ruler.set_offset(0.0)
    QTest.mouseClick(ed.ruler, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(81, 5))
    assert doc.cursor_s == 4.0


def test_navigation_and_loop_between(window):
    ed, doc = window.editor, window.doc
    doc.set_markers([Marker(2.0), Marker(6.0), Marker(9.0)])
    doc.set_cursor(4.0)
    window.a_next_marker.trigger()
    assert doc.cursor_s == 6.0
    window.a_next_marker.trigger()
    assert doc.cursor_s == 9.0
    window.a_next_marker.trigger()                     # none after: stays
    assert doc.cursor_s == 9.0
    window.a_prev_marker.trigger()
    window.a_prev_marker.trigger()
    assert doc.cursor_s == 2.0
    doc.set_cursor(7.0)
    window.a_loop_markers.trigger()
    assert window.playback.loop_region == (6.0, 9.0)
    assert window.playback.loop_enabled
    doc.set_cursor(1.0)
    assert ed.span_between_markers(1.0) == (0.0, 2.0)


def test_add_marker_action_and_clear(window):
    doc = window.doc
    doc.set_markers([])
    doc.set_cursor(3.5)
    window.a_add_marker.trigger()
    window.a_add_marker.trigger()
    assert [m.t_s for m in doc.project.markers] == [3.5, 3.5]
    assert doc.project.markers[1].name == "Marker 2"
    window.a_clear_markers.trigger()
    assert doc.project.markers == []
    doc.undo.undo()
    assert len(doc.project.markers) == 2
