"""PANIC reaches a running output from everywhere the keyboard can be.

Esc is sent through the window-system path (shortcut map first, then the key event),
as a real key is. Runs on the dry-run sink.
"""

import time

import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (QApplication, QColorDialog, QDialog, QFileDialog, QInputDialog, QMenu,
                             QProgressDialog)

from stimstudio.gui import export_dialog
from stimstudio.gui.export_dialog import ExportDialog

NATIVE_OFF = Qt.ApplicationAttribute.AA_DontUseNativeDialogs


def wait_until(qapp, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def press_esc(qapp, top):
    top.activateWindow()
    QTest.qWaitForWindowActive(top, 500)
    QTest.keyClick(top.windowHandle(), Qt.Key.Key_Escape)
    qapp.processEvents()


def active_top():
    return QApplication.activePopupWidget() or QApplication.activeModalWidget()


def during(opener, check):
    """Run *opener*, which blocks in its own event loop, and *check* its window once it is
    up. Whatever check finds, the window gets closed, so a failure can't hang the test."""
    failure = []

    def step():
        top = active_top()
        if top is None:
            QTimer.singleShot(10, step)
            return
        try:
            check(top)
        except BaseException as e:  # noqa: BLE001  raised again outside the event loop
            failure.append(e)
        while (left := active_top()) is not None:
            left.close()

    QTimer.singleShot(10, step)
    opener()
    if failure:
        raise failure[0]


def show_menu(window):
    menu = QMenu(window)
    menu.addAction("Item")
    menu.exec(window.mapToGlobal(window.rect().center()))


def show_progress(window):
    dialog = QProgressDialog("Working", "Cancel", 0, 100, window)
    dialog.setMinimumDuration(0)
    dialog.exec()


OPENERS = {
    "dialog": lambda w: QDialog(w).exec(),
    "message_box": lambda w: w.show_about(),
    "input": lambda w: QInputDialog.getText(w, "Title", "Label"),
    "colour": lambda w: QColorDialog.getColor(parent=w),
    "file": lambda w: QFileDialog.getOpenFileName(w, "Title"),
    "progress": show_progress,
    "export": lambda w: w.show_export_dialog(),
    "menu": show_menu,
}


@pytest.fixture
def panics(window):
    seen = []
    window.a_panic.triggered.connect(lambda: seen.append(1))
    return seen


@pytest.mark.parametrize("name", OPENERS)
def test_esc_stops_the_output_and_leaves_the_window(window, qapp, panics, name):
    assert window.playback.play()

    def check(top):
        press_esc(qapp, top)
        assert not window.playback.playing
        assert panics == [1]
        assert top.isVisible()
        press_esc(qapp, top)          # nothing plays now: Esc is the window's again
        assert panics == [1]
        assert not top.isVisible()

    during(lambda: OPENERS[name](window), check)


@pytest.mark.parametrize("name", OPENERS)
def test_esc_only_closes_the_window_when_nothing_plays(window, qapp, panics, name):
    def check(top):
        press_esc(qapp, top)
        assert not top.isVisible()

    during(lambda: OPENERS[name](window), check)
    assert panics == []


def test_esc_in_a_popup_list(window, qapp, panics):
    assert window.playback.play()
    window.snap_combo.showPopup()
    press_esc(qapp, QApplication.activePopupWidget())
    assert not window.playback.playing and panics == [1]
    window.snap_combo.hidePopup()


def test_esc_in_the_main_window_is_one_panic(window, qapp, panics):
    assert window.playback.play()
    press_esc(qapp, window)
    assert not window.playback.playing and panics == [1]


def test_other_keys_pass(window, qapp, panics):
    assert window.playback.play()

    def check(top):
        QTest.keyClick(top.windowHandle(), Qt.Key.Key_A)
        qapp.processEvents()
        assert window.playback.playing and panics == []

    during(lambda: OPENERS["dialog"](window), check)
    window.playback.panic()


# ── native dialogs: their keys never reach Qt, so none while output runs ────
def test_no_native_dialogs_while_output_runs(window, qapp):
    pb = window.playback
    assert not QApplication.testAttribute(NATIVE_OFF)
    assert pb.play()
    assert QApplication.testAttribute(NATIVE_OFF)
    pb.panic()
    assert not QApplication.testAttribute(NATIVE_OFF)
    assert pb.play()
    pb.stop()                         # graceful: ends on its own after the fade
    assert wait_until(qapp, lambda: not QApplication.testAttribute(NATIVE_OFF))


# ── exports on the GUI thread: nothing reads the keyboard until they end ────
@pytest.fixture
def stopping_at_export(window, tmp_path, monkeypatch):
    """For each export: was the output already on its way out when it began?"""
    seen = []

    def note():
        sink = window.playback.sink
        seen.append(bool(sink._stopping_at) or not sink.playing)

    def fake_mix(*_args, **_kwargs):
        note()
        return 0

    def fake_stems(*_args, **_kwargs):
        note()
        return [tmp_path / "stem.hwl"]

    monkeypatch.setattr(export_dialog, "export_mix", fake_mix)
    monkeypatch.setattr(export_dialog, "export_stems", fake_stems)
    monkeypatch.setattr("stimstudio.core.export.export_mix", fake_mix)
    return seen


@pytest.mark.parametrize("fmt, stems, stops", [
    ("wav", False, True),
    ("flac", False, True),
    ("wav", True, True),
    ("hwl", True, True),
    ("hwl", False, False),            # about a second per hour of project: keeps playing
])
def test_export_dialog(window, tmp_path, stopping_at_export, fmt, stems, stops):
    assert window.playback.play()
    dialog = ExportDialog(window)
    dialog.format.setCurrentIndex(dialog.format.findData(fmt))
    dialog.stems.setChecked(stems)
    dialog.path.setText(str(tmp_path if stems else tmp_path / f"mix.{fmt}"))
    dialog.run()
    assert stopping_at_export == [stops]
    window.playback.panic()


def test_menu_audio_export_stops_the_output_first(window, tmp_path, stopping_at_export,
                                                  monkeypatch):
    target = str(tmp_path / "out.wav")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (target, ""))
    assert window.playback.play()
    window.export_audio()
    assert stopping_at_export == [True]
    window.playback.panic()
