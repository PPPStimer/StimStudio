"""Startup donation / hero window: data files, the behaviour of both
windows, and their manners inside the app (PANIC, running output).

Never touches the network or the real ~/.pppstimer (see the ``server`` fixture).
"""

import json
import sys
import threading
import time

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QProgressBar, QPushButton

from stimstudio.gui import donate_dialog as dd

DONATE_YML = """\
# comment
title: Support us
goal:
  raised: 250
  target: 1000
  currency: "$"
  note: Monthly goal
platforms:
  - name: Ko-fi
    url: https://ko-fi.com/
  - name: Patreon
    url: https://patreon.com/
wallets:
  - name: BTC
    address: bc1abc
  - name: ETH
    address: "0x0"
"""
HEROES_CSV = "TESTUSER,\nTESTUSER2,extra\nTESTUSER10,\n"


def wait_until(qapp, predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def flush_deletes(qapp):
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


@pytest.fixture
def server(monkeypatch, tmp_path):
    """URL -> text of the site; a missing URL is unreachable. donor.json in tmp_path.
    Fails the test on an exception inside a Qt slot or timer."""
    files = {dd.DONATE_URL: DONATE_YML, dd.HEROES_URL: HEROES_CSV}
    monkeypatch.setattr(dd, "_fetch", lambda url, timeout=5.0: files.get(url))
    monkeypatch.setattr(dd, "_CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr(dd, "DONOR_PATH", tmp_path / "cfg" / "donor.json")
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    yield files
    assert not errors


def saved():
    return json.loads(dd.DONOR_PATH.read_text(encoding="utf-8"))


def startup(window):
    """The startup lookup, synchronously; the window it opened, if any."""
    dd._look_up(dd._LookupBridge(window))
    return shown_dialog(window)


def shown_dialog(window):
    found = [d for d in window.findChildren(QDialog)
             if isinstance(d, (dd.DonateDialog, dd.HeroDialog)) and d.isVisible()]
    assert len(found) <= 1
    return found[0] if found else None


@pytest.fixture
def donate(window, server):
    dialog = startup(window)
    assert isinstance(dialog, dd.DonateDialog)
    return dialog


# ── data files ──────────────────────────────────────────────────────────────
def test_parse_donate_yml():
    assert dd._parse_donate_yml(DONATE_YML) == {
        "title": "Support us",
        "goal": {"raised": "250", "target": "1000", "currency": "$", "note": "Monthly goal"},
        "platforms": [{"name": "Ko-fi", "url": "https://ko-fi.com/"},
                      {"name": "Patreon", "url": "https://patreon.com/"}],
        "wallets": [{"name": "BTC", "address": "bc1abc"}, {"name": "ETH", "address": "0x0"}],
    }


def test_parse_heroes_first_column_casefolded():
    assert dd._parse_heroes(HEROES_CSV + "\n  Spaced ,\n") == {
        "testuser", "testuser2", "testuser10", "spaced"}


def test_hero_match_is_exact_but_ignores_case(server):
    assert dd._check_hero("testUser2") is True
    assert dd._check_hero("TESTUSER1") is False      # not TESTUSER10
    del server[dd.HEROES_URL]
    assert dd._check_hero("TESTUSER") is None


# ── startup table ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("donor, online, kind, hero_after", [
    (None, True, dd.DonateDialog, None),
    ({"nickname": "testuser2", "hero": False}, True, dd.HeroDialog, True),
    ({"nickname": "stranger", "hero": True}, True, dd.DonateDialog, False),
    ({"nickname": "testuser2", "hero": True}, False, dd.HeroDialog, True),
    ({"nickname": "testuser2", "hero": False}, False, None, False),
    (None, False, None, None),
])
def test_startup_table(window, server, donor, online, kind, hero_after):
    if donor:
        dd.save_nickname(donor["nickname"], donor["hero"])
    if not online:
        server.clear()
    dialog = startup(window)
    assert dialog is None if kind is None else isinstance(dialog, kind)
    if donor:
        assert saved() == {"nickname": donor["nickname"], "hero": hero_after}
    else:
        assert not dd.DONOR_PATH.exists()


def test_startup_runs_in_the_background(window, server, qapp, monkeypatch):
    thread = []
    release = threading.Event()

    def slow_fetch(url, timeout=5.0):
        thread.append(threading.current_thread())
        release.wait(5.0)
        return server.get(url)

    monkeypatch.setattr(dd, "_fetch", slow_fetch)
    dd.show_startup_dialog(window)        # returns while the "network" is still busy
    assert shown_dialog(window) is None
    release.set()
    assert wait_until(qapp, lambda: shown_dialog(window) is not None)
    assert threading.main_thread() not in thread


# ── donation window ─────────────────────────────────────────────────────────
def test_window_shows_the_data(donate):
    assert donate.windowTitle() == "Support us"
    bar = donate.findChild(QProgressBar)
    assert bar.value() == 250
    texts = {b.text() for b in donate.findChildren(QPushButton)}
    assert {"Ko-fi", "Patreon", "Apply", "Ok, later"} <= texts
    assert sum(b.text() == "Copy" for b in donate.findChildren(QPushButton)) == 2


def test_empty_blocks_are_hidden(window, server):
    server[dd.DONATE_URL] = "title: T\ngoal:\n  raised: 5\n  target: 0\n"
    dialog = startup(window)
    assert dialog.findChild(QProgressBar) is None
    assert {b.text() for b in dialog.findChildren(QPushButton)} == {"Apply", "Ok, later"}


@pytest.mark.parametrize("nickname, online, status, hero_after", [
    ("testuser", True, "✓ Found", True),
    ("stranger", True, "Saved. Not in the list yet", False),
    ("testuser", False, "Saved. Couldn't reach", False),
])
def test_apply_messages(window, server, qapp, donate, nickname, online, status, hero_after):
    if not online:
        del server[dd.HEROES_URL]
    donate._edit_nick.setText(nickname)
    donate._on_apply()
    assert wait_until(qapp, lambda: donate._lbl_status.text().startswith(status))
    assert saved() == {"nickname": nickname, "hero": hero_after}


def test_apply_empty_clears(window, server, donate):
    dd.save_nickname("someone", False)
    donate._edit_nick.setText("  ")
    donate._on_apply()
    assert donate._lbl_status.text() == "Nickname cleared"
    assert dd.load_nickname() == ""


def test_reapply_keeps_hero_offline(window, server, qapp, donate):
    dd.save_nickname("testuser", True)
    del server[dd.HEROES_URL]
    donate._edit_nick.setText("testuser")
    donate._on_apply()
    assert wait_until(qapp, lambda: "Couldn't reach" in donate._lbl_status.text())
    assert saved()["hero"] is True


def test_enter_in_nickname_applies_and_keeps_the_window(window, server, qapp, donate):
    donate._edit_nick.setFocus()
    donate._edit_nick.setText("testuser")
    QTest.keyClick(donate._edit_nick, Qt.Key.Key_Return)
    assert donate.isVisible()
    assert wait_until(qapp, lambda: donate._lbl_status.text().startswith("✓ Found"))


def test_copy_then_close_at_once(window, server, qapp, donate, monkeypatch):
    monkeypatch.setattr(dd, "_COPIED_MS", 20)
    copy = next(b for b in donate.findChildren(QPushButton) if b.text() == "Copy")
    copy.click()
    assert copy.text() == "Copied"
    assert QApplication.clipboard().text() == "bc1abc"
    donate.accept()                       # deleted before the button's text comes back
    flush_deletes(qapp)
    assert sip.isdeleted(copy)
    QTest.qWait(60)                       # a timer that outlived the button would raise here


def test_copy_label_comes_back(window, server, qapp, donate, monkeypatch):
    monkeypatch.setattr(dd, "_COPIED_MS", 20)
    copy = next(b for b in donate.findChildren(QPushButton) if b.text() == "Copy")
    copy.click()
    assert wait_until(qapp, lambda: copy.text() == "Copy")


# ── hero window ─────────────────────────────────────────────────────────────
def test_hero_closes_by_itself(window, server, qapp, monkeypatch):
    monkeypatch.setattr(dd, "HERO_CLOSE_MS", 20)
    dd.save_nickname("testuser2", True)
    dialog = startup(window)
    assert isinstance(dialog, dd.HeroDialog)
    assert wait_until(qapp, lambda: shown_dialog(window) is None)
    flush_deletes(qapp)
    assert sip.isdeleted(dialog)


def test_hero_button_closes_before_the_timer(window, server, qapp, monkeypatch):
    monkeypatch.setattr(dd, "HERO_CLOSE_MS", 30)
    dd.save_nickname("testuser2", True)
    dialog = startup(window)
    name = dialog.findChildren(type(dialog.layout().itemAt(0).widget()))[0]
    assert name.text() == "TESTUSER2"
    next(b for b in dialog.findChildren(QPushButton) if b.text() == "I am a hero").click()
    assert shown_dialog(window) is None
    flush_deletes(qapp)
    QTest.qWait(60)                       # the timer fires on a deleted window


# ── inside the app ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("nickname", ["", "testuser2"])
def test_esc_is_panic_while_the_window_is_open(window, server, qapp, nickname):
    if nickname:
        dd.save_nickname(nickname, True)
    dialog = startup(window)
    dialog.activateWindow()
    assert QTest.qWaitForWindowActive(dialog, 1000)
    assert QApplication.activeModalWidget() is None
    panics = []
    window.a_panic.triggered.connect(lambda: panics.append(1))
    QTest.keyClick(dialog, Qt.Key.Key_Escape)
    assert panics == [1]
    assert dialog.isVisible()


def test_startup_window_waits_for_the_output_to_stop(window, server, qapp, monkeypatch):
    monkeypatch.setattr(dd, "_BUSY_RECHECK_MS", 10)
    assert window.playback.play()
    assert startup(window) is None
    QTest.qWait(50)
    assert shown_dialog(window) is None
    window.playback.panic()
    assert wait_until(qapp, lambda: shown_dialog(window) is not None)


def test_startup_window_waits_for_a_modal_dialog(window, server, qapp, monkeypatch):
    monkeypatch.setattr(dd, "_BUSY_RECHECK_MS", 10)
    modal = QDialog(window)
    modal.open()
    assert startup(window) is None
    QTest.qWait(50)
    assert shown_dialog(window) is None
    modal.reject()
    assert wait_until(qapp, lambda: shown_dialog(window) is not None)


def test_help_menu_opens_at_once_even_while_playing(window, server, qapp):
    assert window.playback.play()
    window.a_support.trigger()
    assert wait_until(qapp, lambda: isinstance(shown_dialog(window), dd.DonateDialog))
    window.playback.panic()


def test_help_menu_keeps_one_window(window, server, qapp):
    window.a_support.trigger()
    assert wait_until(qapp, lambda: shown_dialog(window) is not None)
    first = shown_dialog(window)
    window.a_support.trigger()
    bridges = window.findChildren(dd._LookupBridge)
    assert wait_until(qapp, lambda: all(sip.isdeleted(b) or b._result for b in bridges))
    assert shown_dialog(window) is first


def test_help_menu_offline_says_so(window, server, qapp):
    server.clear()
    window.a_support.trigger()
    assert wait_until(qapp, lambda: "no connection" in window.statusBar().currentMessage())
    assert shown_dialog(window) is None


def test_finished_lookups_leave_nothing_behind(window, server, qapp):
    dialog = startup(window)
    dialog.accept()
    flush_deletes(qapp)
    assert window.findChild(dd.DonateDialog) is None
    assert window.findChild(dd._LookupBridge) is None
