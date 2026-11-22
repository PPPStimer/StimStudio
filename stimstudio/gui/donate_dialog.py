"""Donation window shown at startup, or a thank-you window for a known donor.

The donation details come from donate.yml on the project site; the donors who
have been confirmed are listed in Heroes.csv there. A donor enters the nickname
they used once; on the next launches it is looked up in Heroes.csv.

Usage, after the main window is shown:

    from .donate_dialog import show_startup_dialog
    show_startup_dialog(window)

Shared drop-in module, with these changes for StimStudio: the windows are
not modal, so the main window and its PANIC button stay within reach of the mouse;
the startup window waits for a running output to end; the Help menu loads the data
in the background; Enter in the nickname field doesn't close the window.
"""

from __future__ import annotations

import csv
import json
import threading
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------- settings

APP_NAME = "StimStudio"   # sent as the User-Agent

_RAW = "https://raw.githubusercontent.com/PPPStimer/pppstimer.github.io/main/_data/"
DONATE_URL = _RAW + "donate.yml"
HEROES_URL = _RAW + "Heroes.csv"

# Where the nickname is kept. Apps that use the same
# folder share the nickname: a donor enters it once for all of them.
_CONFIG_DIR = Path.home() / ".pppstimer"
DONOR_PATH = _CONFIG_DIR / "donor.json"

HERO_CLOSE_MS = 3000     # the thank-you window closes by itself after this
_COPIED_MS = 1500        # how long a wallet's button says "Copied"
_BUSY_RECHECK_MS = 500   # how often a waiting startup window looks for its moment

_ACCENT = "#ffc800"

# The dialogs carry their own dark theme so they look the same in any app
_DIALOG_STYLE = """
    QDialog { background-color: #1e1e1e; }
    QLabel { color: #d4d4d4; font-size: 13px; background: transparent; }
    QPushButton {
        background-color: #0e639c; color: white; border: none;
        padding: 6px 14px; border-radius: 3px; min-width: 60px; font-size: 13px;
    }
    QPushButton:hover { background-color: #1177bb; }
    QPushButton:pressed { background-color: #094771; }
    QToolTip { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #3d3d3d; }
"""
_LINE_EDIT_STYLE = (
    "QLineEdit { background: #3c3c3c; color: #d4d4d4; border: 1px solid #555; "
    "padding: 4px; border-radius: 3px; font-size: 13px; }"
)
_PRIMARY_BUTTON_STYLE = (
    "QPushButton { background: #0e639c; color: white; padding: 6px 20px; "
    "border: none; border-radius: 3px; font-weight: bold; }"
    "QPushButton:hover { background: #1177bb; }"
)


# ---------------------------------------------------------------- data

def _fetch(url: str, timeout: float = 5.0) -> str | None:
    """The text at url, or None if it can't be had."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": APP_NAME})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8-sig")
    except Exception:
        return None


def _load_donor() -> dict:
    """donor.json: the nickname the user entered and whether the last check
    found it in Heroes.csv."""
    try:
        data = json.loads(DONOR_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_nickname() -> str:
    return str(_load_donor().get("nickname", "")).strip()


def save_nickname(nickname: str, hero: bool = False):
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DONOR_PATH.write_text(
        json.dumps({"nickname": nickname, "hero": hero}, ensure_ascii=False), encoding="utf-8")


def _check_hero(nickname: str) -> bool | None:
    """Whether nickname is in Heroes.csv; None if the list can't be fetched."""
    text = _fetch(HEROES_URL)
    if text is None:
        return None
    return nickname.casefold() in _parse_heroes(text)


def _remember_hero(nickname: str, hero: bool):
    """Keep the result of a check, unless another nickname was entered meanwhile."""
    try:
        if load_nickname() == nickname:
            save_nickname(nickname, hero)
    except OSError:
        pass


def _parse_heroes(text: str) -> set[str]:
    """The nicknames in Heroes.csv (its first column), casefolded."""
    names = set()
    for row in csv.reader(text.splitlines()):
        if row and row[0].strip():
            names.add(row[0].strip().casefold())
    return names


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_donate_yml(text: str) -> dict:
    """donate.yml as a dict. Only the shape that file uses is understood:
    top-level `key: value`, a nested mapping (`goal:` followed by indented
    `key: value`) and lists of mappings (`platforms:` followed by `- name: ...`
    and indented `key: value`)."""
    data: dict = {}
    block = None   # top-level key whose indented lines are being read
    item = None    # the list entry being read
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line[0].isspace():
            key, _, value = stripped.partition(":")
            key = key.strip()
            if value.strip():
                data[key] = _scalar(value)
                block = None
            else:
                data[key] = None
                block = key
            item = None
            continue
        if block is None:
            continue
        if stripped == "-" or stripped.startswith("- "):
            if not isinstance(data[block], list):
                data[block] = []
            item = {}
            data[block].append(item)
            stripped = stripped[1:].strip()
            if not stripped:
                continue
            target = item
        elif isinstance(data[block], list):
            target = item
        else:
            if data[block] is None:
                data[block] = {}
            target = data[block]
        key, sep, value = stripped.partition(":")
        if sep and target is not None:
            target[key.strip()] = _scalar(value)
    return data


def _load_donate_data() -> dict | None:
    """donate.yml from the site; None offline, so that there is no window then."""
    text = _fetch(DONATE_URL)
    return None if text is None else _parse_donate_yml(text)


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- dialogs

class HeroDialog(QDialog):
    """Thanks a known donor; closes by itself after HERO_CLOSE_MS."""

    def __init__(self, nickname: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_DIALOG_STYLE)
        self.setWindowTitle("Thank you!")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 20)
        layout.setSpacing(16)

        lbl_name = QLabel(nickname.upper())
        lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_name.setWordWrap(True)
        lbl_name.setStyleSheet(f"color: {_ACCENT}; font-size: 44px; font-weight: bold;")
        layout.addWidget(lbl_name)

        lbl_text = QLabel(
            "Thank you for your invaluable contribution to the development "
            "of tools for creating PMVs")
        lbl_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_text.setWordWrap(True)
        lbl_text.setStyleSheet("font-size: 15px;")
        layout.addWidget(lbl_text)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_hero = QPushButton("I am a hero")
        btn_hero.setStyleSheet(_PRIMARY_BUTTON_STYLE)
        btn_hero.setDefault(True)
        btn_hero.clicked.connect(self.accept)
        btn_layout.addWidget(btn_hero)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        QTimer.singleShot(HERO_CLOSE_MS, self.accept)


class DonateDialog(QDialog):
    """Ways to support the development, from donate.yml."""

    # Result of checking an applied nickname: (nickname, True / False / None = list unreachable)
    hero_checked = pyqtSignal(str, object)

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self.hero_checked.connect(self._on_hero_checked)
        self.setStyleSheet(_DIALOG_STYLE)
        self.setWindowTitle(str(data.get("title") or "Support development"))
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        lbl_title = QLabel(str(data.get("title") or "Support development"))
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(lbl_title)

        goal = data.get("goal")
        if isinstance(goal, dict):
            self._add_goal(layout, goal)

        platforms = [p for p in data.get("platforms") or [] if isinstance(p, dict) and p.get("url")]
        if platforms:
            grid = QGridLayout()
            for i, platform in enumerate(platforms):
                btn = QPushButton(platform.get("name") or platform["url"])
                btn.setToolTip(platform["url"])
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.clicked.connect(
                    lambda _=False, url=platform["url"]: QDesktopServices.openUrl(QUrl(url)))
                grid.addWidget(btn, i // 3, i % 3)
            layout.addLayout(grid)

        wallets = [w for w in data.get("wallets") or [] if isinstance(w, dict) and w.get("address")]
        if wallets:
            grid = QGridLayout()
            for row, wallet in enumerate(wallets):
                lbl = QLabel(wallet.get("name", ""))
                lbl.setStyleSheet("font-weight: bold;")
                grid.addWidget(lbl, row, 0)
                edit = QLineEdit(wallet["address"])
                edit.setReadOnly(True)
                edit.setStyleSheet(_LINE_EDIT_STYLE)
                grid.addWidget(edit, row, 1)
                btn = QPushButton("Copy")
                btn.clicked.connect(
                    lambda _=False, b=btn, address=wallet["address"]: self._copy(b, address))
                grid.addWidget(btn, row, 2)
            grid.setColumnStretch(1, 1)
            layout.addLayout(grid)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #3d3d3d;")
        layout.addWidget(line)

        layout.addWidget(QLabel("Already donated? Enter the nickname you used:"))
        nick_row = QHBoxLayout()
        self._edit_nick = QLineEdit(load_nickname())
        self._edit_nick.setPlaceholderText("Nickname")
        self._edit_nick.setStyleSheet(_LINE_EDIT_STYLE)
        self._edit_nick.returnPressed.connect(self._on_apply)
        nick_row.addWidget(self._edit_nick, stretch=1)
        btn_apply = QPushButton("Apply")
        btn_apply.setAutoDefault(False)
        btn_apply.clicked.connect(self._on_apply)
        nick_row.addWidget(btn_apply)
        layout.addLayout(nick_row)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(self._lbl_status)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_later = QPushButton("Ok, later")
        btn_later.setStyleSheet(_PRIMARY_BUTTON_STYLE)
        btn_later.setDefault(True)
        btn_later.clicked.connect(self.accept)
        btn_layout.addWidget(btn_later)
        layout.addLayout(btn_layout)

    def _add_goal(self, layout: QVBoxLayout, goal: dict):
        raised = _number(goal.get("raised"))
        target = _number(goal.get("target"))
        currency = goal.get("currency", "")

        row = QHBoxLayout()
        row.addWidget(QLabel(str(goal.get("note") or "Goal")))
        row.addStretch()
        row.addWidget(QLabel(f"{currency}{raised:g} / {currency}{target:g}"))
        layout.addLayout(row)

        if target > 0:
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(min(raised / target, 1.0) * 1000))
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(
                "QProgressBar { background: #3c3c3c; border: none; border-radius: 5px; }"
                f"QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 5px; }}"
            )
            layout.addWidget(bar)

    def keyPressEvent(self, event):
        # Enter in the nickname field applies it; it must not press "Ok, later" as well
        if self.focusWidget() is self._edit_nick and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return
        super().keyPressEvent(event)

    def _copy(self, button: QPushButton, address: str):
        QApplication.clipboard().setText(address)
        button.setText("Copied")
        # Owned by the button: the window may be closed and deleted before this fires
        timer = QTimer(button)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: button.setText("Copy"))
        timer.start(_COPIED_MS)

    def _set_status(self, text: str, color: str = "#999"):
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _on_apply(self):
        nickname = self._edit_nick.text().strip()
        donor = _load_donor()
        # Re-applying the same nickname keeps its status, so it isn't lost while offline
        hero = donor.get("hero") is True and str(donor.get("nickname", "")).strip() == nickname
        try:
            save_nickname(nickname, hero)
        except OSError as e:
            self._set_status(f"Could not save the nickname: {e}", "#e06c6c")
            return
        if not nickname:
            self._set_status("Nickname cleared")
            return
        self._set_status("Checking…")
        threading.Thread(target=self._check_nickname, args=(nickname,), daemon=True).start()

    def _check_nickname(self, nickname: str):
        """Runs in a background thread."""
        hero = _check_hero(nickname)
        if hero is not None:
            _remember_hero(nickname, hero)
        try:
            self.hero_checked.emit(nickname, hero)
        except RuntimeError:
            pass  # the dialog is gone

    @pyqtSlot(str, object)
    def _on_hero_checked(self, nickname: str, hero):
        if nickname != load_nickname():
            return  # another nickname was applied meanwhile
        if hero:
            self._set_status("✓ Found — thank you, hero!", _ACCENT)
        elif hero is None:
            self._set_status(
                "Saved. Couldn't reach the list of heroes now — it will be checked on the next launch")
        else:
            self._set_status(
                "Saved. Not in the list yet — it will be checked again on the next launch "
                "(the list takes a few minutes to update)")


# ---------------------------------------------------------------- startup

class _LookupBridge(QObject):
    """Hands the result of a background lookup to the GUI thread."""
    loaded = pyqtSignal(object)

    def __init__(self, window: QWidget, requested: bool = False):
        super().__init__(window)
        self._window = window
        self._requested = requested   # from the Help menu, not at startup
        self._result = None
        self.loaded.connect(self._on_loaded)

    @pyqtSlot(object)
    def _on_loaded(self, result):
        self._result = result
        self._show()

    def _show(self):
        # Nobody asked for the startup window: it doesn't take the keyboard from a
        # running output, and it would be dead under a modal dialog
        if not self._requested and (self._window.playback.playing
                                    or QApplication.activeModalWidget() is not None):
            QTimer.singleShot(_BUSY_RECHECK_MS, self._show)
            return
        if self._result:
            kind, payload = self._result
            if kind == "hero":
                dialog = HeroDialog(payload, self._window)
            else:
                dialog = (self._window.findChild(DonateDialog)
                          or DonateDialog(payload, self._window))
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.show()   # not exec(): see the module docstring
            dialog.raise_()
            dialog.activateWindow()
        elif self._requested:
            self._window.statusBar().showMessage(
                "Couldn't load the donation details: no connection.", 5000)
        self.deleteLater()


def _look_up(bridge: _LookupBridge):
    donor = _load_donor()
    nickname = str(donor.get("nickname", "")).strip()
    result = None
    if nickname:
        hero = _check_hero(nickname)
        if hero is None:
            hero = donor.get("hero") is True   # offline: the last known status
        else:
            _remember_hero(nickname, hero)
        if hero:
            result = ("hero", nickname)
    if result is None:
        data = _load_donate_data()
        if data:
            result = ("donate", data)
    try:
        bridge.loaded.emit(result)
    except RuntimeError:
        pass  # the window was closed meanwhile


def show_startup_dialog(window: QWidget):
    """Once the data is loaded, show the thank-you window to a known donor and
    the donation window to everyone else. The lookup runs in the background so
    a slow network doesn't hold up the start."""
    bridge = _LookupBridge(window)
    threading.Thread(target=_look_up, args=(bridge,), daemon=True).start()


def _look_up_donate(bridge: _LookupBridge):
    data = _load_donate_data()
    try:
        bridge.loaded.emit(("donate", data) if data else None)
    except RuntimeError:
        pass  # the window was closed meanwhile


def show_donate_dialog(window: QWidget):
    """Help → Support development. Also loads in the background: a slow network
    must not freeze the GUI thread, or PANIC with it."""
    bridge = _LookupBridge(window, requested=True)
    threading.Thread(target=_look_up_donate, args=(bridge,), daemon=True).start()
