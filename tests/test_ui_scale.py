"""The interface on small logical screens (large display scale) and View → Interface size."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── --scale and the Interface size setting ─────────────────────────────────
def test_parse_scale():
    from stimstudio.gui.metrics import parse_scale
    assert parse_scale("1.5") == parse_scale("150") == parse_scale("150%") == 1.5
    assert parse_scale(2) == 2.0
    assert parse_scale("abc") is None and parse_scale("0.1") is None
    assert parse_scale("900") is None


def test_apply_ui_scale(qapp, monkeypatch):
    from stimstudio.gui import metrics
    from stimstudio.gui.settings import app_settings
    env = {k: v for k, v in os.environ.items() if k != "QT_SCALE_FACTOR"}
    monkeypatch.setattr(os, "environ", env)              # a private copy, restored after
    assert metrics.apply_ui_scale(["x", "--scale", "125", "p.stimproj"]) == ["x", "p.stimproj"]
    assert env["QT_SCALE_FACTOR"] == "1.25"
    assert metrics.ui_scale_percent() == 125

    del env["QT_SCALE_FACTOR"]
    settings = app_settings()
    settings.setValue(metrics.UI_SCALE_KEY, 150)                 # View → Interface size
    try:
        assert metrics.apply_ui_scale(["x", "--other"]) == ["x", "--other"]
        assert env["QT_SCALE_FACTOR"] == "1.5"
        env["QT_SCALE_FACTOR"] = "2"                             # the environment wins
        metrics.apply_ui_scale(["x"])
        assert env["QT_SCALE_FACTOR"] == "2"
        del env["QT_SCALE_FACTOR"]
        settings.setValue(metrics.UI_SCALE_KEY, 100)             # as Windows: nothing set
        metrics.apply_ui_scale(["x"])
        assert "QT_SCALE_FACTOR" not in env
    finally:
        settings.remove(metrics.UI_SCALE_KEY)


def test_below_100_percent_only_shrinks_text(monkeypatch):
    """A device pixel ratio under 1 draws wrong (tree arrows on other rows, stale
    tabs with OpenGL views): under 100 % only the theme's text gets smaller."""
    import re

    from stimstudio.gui import metrics
    from stimstudio.gui.theme import load_stylesheet
    env = {k: v for k, v in os.environ.items() if k != "QT_SCALE_FACTOR"}
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(metrics, "_text_scale", 1.0)
    metrics.apply_ui_scale(["x", "--scale", "80%"])
    assert "QT_SCALE_FACTOR" not in env
    assert metrics.text_scale() == 0.8 and metrics.ui_scale_percent() == 80
    sizes = {int(n) for n in re.findall(r"font-size:\s*(\d+)px", load_stylesheet())}
    assert sizes and max(sizes) < 18 and min(sizes) >= metrics.MIN_TEXT_PX
    assert "font-size: 13px" in load_stylesheet(1.0)

    env["QT_SCALE_FACTOR"] = "0.9"              # from outside: turned into text too
    metrics.apply_ui_scale(["x"])
    assert "QT_SCALE_FACTOR" not in env and metrics.text_scale() == 0.9
    metrics.apply_ui_scale(["x", "--scale", "1.25"])       # above 100 %: back to scaling
    assert env["QT_SCALE_FACTOR"] == "1.25" and metrics.text_scale() == 1.0


def test_interface_size_menu(window, monkeypatch):
    from stimstudio.gui import metrics
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items()
                                        if k != "QT_SCALE_FACTOR"})       # this session: 100 %
    menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    sub = next(a.menu() for a in menu.actions() if a.menu() and "size" in a.text())
    choice = next(a for a in sub.actions() if a.text().startswith("125 %"))
    try:
        choice.trigger()
        assert int(window.settings.value(metrics.UI_SCALE_KEY)) == 125
        assert "restart" in window.statusBar().currentMessage()
    finally:
        window.settings.remove(metrics.UI_SCALE_KEY)


def test_fit_and_keep_on_screen(qapp):
    from PyQt6.QtWidgets import QWidget
    from stimstudio.gui import metrics
    avail = metrics.available()
    w = QWidget()
    size = metrics.fit_to_screen(w, 10_000, 10_000, frac=0.9)
    assert size.width() <= avail.width() * 0.9 and size.height() <= avail.height() * 0.9
    assert metrics.fit_size(None, 300, 200).width() == 300                # room: unchanged
    w.resize(avail.width() + 400, avail.height() + 400)
    w.move(avail.right() - 50, avail.bottom() - 50)
    w.show()
    metrics.keep_on_screen(w)
    assert avail.contains(w.frameGeometry())
    w.close()


def test_paint_font_minimum(qapp):
    from stimstudio.gui.metrics import MIN_TEXT_PX, paint_font
    assert paint_font(6).pixelSize() == MIN_TEXT_PX
    assert paint_font(14, bold=True).pixelSize() == 14


def test_panic_has_its_own_bar(window):
    """A narrow window folds the transport bar's tail into its » menu: PANIC
    must not be part of it."""
    from PyQt6.QtWidgets import QToolBar
    bar = window.findChild(QToolBar, "panicBar")
    assert bar is not None and window.panic_button.parent() is bar
    assert not bar.toggleViewAction().isEnabled()


# ── the audit on virtual screens ───────────────────────────────────────────
def _ui_audit():
    spec = importlib.util.spec_from_file_location("ui_audit", ROOT / "tools" / "ui_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: always checked; "text cut" and "font too small" depend on the fonts installed,
#: so they are checked where the interface is designed (Windows, Segoe UI)
STRUCTURAL = {"window larger than screen", "window off screen", "PANIC hidden",
              "pushed out of parent", "audit error"}


def test_audit_profiles():
    """Every resolution at 100 % and at 80 % (Interface size), plus the usual scales."""
    profiles = _ui_audit().PROFILES
    for res in ("1920x1080", "1366x768", "1600x900", "2560x1440", "3840x2160"):
        assert f"{res}@100" in profiles and f"{res}@80" in profiles
        assert profiles[f"{res}@80"][2:] == (100, 80)


def test_audit_sees_contents_cut_by_a_scroll_area(qapp):
    """A scroll area that may not scroll sideways cuts what is wider than it."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget
    top = QWidget()
    lay = QVBoxLayout(top)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    body = QWidget()
    QVBoxLayout(body).addWidget(QLabel("a long line of text that does not wrap " * 4))
    area.setWidget(body)
    lay.addWidget(area)
    top.resize(200, 200)
    top.show()
    auditor = _ui_audit().Auditor("test")
    auditor.check(top, "scroll")
    kinds = {i["kind"] for i in auditor.issues.values()}
    top.close()
    assert "cut by scroll area" in kinds


@pytest.mark.parametrize("profile", ["1920x1080@175", "1366x768@100", "1920x1080@80"])
def test_audit_small_screens(profile, tmp_path):
    audit = _ui_audit()
    results = audit.run_profiles([profile], tmp_path, timeout=300)
    res = results[0]
    assert not res.get("error"), res.get("error")
    issues = [i for i in res["issues"] if sys.platform == "win32" or i["kind"] in STRUCTURAL]
    assert not issues, audit.summarize(results)
    assert len(res["shots"]) > 20
