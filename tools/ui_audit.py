r"""Interface audit at other screen sizes and DPI scales, on any one monitor.

Opens the main window (every dock, the Pattern editor and Output pages) and every
dialog / Tools window on a virtual screen, saves screenshots and lists what is
unreadable: text that does not fit its widget, widgets pushed out of their
parent, contents cut by a scroll area that does not scroll, windows larger than
the screen, a hidden PANIC button, fonts that are too small.

    python tools/ui_audit.py                       # every profile, 4 at a time
    python tools/ui_audit.py -p 1920x1080@150      # one profile (repeatable)
    python tools/ui_audit.py --fail                # exit code 1 when anything is found
    python tools/ui_audit.py --list                # the profiles

Profiles: 1920×1080, 1366×768, 1600×900, 2560×1440 and 3840×2160, each at
100 % and at the Windows display scales people use on it, and each at 80 %
(Windows 100 % + View → Interface size 80 %: smaller text).

Each profile runs in its own process on Qt's offscreen platform with a virtual
screen (QT_QPA_PLATFORM=offscreen:configfile=…): ``logicalDpi`` = 96 × scale is
exactly what Windows reports at that display scale, so Qt lays out on the same
logical screen (1920×1080 at 150 % → 1280×720) with the same device pixel ratio;
Interface size is applied by the app's own ``metrics.apply_ui_scale``. A Windows taskbar
(40 px) is taken off the screen height. Screenshots are in device pixels, as
they would look on that monitor. Results: <out>/index.html.

The offscreen platform draws with the Fusion style; the real app uses the
native Windows style, whose metrics differ a little (e.g. wider spin boxes).

Settings go to a temporary ini file (a first run: no saved window layout),
autosave and presets to temporary folders: the real ones are never touched.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKBAR_PX = 40          # logical px, Windows 10

#: name → (native width, native height, Windows display scale %, Interface size %).
#: "@80" is Windows at 100 % with View → Interface size 80 % (Windows itself
#: has no 80 %): smaller text, set the way the app sets it.
PROFILES = {
    "1920x1080@80": (1920, 1080, 100, 80),
    "1920x1080@100": (1920, 1080, 100, 100),
    "1920x1080@125": (1920, 1080, 125, 100),
    "1920x1080@150": (1920, 1080, 150, 100),
    "1920x1080@175": (1920, 1080, 175, 100),
    "1366x768@80": (1366, 768, 100, 80),
    "1366x768@100": (1366, 768, 100, 100),
    "1600x900@80": (1600, 900, 100, 80),
    "1600x900@100": (1600, 900, 100, 100),
    "1600x900@125": (1600, 900, 125, 100),
    "2560x1440@80": (2560, 1440, 100, 80),
    "2560x1440@100": (2560, 1440, 100, 100),
    "2560x1440@125": (2560, 1440, 125, 100),
    "2560x1440@150": (2560, 1440, 150, 100),
    "3840x2160@80": (3840, 2160, 100, 80),
    "3840x2160@100": (3840, 2160, 100, 100),
    "3840x2160@150": (3840, 2160, 150, 100),
    "3840x2160@200": (3840, 2160, 200, 100),
}
#: fonts smaller than this (logical px) are reported
MIN_FONT_PX = 10
#: px of slack before a text is reported as cut
SLACK = 2


def describe_profile(name: str) -> str:
    w, h, win, ui = PROFILES[name]
    f = win * ui / 10000.0
    how = f"Windows {win} %" + (f" + Interface size {ui} %" if ui != 100 else "")
    if ui < 100:                       # smaller text only: the device pixel ratio stays
        f = win / 100.0
        how += " (smaller text)"
    return f"{w}×{h}, {how} → logical {round(w / f)}×{round(h / f)}"


# ════════════════════════════════════════════════════════════════════ parent
def run_profile(name: str, out: Path, timeout: float) -> dict:
    w, h, win, ui = PROFILES[name]
    pdir = out / name
    pdir.mkdir(parents=True, exist_ok=True)
    for old in list(pdir.glob("*.png")) + list(pdir.glob("result.json")):
        old.unlink()
    f = win / 100.0
    screen = {"screens": [{
        "name": name, "x": 0, "y": 0, "width": w,
        "height": h - round(TASKBAR_PX * f),
        "logicalDpi": 96 * f, "logicalBaseDpi": 96, "dpr": 1}]}
    (pdir / "screen.json").write_text(json.dumps(screen), encoding="utf-8")
    env = dict(os.environ)
    # a relative path: a drive letter's colon would end the platform argument
    env["QT_QPA_PLATFORM"] = "offscreen:configfile=screen.json"
    if sys.platform == "win32":
        env["QT_QPA_FONTDIR"] = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    env["STIMSTUDIO_SOFTWARE_RENDER"] = "1"
    for k in ("QT_SCALE_FACTOR", "QT_FONT_DPI", "QT_SCREEN_SCALE_FACTORS"):
        env.pop(k, None)
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", name],
                              cwd=pdir, env=env, timeout=timeout,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        log = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as e:
        log = f"timeout after {timeout:.0f} s\n{e.stdout or ''}{e.stderr or ''}"
    (pdir / "log.txt").write_text(log, encoding="utf-8")
    rfile = pdir / "result.json"
    if rfile.exists():
        res = json.loads(rfile.read_text(encoding="utf-8"))
    else:
        res = {"profile": name, "issues": [], "shots": [],
               "error": "the audit process failed — see log.txt\n" + log[-3000:]}
    res["seconds"] = round(time.time() - t0, 1)
    res["describe"] = describe_profile(name)
    return res


def run_profiles(names: list[str], out: Path, timeout: float, jobs: int = 1) -> list[dict]:
    """Every profile in its own process, *jobs* at a time; results in *names* order."""
    from concurrent.futures import ThreadPoolExecutor
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(run_profile, n, out, timeout): n for n in names}
        for fut in futures:
            name = futures[fut]
            res = results[name] = fut.result()
            print(f"── {name:15} {res['describe']}: {len(res['issues'])} issue(s), "
                  f"{len(res['shots'])} screenshot(s), {res['seconds']} s"
                  + (f"  ERROR: {res['error'].splitlines()[0]}" if res.get("error") else ""),
                  flush=True)
    return [results[n] for n in names]


def summarize(results: list[dict]) -> str:
    lines = []
    for res in results:
        lines.append(f"\n=== {res['profile']}  ({res.get('describe', '')}, "
                     f"dpr {res.get('dpr', '?')})")
        if res.get("error"):
            lines.append("  ERROR: " + res["error"].strip().replace("\n", "\n         "))
        by_kind: dict[str, list] = {}
        for it in res["issues"]:
            by_kind.setdefault(it["kind"], []).append(it)
        for kind, items in sorted(by_kind.items()):
            lines.append(f"  [{kind}] {len(items)}")
            for it in items[:40]:
                count = f" ×{it['count']}" if it.get("count", 1) > 1 else ""
                lines.append(f"    {it['scenario']}: {it['where']} — {it['detail']}{count}")
            if len(items) > 40:
                lines.append(f"    … {len(items) - 40} more")
    return "\n".join(lines)


def write_html(results: list[dict], out: Path) -> Path:
    e = html.escape
    parts = ["<!doctype html><meta charset=utf-8><title>StimStudio UI audit</title>",
             "<style>body{background:#1e1e1e;color:#ccc;font:13px Segoe UI,sans-serif;"
             "margin:16px}h2{color:#9cdcfe;margin-top:32px}a{color:#4ec9b0}"
             "table{border-collapse:collapse}td,th{border-bottom:1px solid #333;"
             "padding:2px 8px;text-align:left;vertical-align:top}"
             ".k{color:#dcdcaa}.shots{display:flex;flex-wrap:wrap;gap:12px}"
             "figure{margin:0}figcaption{color:#888}img{max-width:640px;max-height:420px;"
             "border:1px solid #444;background:#000}</style>",
             "<h1>StimStudio UI audit</h1><ul>"]
    for res in results:
        parts.append(f"<li><a href='#{e(res['profile'])}'>{e(res['profile'])}</a> — "
                     f"{len(res['issues'])} issue(s)</li>")
    parts.append("</ul>")
    for res in results:
        p = res["profile"]
        parts.append(f"<h2 id='{e(p)}'>{e(p)} — {e(res.get('describe', ''))}, "
                     f"device pixel ratio {e(str(res.get('dpr', '?')))}</h2>")
        if res.get("error"):
            parts.append(f"<pre>{e(res['error'])}</pre>")
        if res["issues"]:
            parts.append("<table><tr><th>kind</th><th>window</th><th>widget</th>"
                         "<th>detail</th></tr>")
            for it in res["issues"]:
                count = f" ×{it['count']}" if it.get("count", 1) > 1 else ""
                parts.append(f"<tr><td class=k>{e(it['kind'])}</td><td>{e(it['scenario'])}</td>"
                             f"<td>{e(it['where'])}</td><td>{e(it['detail'])}{count}</td></tr>")
            parts.append("</table>")
        parts.append("<div class=shots>")
        for shot in res["shots"]:
            src = f"{p}/{shot}"
            parts.append(f"<figure><a href='{e(src)}'><img src='{e(src)}' loading=lazy></a>"
                         f"<figcaption>{e(shot)}</figcaption></figure>")
        parts.append("</div>")
    path = out / "index.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# ═════════════════════════════════════════════════════════════════════ child
class Auditor:
    def __init__(self, profile: str):
        from PyQt6.QtWidgets import QApplication
        self.app = QApplication.instance()
        self.profile = profile
        self.issues: dict[tuple, dict] = {}
        self.shots: list[str] = []
        self.paint_fonts: dict[str, float] = {}

    # ── helpers ──
    def settle(self, ms: int = 150):
        from PyQt6.QtCore import QElapsedTimer
        t = QElapsedTimer()
        t.start()
        while t.elapsed() < ms:
            self.app.processEvents()
            time.sleep(0.01)

    def shot(self, widget, name: str):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name) + ".png"
        if widget.grab().save(safe):
            self.shots.append(safe)

    def add(self, scenario: str, kind: str, where: str, detail: str):
        key = (scenario, kind, where, detail)
        if key in self.issues:
            self.issues[key]["count"] += 1
        else:
            self.issues[key] = {"scenario": scenario, "kind": kind, "where": where,
                                "detail": detail, "count": 1}

    @staticmethod
    def describe(w) -> str:
        from PyQt6.QtWidgets import (QAbstractButton, QComboBox, QDockWidget, QGroupBox,
                                     QLabel, QLineEdit, QAbstractSpinBox)
        chain = []
        node = w
        while node is not None and len(chain) < 4:
            name = type(node).__name__
            text = ""
            if isinstance(node, (QLabel, QAbstractButton)):
                text = node.text()
            elif isinstance(node, QGroupBox):
                text = node.title()
            elif isinstance(node, QDockWidget):
                text = node.windowTitle()
            elif isinstance(node, QComboBox):
                text = node.currentText()
            elif isinstance(node, (QLineEdit, QAbstractSpinBox)):
                text = node.text()
            text = " ".join(text.split())[:32]
            label = name + (f"#{node.objectName()}" if node.objectName()
                            and not node.objectName().startswith("qt_") else "")
            if text:
                label += f"({text!r})"
            chain.append(label)
            if isinstance(node, QDockWidget) or node.isWindow():
                break
            node = node.parentWidget()
        return " < ".join(chain)

    @staticmethod
    def describe_row(w) -> str:
        """A row of a panel by what it holds: its classes and first texts."""
        from PyQt6.QtWidgets import QAbstractButton, QComboBox, QLabel, QWidget
        parts = []
        for c in [w] + w.findChildren(QWidget):
            if isinstance(c, (QLabel, QAbstractButton)) and c.text():
                parts.append(repr(" ".join(c.text().split())[:24]))
            elif isinstance(c, QComboBox):
                parts.append(f"QComboBox({c.currentText()[:24]!r}, min "
                             f"{c.minimumSizeHint().width()}px)")
            if len(parts) >= 3:
                break
        return type(w).__name__ + ("[" + ", ".join(parts) + "]" if parts else "")

    # ── checks ──
    def check(self, top, scenario: str):
        from PyQt6.QtGui import QFontInfo
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QAbstractSpinBox,
                                     QCheckBox, QComboBox, QLabel, QLineEdit, QRadioButton,
                                     QScrollArea, QStyle, QStyleOptionComboBox, QWidget)
        from PyQt6.QtWidgets import QDockWidget, QMainWindow
        self.settle(60)
        if top.isWindow():
            self.check_window(top, scenario)

        for w in top.findChildren(QWidget):
            if not w.isVisibleTo(top) or w.isWindow() or w.width() <= 0:
                continue
            if w.visibleRegion().isEmpty() and not isinstance(w.parentWidget(),
                                                              QAbstractScrollArea):
                continue                    # a dock in a tab behind, a hidden page
            if isinstance(w, QDockWidget) and isinstance(top, QMainWindow):
                continue                    # placed by the dock layout, checked one by one
            name = w.objectName()
            if name.startswith("qt_"):
                continue
            where = None

            # out of the parent
            # (only in a layout: absolutely placed children scroll by design)
            parent = w.parentWidget()
            if parent is not None and QWidget.layout(parent) is not None \
                    and not parent.objectName().startswith("qt_scrollarea") \
                    and not isinstance(parent, QAbstractScrollArea):
                g, r = w.geometry(), parent.rect()
                over_x = max(0, g.right() - r.right(), r.left() - g.left())
                over_y = max(0, g.bottom() - r.bottom(), r.top() - g.top())
                if over_x > SLACK or over_y > SLACK:
                    where = self.describe(w)
                    self.add(scenario, "pushed out of parent", where,
                             f"{over_x}px horizontally, {over_y}px vertically past "
                             f"{type(parent).__name__} {r.width()}×{r.height()}")

            # contents wider (or taller) than a scroll area that does not scroll that way
            if isinstance(w, QScrollArea) and w.widget() is not None and w.widgetResizable():
                body = w.widget()
                need = body.minimumSizeHint().expandedTo(body.minimumSize())
                view = w.viewport().size()
                off = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                for axis, n, v, policy in (("wide", need.width(), view.width(),
                                            w.horizontalScrollBarPolicy()),
                                           ("tall", need.height(), view.height(),
                                            w.verticalScrollBarPolicy())):
                    if policy == off and n > v + SLACK:
                        kids = [c for c in body.findChildren(QWidget)
                                if c.parentWidget() is body and c.isVisibleTo(body)]
                        size = (lambda c: c.minimumSizeHint().width()) if axis == "wide" \
                            else (lambda c: c.minimumSizeHint().height())
                        widest = sorted(kids, key=size, reverse=True)[:2]
                        who = "; ".join(f"{self.describe_row(c)} {size(c)}px" for c in widest)
                        self.add(scenario, "cut by scroll area", self.describe(w),
                                 f"contents {n}px {axis}, room {v}px, no scroll bar"
                                 + (f" (widest: {who})" if who else ""))

            # too small to hold its text
            need = have = None
            text = ""
            if isinstance(w, QLabel):
                text = w.text()
                if text and w.pixmap().isNull():
                    if w.wordWrap():
                        hfw = w.heightForWidth(w.width())
                        if hfw > w.height() + SLACK:
                            self.add(scenario, "text cut", where or self.describe(w),
                                     f"wrapped text needs {hfw}px height, has {w.height()}px")
                    else:
                        msh = w.minimumSizeHint()
                        need, have = msh, w.size()
            elif isinstance(w, (QCheckBox, QRadioButton)) and w.text():
                text = w.text()
                need, have = w.minimumSizeHint(), w.size()
            elif isinstance(w, QAbstractButton) and w.text():
                # the glyphs themselves (style padding may be trimmed on purpose)
                text = w.text().replace("&&", "\0").replace("&", "").replace("\0", "&")
                fm = w.fontMetrics()
                tw = max(fm.horizontalAdvance(line) for line in text.split("\n"))
                if not w.icon().isNull():
                    tw += w.iconSize().width() + 4
                if tw > w.width() - 2:
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"needs {tw}px for the text, button is {w.width()}px")
                if fm.height() > w.height():
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"text {fm.height()}px high, button {w.height()}px")
            elif isinstance(w, QComboBox) and w.currentText():
                text = w.currentText()
                opt = QStyleOptionComboBox()
                w.initStyleOption(opt)
                field = w.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                                 QStyle.SubControl.SC_ComboBoxEditField, w)
                tw = w.fontMetrics().horizontalAdvance(text)
                if not w.iconSize().isEmpty() and not w.itemIcon(w.currentIndex()).isNull():
                    tw += w.iconSize().width() + 4
                if tw > field.width() + SLACK:
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"needs {tw}px for the text, field is {field.width()}px")
                if w.fontMetrics().height() > field.height() + SLACK:
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"text {w.fontMetrics().height()}px high, field {field.height()}px")
            elif isinstance(w, QAbstractSpinBox):
                edit = w.findChild(QLineEdit)
                text = w.text()
                if edit is not None and text:
                    tw = edit.fontMetrics().horizontalAdvance(text) + 4
                    if tw > edit.width() + SLACK:
                        self.add(scenario, "text cut", where or self.describe(w),
                                 f"value needs {tw}px, field is {edit.width()}px")
            elif isinstance(w, QLineEdit) and w.isReadOnly() and w.text() \
                    and not isinstance(parent, (QAbstractSpinBox, QComboBox)):
                text = w.text()
                tw = w.fontMetrics().horizontalAdvance(text) + 8
                if tw > w.width() + SLACK:
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"read-only text needs {tw}px, field is {w.width()}px")
            if need is not None and need.isValid():
                dx, dy = need.width() - have.width(), need.height() - have.height()
                if dx > SLACK or dy > SLACK:
                    self.add(scenario, "text cut", where or self.describe(w),
                             f"needs {need.width()}×{need.height()}, has "
                             f"{have.width()}×{have.height()}")

            # tiny font
            if text:
                px = QFontInfo(w.font()).pixelSize()
                if 0 < px < MIN_FONT_PX:
                    self.add(scenario, "font too small", self.describe(w), f"{px}px")

    def check_window(self, top, scenario: str):
        from PyQt6.QtWidgets import QDockWidget, QWidget
        avail = top.screen().availableGeometry()
        frame = top.frameGeometry()
        too_big = frame.width() > avail.width() + SLACK or frame.height() > avail.height() + SLACK
        if too_big:
            self.add(scenario, "window larger than screen", self.describe(top),
                     f"{frame.width()}×{frame.height()} on {avail.width()}×{avail.height()}")
        elif not avail.adjusted(-SLACK, -SLACK, SLACK, SLACK).contains(frame):
            self.add(scenario, "window off screen", self.describe(top),
                     f"at {frame.x()},{frame.y()} {frame.width()}×{frame.height()} on "
                     f"{avail.width()}×{avail.height()}")
        mins = top.minimumSizeHint().expandedTo(top.minimumSize())
        if mins.width() > avail.width() or mins.height() > avail.height():
            # who asks for that much: the largest minimums among the docks and panels
            parts = []
            for d in top.findChildren(QDockWidget):
                if d.isVisibleTo(top):
                    m = d.minimumSizeHint().expandedTo(d.minimumSize())
                    parts.append((m.height(), m.width(), d.windowTitle()))
            central = getattr(top, "centralWidget", lambda: None)()
            if isinstance(central, QWidget):
                m = central.minimumSizeHint().expandedTo(central.minimumSize())
                parts.append((m.height(), m.width(), "central widget"))
            parts.sort(reverse=True)
            who = ", ".join(f"{t} {w}×{h}" for h, w, t in parts[:5])
            self.add(scenario, "window larger than screen", self.describe(top),
                     f"minimum {mins.width()}×{mins.height()} on "
                     f"{avail.width()}×{avail.height()}" + (f" (minimums: {who})" if who else ""))

    def check_paint_fonts(self):
        for where, px in sorted(self.paint_fonts.items()):
            if px < MIN_FONT_PX:
                self.add("painted text", "font too small", where, f"{px:g}px")

    # ── scenarios ──
    def scenario(self, name: str, fn):
        try:
            fn()
        except Exception:
            self.add(name, "audit error", name, traceback.format_exc(limit=4).strip()[-600:])

    def run(self, tmp: Path) -> dict:
        from PyQt6.QtWidgets import QDockWidget

        from stimstudio.demo import build_demo_project
        from stimstudio.gui.document import ProjectDocument
        from stimstudio.gui.main_window import MainWindow

        window = MainWindow(ProjectDocument(build_demo_project()),
                            user_preset_dir=tmp / "presets", autosave_dir=tmp / "autosave")
        self.window = window
        window.show()                                   # as the app does: its own first size
        window.doc.wait_for_render()
        self.settle(400)                                # first-run dock layout (a timer)
        screen = window.screen()
        geo = screen.geometry()
        result_head = {"logical": f"{geo.width()}×{geo.height()}",
                       "dpr": round(screen.devicePixelRatio(), 3)}

        def main():
            self.check(window, "main window")
            panic = getattr(window, "panic_button", None)
            if panic is None or not panic.isVisible() or panic.visibleRegion().isEmpty():
                self.add("main window", "PANIC hidden", "PANIC button",
                         "not in sight (folded into a toolbar's » menu?)")
            self.shot(window, "00 main window")
        self.scenario("main window", main)

        # every dock, raised in its tab group
        docks = [d for d in window.findChildren(QDockWidget) if d.isVisible()
                 or window.tabifiedDockWidgets(d)]
        for dock in docks:
            def one(dock=dock):
                dock.show()
                dock.raise_()
                self.settle(120)
                title = dock.windowTitle()
                self.check(dock, f"dock: {title}")
                self.shot(dock, f"10 dock {title}")
            self.scenario(f"dock {dock.windowTitle()}", one)

        self.scenario("pattern editor", lambda: self.pattern_pages(window))
        self.scenario("output devices", lambda: self.output_devices(window))
        self.scenario("track headers", lambda: self.track_headers(window))
        self.dialogs(window, tmp)
        self.check_paint_fonts()

        window.doc.undo.setClean()
        window.close()
        return result_head

    def pattern_pages(self, window):
        """Pattern editor + Inspector for one clip of each kind."""
        seen = set()
        dock = window.pattern_dock
        for track in window.doc.project.tracks:
            for clip in track.clips:
                kind = (track.type, clip.source.kind)
                if kind in seen:
                    continue
                seen.add(kind)
                window.editor.select_clips([clip.id])
                dock.show()
                dock.raise_()
                self.settle(200)
                label = f"{track.type}-{clip.source.kind}"
                self.check(dock, f"pattern editor ({label})")
                self.shot(dock, f"20 pattern {label}")
                for d in window.findChildren(type(dock)):
                    if d.windowTitle() == "Inspector":
                        d.raise_()
                        self.settle(100)
                        self.check(d, f"inspector ({label})")
                        self.shot(d, f"21 inspector {label}")

    def output_devices(self, window):
        panel = window.device_panel
        combo = panel.device
        dock = next(d for d in window.findChildren(type(window.pattern_dock))
                    if d.widget() is panel)
        dock.raise_()
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            self.settle(150)
            label = combo.itemData(i)
            self.check(dock, f"output panel ({label})")
            self.shot(dock, f"30 output {label}")
        combo.setCurrentIndex(combo.findData("dryrun"))

    def track_headers(self, window):
        ed = window.editor
        for track in window.doc.project.tracks:
            try:
                ed.toggle_expanded(track.id)
            except Exception:
                pass
        self.settle(250)
        self.check(window, "main window (tracks expanded)")
        self.shot(window, "01 main window, tracks expanded")

    def dialogs(self, window, tmp: Path):
        import numpy as np

        def show(name, make):
            def go():
                dlg = make()
                dlg.show()
                self.settle(250)
                self.check(dlg, name)
                self.shot(dlg, f"40 {name}")
                dlg.close()
                dlg.deleteLater()
                self.settle(30)
            self.scenario(name, go)

        from stimstudio.gui.export_dialog import ExportDialog
        for fmt in (None, "wav", "funscript"):
            show(f"export {fmt or 'default'}", lambda fmt=fmt: ExportDialog(window, fmt=fmt))

        script = tmp / "audit.funscript"
        script.write_text(json.dumps({"actions": [{"at": 0, "pos": 0}, {"at": 1000, "pos": 100},
                                                  {"at": 2000, "pos": 0}]}), encoding="utf-8")
        from stimstudio.gui.funscript_import import FunscriptImportDialog
        show("funscript import", lambda: FunscriptImportDialog([str(script)], "standard", window))
        show("funscript import triphase",
             lambda: FunscriptImportDialog([str(script)], "triphase", window))

        csv = tmp / "beats.csv"
        csv.write_text("Kick,Kick mark,Snare,Snare mark\n0.500,,0.750,\n2.000,pattern start,,\n"
                       "2.500,,,\n3.500,pattern end,,\n", encoding="utf-8")
        from stimstudio.gui.beat_import import BeatImportDialog
        show("beat import", lambda: BeatImportDialog([str(csv)], 2.0, window))

        from stimstudio.gui.peak_marks import PeakMarksDialog
        sig = np.abs(np.sin(np.linspace(0, 40, 400)))
        show("peak marks", lambda: PeakMarksDialog(sig, "Clip · envelope", window))

        from stimstudio.gui.session_dialog import SessionDialog
        show("session", lambda: SessionDialog(window, 120.0, 300.0, 0.0, markers=[50.0]))

        def audio_import():
            from stimstudio.audio import triphase_probe as tp
            from stimstudio.gui.audio_import import AudioImportDialog
            sr = 8000
            t = np.arange(sr * 5) / sr
            x = (np.sin(2 * np.pi * 700 * t) * (0.5 + 0.4 * np.sin(np.pi * t))).astype(np.float32)
            data = np.stack([x, x], 1)
            probe = tp.probe([data], sr, len(data) / sr, [(0.0, len(data) / sr)])
            return AudioImportDialog("music.wav", probe, window)
        show("audio import", audio_import)

        from stimstudio.gui.marker_dialog import MarkerDialog
        show("marker", lambda: MarkerDialog(window, "New marker", "Intro", "#569cd6"))

        from stimstudio.gui.media_dialog import CollectDialog, ProjectMediaDialog
        show("project media", lambda: ProjectMediaDialog(window.doc, window))
        def collect():
            if not window.doc.path:
                window.doc.save(str(tmp / "audit.stimproj"))
            return CollectDialog(window.doc, window)
        show("collect media", collect)

        from stimstudio.gui.help import ShortcutsDialog
        show("shortcuts", lambda: ShortcutsDialog(window))

        from stimstudio.gui.widgets.triphase_cal import TriphaseCalibrationDialog
        show("triphase calibration", lambda: TriphaseCalibrationDialog(window.doc, window))

        from stimstudio.gui.donate_dialog import DonateDialog, HeroDialog
        show("hero", lambda: HeroDialog("SomeLongDonorNickname", window))
        show("donate", lambda: DonateDialog({
            "title": "Support development",
            "goal": {"raised": 120, "target": 500, "currency": "USD", "note": "Next goal"},
            "platforms": [{"name": n, "url": "https://example.org"} for n in
                          ("Boosty", "Patreon", "Ko-fi", "Buy me a coffee")],
            "wallets": [{"name": n, "address": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"}
                        for n in ("BTC", "USDT TRC20")]}, window))

        def video_analysis():
            import av  # noqa: F401  (PyAV: skip the dialog without it)
            from stimstudio.gui.video_analysis import VideoAnalysisDialog
            path = tmp / "scenes.mp4"
            with av.open(str(path), "w") as c:
                s = c.add_stream("mpeg4", rate=25)
                s.width, s.height, s.pix_fmt = 160, 96, "yuv420p"
                for i in range(50):
                    g = np.full((96, 160, 3), (i * 5) % 255, np.uint8)
                    for pk in s.encode(av.VideoFrame.from_ndarray(g, format="rgb24")):
                        c.mux(pk)
                for pk in s.encode():
                    c.mux(pk)
            return VideoAnalysisDialog(str(path), 2.0, window, True, True)
        try:
            import av  # noqa: F401
            show("video analysis", video_analysis)
        except ImportError:
            pass

        from stimstudio.gui.tools import TOOLS
        for tool in TOOLS:
            def go(tool=tool):
                win = window.show_tool(tool.id)
                self.settle(300)
                self.check(win, f"tool: {tool.id}")
                self.shot(win, f"50 tool {tool.id}")
                win.close()
            self.scenario(f"tool {tool.id}", go)


def watch_paint_fonts(store: dict):
    """Record every font size set from StimStudio code (QPainter text, mostly),
    by the calling line, so that tiny painted text is reported too."""
    from PyQt6.QtGui import QFont

    def caller():
        f = sys._getframe(2)
        while f is not None and "stimstudio" not in f.f_code.co_filename:
            f = f.f_back
        if f is None:
            return None
        rel = os.path.relpath(f.f_code.co_filename, ROOT).replace("\\", "/")
        return f"{rel}:{f.f_lineno}"

    orig_px, orig_pt = QFont.setPixelSize, QFont.setPointSizeF
    orig_pti = QFont.setPointSize

    def set_px(self, px):
        where = caller()
        if where:
            store[where] = min(store.get(where, 1e9), float(px))
        return orig_px(self, px)

    def set_pt(self, pt, _orig=orig_pt):
        where = caller()
        if where:
            store[where] = min(store.get(where, 1e9), float(pt) * 96 / 72)
        return _orig(self, pt)

    QFont.setPixelSize = set_px
    QFont.setPointSizeF = set_pt
    QFont.setPointSize = lambda self, pt: set_pt(self, pt, orig_pti)


def child(profile: str) -> int:
    sys.path.insert(0, str(ROOT))
    tmp = Path(tempfile.mkdtemp(prefix="stimstudio-ui-audit-"))
    os.environ["STIMSTUDIO_AUTOSAVE_DIR"] = str(tmp / "autosave")
    out = {"profile": profile, "issues": [], "shots": []}
    try:
        ui = PROFILES[profile][3]
        if ui != 100:              # View → Interface size, the way the app applies it
            from stimstudio.gui.metrics import apply_ui_scale
            apply_ui_scale(["audit", "--scale", f"{ui / 100.0:g}"])
        from PyQt6.QtCore import QSettings

        from stimstudio.gui.gl import configure_opengl
        configure_opengl()
        from PyQt6.QtWidgets import QApplication
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp / "ini"))
        app = QApplication([sys.argv[0]])
        app.setApplicationName("StimStudio")
        app.setOrganizationName("StimStudio")
        from stimstudio.gui.theme import load_stylesheet
        app.setStyleSheet(load_stylesheet())
        auditor = Auditor(profile)
        watch_paint_fonts(auditor.paint_fonts)
        out.update(auditor.run(tmp))
        out["issues"] = list(auditor.issues.values())
        out["shots"] = auditor.shots
        out["paint_fonts"] = auditor.paint_fonts
    except Exception:
        out["error"] = traceback.format_exc()
    Path("result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)   # skip Qt / thread teardown; the result is written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-p", "--profile", action="append", choices=sorted(PROFILES),
                    help="profile to run (default: all)")
    ap.add_argument("-o", "--out", default=str(ROOT / "ui_audit"), help="output folder")
    ap.add_argument("--fail", action="store_true", help="exit 1 if any issue is found")
    ap.add_argument("--list", action="store_true", help="list the profiles")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per profile")
    ap.add_argument("-j", "--jobs", type=int, default=min(4, os.cpu_count() or 1),
                    help="profiles run at once (default: up to 4)")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):     # a cp1251 / cp866 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if args.child:
        return child(args.child)
    if args.list:
        for name in PROFILES:
            print(f"{name:16} {describe_profile(name)}")
        return 0
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = run_profiles(args.profile or list(PROFILES), out, args.timeout, args.jobs)
    report = summarize(results)
    (out / "report.txt").write_text(report, encoding="utf-8")
    page = write_html(results, out)
    print(report)
    print(f"\nScreenshots and report: {page}")
    bad = any(r["issues"] or r.get("error") for r in results)
    return 1 if (args.fail and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
