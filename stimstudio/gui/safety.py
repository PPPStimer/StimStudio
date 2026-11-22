"""Safety check in the GUI: a monitor that re-checks every render, the Safety
dock (issues, Fix / Ignore) and the prompt before Play / export.

The analysis itself is ``core.safety`` (fast, numpy only), so it runs on the GUI
thread right after a render, coalesced by a short timer.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import safety as sc
from ..core.model import Envelope, Lane
from . import commands as cmd
from . import theme

SEVERITY_COLORS = {sc.HIGH: theme.DANGER, sc.WARN: "#d7ba7d", sc.INFO: theme.TEXT_DIM}
SEVERITY_ICONS = {sc.HIGH: "●", sc.WARN: "▲", sc.INFO: "i"}
IGNORE_KEY = "safety_ignored"
NO_ASK_KEY = "safety_no_prompt"
FIXABLE = ("jump", "cold_start", "sustained")
CHECK_DELAY_MS = 250


class SafetyMonitor(QObject):
    """Keeps an up-to-date :class:`~stimstudio.core.safety.SafetyReport`."""

    reportChanged = pyqtSignal(object)

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.rules = sc.SafetyRules()
        self.report = sc.SafetyReport()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(CHECK_DELAY_MS)
        self._timer.timeout.connect(self.recheck)
        doc.renderReady.connect(lambda *_: self._timer.start())

    def ignored(self) -> list[str]:
        return list(self.doc.project.meta.get(IGNORE_KEY, []))

    def recheck(self) -> sc.SafetyReport:
        render = self.doc.last_render
        if render is None:
            return self.report
        self.report = sc.check(render, self.doc.project, self.rules, self.ignored())
        self.reportChanged.emit(self.report)
        return self.report

    def current(self) -> sc.SafetyReport:
        """A report for the current model (renders now if the last one is stale)."""
        if not self.doc.render_is_current:
            self.doc.render_now()
        self._timer.stop()
        return self.recheck()

    # ── actions ─────────────────────────────────────────────────────────────
    def set_ignored(self, keys) -> None:
        meta = dict(self.doc.project.meta)
        meta[IGNORE_KEY] = sorted(set(keys))
        self.doc.set_project_attrs(text="Ignore safety issue", meta=meta)

    def ignore(self, issue: sc.Issue) -> None:
        self.set_ignored([*self.ignored(), issue.key])

    def reset_ignored(self) -> None:
        self.set_ignored([])

    def fix(self, issue: sc.Issue) -> str:
        """Apply the fix for *issue*; returns a short description."""
        project = self.doc.project
        track = None
        if issue.track_id is not None:
            try:
                track = project.track_by_id(issue.track_id)
            except KeyError:
                track = None
        if issue.kind in ("jump", "cold_start"):
            if track is not None:
                self.doc.set_track_attrs(track.id, text="Add slew limiter",
                                         effects=sc.apply_limiter(track.effects, issue))
                return f"Slew limiter on “{track.name}”"
            self.doc.set_project_attrs(text="Add master slew limiter",
                                       master_effects=sc.apply_limiter(project.master_effects,
                                                                       issue))
            return "Slew limiter on the master"
        if issue.kind == "sustained" and track is not None:
            keys = ("amp_l",) if track.type == "triphase" else ("amp_l", "amp_r")
            new = sc.rest_points(issue)
            self.doc.undo.beginMacro("Add rests")
            try:
                for key in keys:
                    lane = track.lanes[key].clone()
                    env = lane.envelope
                    if env is None or env.mode != "multiply":
                        env = Envelope([], "multiply", "linear")
                    lane.envelope = Envelope(sc.merge_points(env.points, new), "multiply",
                                             env.interp, True)
                    self.doc.push(cmd.SetLane(self.doc, track.id, key, lane, "Add rests"))
            finally:
                self.doc.undo.endMacro()
            return f"Rests added on “{track.name}” (amplitude envelope)"
        return ""

    def add_master_limiter(self) -> None:
        issue = sc.Issue(0, 0, "jump", sc.HIGH, "")
        self.doc.set_project_attrs(text="Add master slew limiter",
                                   master_effects=sc.apply_limiter(
                                       self.doc.project.master_effects, issue))


class SafetyPanel(QWidget):
    """The Safety dock: the issue list with Fix / Ignore."""

    issueActivated = pyqtSignal(object)        # Issue

    def __init__(self, monitor: SafetyMonitor, parent=None):
        super().__init__(parent)
        self.monitor = monitor
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["", "Time", "Issue", "Track"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 22)
        self.tree.setColumnWidth(1, 70)
        self.tree.setColumnWidth(2, 150)
        self.tree.itemSelectionChanged.connect(self._sync_buttons)
        self.tree.itemClicked.connect(lambda it, _c: self.issueActivated.emit(self._issue(it)))
        self.tree.itemDoubleClicked.connect(lambda it, _c: self.fix_selected())
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        self.fix_btn = QPushButton("Fix")
        self.fix_btn.setToolTip("Jump / no warm-up: add a slew limiter · long at the top: "
                                "add short rests to the amplitude envelope")
        self.fix_btn.clicked.connect(self.fix_selected)
        self.ignore_btn = QPushButton("Ignore")
        self.ignore_btn.clicked.connect(self.ignore_selected)
        self.reset_btn = QPushButton("Show ignored")
        self.reset_btn.clicked.connect(monitor.reset_ignored)
        for b in (self.fix_btn, self.ignore_btn, self.reset_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self._issues: list[sc.Issue] = []
        monitor.reportChanged.connect(self.set_report)
        self.set_report(monitor.report)

    def _issue(self, item) -> sc.Issue | None:
        if item is None:
            return None
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        return self._issues[idx] if idx is not None and idx < len(self._issues) else None

    def selected_issue(self) -> sc.Issue | None:
        items = self.tree.selectedItems()
        return self._issue(items[0]) if items else None

    def set_report(self, report: sc.SafetyReport) -> None:
        self._issues = list(report.issues)
        names = {t.id: t.name for t in self.monitor.doc.project.tracks}
        self.tree.clear()
        for idx, issue in enumerate(self._issues):
            it = QTreeWidgetItem(["", theme.format_time(issue.t0, 1), issue.label,
                                  names.get(issue.track_id, "master" if issue.kind != "dose"
                                            else "")])
            it.setText(0, SEVERITY_ICONS[issue.severity])
            it.setForeground(0, QBrush(QColor(SEVERITY_COLORS[issue.severity])))
            it.setToolTip(2, issue.message)
            it.setToolTip(3, issue.message)
            it.setData(0, Qt.ItemDataRole.UserRole, idx)
            self.tree.addTopLevelItem(it)
        n_ignored = len(self.monitor.ignored())
        high, warn = report.count(sc.HIGH), report.count(sc.WARN)
        if not report.issues:
            text = f"<span style='color:{theme.ACCENT}'>✓ No safety issues found.</span>"
        else:
            parts = []
            if high:
                parts.append(f"<b style='color:{theme.DANGER}'>{high} serious</b>")
            if warn:
                parts.append(f"{warn} warning{'s' if warn != 1 else ''}")
            info = report.count(sc.INFO)
            if info:
                parts.append(f"{info} note{'s' if info != 1 else ''}")
            text = " · ".join(parts)
        if n_ignored:
            text += f" <span style='color:{theme.TEXT_DIM}'>({n_ignored} ignored)</span>"
        self.summary.setText(text)
        self.reset_btn.setEnabled(n_ignored > 0)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        issue = self.selected_issue()
        self.ignore_btn.setEnabled(issue is not None)
        self.fix_btn.setEnabled(issue is not None and issue.kind in FIXABLE
                                and (issue.kind != "sustained" or issue.track_id is not None))

    def fix_selected(self) -> None:
        issue = self.selected_issue()
        if issue is None or issue.kind not in FIXABLE:
            return
        done = self.monitor.fix(issue)
        if done:
            self.summary.setText(f"Fixed: {done}. Re-checking…")

    def ignore_selected(self) -> None:
        issue = self.selected_issue()
        if issue is not None:
            self.monitor.ignore(issue)


def prompt_before(parent, monitor: SafetyMonitor, what: str = "play") -> str:
    """Ask before playing / exporting a project with serious issues.

    Returns ``"go"`` (proceed), ``"limit"`` (a master limiter was added — proceed),
    ``"show"`` (show the issues instead) or ``"cancel"``.
    """
    project = monitor.doc.project
    if project.meta.get(NO_ASK_KEY):
        return "go"
    report = monitor.current()
    high = [i for i in report.issues if i.severity == sc.HIGH]
    if not high:
        return "go"
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Safety check")
    lines = "".join(f"<li>{theme.format_time(i.t0, 1)} — {i.message}</li>" for i in high[:5])
    more = f"<p>…and {len(high) - 5} more.</p>" if len(high) > 5 else ""
    box.setText(f"<b>{len(high)} serious issue{'s' if len(high) != 1 else ''}</b> "
                f"found before you {what}:<ul>{lines}</ul>{more}")
    go = box.addButton("Play anyway" if what == "play" else "Export anyway",
                       QMessageBox.ButtonRole.AcceptRole)
    limit = box.addButton("Add master limiter", QMessageBox.ButtonRole.ActionRole)
    show = box.addButton("Show issues", QMessageBox.ButtonRole.HelpRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    dont = QCheckBox("Don't ask again for this project")
    box.setCheckBox(dont)
    box.exec()
    clicked = box.clickedButton()
    if dont.isChecked():
        meta = dict(project.meta)
        meta[NO_ASK_KEY] = True
        monitor.doc.set_project_attrs(text="Safety prompt off", meta=meta)
    if clicked is go:
        return "go"
    if clicked is limit:
        monitor.add_master_limiter()
        return "limit"
    if clicked is show:
        return "show"
    return "cancel"
