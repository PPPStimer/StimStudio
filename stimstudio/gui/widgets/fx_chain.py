"""Effect chain editor for a clip, a track or the master bus.

Each effect is a card: on/off, channel A/B switches, move up/down, remove, its
parameters, and (on tracks) automation toggles for automatable parameters —
an automated parameter gets its own lane under the track in the song editor.
Every edit replaces the target's effect list through one undoable command;
consecutive edits of the same parameter merge into one undo step.
"""

from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea,
                             QToolButton, QVBoxLayout, QWidget)

from ...effects import get_effect, list_effects
from .. import theme
from .param_form import ParamForm

CATEGORY_TITLES = (("amplitude", "Amplitude"), ("frequency", "Frequency"),
                   ("channels", "Channels"), ("time", "Time / rhythm"), ("safety", "Safety"))


def _tool(text, tip, checkable=False, name="") -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    if name:
        b.setObjectName(name)
    return b


class EffectCard(QFrame):
    def __init__(self, chain: "FxChainEditor", index: int, eff: dict):
        super().__init__()
        self.chain = chain
        self.index = index
        self.type_id = eff.get("type")
        self.setObjectName("fxCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(3)
        try:
            self.definition = get_effect(self.type_id)
            title = self.definition.label
        except KeyError:
            self.definition = None
            title = f"{self.type_id} (unknown)"
        self.enabled = QCheckBox(title)
        self.enabled.setToolTip("Enable / bypass this effect")
        self.enabled.toggled.connect(lambda on: chain.edit(index, lambda e: e.__setitem__(
            "enabled", on), "Bypass effect" if not on else "Enable effect"))
        head.addWidget(self.enabled, 1)
        self.ch_a = _tool("A", "Apply to channel A", True, "chan")
        self.ch_b = _tool("B", "Apply to channel B", True, "chan")
        for btn in (self.ch_a, self.ch_b):
            btn.toggled.connect(lambda _on: self._channels_changed())
            head.addWidget(btn)
        up = _tool("▲", "Move up")
        down = _tool("▼", "Move down")
        rm = _tool("✕", "Remove effect")
        up.clicked.connect(lambda: chain.move(index, -1))
        down.clicked.connect(lambda: chain.move(index, +1))
        rm.clicked.connect(lambda: chain.remove(index))
        for b in (up, down, rm):
            head.addWidget(b)
        lay.addLayout(head)

        self.form = ParamForm()
        self.form.paramChanged.connect(self._param_changed)
        lay.addWidget(self.form)
        self.auto_buttons: dict[str, QToolButton] = {}
        if self.definition is not None and self.definition.automatable and chain.allow_automation:
            row = QHBoxLayout()
            row.setSpacing(3)
            lbl = QLabel("Automate:")
            lbl.setStyleSheet(f"color: {theme.TEXT_DIM};")
            row.addWidget(lbl)
            for key in self.definition.automatable:
                b = _tool(self.definition.spec(key).label, "Draw this parameter over time in "
                          "a lane under the track", True, "auto")
                b.toggled.connect(lambda on, k=key: chain.set_automation(index, k, on))
                row.addWidget(b)
                self.auto_buttons[key] = b
            row.addStretch(1)
            lay.addLayout(row)
        self.sync(eff)

    def sync(self, eff: dict) -> None:
        widgets = (self.enabled, self.ch_a, self.ch_b, *self.auto_buttons.values())
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(eff.get("enabled", True))
        chs = set(eff.get("channels", ["A", "B"]))
        self.ch_a.setChecked("A" in chs)
        self.ch_b.setChecked("B" in chs)
        automation = eff.get("automation") or {}
        for key, b in self.auto_buttons.items():
            b.setChecked(key in automation)
        for w in widgets:
            w.blockSignals(False)
        if self.definition is not None:
            values = {**self.definition.default_params(), **eff}
            self.form.set_specs(self.definition.params, values)
            for key in self.definition.automatable:
                self.form.set_enabled(key, key not in automation)

    def _channels_changed(self) -> None:
        chs = [c for c, b in (("A", self.ch_a), ("B", self.ch_b)) if b.isChecked()]
        self.chain.edit(self.index, lambda e: e.__setitem__("channels", chs), "Effect channels")

    def _param_changed(self, key: str, value) -> None:
        self.chain.edit(self.index, lambda e: e.__setitem__(key, value), "Edit effect",
                        merge_key=("fx", self.index, key))


class FxChainEditor(QScrollArea):
    """Edits ``effects`` of one target: ``kind`` is "clip", "track" or "master"."""

    automationAdded = pyqtSignal(str)   # track id: show its lanes

    def __init__(self, doc, kind: str, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.kind = kind
        self.target_id: str | None = None
        self.allow_automation = kind == "track"
        self.cards: list[EffectCard] = []
        self._sig: tuple | None = None
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        self.box = QVBoxLayout(body)
        self.box.setContentsMargins(6, 6, 6, 6)
        self.box.setSpacing(6)

        top = QHBoxLayout()
        self.title = QLabel()
        self.title.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.add_btn = QToolButton()
        self.add_btn.setText("+ Add effect")
        self.add_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_btn.setMenu(self._build_add_menu())
        top.addWidget(self.title, 1)
        top.addWidget(self.add_btn)
        self.box.addLayout(top)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(6)
        self.box.addLayout(self.cards_box)
        self.empty = QLabel("No effects. Effects run top to bottom.")
        self.empty.setStyleSheet(f"color: {theme.TEXT_DIM}; padding: 8px 2px;")
        self.box.addWidget(self.empty)
        self.box.addStretch(1)
        doc.modelChanged.connect(self.refresh)
        self.refresh()

    def _build_add_menu(self) -> QMenu:
        menu = QMenu(self)
        by_cat: dict[str, list] = {}
        for d in list_effects():
            by_cat.setdefault(d.category, []).append(d)
        for cat, title in CATEGORY_TITLES:
            if cat not in by_cat:
                continue
            sub = menu.addMenu(title)
            for d in by_cat[cat]:
                a = QAction(d.label + ("  ⟿" if d.automatable and self.allow_automation else ""),
                            sub)
                a.triggered.connect(lambda _c=False, t=d.type_id: self.add(t))
                sub.addAction(a)
        return menu

    # ── target ──────────────────────────────────────────────────────────────
    def set_target(self, target_id: str | None) -> None:
        if target_id != self.target_id:
            self.target_id = target_id
            self._sig = None
        self.refresh()

    def effects(self) -> list[dict] | None:
        p = self.doc.project
        if self.kind == "master":
            return p.master_effects
        if self.target_id is None:
            return None
        if self.kind == "track":
            try:
                return p.track_by_id(self.target_id).effects
            except KeyError:
                return None
        found = p.find_clip(self.target_id)
        return found[1].effects if found else None

    def _target_name(self) -> str:
        p = self.doc.project
        if self.kind == "master":
            return "Master bus (after all tracks)"
        if self.kind == "track":
            return f"Track: {p.track_by_id(self.target_id).name}"
        return f"Clip: {p.find_clip(self.target_id)[1].name or '(unnamed)'}"

    def _commit(self, effects: list[dict], text: str, merge_key=None) -> None:
        if self.kind == "master":
            self.doc.set_project_attrs(merge_key=merge_key, text=text, master_effects=effects)
        elif self.kind == "track":
            self.doc.set_track_attrs(self.target_id, merge_key=merge_key, text=text,
                                     effects=effects)
        else:
            self.doc.set_clip_attrs(self.target_id, merge_key=merge_key, text=text,
                                    effects=effects)

    # ── edits ───────────────────────────────────────────────────────────────
    def edit(self, index: int, fn, text: str, merge_key=None) -> None:
        effects = copy.deepcopy(self.effects())
        fn(effects[index])
        self._commit(effects, text, merge_key)

    def add(self, type_id: str) -> None:
        if self.effects() is None:
            return
        effects = copy.deepcopy(self.effects())
        effects.append(get_effect(type_id).make())
        self._commit(effects, f"Add {get_effect(type_id).label}")

    def remove(self, index: int) -> None:
        effects = copy.deepcopy(self.effects())
        del effects[index]
        self._commit(effects, "Remove effect")

    def move(self, index: int, delta: int) -> None:
        effects = copy.deepcopy(self.effects())
        j = index + delta
        if 0 <= j < len(effects):
            effects[index], effects[j] = effects[j], effects[index]
            self._commit(effects, "Reorder effects")

    def set_automation(self, index: int, key: str, on: bool) -> None:
        """Start automating a parameter (flat at its current value) or stop."""
        def fn(eff):
            definition = get_effect(eff["type"])
            automation = eff.setdefault("automation", {})
            if on:
                value = eff.get(key, definition.default_params()[key])
                automation[key] = {"points": [[0.0, round(definition.to_unit(key, value), 4)]],
                                   "interp": "linear", "enabled": True}
            else:
                automation.pop(key, None)
                if not automation:
                    eff.pop("automation", None)
        self.edit(index, fn, "Automate parameter" if on else "Remove automation")
        if on and self.kind == "track":
            self.automationAdded.emit(self.target_id)

    # ── view ────────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        effects = self.effects()
        has_target = effects is not None
        self.add_btn.setEnabled(has_target)
        if not has_target:
            self.title.setText({"clip": "Select one clip", "track": "Select a track"}
                               .get(self.kind, ""))
            self._rebuild([])
            return
        self.title.setText(self._target_name())
        sig = tuple(e.get("type") for e in effects)
        if sig != self._sig:
            self._rebuild(effects)
            self._sig = sig
        else:
            for card, eff in zip(self.cards, effects):
                card.sync(eff)
        self.empty.setVisible(not effects)

    def _rebuild(self, effects) -> None:
        for card in self.cards:
            card.deleteLater()
        self.cards = [EffectCard(self, i, eff) for i, eff in enumerate(effects)]
        for card in self.cards:
            self.cards_box.addWidget(card)
        self.empty.setVisible(not effects)
        self._sig = tuple(e.get("type") for e in effects) if effects else None


class EffectsPanel(QWidget):
    """The Effects dock: Clip / Track / Master chains."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QTabWidget
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.clip = FxChainEditor(doc, "clip")
        self.track = FxChainEditor(doc, "track")
        self.master = FxChainEditor(doc, "master")
        self.tabs.addTab(self.track, "Track")
        self.tabs.addTab(self.clip, "Clip")
        self.tabs.addTab(self.master, "Master")
        lay.addWidget(self.tabs)

    def set_clip_selection(self, clip_ids: list[str]) -> None:
        self.clip.set_target(clip_ids[0] if len(clip_ids) == 1 else None)

    def set_track(self, track_id: str) -> None:
        self.track.set_target(track_id or None)
