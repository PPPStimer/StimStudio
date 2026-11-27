"""Marker window: name and colour of a timeline flag, in one undo step."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QButtonGroup, QColorDialog, QDialog, QDialogButtonBox, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QToolButton, QVBoxLayout)

from . import theme

DEFAULT_COLOR = "#dcdcaa"
#: The swatches offered, the model's default first.
SWATCHES = (DEFAULT_COLOR, "#4ec9b0", "#569cd6", "#c586c0", "#ce9178", "#d16969",
            "#b5cea8", "#9cdcfe", "#f14c4c", "#e8e8e8")
LAST_COLOR_KEY = "markers/last_color"


class MarkerDialog(QDialog):
    def __init__(self, parent=None, title: str = "Marker", name: str = "",
                 color: str = DEFAULT_COLOR):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._color = QColor(color).name() if QColor(color).isValid() else DEFAULT_COLOR
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("Name"))
        self.name = QLineEdit(name)
        self.name.selectAll()
        lay.addWidget(self.name)

        lay.addWidget(QLabel("Colour"))
        row = QHBoxLayout()
        row.setSpacing(4)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.swatches: dict[str, QToolButton] = {}
        for c in SWATCHES:
            b = QToolButton(checkable=True)
            b.setFixedSize(24, 24)
            b.setToolTip(c)
            b.setStyleSheet(self._swatch_css(c))
            b.clicked.connect(lambda _c=False, col=c: self.set_color(col))
            self.group.addButton(b)
            row.addWidget(b)
            self.swatches[c] = b
        self.other = QPushButton("Other…")
        self.other.setObjectName("secondary")
        self.other.clicked.connect(self._pick_other)
        row.addWidget(self.other)
        row.addStretch()
        lay.addLayout(row)

        self.preview = QLabel()
        self.preview.setMinimumHeight(22)
        lay.addWidget(self.preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.set_color(self._color)
        self.name.setFocus()

    @staticmethod
    def _swatch_css(c: str) -> str:
        return (f"QToolButton {{ background: {c}; border: 1px solid {theme.BORDER}; "
                "border-radius: 12px; }"
                f"QToolButton:checked {{ border: 2px solid {theme.TEXT}; }}")

    def set_color(self, color: str) -> None:
        q = QColor(color)
        if not q.isValid():
            return
        self._color = q.name()
        for c, b in self.swatches.items():
            b.setChecked(c == self._color)
        if self._color not in self.swatches:
            checked = self.group.checkedButton()
            if checked is not None:          # exclusive groups refuse a plain uncheck
                self.group.setExclusive(False)
                checked.setChecked(False)
                self.group.setExclusive(True)
        self.preview.setText(f"<span style='color:{self._color}; font-size:14pt'>⚑</span>"
                             f"  <span style='color:{theme.TEXT_DIM}'>{self._color}</span>")

    def _pick_other(self) -> None:
        q = QColorDialog.getColor(QColor(self._color), self, "Marker colour")
        if q.isValid():
            self.set_color(q.name())

    def color(self) -> str:
        return self._color

    def values(self) -> tuple[str, str]:
        return self.name.text().strip(), self._color
