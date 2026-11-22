"""Form generated from a list of ParamSpecs (generators now, effects later).

Ported from WaveHowl's ParameterPanel widget factory. Emits ``paramChanged``
per edit, carrying the key, so callers can turn every edit into a mergeable
undo command.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QWidget)


class ParamForm(QWidget):
    paramChanged = pyqtSignal(str, object)   # key, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setVerticalSpacing(4)
        self._widgets: dict[str, tuple[str, QWidget]] = {}
        self._specs_key: tuple = ()

    def set_specs(self, specs, values: dict) -> None:
        """Build the form; rebuilt only when the spec list itself changes."""
        key = tuple(s.key for s in specs)
        if key != self._specs_key:
            while self.form.rowCount():
                self.form.removeRow(0)
            self._widgets.clear()
            for spec in specs:
                self.form.addRow(spec.label, self._make(spec, values.get(spec.key, spec.default)))
            self._specs_key = key
        self.set_values(values)

    def set_values(self, values: dict) -> None:
        for key, (kind, w) in self._widgets.items():
            if key not in values or w.hasFocus():
                continue
            v = values[key]
            w.blockSignals(True)
            if kind == "choice":
                w.setCurrentText(str(v))
            elif kind == "bool":
                w.setChecked(bool(v))
            elif kind == "int":
                w.setValue(int(v))
            elif kind == "str":
                w.setText(str(v))
            else:
                w.setValue(float(v))
            w.blockSignals(False)

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Grey out one parameter (e.g. while an automation envelope drives it)."""
        entry = self._widgets.get(key)
        if entry is not None:
            entry[1].setEnabled(enabled)
            entry[1].setToolTip("" if enabled else "Automated — edit its lane in the song editor")

    def _make(self, spec, value) -> QWidget:
        kind = spec.kind
        emit = lambda v, k=spec.key: self.paramChanged.emit(k, v)  # noqa: E731
        if kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in spec.choices])
            w.setCurrentText(str(value))
            w.currentTextChanged.connect(emit)
            self._widgets[spec.key] = ("choice", w)
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(emit)
            self._widgets[spec.key] = ("bool", w)
            return w
        if kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.min if spec.min is not None else 0),
                       int(spec.max if spec.max is not None else 1_000_000))
            if spec.step:
                w.setSingleStep(int(spec.step))
            w.setValue(int(value))
            w.valueChanged.connect(emit)
            self._widgets[spec.key] = ("int", w)
            return w
        if kind in ("str", "path"):
            container = QWidget()
            h = QHBoxLayout(container)
            h.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(str(value))
            edit.editingFinished.connect(lambda e=edit, k=spec.key: self.paramChanged.emit(k, e.text()))
            h.addWidget(edit)
            if kind == "path":
                btn = QPushButton("…")
                btn.setObjectName("secondary")
                btn.setMaximumWidth(32)
                btn.clicked.connect(lambda _=False, e=edit, k=spec.key: self._browse(e, k))
                h.addWidget(btn)
            self._widgets[spec.key] = ("str", edit)
            return container
        w = QDoubleSpinBox()
        w.setRange(spec.min if spec.min is not None else -1e6,
                   spec.max if spec.max is not None else 1e6)
        w.setSingleStep(spec.step or 0.01)
        w.setDecimals(3)
        w.setKeyboardTracking(False)
        w.setValue(float(value))
        w.valueChanged.connect(emit)
        self._widgets[spec.key] = ("float", w)
        return w

    def _browse(self, edit: QLineEdit, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "",
                                              "Audio (*.wav *.flac *.ogg *.mp3);;All files (*)")
        if path:
            edit.setText(path)
            self.paramChanged.emit(key, path)
