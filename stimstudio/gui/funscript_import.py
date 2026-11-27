"""Funscript import dialog: map funscript axes onto the four columns of a clip.

Adapted from HWL Viewer's FunscriptImportDialog. On a triphase track the four
columns are volume / position X / position Y / pulse frequency.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QGridLayout, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout)

from ..core.pulse import FS
from ..io.funscript import (INTERP_HOLD, INTERP_LINEAR, INTERP_SMOOTH, VALUE_POSITION,
                            VALUE_SPEED, ChannelMap, auto_mapping, axes_duration_ms,
                            funscript_to_pulses, load_funscript_axes)
from . import metrics, theme

PULSE_TARGETS = ("Amp L", "Amp R", "Freq L", "Freq R")


def load_axes(paths) -> tuple[list, list[str]]:
    axes, errors = [], []
    for p in paths:
        try:
            axes.extend(load_funscript_axes(p))
        except (OSError, ValueError) as exc:
            errors.append(f"{Path(p).name}: {exc}")
    return axes, errors


class FunscriptImportDialog(QDialog):
    def __init__(self, paths, track_kind: str = "standard", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import funscript")
        self.setMinimumWidth(metrics.cap_to_screen(640, parent))
        self.axes, errors = load_axes(paths)
        if errors:
            QMessageBox.warning(self, "Import funscript", "\n".join(errors))
        lay = QVBoxLayout(self)
        info = QLabel(f"{len(self.axes)} axis/axes from {len(paths)} file(s), "
                      f"{theme.format_time(axes_duration_ms(self.axes) / 1000.0)}")
        info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(info)

        if track_kind == "triphase":
            from ..core.triphase import TRI_LANE_NAMES
            targets = TRI_LANE_NAMES
        else:
            targets = PULSE_TARGETS
        grid = QGridLayout()
        for c, h in enumerate(("Column", "Source axis", "Value", "Invert", "Min", "Max")):
            grid.addWidget(QLabel(f"<b>{h}</b>"), 0, c)
        self.rows = []
        defaults = auto_mapping(self.axes)
        for r, (name, d) in enumerate(zip(targets, defaults), start=1):
            src = QComboBox()
            src.addItem("— constant (Min) —", None)
            for i, ax in enumerate(self.axes):
                src.addItem(ax.label, i)
            src.setCurrentIndex(0 if d.axis is None else d.axis + 1)
            value = QComboBox()
            value.addItem("position", VALUE_POSITION)
            value.addItem("speed", VALUE_SPEED)
            inv = QCheckBox()
            lo, hi = QDoubleSpinBox(), QDoubleSpinBox()
            for s, v in ((lo, d.out_min), (hi, d.out_max)):
                s.setRange(0.0, 1.0)
                s.setSingleStep(0.05)
                s.setValue(v)
            for c, w in enumerate((QLabel(name), src, value, inv, lo, hi)):
                grid.addWidget(w, r, c)
            self.rows.append((src, value, inv, lo, hi))
        lay.addLayout(grid)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Interpolation"))
        self.interp = QComboBox()
        for label, key in (("linear", INTERP_LINEAR), ("smooth", INTERP_SMOOTH),
                           ("hold", INTERP_HOLD)):
            self.interp.addItem(label, key)
        opts.addWidget(self.interp)
        opts.addSpacing(16)
        opts.addWidget(QLabel("Offset"))
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-60000, 60000)
        self.offset.setSuffix(" ms")
        self.offset.setSingleStep(25)
        opts.addWidget(self.offset)
        opts.addStretch(1)
        lay.addLayout(opts)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.axes))
        lay.addWidget(buttons)

    def mapping(self) -> list[ChannelMap]:
        return [ChannelMap(axis=src.currentData(), value=value.currentData(),
                           invert=inv.isChecked(), out_min=lo.value(), out_max=hi.value())
                for src, value, inv, lo, hi in self.rows]

    def pulses(self):
        return funscript_to_pulses(self.axes, self.mapping(), FS, self.interp.currentData(),
                                   self.offset.value())
