"""Triphase calibration dialog (project setting).

Centre offset, per-electrode gain and reach. Drag the test point on the pad to
see how the current divides between the electrodes; every change is applied
to the project immediately (one undo step per control while it is moved).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                             QLabel, QVBoxLayout)

from ...core.triphase import ELECTRODES, Calibration, electrode_levels
from .. import theme
from .position_pad import PositionPad

FIELDS = (("center_x", "Centre X", -0.5, 0.5, 0.01),
          ("center_y", "Centre Y", -0.5, 0.5, 0.01),
          ("gain_n", "Gain N (common)", 0.2, 2.0, 0.05),
          ("gain_l", "Gain L", 0.2, 2.0, 0.05),
          ("gain_r", "Gain R", 0.2, 2.0, 0.05),
          ("reach", "Reach", 0.25, 1.0, 0.05))


class TriphaseCalibrationDialog(QDialog):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Triphase calibration")
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.pad = PositionPad(editable=True)
        self.pad.set_point(0.0, 0.0)
        self.pad.pointDragged.connect(lambda _x, _y: self._update_levels())
        left.addWidget(self.pad, 1)
        self.levels = QLabel()
        left.addWidget(self.levels)
        lay.addLayout(left, 1)

        right = QVBoxLayout()
        info = QLabel("Wiring for Howl / Coyote: channel A → electrode L, channel B → "
                      "electrode R, both returns → common electrode N.\n\nAudio output plays "
                      "the true three-phase signal on the stereo channels with N as ground.")
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {theme.TEXT_DIM};")
        right.addWidget(info)
        form = QFormLayout()
        self.spins: dict[str, QDoubleSpinBox] = {}
        for key, label, lo, hi, step in FIELDS:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(2)
            s.setKeyboardTracking(False)
            s.valueChanged.connect(lambda v, k=key: self._changed(k, v))
            form.addRow(label, s)
            self.spins[key] = s
        right.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close
                                   | QDialogButtonBox.StandardButton.Reset)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset)
        right.addStretch(1)
        right.addWidget(buttons)
        lay.addLayout(right, 1)
        doc.modelChanged.connect(self.sync)
        self.sync()

    def calibration(self) -> Calibration:
        return Calibration.from_dict(self.doc.project.triphase_cal)

    def sync(self) -> None:
        cal = self.calibration()
        for key, s in self.spins.items():
            if not s.hasFocus():
                s.blockSignals(True)
                s.setValue(getattr(cal, key))
                s.blockSignals(False)
        self.pad.set_calibration(cal)
        self._update_levels()

    def _changed(self, key: str, value: float) -> None:
        cal = {**self.calibration().to_dict(), key: float(value)}
        self.doc.set_project_attrs(merge_key=("triphase_cal", key), text="Triphase calibration",
                                   triphase_cal=cal)

    def reset(self) -> None:
        self.doc.set_project_attrs(text="Reset triphase calibration", triphase_cal={})

    def _update_levels(self) -> None:
        if self.pad.point is None:
            return
        x, y = self.pad.point
        lv = electrode_levels(1.0, x, y, self.calibration())
        self.levels.setText("  ".join(f"{k}: {float(lv[k]):.2f}" for k in ELECTRODES))
