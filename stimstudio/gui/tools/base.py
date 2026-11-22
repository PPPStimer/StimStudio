"""Base class of the Tools windows."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget


class ToolWindow(QWidget):
    """A non-modal top-level window owned by the main window (stays above it,
    closes with it). Works on ``main_window.doc`` when it creates something."""

    title = "Tool"

    def __init__(self, main_window):
        super().__init__(main_window, Qt.WindowType.Window)
        self.main_window = main_window
        self.setWindowTitle(self.title)

    @property
    def doc(self):
        return self.main_window.doc
