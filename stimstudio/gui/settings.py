"""Application settings (window layout, output device, Howl address)."""

from __future__ import annotations

from PyQt6.QtCore import QSettings


def app_settings() -> QSettings:
    """Settings store honouring ``QSettings.setDefaultFormat``.

    ``QSettings(org, app)`` would always use the native store (the Windows
    registry); building it from the default format lets the tests redirect
    everything to a temporary ini file.
    """
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope,
                     "StimStudio", "StimStudio")
