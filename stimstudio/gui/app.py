"""GUI entry point."""

from __future__ import annotations

import sys


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    from .gl import configure_opengl

    configure_opengl()  # before the QApplication: GL surface format, shared contexts
    if sys.platform == "win32":
        # own taskbar button with our icon, instead of grouping under python.exe
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StimStudio.StimStudio")

    from PyQt6.QtWidgets import QApplication

    from .donate_dialog import show_startup_dialog
    from .main_window import MainWindow
    from .theme import app_icon, load_stylesheet

    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("StimStudio")
    app.setOrganizationName("StimStudio")
    app.setWindowIcon(app_icon())  # inherited by every window, dialog and floating dock
    app.setStyleSheet(load_stylesheet())
    window = MainWindow()
    paths = [a for a in argv[1:] if not a.startswith("-")]
    window.show()
    if paths:
        window.file_open(paths[0])
    else:
        window.offer_recovery()
    show_startup_dialog(window)  # after the questions above, so as not to stack on them
    return app.exec()
