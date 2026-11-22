import os
import sys
from pathlib import Path

# Qt must be configured before PyQt6 is imported anywhere: offscreen platform,
# raster rendering (no GL context exists offscreen).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["STIMSTUDIO_SOFTWARE_RENDER"] = "1"
# never touch the user's real autosave folder
os.environ.setdefault("STIMSTUDIO_AUTOSAVE_DIR", os.path.join(os.environ.get("TEMP", "/tmp"),
                                                              "stimstudio-test-autosave"))

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "NotInclude" / "External Data"


@pytest.fixture(scope="session")
def wavehowl():
    """The original WaveHowl package, for parity checks (skipped if absent)."""
    path = EXTERNAL / "WaveHowl"
    if not path.is_dir():
        pytest.skip("External Data/WaveHowl not present")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import wavehowl.core
    import wavehowl.generators  # noqa: F401
    return wavehowl


@pytest.fixture(scope="session")
def qapp(tmp_path_factory):
    """The QApplication; QSettings redirected to a temporary ini file (never the registry)."""
    from PyQt6.QtCore import QSettings
    from stimstudio.gui.gl import configure_opengl
    configure_opengl()
    from PyQt6.QtWidgets import QApplication
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path_factory.mktemp("settings")))
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp, tmp_path):
    """Main window on the demo project, output switched to the dry run (no sound card),
    user presets in a temporary folder."""
    from stimstudio.demo import build_demo_project
    from stimstudio.gui.document import ProjectDocument
    from stimstudio.gui.main_window import MainWindow
    w = MainWindow(ProjectDocument(build_demo_project()), user_preset_dir=tmp_path / "presets",
                   autosave_dir=tmp_path / "autosave")
    w.device_panel.device.setCurrentIndex(w.device_panel.device.findData("dryrun"))
    w.resize(1400, 800)
    w.show()
    assert w.doc.wait_for_render()
    qapp.processEvents()
    yield w
    w.doc.undo.setClean()
    w.close()
    w.deleteLater()
    qapp.processEvents()
