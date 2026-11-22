"""Hardware (OpenGL) rendering for every plot and graphics view.

Call :func:`configure_opengl` once, before the QApplication exists. Then:

* pyqtgraph widgets render through OpenGL (``useOpenGL=True``);
* custom ``QGraphicsView`` widgets get a ``QOpenGLWidget`` viewport via
  :func:`attach_gl_viewport` (or derive from :class:`GLGraphicsView`).

Set ``STIMSTUDIO_SOFTWARE_RENDER=1`` to fall back to the raster engine (used by
the headless test suite, where no GL context is available).
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QSurfaceFormat
from PyQt6.QtWidgets import QApplication, QGraphicsView

from . import theme

#: Multisample anti-aliasing samples for GL surfaces.
MSAA_SAMPLES = 4


def gl_enabled() -> bool:
    return os.environ.get("STIMSTUDIO_SOFTWARE_RENDER", "") not in ("1", "true", "yes")


def configure_opengl() -> None:
    """Global GL setup. Must run before the QApplication is created."""
    import pyqtgraph as pg

    use_gl = gl_enabled()
    if use_gl:
        fmt = QSurfaceFormat()
        fmt.setSamples(MSAA_SAMPLES)
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)
        # docks can float into their own top-level windows: share one GL context
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    pg.setConfigOptions(useOpenGL=use_gl, antialias=True,
                        background=theme.BG, foreground=theme.TEXT)


def attach_gl_viewport(view: QGraphicsView) -> None:
    """Give *view* an OpenGL viewport (no-op in software mode)."""
    view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    view.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    if not gl_enabled():
        return
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget

    vp = QOpenGLWidget()
    fmt = QSurfaceFormat.defaultFormat()
    fmt.setSamples(MSAA_SAMPLES)
    vp.setFormat(fmt)
    view.setViewport(vp)
    # a GL viewport repaints whole frames; partial updates only cost bookkeeping
    view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)


class GLGraphicsView(QGraphicsView):
    """QGraphicsView preconfigured for hardware rendering."""

    def __init__(self, scene=None, parent=None):
        super().__init__(parent)
        if scene is not None:
            self.setScene(scene)
        attach_gl_viewport(self)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheNone)
