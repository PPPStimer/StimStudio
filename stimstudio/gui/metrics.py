"""Sizes that follow the screen.

Qt already scales logical pixels by the display scale (125 % → ×1.25), so a size
in logical pixels looks the same on every monitor. What does not follow is the
room on the screen: 1920×1080 at 150 % is only 1280×720 logical pixels, so a
window or dialog sized for a FullHD monitor at 100 % runs off the screen — use
:func:`fit_to_screen`, :func:`cap_to_screen` and :func:`scroll_body`. Text drawn
with QPainter takes :func:`paint_font`, which never goes below a readable size.
"""

from __future__ import annotations

from PyQt6.QtCore import QRect, QSize
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import QApplication, QWidget

#: smallest text size (logical px) still readable at 100 % on a FullHD monitor
MIN_TEXT_PX = 10


#: View → Interface size: % on top of the Windows display scale (100 = as Windows)
UI_SCALES = (80, 90, 100, 110, 125, 150, 175, 200)
UI_SCALE_KEY = "ui/scale"


def parse_scale(text) -> float | None:
    """``1.5``, ``150`` or ``150%`` → 1.5; None when it is not a sane factor."""
    try:
        value = float(str(text).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if value > 4:                       # given in percent
        value /= 100.0
    return value if 0.5 <= value <= 4.0 else None


_text_scale = 1.0


def apply_ui_scale(argv: list[str]) -> list[str]:
    """Before the QApplication exists: the interface size from ``--scale X`` (e.g.
    ``--scale 1.5`` to see a 150 % screen on a 100 % one), else ``QT_SCALE_FACTOR``
    from the environment, else View → Interface size. Returns *argv* without the
    option.

    Above 100 % it sets ``QT_SCALE_FACTOR``, which Qt multiplies with the display
    scale: everything grows. Below 100 % it only makes the text smaller
    (:func:`text_scale`, applied by the stylesheet): Qt cannot draw below one
    device pixel per logical pixel — at a device pixel ratio under 1 tree arrows
    land on the wrong rows and, with OpenGL views in the window, old pixels stay
    on screen (tabs, panels) until something repaints the whole window."""
    global _text_scale
    rest, flag = [], None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--scale" and i + 1 < len(argv):
            flag = parse_scale(argv[i + 1])
            i += 2
            continue
        if a.startswith("--scale="):
            flag = parse_scale(a.split("=", 1)[1])
            i += 1
            continue
        rest.append(a)
        i += 1
    import os
    factor = flag
    if factor is None and "QT_SCALE_FACTOR" in os.environ:
        factor = parse_scale(os.environ["QT_SCALE_FACTOR"])
    if factor is None and "QT_SCALE_FACTOR" not in os.environ:
        from .settings import app_settings
        factor = parse_scale(app_settings().value(UI_SCALE_KEY, 100))
    if factor is None:
        return rest
    if factor >= 1.0:
        _text_scale = 1.0
        if abs(factor - 1.0) > 1e-3:
            os.environ["QT_SCALE_FACTOR"] = f"{factor:g}"
        else:
            os.environ.pop("QT_SCALE_FACTOR", None)
    else:
        os.environ.pop("QT_SCALE_FACTOR", None)        # never a device pixel ratio under 1
        _text_scale = factor
    return rest


def text_scale() -> float:
    """Below 100 % Interface size: the factor for the stylesheet's font sizes."""
    return _text_scale


def ui_scale_percent() -> int:
    """The interface size this session runs at (% on top of the display scale)."""
    import os
    if _text_scale < 1.0:
        return round(_text_scale * 100)
    factor = parse_scale(os.environ.get("QT_SCALE_FACTOR", "1")) or 1.0
    return round(factor * 100)


def paint_font(size_px: float, *, bold: bool = False, base: QFont | None = None) -> QFont:
    """A font for QPainter text: *base* (default: the application font) at
    *size_px* logical px, never below :data:`MIN_TEXT_PX`."""
    app = QApplication.instance()
    font = QFont(base) if base is not None else (QFont(app.font()) if app is not None
                                                  else QFont())
    font.setPixelSize(max(MIN_TEXT_PX, round(size_px)))
    if bold:
        font.setBold(True)
    return font


def available(widget: QWidget | None = None) -> QRect:
    """Available geometry (without the taskbar) of the screen *widget* is on,
    else of the screen under its parent, else the primary screen."""
    screen = None
    if widget is not None:
        target = widget.parentWidget().window() if (widget.parentWidget() is not None
                                                    and not widget.isVisible()) else widget
        screen = target.screen()
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)


def fit_size(widget: QWidget | None, width: int, height: int, frac: float = 0.92) -> QSize:
    """*width* × *height*, but at most *frac* of the available screen."""
    avail = available(widget)
    return QSize(min(int(width), int(avail.width() * frac)),
                 min(int(height), int(avail.height() * frac)))


def fit_to_screen(widget: QWidget, width: int, height: int, frac: float = 0.92) -> QSize:
    """``widget.resize(width, height)``, shrunk to fit the screen."""
    size = fit_size(widget, width, height, frac)
    widget.resize(size)
    return size


def cap_to_screen(n: int, widget: QWidget | None = None, *, vertical: bool = False,
                  frac: float = 0.9) -> int:
    """A minimum width (or height) that still fits the screen."""
    avail = available(widget)
    room = avail.height() if vertical else avail.width()
    return min(int(n), int(room * frac))


def scroll_body(window: QWidget) -> QWidget:
    """A scrolling body for a big window: lay the contents out on the widget this
    returns. With room it looks the same; on a small screen (1280×720 logical is
    1920×1080 at 150 %) the window still fits and its contents scroll.

    The window's size hint no longer follows its contents: size it explicitly,
    e.g. with :func:`fit_to_screen`."""
    from PyQt6.QtWidgets import QFrame, QScrollArea, QVBoxLayout
    outer = QVBoxLayout(window)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    scroll = QScrollArea()
    scroll.setObjectName("windowScroll")
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    body = QWidget()
    scroll.setWidget(body)
    outer.addWidget(scroll)
    return body


def shrinkable_combo(combo, min_chars: int = 10):
    """Let *combo* narrow below its longest item (a sound card name can be 60
    characters long): the field shows the start, the list and the tooltip the
    whole name. Call :func:`widen_popup` after filling it."""
    from PyQt6.QtWidgets import QComboBox
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    combo.currentTextChanged.connect(combo.setToolTip)
    return combo


def widen_popup(combo) -> None:
    """The drop-down list of *combo* as wide as its longest item (not the field)."""
    view = combo.view()
    need = view.sizeHintForColumn(0) + view.verticalScrollBar().sizeHint().width() + 8
    view.setMinimumWidth(cap_to_screen(need, combo))


def fit_wrapped_height(window: QWidget) -> None:
    """Grow *window* to the height its word-wrapped labels need at its width.

    ``adjustSize()`` sizes a window by its size hint, which ignores
    height-for-width: a wrapped label then gets one line and is cut."""
    layout = window.layout()
    if layout is None:
        return
    layout.activate()
    if not layout.hasHeightForWidth():
        return
    need = layout.totalHeightForWidth(window.width())
    room = available(window).height() - (window.frameGeometry().height() - window.height())
    need = min(need, room)
    if need > window.height():
        window.resize(window.width(), need)


def keep_on_screen(widget: QWidget) -> None:
    """Shrink and move a top-level window so that all of it is on its screen
    (after restoring a size saved on a bigger monitor or at a smaller scale)."""
    avail = available(widget)
    frame = widget.frameGeometry()
    extra_w = frame.width() - widget.width()
    extra_h = frame.height() - widget.height()
    w = min(widget.width(), avail.width() - extra_w)
    h = min(widget.height(), avail.height() - extra_h)
    if (w, h) != (widget.width(), widget.height()):
        widget.resize(w, h)
        frame = widget.frameGeometry()
    x = min(max(frame.x(), avail.left()), avail.right() - frame.width() + 1)
    y = min(max(frame.y(), avail.top()), avail.bottom() - frame.height() + 1)
    if (x, y) != (frame.x(), frame.y()):
        widget.move(x, y)
