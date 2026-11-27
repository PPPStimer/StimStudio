"""Help: keyboard / mouse reference and the About text."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from . import metrics

from .. import __version__

MOUSE = (
    ("Add a clip", "Double-click a track · drag from the Browser · double-click in the Browser"),
    ("Move / copy clips", "Drag · Ctrl+drag copies · drag onto another track of the same kind"),
    ("Resize / trim", "Drag the right / left clip edge"),
    ("Fades", "Drag the small squares in the clip's top corners"),
    ("Free movement", "Hold Alt (no grid snap, no sticking to markers / clip edges)"),
    ("Crossfade", "Drag a clip over its neighbour: the overlap becomes a crossfade (track "
                  "header right-click → Clip overlaps: Crossfade / Mix / Max, Smooth joins)"),
    ("Select", "Click · Shift+click toggles · drag a rubber band on empty space"),
    ("Edit cursor", "Click empty space or the ruler"),
    ("Loop region", "Right-drag on the ruler · right-click clears"),
    ("Zoom / scroll", "Ctrl+wheel · Shift+wheel · wheel"),
    ("Envelope points", "Click a lane to add · drag to move · right-click to delete"),
    ("Show sub-lanes", "▶ in the track header"),
    ("Import", "Drop audio files, .funscript or beat timestamp .csv files onto a track"),
    ("Beat marks", "Clip right-click → Beat marks from peaks… · select marks + clips → "
                   "Convert to pattern"),
    ("Pattern marks", "Pattern editor: double-click adds · drag moves · drag the tick top / "
                      "Shift sets velocity · Del deletes · right-click pins a pool clip"),
    ("Video", "Drop a video file on the timeline · drag the video clip to line it up · "
              "double-click it for the Video window · ▦ / ♫ in its header hide frames / sound · "
              "F11 full screen on the dock's monitor · x2 / x5 / x10 fast-forward (output "
              "pauses, resumes when you press the button again)"),
    ("Color rhythm", "Tools → Color rhythm from video: drag around the outer edge of the "
                     "target · click a colour on the frame to pick / drop it · Create clip → "
                     "right-click → Beat marks from peaks"),
    ("Markers", "Double-click the ruler (or Shift+M) to add with a name and colour · drag a "
                "flag to move · click it to jump · double-click to edit name and colour · "
                "right-click for loop / delete · clips and the cursor stick to markers"),
    ("Output signal", "Output dock, audio outputs: scope (window, trigger A/B, Hold), "
                      "spectrum, XY (Lissajous, phase of B vs A), RMS / peak meters of what the "
                      "sound card receives"),
    ("Overview bar", "Click or drag to scroll the timeline · wheel to zoom"),
    ("Record pad", "Hold the left button and move: up = level, right = frequency (triphase: "
                   "position) · release springs to 0 unless Hold · keys 1–9 = 10–90 %, 0 = off "
                   "· wheel = frequency (triphase: volume)"),
    ("Safety list", "Click an issue to jump there · double-click (or Fix) to repair it"),
)


def shortcut_rows(window) -> list[tuple[str, str]]:
    rows = []
    for a in window.findChildren(type(window.a_play)):
        keys = ", ".join(s.toString() for s in a.shortcuts() if not s.isEmpty())
        text = a.text().replace("&", "").rstrip("…").strip()
        if keys and text:
            rows.append((text, keys))
    return sorted(set(rows), key=lambda r: r[0].lower())


class ShortcutsDialog(QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("Keyboard and mouse")
        metrics.fit_to_screen(self, 560, 640)
        lay = QVBoxLayout(self)
        view = QTextBrowser()
        lay.addWidget(view)
        css = ("<style>td{padding:2px 12px 2px 0;} th{text-align:left;padding-top:10px;}"
               "kbd{color:#4ec9b0;font-family:Consolas,monospace;}</style>")
        html = [css, "<table>", "<tr><th colspan=2>Keyboard</th></tr>"]
        html += [f"<tr><td>{t}</td><td><kbd>{k}</kbd></td></tr>" for t, k in shortcut_rows(window)]
        html.append("<tr><th colspan=2>Mouse</th></tr>")
        html += [f"<tr><td>{t}</td><td>{k}</td></tr>" for t, k in MOUSE]
        html.append("</table>")
        view.setHtml("".join(html))
        self.view = view


def about_text() -> str:
    return (f"<b>StimStudio {__version__}</b><br>Multitrack e-stim pattern studio.<br><br>"
            "Outputs: Coyote 3 via the Howl app (remote API), audio (continuous / wavelet, "
            "triphase with phase), Intiface / Buttplug devices, .hwl, WAV/FLAC and funscript "
            "export.<br><br>"
            "Built on ports of <i>Howl</i> (activities, output formulas), <i>WaveHowl</i> "
            "(generators) and <i>HWL Viewer</i> (outputs, effects, audio conversion, "
            "funscript).<br><br><b>Esc</b> stops all output immediately.")
