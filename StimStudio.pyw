r"""Start the StimStudio GUI: ``StimStudio.pyw [project.stimproj]``.

A double-click runs it with pythonw, without a console window; output and crash
tracebacks then go to %LOCALAPPDATA%\StimStudio\StimStudio.log. Started as
``python StimStudio.pyw`` in a terminal, they stay in the terminal.

``--scale 1.5`` makes the whole interface 1.5× larger for this run (View →
Interface size keeps a size): on a 100 % monitor it shows what a 150 % one sees.
"""

import os
import sys
from pathlib import Path

# the package next to this file, whatever folder StimStudio is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

if sys.stderr is None:  # pythonw: no console to print to
    log_dir = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "StimStudio"
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(log_dir / "StimStudio.log", "w", encoding="utf-8", buffering=1)

from stimstudio.gui.app import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
