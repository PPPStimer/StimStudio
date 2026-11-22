"""Command line: ``python -m stimstudio <command> ...``.

    render  PROJECT.stimproj OUT.hwl|.wav|.flac [--start S] [--end S] [--mode M]
                                          render the mix (format from the extension)
    stems   PROJECT.stimproj OUT_DIR [--format hwl|wav|flac] [--start S] [--end S]
                                          one file per track
    demo    OUT_DIR                       write demo.stimproj + demo.hwl
    info    FILE                          summarize a .hwl / .parthwl / .stimproj
    gui     [PROJECT.stimproj]            start the editor (same as ``StimStudio.pyw``)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .core.pulse import FS


def _summary(pulses: np.ndarray) -> str:
    n = len(pulses)
    if n == 0:
        return "0 pulses"
    names = ("L amp", "R amp", "L freq", "R freq")
    cols = ", ".join(f"{name} {pulses[:, i].min():.2f}..{pulses[:, i].max():.2f}"
                     for i, name in enumerate(names))
    return f"{n} pulses ({n / FS:.2f} s); {cols}"


def _audio(args):
    from .core.export import AudioSettings
    return AudioSettings(mode=args.mode, sample_rate=args.rate)


def cmd_render(args) -> int:
    from .core.export import export_mix
    from .core.render import RenderCache
    from .io.project_file import load_project

    project = load_project(args.project)
    t = time.perf_counter()
    n = export_mix(project, args.out, args.start, args.end, _audio(args), RenderCache())
    unit = "pulses" if str(args.out).lower().endswith(".hwl") else "samples"
    print(f"wrote {args.out}: {n} {unit} in {time.perf_counter() - t:.2f} s")
    return 0


def cmd_stems(args) -> int:
    from .core.export import export_stems
    from .core.render import RenderCache
    from .io.project_file import load_project

    project = load_project(args.project)
    paths = export_stems(project, args.out_dir, Path(args.project).stem, args.format,
                         args.start, args.end, _audio(args), RenderCache())
    for path in paths:
        print("wrote", path)
    return 0


def cmd_demo(args) -> int:
    from .core.render import RenderCache, render_master
    from .demo import build_demo_project
    from .io.hwl import write_hwl
    from .io.project_file import save_project

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    project = build_demo_project()
    save_project(project, out / "demo.stimproj")
    master = render_master(project, RenderCache())
    write_hwl(out / "demo.hwl", master)
    print(f"wrote {out / 'demo.stimproj'} and {out / 'demo.hwl'}: {_summary(master)}")
    return 0


def cmd_info(args) -> int:
    path = Path(args.file)
    suffix = path.suffix.lower()
    if suffix == ".hwl":
        from .io.hwl import read_hwl
        print(_summary(read_hwl(path)))
    elif suffix == ".parthwl":
        from .core.render import render_clip
        from .io.parthwl import load_parthwl, part_to_clip
        clip = part_to_clip(load_parthwl(path))
        print(f"clip {clip.name!r}: source={clip.source.kind}, {clip.length_s:.2f} s, "
              f"{len(clip.effects)} effect(s)")
        print(_summary(render_clip(clip)))
    elif suffix == ".stimproj":
        from .io.project_file import load_project
        p = load_project(path)
        print(f"project {p.name!r}: {len(p.tracks)} track(s), {p.duration_s():.2f} s, bpm {p.bpm}")
        for t in p.tracks:
            print(f"  [{t.type}] {t.name}: {len(t.clips)} clip(s), {len(t.effects)} fx, "
                  f"blend={t.blend}, gain={t.gain}{' (muted)' if t.muted else ''}")
    else:
        print(f"unsupported file type: {suffix}", file=sys.stderr)
        return 2
    return 0


def cmd_gui(args) -> int:
    from .gui.app import run
    return run(["stimstudio"] + ([args.project] if args.project else []))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="stimstudio")
    sub = parser.add_subparsers(dest="cmd", required=True)
    def range_and_audio(p):
        p.add_argument("--start", type=float, default=None, help="range start (s)")
        p.add_argument("--end", type=float, default=None, help="range end (s)")
        p.add_argument("--mode", choices=("continuous", "wavelet"), default="continuous",
                       help="audio synthesis (audio formats only)")
        p.add_argument("--rate", type=int, default=48000, help="audio sample rate")

    p = sub.add_parser("render", help="render the mix to .hwl / .wav / .flac")
    p.add_argument("project")
    p.add_argument("out")
    range_and_audio(p)
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("stems", help="one file per track")
    p.add_argument("project")
    p.add_argument("out_dir")
    p.add_argument("--format", choices=("hwl", "wav", "flac"), default="hwl")
    range_and_audio(p)
    p.set_defaults(fn=cmd_stems)
    p = sub.add_parser("demo", help="write a demo project and its .hwl")
    p.add_argument("out_dir")
    p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("info", help="summarize a file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("gui", help="start the editor")
    p.add_argument("project", nargs="?")
    p.set_defaults(fn=cmd_gui)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
