"""Clip presets: ``.parthwl`` files in the bundled folder and in a user folder."""

from __future__ import annotations

from pathlib import Path

from ..core.model import Clip
from ..core.pulse import n_pulses
from ..core.render import RenderCache, render_source
from .parthwl import clip_to_part, load_parthwl, part_to_clip, save_parthwl

EXTENSION = ".parthwl"
BUNDLED_DIR = Path(__file__).resolve().parent.parent / "resources" / "presets"


def list_presets(folder) -> list[Path]:
    """Every preset under *folder* (recursive), sorted by relative path."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(folder.rglob(f"*{EXTENSION}"),
                  key=lambda p: str(p.relative_to(folder)).lower())


def load_preset(path, start_s: float = 0.0) -> Clip:
    clip = part_to_clip(load_parthwl(path), start_s=start_s)
    if not clip.name:
        clip.name = Path(path).stem
    return clip


def save_preset(clip: Clip, path, cache: RenderCache | None = None, bpm: float = 120.0,
                domain: str = "pulse") -> Path:
    """Write *clip* as a preset (with an embedded render cache; a pattern is
    stored as the pulses it renders)."""
    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_suffix(EXTENSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_source(clip.source, n_pulses(clip.length_s), n_pulses(clip.trim_s),
                             cache, bpm, domain)
    part = clip_to_part(clip, rendered)
    part["name"] = clip.name or path.stem
    save_parthwl(path, part)
    return path


def safe_filename(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_()." else "_" for c in name).strip()
    return keep or "preset"
