"""``.parthwl`` segment files ↔ clips.

A ``.parthwl`` is JSON: a generator definition plus an optional base64 cache of
rendered pulses (see ``docs/PARTHWL_FORMAT.md``). Two producers exist
and they use it differently:

* **WaveHowl** — the generator definition is the source of truth; effects are
  WaveHowl-style (no ``channels`` key, fixed canonical order, ``feel`` means
  exponent ``1/feel``).
* **HWL Viewer** — the cache holds the real (un-effected) pulses and the
  generator is a placeholder ``waveform`` with empty params; effects are
  HWL-Viewer-style (with ``channels``), which is also StimStudio's format.

Both import as clips; the export side writes the HWL Viewer convention for
pulse clips and a real generator for generator clips, always with a cache.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from ..core.model import Clip, GeneratorSource, PulseDataSource
from ..core.pulse import FS, array_from_bytes, array_to_bytes

FORMAT = "parthwl"
VERSION = 1
CACHE_ENCODING = "base64-f32le"

#: WaveHowl's fixed effect order (its effects ignore list order).
WAVEHOWL_ORDER = (
    "channel_swap", "invert_frequencies", "frequency_adjust",
    "random_amplitude_noise", "random_frequency_noise",
    "amplitude_feel", "frequency_feel", "scale_frequency", "scale_amplitude",
)


def load_parthwl(path) -> dict:
    part = json.loads(Path(path).read_text(encoding="utf-8"))
    if part.get("format") != FORMAT:
        raise ValueError("not a .parthwl file (wrong or missing 'format')")
    return part


def decode_cache(part: dict):
    cache = part.get("cache")
    if not cache:
        return None
    return array_from_bytes(base64.b64decode(cache["data"]))


def _is_placeholder_generator(part: dict) -> bool:
    gen = part.get("generator") or {}
    return gen.get("type") in (None, "waveform") and not gen.get("params")


def convert_wavehowl_effects(effects: list[dict], seed: int = 0) -> list[dict]:
    """Translate a WaveHowl effect list into StimStudio's ordered chain."""
    order = {t: i for i, t in enumerate(WAVEHOWL_ORDER)}
    out = []
    for fx in sorted((e for e in effects if e.get("type") in order),
                     key=lambda e: order[e["type"]]):
        fx = dict(fx)
        if fx["type"] in ("amplitude_feel", "frequency_feel"):
            feel = float(fx.get("feel", 1.0))
            fx["feel"] = 1.0 / feel if feel > 0 else 1.0
        if fx["type"] in ("random_amplitude_noise", "random_frequency_noise"):
            fx.setdefault("seed", int(seed))
        fx.setdefault("channels", ["A", "B"])
        out.append(fx)
    return out


def part_to_clip(part: dict, start_s: float = 0.0) -> Clip:
    """Build a clip from a loaded part."""
    gen = part.get("generator") or {}
    effects = list(part.get("effects") or [])
    hwl_viewer_style = any("channels" in e for e in effects)
    cached = decode_cache(part)
    duration = float(part.get("duration_s", 0.0))

    # StimStudio writes an explicit "source"; files without it (HWL Viewer) store
    # their real pulses in the cache under a placeholder generator
    explicit = part.get("source")
    is_pulses = explicit == "pulses" if explicit else _is_placeholder_generator(part)
    if cached is not None and is_pulses:
        source = PulseDataSource(cached, origin=f"parthwl:{part.get('name', '')}")
        trim_s = 0.0
        duration = duration or len(cached) / FS
    else:
        source = GeneratorSource(gen.get("type", "waveform"), dict(gen.get("params", {})),
                                 int(gen.get("seed", 0)))
        trim_s = float(part.get("trim_start_s", 0.0))
    if not hwl_viewer_style:
        effects = convert_wavehowl_effects(effects, int(gen.get("seed", 0)))

    return Clip(
        source=source, start_s=start_s, length_s=duration, trim_s=trim_s,
        fade_in_s=float(part.get("fade_in_seconds", 0.3)) if part.get("start_at_zero") else 0.0,
        fade_out_s=float(part.get("fade_seconds", 0.3)) if part.get("end_at_zero") else 0.0,
        effects=effects, name=part.get("name", ""),
    )


def clip_to_part(clip: Clip, rendered=None) -> dict:
    """A .parthwl dict for *clip*. *rendered* is the clip's source render to
    embed as cache (pass ``render_source`` output; fades are stored as flags)."""
    if clip.source.kind == "audio":
        raise ValueError("audio modulator clips cannot be saved as .parthwl")
    if isinstance(clip.source, GeneratorSource):
        generator = {"type": clip.source.type_id, "seed": int(clip.source.seed),
                     "params": dict(clip.source.params)}
        source = "generator"
    else:
        generator = {"type": "waveform", "seed": 0, "params": {}}
        source = "pulses"
    part = {
        "format": FORMAT, "version": VERSION, "fs": FS,
        "name": clip.name, "duration_s": float(clip.length_s),
        "trim_start_s": float(clip.trim_s) if source == "generator" else 0.0,
        "end_at_zero": clip.fade_out_s > 0, "fade_seconds": float(clip.fade_out_s or 0.3),
        "start_at_zero": clip.fade_in_s > 0, "fade_in_seconds": float(clip.fade_in_s or 0.3),
        "generator": generator, "source": source,
        "effects": [dict(e) for e in clip.effects],
        "channel_map": {"left": "A", "right": "B"},
    }
    if rendered is not None:
        part["cache"] = {"encoding": CACHE_ENCODING, "n": int(len(rendered)), "fs": FS,
                         "data": base64.b64encode(array_to_bytes(rendered)).decode("ascii")}
    return part


def save_parthwl(path, part: dict) -> None:
    Path(path).write_text(json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8")
