"""Project and .parthwl persistence, including WaveHowl / HWL Viewer interop."""

import base64
import json

import numpy as np
import pytest

from stimstudio.core.model import Clip, GeneratorSource, Project, PulseDataSource, Track
from stimstudio.core.render import render_clip, render_master, render_source
from stimstudio.demo import build_demo_project
from stimstudio.io.parthwl import (clip_to_part, convert_wavehowl_effects, load_parthwl,
                                   part_to_clip, save_parthwl)
from stimstudio.io.project_file import load_project, save_project


def test_project_roundtrip_renders_identically(tmp_path):
    p = build_demo_project()
    data = np.random.default_rng(0).random((90, 4)).astype(np.float32)
    p.tracks.append(Track(name="imported", type="standard",
                          clips=[Clip(PulseDataSource(data, loop=True), start_s=5, length_s=4)]))
    p.tracks.append(Track(name="later", type="triphase"))
    path = tmp_path / "p.stimproj"
    save_project(p, path)
    q = load_project(path)
    assert q.to_dict() == p.to_dict()
    np.testing.assert_array_equal(render_master(q), render_master(p))


def test_project_rejects_foreign_and_future_files(tmp_path):
    path = tmp_path / "x.stimproj"
    path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(path)
    path.write_text(json.dumps({"format": "stimproj", "version": 999}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(path)


def test_parthwl_roundtrip_generator_clip(tmp_path):
    clip = Clip(GeneratorSource("licks", {"speed": 0.7}, seed=4), length_s=3.0, trim_s=1.0,
                fade_out_s=0.5, name="lick")
    part = clip_to_part(clip, render_source(clip.source, 120, 40))
    save_parthwl(tmp_path / "c.parthwl", part)
    back = part_to_clip(load_parthwl(tmp_path / "c.parthwl"))
    assert isinstance(back.source, GeneratorSource) and back.source.seed == 4
    np.testing.assert_array_equal(render_clip(back), render_clip(clip))


def test_parthwl_roundtrip_pulse_clip(tmp_path):
    data = np.random.default_rng(2).random((50, 4)).astype(np.float32)
    clip = Clip(PulseDataSource(data), length_s=50 / 40)
    part = clip_to_part(clip, data)
    save_parthwl(tmp_path / "p.parthwl", part)
    back = part_to_clip(load_parthwl(tmp_path / "p.parthwl"))
    assert isinstance(back.source, PulseDataSource)
    np.testing.assert_array_equal(render_clip(back), render_clip(clip))


def test_hwl_viewer_style_part_imports_cache_as_pulses():
    """HWL Viewer stores real pulses in the cache under a placeholder generator."""
    data = np.random.default_rng(3).random((40, 4)).astype(np.float32)
    part = {
        "format": "parthwl", "version": 1, "fs": 40, "name": "sel", "duration_s": 1.0,
        "trim_start_s": 0.0, "end_at_zero": False, "fade_seconds": 0.3,
        "generator": {"type": "waveform", "seed": 0, "params": {}},
        "effects": [{"type": "scale_amplitude", "value": 0.5, "channels": ["A"]}],
        "cache": {"encoding": "base64-f32le", "n": 40, "fs": 40,
                  "data": base64.b64encode(data.astype("<f4").tobytes()).decode()},
    }
    clip = part_to_clip(part)
    assert isinstance(clip.source, PulseDataSource)
    assert clip.effects == part["effects"]           # already StimStudio format
    out = render_clip(clip)
    np.testing.assert_allclose(out[:, 0], np.minimum(data[:, 0] * 0.5, 1.0), atol=1e-6)
    np.testing.assert_array_equal(out[:, 1], data[:, 1])


def test_wavehowl_effects_are_converted():
    fx = convert_wavehowl_effects([
        {"type": "scale_amplitude", "value": 0.5},
        {"type": "amplitude_feel", "feel": 2.0},
        {"type": "channel_swap"},
    ], seed=9)
    assert [f["type"] for f in fx] == ["channel_swap", "amplitude_feel", "scale_amplitude"]
    assert fx[1]["feel"] == pytest.approx(0.5)
    assert all(f["channels"] == ["A", "B"] for f in fx)


def test_wavehowl_part_renders_like_wavehowl(wavehowl, tmp_path):
    """A part saved by WaveHowl renders the same here (effects with equal semantics)."""
    from wavehowl.core import parthwl as wh
    part = wh.build_part("x", "succubus", {}, 6.0, seed=12,
                         effects=[{"type": "channel_swap"},
                                  {"type": "amplitude_feel", "feel": 2.0},
                                  {"type": "scale_amplitude", "value": 0.7}],
                         end_at_zero=True, fade_seconds=0.5, trim_start_s=2.0)
    path = tmp_path / "wh.parthwl"
    wh.save_parthwl(path, part)
    theirs = wh.decode_cache(wh.load_parthwl(path))
    ours = render_clip(part_to_clip(load_parthwl(path)))
    np.testing.assert_allclose(ours, np.stack(theirs.channels, axis=1), atol=1e-5)
