"""M8: exports (range, stems, audio), funscript import, autosave / recovery, recent
files, help."""

import json

import numpy as np
import pytest
import soundfile as sf
from PyQt6.QtCore import QPointF

from stimstudio.core.export import (AudioSettings, export_mix, export_stems, slice_range,
                                    stem_project, stem_tracks)
from stimstudio.core.model import Clip, GeneratorSource, Project, Track
from stimstudio.core.pulse import FS
from stimstudio.core.render import render_project
from stimstudio.io.funscript import (VALUE_SPEED, ChannelMap, auto_mapping, funscript_to_pulses,
                                     load_funscript_axes)
from stimstudio.io.hwl import read_hwl
from stimstudio.io.project_file import save_project


def flat(level, start=0.0, length=4.0):
    src = GeneratorSource("waveform", {"amp_speed": 0, "amp_min": level, "amp_max": level,
                                       "freq_speed": 0, "freq_min": 0.5, "freq_max": 0.5})
    return Clip(src, start_s=start, length_s=length)


def two_tracks():
    return Project(tracks=[Track(name="a/b", clips=[flat(0.2)]),
                           Track(name="two", clips=[flat(0.3, 1.0, 2.0)]),
                           Track(name="muted", muted=True, clips=[flat(0.9)]),
                           Track(name="tri", type="triphase",
                                 clips=[Clip(GeneratorSource("tri_circle", {}), length_s=3.0)])])


# ── core exports ────────────────────────────────────────────────────────────
def test_range_export_matches_render(tmp_path):
    p = two_tracks()
    n = export_mix(p, tmp_path / "x.hwl", 1.0, 3.0)
    assert n == 80
    np.testing.assert_array_equal(read_hwl(tmp_path / "x.hwl"),
                                  np.clip(render_project(p).master[40:120], 0, 1))
    with pytest.raises(ValueError):
        slice_range(100, 3.0, 2.0)
    with pytest.raises(ValueError):
        export_mix(p, tmp_path / "x.mp4")


@pytest.mark.parametrize("ext,mode", [("wav", "continuous"), ("flac", "wavelet")])
def test_audio_export(tmp_path, ext, mode):
    p = two_tracks()
    path = tmp_path / f"x.{ext}"
    n = export_mix(p, path, 0.0, 2.0, AudioSettings(mode=mode, sample_rate=44100))
    info = sf.info(path)
    assert info.samplerate == 44100 and info.frames == n == 2 * 44100 and info.channels == 2


def test_stems(tmp_path):
    p = two_tracks()
    assert [t.name for t in stem_tracks(p)] == ["a/b", "two", "tri"]
    paths = export_stems(p, tmp_path / "stems", "proj")
    assert [x.name for x in paths] == ["proj - 01 a_b.hwl", "proj - 02 two.hwl",
                                       "proj - 03 tri.hwl"]
    stems = [read_hwl(x) for x in paths]
    assert all(len(s) == 4 * FS for s in stems)                  # full project length
    mix = render_project(p).master
    # additive tracks without master effects: the stems sum to the mix (which clips at 1)
    np.testing.assert_allclose(np.clip(stems[0][:, 0] + stems[1][:, 0] + stems[2][:, 0], 0, 1),
                               mix[:, 0], atol=1e-5)
    solo = stem_project(p, p.tracks[1].id)
    assert [t.solo for t in solo.tracks] == [False, True, False, False]
    assert not p.tracks[1].solo                                   # original untouched


# ── funscript ───────────────────────────────────────────────────────────────
def write_script(path, actions, **extra):
    path.write_text(json.dumps({"actions": [{"at": t, "pos": v} for t, v in actions], **extra}),
                    encoding="utf-8")
    return path


def test_funscript_loading_and_rendering(tmp_path):
    a = write_script(tmp_path / "clip.funscript", [(0, 0), (1000, 100), (2000, 0)])
    b = write_script(tmp_path / "clip.roll.funscript", [(0, 50), (2000, 50)], inverted=True)
    axes = load_funscript_axes(a) + load_funscript_axes(b)
    assert [ax.axis_id for ax in axes] == ["L0", "R1"]
    mapping = auto_mapping(axes)
    assert [m.axis for m in mapping] == [0, 1, None, None] and mapping[2].out_min == 0.5
    p = funscript_to_pulses(axes, mapping)
    assert p.shape == (80, 4)
    assert p[0, 0] == 0.0 and p[40, 0] == pytest.approx(1.0) and p[20, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(p[:, 1], 0.5)                      # inverted 50 → 50
    np.testing.assert_allclose(p[:, 2], 0.5)
    speed = funscript_to_pulses(axes, [ChannelMap(0, VALUE_SPEED)] * 4, speed_ref=100.0)
    assert speed[20, 0] == pytest.approx(1.0)                     # 100 units/s
    shifted = funscript_to_pulses(axes, mapping, offset_ms=500)
    assert len(shifted) == 100 and shifted[20 + 20, 0] == pytest.approx(p[20, 0], abs=1e-6)


def test_import_funscript_into_tracks(window, tmp_path):
    script = write_script(tmp_path / "s.funscript", [(0, 0), (1000, 100), (2000, 0)])
    doc = window.doc
    clip = window.import_funscript([str(script)], start_s=55.0, track_id=doc.project.tracks[1].id,
                                   mapping=[ChannelMap(0)] * 2 + [ChannelMap(None, out_min=0.4)] * 2)
    assert clip in doc.project.tracks[1].clips and clip.length_s == pytest.approx(2.0)
    assert clip.source.data[40, 0] == pytest.approx(1.0) and clip.source.data[0, 2] == 0.4
    tri = window.add_triphase_track()
    clip2 = window.import_funscript([str(script)], start_s=0.0, track_id=tri.id,
                                    mapping=auto_mapping(load_funscript_axes(script)))
    assert clip2 in tri.clips


def test_funscript_dialog_and_drop(window, tmp_path):
    from stimstudio.gui.funscript_import import FunscriptImportDialog
    script = write_script(tmp_path / "d.funscript", [(0, 0), (500, 100)])
    dlg = FunscriptImportDialog([str(script)], "triphase", window)
    assert len(dlg.rows) == 4 and dlg.rows[0][0].currentData() == 0
    assert dlg.pulses().shape == (20, 4)
    got = []
    window.editor.funscriptDropped.disconnect()
    window.editor.funscriptDropped.connect(lambda paths, t, tid: got.append((paths, tid)))
    ed = window.editor
    ed.drop_funscripts([str(script)], QPointF(5 * ed.pps, ed.layout.row(
        window.doc.project.tracks[0].id).y + 5))
    assert got == [([str(script)], window.doc.project.tracks[0].id)]


# ── export dialog ───────────────────────────────────────────────────────────
def test_export_dialog_loop_region_stems(window, tmp_path):
    from stimstudio.gui.export_dialog import ExportDialog
    window.playback.set_loop_region((10.0, 15.0))
    dlg = ExportDialog(window)
    assert dlg.range.count() == 2
    dlg.range.setCurrentIndex(1)
    dlg.stems.setChecked(True)
    dlg.path.setText(str(tmp_path / "stems"))
    written = dlg.run()
    assert len(written) == 2 and all(len(read_hwl(p)) == 5 * FS for p in written)

    dlg = ExportDialog(window)
    dlg.format.setCurrentIndex(dlg.format.findData("wav"))
    dlg.path.setText(str(tmp_path / "mix"))
    (path,) = dlg.run()
    assert path.suffix == ".wav" and sf.info(path).frames == 50 * 48000


# ── recent files, autosave, recovery ────────────────────────────────────────
def test_recent_files(window, tmp_path):
    path = str(tmp_path / "r.stimproj")
    window.doc.save(path)
    window._add_recent(path)
    window._add_recent(str(tmp_path / "other.stimproj"))
    window._add_recent(path)
    assert window.recent_files()[:2] == [path, str(tmp_path / "other.stimproj")]
    window._fill_recent()
    texts = [a.text() for a in window.recent_menu.actions()]
    assert path in texts and "Clear list" in texts


def test_autosave_and_recovery(qapp, tmp_path):
    from stimstudio.demo import build_demo_project
    from stimstudio.gui.document import ProjectDocument
    from stimstudio.gui.main_window import MainWindow
    folder = tmp_path / "autosave"
    w1 = MainWindow(ProjectDocument(build_demo_project()), user_preset_dir=tmp_path / "p",
                    autosave_dir=folder)
    assert w1.autosave.save_now() is None                       # nothing unsaved yet
    w1.doc.add_track("crash test")
    saved = w1.autosave.save_now()
    assert saved.exists() and saved.with_suffix(".json").exists()
    proj_file = str(tmp_path / "w1.stimproj")
    w1.doc.save(proj_file)                                      # clean → autosave removed
    assert not saved.exists()
    w1.doc.add_track("after save")
    crashed = w1.autosave.save_now()
    w1.autosave.timer.stop()                                    # "crash": leave it behind

    w2 = MainWindow(ProjectDocument(), user_preset_dir=tmp_path / "p", autosave_dir=folder)
    assert w2.offer_recovery(ask=lambda p, m: "recover")
    assert [t.name for t in w2.doc.project.tracks][-2:] == ["crash test", "after save"]
    assert w2.doc.path == proj_file and w2.doc.dirty
    assert not crashed.exists()
    assert not w2.offer_recovery(ask=lambda p, m: "recover")     # nothing left
    for w in (w1, w2):
        w.doc.undo.setClean()
        w.close()


def test_discard_recovery(qapp, tmp_path):
    from stimstudio.gui.document import ProjectDocument
    from stimstudio.gui.main_window import MainWindow
    folder = tmp_path / "autosave"
    folder.mkdir()
    save_project(Project(name="lost"), folder / "lost-abc.stimproj")
    w = MainWindow(ProjectDocument(), user_preset_dir=tmp_path / "p", autosave_dir=folder)
    assert not w.offer_recovery(ask=lambda p, m: "discard")
    assert not list(folder.glob("*.stimproj"))
    w.close()


# ── help ────────────────────────────────────────────────────────────────────
def test_help(window):
    from stimstudio.gui.help import about_text, shortcut_rows
    rows = dict(shortcut_rows(window))
    assert rows["Play"] == "Space" and "Esc" in rows["PANIC — instant stop"]
    assert rows["Keyboard shortcuts"] == "F1"
    dlg = window.show_shortcuts()
    assert "Ctrl+S" in dlg.view.toPlainText()
    dlg.close()
    assert "StimStudio" in about_text()
