"""Relative media paths in project files, Collect files, missing-file relinking."""

import json
import os
import shutil

import pytest

from stimstudio.core import media as md
from stimstudio.core.model import (AudioFileSource, Clip, GeneratorSource, Project, Track,
                                   VideoFileSource)
from stimstudio.io.project_file import load_project, save_project


def touch(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def media_project(video, audio=None, wav=None) -> Project:
    tracks = [Track(name="Video", type="video",
                    clips=[Clip(VideoFileSource(str(video), 10.0), length_s=10.0, name="vid")])]
    if audio is not None:
        tracks.append(Track(name="Mod", type="audio_mod",
                            clips=[Clip(AudioFileSource(str(audio)), length_s=5.0, name="song")]))
    if wav is not None:
        tracks.append(Track(name="Wav", clips=[Clip(GeneratorSource("wav", {"path": str(wav)}),
                                                    length_s=5.0, name="wav")]))
    return Project(tracks=tracks)


def stored_source(path, track=0):
    return json.loads(path.read_text("utf-8"))["tracks"][track]["clips"][0]["source"]


# ── project files ───────────────────────────────────────────────────────────
def test_paths_are_saved_relative_with_absolute_fallback(tmp_path):
    video = touch(tmp_path / "proj" / "media" / "movie.mp4")
    wav = touch(tmp_path / "proj" / "sounds" / "a.wav")
    path = tmp_path / "proj" / "p.stimproj"
    save_project(media_project(video, wav=wav), path)
    src = stored_source(path)
    assert src["path"] == "media/movie.mp4"
    assert src["abs_paths"]["path"] == md.norm(video)
    gen = stored_source(path, 1)
    assert gen["params"]["path"] == "sounds/a.wav"
    assert gen["abs_paths"]["param:path"] == md.norm(wav)
    p = load_project(path)
    assert p.tracks[0].clips[0].source.path == md.norm(video)
    assert p.tracks[1].clips[0].source.params["path"] == md.norm(wav)
    assert not hasattr(p.tracks[0].clips[0].source, "abs_paths")


def test_moving_the_project_folder_keeps_media(tmp_path):
    video = touch(tmp_path / "proj" / "media" / "movie.mp4")
    audio = touch(tmp_path / "proj" / "song.wav")
    save_project(media_project(video, audio), tmp_path / "proj" / "p.stimproj")
    shutil.move(str(tmp_path / "proj"), str(tmp_path / "moved"))
    p = load_project(tmp_path / "moved" / "p.stimproj")
    assert p.tracks[0].clips[0].source.path == md.norm(tmp_path / "moved" / "media" / "movie.mp4")
    assert p.tracks[1].clips[0].source.path == md.norm(tmp_path / "moved" / "song.wav")
    assert md.missing_files(p) == []


def test_project_file_alone_falls_back_to_absolute(tmp_path):
    video = touch(tmp_path / "lib" / "movie.mp4")
    (tmp_path / "proj").mkdir()
    save_project(media_project(video), tmp_path / "proj" / "p.stimproj")
    (tmp_path / "elsewhere").mkdir()
    shutil.copy(tmp_path / "proj" / "p.stimproj", tmp_path / "elsewhere" / "p.stimproj")
    p = load_project(tmp_path / "elsewhere" / "p.stimproj")
    assert p.tracks[0].clips[0].source.path == md.norm(video)


def test_missing_everywhere_keeps_the_original_location(tmp_path):
    video = touch(tmp_path / "lib" / "movie.mp4")
    (tmp_path / "proj").mkdir()
    save_project(media_project(video), tmp_path / "proj" / "p.stimproj")
    video.unlink()
    p = load_project(tmp_path / "proj" / "p.stimproj")
    assert p.tracks[0].clips[0].source.path == md.norm(video)
    (missing,) = md.missing_files(p)
    assert missing.kind == "video" and missing.refs[0].clip_name == "vid"


def test_other_drive_is_stored_absolute(tmp_path, monkeypatch):
    video = touch(tmp_path / "movie.mp4")

    def no_relpath(*_a, **_k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(md.os.path, "relpath", no_relpath)
    save_project(media_project(video), tmp_path / "p.stimproj")
    assert stored_source(tmp_path / "p.stimproj")["path"] == md.norm(video)


def test_old_absolute_projects_still_load(tmp_path):
    video = touch(tmp_path / "movie.mp4")
    data = media_project(video).to_dict()            # no abs_paths: an old file
    (tmp_path / "old.stimproj").write_text(json.dumps(data), "utf-8")
    p = load_project(tmp_path / "old.stimproj")
    assert p.tracks[0].clips[0].source.path == md.norm(video)


def test_absolute_only_save_for_autosave(tmp_path):
    video = touch(tmp_path / "proj" / "movie.mp4")
    (tmp_path / "auto").mkdir()
    save_project(media_project(video), tmp_path / "auto" / "a.stimproj", base_dir=None)
    assert stored_source(tmp_path / "auto" / "a.stimproj")["path"] == md.norm(video)
    save_project(media_project(video), tmp_path / "auto" / "b.stimproj",
                 base_dir=tmp_path / "proj")
    assert stored_source(tmp_path / "auto" / "b.stimproj")["path"] == "movie.mp4"
    p = load_project(tmp_path / "auto" / "b.stimproj", base_dir=tmp_path / "proj")
    assert p.tracks[0].clips[0].source.path == md.norm(video)


# ── collect / relink (core) ─────────────────────────────────────────────────
def test_plan_collect_names_and_actions(tmp_path):
    proj = tmp_path / "proj"
    a = touch(tmp_path / "x" / "clip.mp4", b"first")
    b = touch(tmp_path / "y" / "clip.wav", b"song")
    inside = touch(proj / "own" / "a.wav", b"inside")
    same_name = touch(tmp_path / "z" / "clip.mp4", b"second")      # name clash with a
    gone = tmp_path / "gone.wav"
    reuse_src = touch(tmp_path / "r" / "keep.wav", b"keep")
    touch(proj / "media" / "keep.wav", b"keep")                      # already collected
    p = Project(tracks=[
        Track(type="video", clips=[Clip(VideoFileSource(str(a), 1.0), length_s=1.0)]),
        Track(type="audio_mod", clips=[Clip(AudioFileSource(str(b))),
                                       Clip(AudioFileSource(str(inside))),
                                       Clip(AudioFileSource(str(gone))),
                                       Clip(AudioFileSource(str(reuse_src))),
                                       Clip(AudioFileSource(str(b)), start_s=20)]),
        Track(clips=[Clip(GeneratorSource("wav", {"path": str(same_name)}))])])
    items = {os.path.basename(os.path.dirname(i.file.path)) + "/" + os.path.basename(i.file.path):
             i for i in md.plan_collect(p, proj, proj / "media")}
    assert len(items) == 6                                   # b is used twice, listed once
    assert items["y/clip.wav"].action == md.COPY
    assert items["own/a.wav"].action == md.INSIDE
    assert items[tmp_path.name + "/gone.wav"].action == md.MISSING
    assert items["r/keep.wav"].action == md.REUSE
    names = sorted(os.path.basename(items[k].target) for k in ("x/clip.mp4", "z/clip.mp4"))
    assert names == ["clip (2).mp4", "clip.mp4"]


def test_run_collect_copies_and_relinks(tmp_path):
    proj = tmp_path / "proj"
    a = touch(tmp_path / "lib" / "movie.mp4", os.urandom(9 * 1024 * 1024 + 7))
    p = media_project(a)
    seen = []
    mapping = md.run_collect(md.plan_collect(p, proj, proj / "media"),
                             lambda name, f: seen.append(f))
    target = proj / "media" / "movie.mp4"
    assert mapping == {md.norm(a): md.norm(target)}
    assert target.read_bytes() == a.read_bytes() and a.exists()
    assert seen[-1] == pytest.approx(1.0) and seen == sorted(seen)
    changed = md.relinked_sources(p, mapping)
    assert changed[p.tracks[0].clips[0].id].path == md.norm(target)


def test_cancelled_copy_leaves_nothing(tmp_path):
    a = touch(tmp_path / "big.bin", os.urandom(9 * 1024 * 1024))
    calls = []
    with pytest.raises(md.Cancelled):
        md.copy_file(str(a), str(tmp_path / "out" / "big.bin"), calls.append,
                     lambda: len(calls) >= 1)
    assert not any((tmp_path / "out").iterdir())


def test_relink_by_name_searches_subfolders(tmp_path):
    touch(tmp_path / "new" / "deep" / "movie.mp4")
    touch(tmp_path / "new" / "Song.WAV")
    mapping = md.relink_by_name([str(tmp_path / "old" / "movie.mp4"),
                                 str(tmp_path / "old" / "song.wav"),
                                 str(tmp_path / "old" / "nope.wav")], tmp_path / "new")
    assert mapping == {str(tmp_path / "old" / "movie.mp4"):
                       str(tmp_path / "new" / "deep" / "movie.mp4"),
                       str(tmp_path / "old" / "song.wav"): str(tmp_path / "new" / "Song.WAV")}


# ── GUI ─────────────────────────────────────────────────────────────────────
def test_media_dialog_collect_and_undo(window, tmp_path):
    from stimstudio.gui.media_dialog import ProjectMediaDialog
    video = touch(tmp_path / "lib" / "movie.mp4", b"video")
    wav = touch(tmp_path / "lib" / "a.wav", b"wav")
    doc = window.doc
    doc.set_video(str(video), 10.0)
    tid = doc.project.tracks[1].id
    doc.add_clip(tid, Clip(GeneratorSource("wav", {"path": str(wav)}), start_s=30, length_s=2))
    (tmp_path / "proj").mkdir()
    doc.save(str(tmp_path / "proj" / "p.stimproj"))
    dlg = ProjectMediaDialog(doc, window, save=window.file_save)
    assert dlg.tree.topLevelItemCount() == 2 and not dlg.missing()
    assert dlg.collect(str(tmp_path / "proj" / "media"), save_after=True) == 2
    assert doc.project.video_track().clips[0].source.path == md.norm(
        tmp_path / "proj" / "media" / "movie.mp4")
    assert not doc.dirty                                        # saved afterwards
    src = stored_source(tmp_path / "proj" / "p.stimproj")
    assert src["path"] == "media/movie.mp4"
    assert dlg.tree.topLevelItem(0).text(4) == "OK · in project folder"
    doc.undo.undo()                                             # one step back to the originals
    assert doc.project.video_track().clips[0].source.path == md.norm(video)
    dlg.deleteLater()


def test_missing_files_found_in_folder(window, tmp_path):
    from stimstudio.gui.media_dialog import ProjectMediaDialog
    doc = window.doc
    doc.set_video(str(tmp_path / "old" / "movie.mp4"), 10.0)
    asked = []
    assert window.check_missing_media(ask=lambda m: asked.append(m) or False) == 1
    assert asked[0][0].path == md.norm(tmp_path / "old" / "movie.mp4")
    new = touch(tmp_path / "found" / "movie.mp4")
    dlg = ProjectMediaDialog(doc, window)
    assert dlg.find_btn.isEnabled()
    assert dlg.find_in_folder(str(tmp_path / "found")) == 1
    assert doc.project.video_track().clips[0].source.path == md.norm(new)
    assert not dlg.missing() and window.check_missing_media(ask=lambda m: True) == 0
    dlg.deleteLater()
