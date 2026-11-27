"""Video track: model, document, song editor rules, timeline sync, media helpers, dock."""

import json
import os

import numpy as np
import pytest
from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal

from stimstudio.core.export import stem_tracks
from stimstudio.core.model import Clip, GeneratorSource, Project, Track, VideoFileSource
from stimstudio.core.render import render_project
from stimstudio.gui.video.sync import AFTER, BEFORE, INSIDE, NUDGE, VideoSync
from stimstudio.io.project_file import load_project, save_project
from stimstudio.io.video import find_sibling_funscripts, is_video_file, thumbnail_times

def _have_av():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


needs_av = pytest.mark.skipif(not _have_av(), reason="PyAV not installed")


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    """3 s, 25 fps, 160×90 mp4 (a white bar growing left to right)."""
    if not _have_av():
        pytest.skip("PyAV not installed")
    import av
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=25)
        s.width, s.height, s.pix_fmt = 160, 90, "yuv420p"
        for i in range(75):
            img = np.zeros((90, 160, 3), np.uint8)
            img[:, :i * 2] = 255
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return path


def flat(level=0.5, start=0.0, length=2.0):
    return Clip(GeneratorSource("waveform", {"amp_speed": 0, "amp_min": level, "amp_max": level}),
                start_s=start, length_s=length)


# ── model / render ──────────────────────────────────────────────────────────
def test_video_source_roundtrip_and_render_ignores_it(tmp_path):
    p = Project(tracks=[Track(name="Video", type="video",
                              clips=[Clip(VideoFileSource("C:/x/movie.mp4", 12.5), start_s=3.0,
                                          length_s=12.5)]),
                        Track(name="stim", clips=[flat()])])
    path = tmp_path / "v.stimproj"
    save_project(p, path)
    q = load_project(path)
    src = q.tracks[0].clips[0].source
    assert isinstance(src, VideoFileSource) and src.path == os.path.normpath("C:/x/movie.mp4")
    assert src.duration_s == 12.5 and q.video_track() is q.tracks[0]
    assert json.loads(path.read_text("utf-8"))["tracks"][0]["clips"][0]["source"]["kind"] == "video"

    r = render_project(q)
    assert q.duration_s() == pytest.approx(15.5)                   # video extends the timeline
    assert len(r.master) == round(15.5 * 40)
    assert r.master[:80, 0].min() > 0.4 and r.master[100:, 0].max() == 0
    q.tracks[0].solo = True                                         # soloing the video ≠ silence
    assert render_project(q).master[:80, 0].min() > 0.4
    assert [t.name for t in stem_tracks(q)] == ["stim"]


# ── document ────────────────────────────────────────────────────────────────
def test_set_video_creates_top_track_and_replaces(window):
    doc = window.doc
    n = len(doc.project.tracks)
    clip = doc.set_video("C:/v/a.mp4", 30.0)
    assert len(doc.project.tracks) == n + 1 and doc.project.tracks[0].type == "video"
    assert clip.start_s == 0.0 and clip.length_s == 30.0 and clip.name == "a"
    doc.set_clip_attrs(clip.id, start_s=4.0)
    clip2 = doc.set_video("C:/v/b.mkv", 10.0)
    track = doc.project.video_track()
    assert len(track.clips) == 1 and clip2.id == clip.id
    assert clip2.start_s == 4.0 and clip2.length_s == 10.0 and clip2.source.path == "C:/v/b.mkv"
    doc.undo.undo()                                                 # back to a.mp4
    assert doc.project.video_track().clips[0].source.path == "C:/v/a.mp4"
    doc.undo.undo()
    doc.undo.undo()                                                 # no video track at all
    assert doc.project.video_track() is None and len(doc.project.tracks) == n
    doc.undo.redo()
    assert doc.project.video_track() is not None


def test_video_clip_is_never_copied_or_split(window):
    doc, ed = window.doc, window.editor
    clip = doc.set_video("C:/v/a.mp4", 30.0)
    ed.refresh()
    item = ed.clip_items[clip.id]
    assert item.video and all(item.zone_at(QPointF(x, 5)) == "move" for x in (0, 1, item.width - 1))
    ed.select_clips([clip.id])
    ed.duplicate_selected()
    ed.split_selected(10.0)
    window.edit_copy()
    window.edit_paste()
    assert len(doc.project.video_track().clips) == 1
    assert doc.project.video_track().clips[0].length_s == 30.0
    ed.toggle_mute_selected()
    assert not doc.project.video_track().clips[0].muted
    # Ctrl+drag moves (no copy)
    row = ed.layout.row(doc.project.video_track().id)
    ed.begin_drag(item, "move", QPointF(10, row.y + 5))
    ed.update_drag(QPointF(10 + 5 * ed.pps, row.y + 5), Qt.KeyboardModifier.AltModifier)
    ed.end_drag(Qt.KeyboardModifier.ControlModifier)
    clips = doc.project.video_track().clips
    assert len(clips) == 1 and clips[0].start_s == pytest.approx(5.0)
    # generators never land on the video row
    assert ed._add_clip_at("waveform", QPointF(50, row.y + 5)) is None


def test_video_drop_and_context(window, monkeypatch):
    got = []
    window.editor.videoDropped.disconnect()
    window.editor.videoDropped.connect(got.append)
    from PyQt6.QtCore import QMimeData, QUrl
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("C:/v/movie.MP4")])
    assert window.editor.view._audio_paths(mime) == ["C:/v/movie.MP4"]
    assert is_video_file("x.mkv") and not is_video_file("x.wav")


# ── sync ────────────────────────────────────────────────────────────────────
class FakePlayer:
    def __init__(self):
        self.pos = 0.0
        self.playing = False
        self.rate = 1.0
        self.seeks = []

    def position(self):
        return self.pos

    def setPosition(self, ms):
        self.pos = ms
        self.seeks.append(ms)

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def setPlaybackRate(self, r):
        self.rate = r


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_sync():
    states = []
    clock = Clock()
    player = FakePlayer()
    sync = VideoSync(player, states.append, clock)
    sync.set_clip(10.0, 20.0)            # video at 10–30 s on the timeline
    return sync, player, states, clock


def test_sync_scrub_and_states():
    sync, player, states, _ = make_sync()
    sync.scrub(15.0)
    assert player.seeks[-1] == 5000 and sync.state == INSIDE
    sync.scrub(5.0)
    assert sync.state == BEFORE and player.seeks[-1] == 0            # first frame
    sync.scrub(40.0)
    assert sync.state == AFTER and player.seeks[-1] == 19999         # last frame
    sync.offset_ms = 250
    sync.scrub(15.0)
    assert player.seeks[-1] == 5250
    assert states == [INSIDE, BEFORE, AFTER, INSIDE]


def test_sync_playback_drift_correction():
    sync, player, _, clock = make_sync()
    sync.playhead(5.0)                                   # before the clip: nothing plays
    assert not player.playing
    sync.playhead(12.0)                                  # enters: seek + play
    assert player.playing and player.seeks[-1] == 2000 and player.rate == 1.0
    n = len(player.seeks)
    clock.t = 1.0
    player.pos = 3000 + 80                               # 80 ms ahead → slow down a little
    sync.playhead(13.0)
    assert player.rate == pytest.approx(1 - NUDGE) and len(player.seeks) == n
    player.pos = 4000 - 60                               # 60 ms behind → speed up
    sync.playhead(14.0)
    assert player.rate == pytest.approx(1 + NUDGE)
    player.pos = 5000 + 10                               # in sync
    sync.playhead(15.0)
    assert player.rate == 1.0
    player.pos = 1000                                    # big jump (loop / seek) → seek
    sync.playhead(16.0)
    assert player.seeks[-1] == 6000 and len(player.seeks) == n + 1
    player.pos = 0                                       # right after a seek: settle, no reseek
    sync.playhead(16.1)
    assert len(player.seeks) == n + 1
    sync.playhead(31.0)                                  # leaves the clip: pause
    assert not player.playing and sync.state == AFTER


def test_sync_stop_returns_to_cursor():
    sync, player, _, _ = make_sync()
    sync.playhead(12.0)
    sync.stopped(20.0)
    assert not player.playing and player.seeks[-1] == 10000 and not sync.running
    sync.clear()
    sync.scrub(15.0)
    assert player.seeks[-1] == 10000                     # no clip: untouched


def test_sync_fast_forward_rate():
    sync, player, _, clock = make_sync()
    sync.playhead(12.0, rate=5.0)                         # enters at 5×
    assert player.playing and player.rate == 5.0 and player.seeks[-1] == 2000
    n = len(player.seeks)
    clock.t = 1.0
    player.pos = 7000 + 400                               # 400 ms ahead: a nudge at 5×, no seek
    sync.playhead(17.0, rate=5.0)
    assert len(player.seeks) == n and player.rate == pytest.approx(5.0 * (1 - NUDGE))
    player.pos = 8000 + 50
    sync.playhead(18.0, rate=5.0)
    assert player.rate == 5.0
    player.pos = 20000                                     # far off: seek
    sync.playhead(19.0, rate=5.0)
    assert player.seeks[-1] == 9000
    clock.t = 2.0
    player.pos = 10000
    sync.playhead(20.0)                                    # back to normal playback
    assert player.rate == 1.0


# ── fast-forward (shuttle) ──────────────────────────────────────────────────
class FakeDoc(QObject):
    cursorChanged = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.cursor_s = 0.0

    def set_cursor(self, s):
        s = max(0.0, round(float(s) * 40) / 40)
        if s != self.cursor_s:
            self.cursor_s = s
            self.cursorChanged.emit(s)


class FakePlayback(QObject):
    playingChanged = pyqtSignal(bool)
    panicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.playing = False
        self.pos = 0.0
        self.stops = 0
        self.plays = 0

    def position_s(self):
        return self.pos if self.playing else -1.0

    def stop(self):
        self.stops += 1                # fades: still "playing" until the test says so

    def play(self):
        self.plays += 1
        self.playing = True
        self.playingChanged.emit(True)
        return True


def make_shuttle(qapp, end=100.0):
    from stimstudio.gui.video.shuttle import Shuttle
    doc, pb, clock = FakeDoc(), FakePlayback(), Clock()
    sh = Shuttle(doc, pb, end_s=lambda: end, clock=clock)
    sh.timer.stop()
    sh.timer.start = lambda *a: None          # ticks are driven by hand
    return sh, doc, pb, clock


def test_shuttle_runs_cursor_at_speed(qapp):
    sh, doc, pb, clock = make_shuttle(qapp)
    doc.set_cursor(10.0)
    seen = []
    sh.positionChanged.connect(seen.append)
    sh.toggle(5)
    assert sh.active and sh.speed == 5 and pb.stops == 0 and not sh.resume
    clock.t = 2.0
    sh.tick()
    assert doc.cursor_s == 20.0 and seen[-1] == pytest.approx(20.0)
    sh.toggle(10)                                          # change speed: re-anchored
    clock.t = 3.0
    sh.tick()
    assert doc.cursor_s == 30.0
    doc.set_cursor(50.0)                                   # the user clicks the timeline
    clock.t = 3.5
    sh.tick()
    assert doc.cursor_s == 55.0
    sh.toggle(10)                                          # same button: off, nothing to resume
    assert not sh.active and pb.plays == 0


def test_shuttle_pauses_output_and_resumes_after_fade(qapp):
    sh, doc, pb, clock = make_shuttle(qapp)
    pb.playing, pb.pos = True, 12.0
    sh.toggle(2)
    assert pb.stops == 1 and sh.resume and doc.cursor_s == 12.0
    clock.t = 1.0
    sh.tick()
    assert doc.cursor_s == 14.0
    sh.toggle(2)                                           # off while the fade still runs
    assert pb.plays == 0 and sh.busy and not sh.active
    sh.tick()
    assert pb.plays == 0
    pb.playing = False                                     # fade done
    sh.tick()
    assert pb.plays == 1 and not sh.busy
    played = []
    sh2, doc2, pb2, _ = make_shuttle(qapp)
    sh2.play_fn = lambda: played.append(doc2.cursor_s)     # MainWindow's safety-checked play
    pb2.playing, pb2.pos = True, 3.0
    sh2.toggle(5)
    pb2.playing = False
    sh2.toggle(5)
    assert played == [3.0] and pb2.plays == 0


def test_shuttle_panic_and_end_never_resume(qapp):
    sh, doc, pb, clock = make_shuttle(qapp, end=20.0)
    pb.playing, pb.pos = True, 5.0
    sh.toggle(10)
    pb.panicked.emit()                                     # Esc
    assert not sh.busy and not sh.resume
    pb.playing = False
    sh.tick()
    assert pb.plays == 0
    pb.playing, pb.pos = True, 5.0
    sh.toggle(10)
    pb.playing = False
    clock.t = 5.0                                          # 5 + 50 s > end
    sh.tick()
    assert not sh.busy and doc.cursor_s == 20.0 and pb.plays == 0


def test_shuttle_ends_when_output_starts(qapp):
    sh, doc, pb, clock = make_shuttle(qapp)
    sh.toggle(2)
    pb.play()                                              # Space / Record while fast-forwarding
    assert not sh.busy and pb.plays == 1


def test_video_panel_fast_forward_buttons(window):
    panel = window.video
    window.doc.set_cursor(1.0)
    panel.ff_buttons[5].click()
    assert panel.shuttle.speed == 5 and panel.ff_buttons[5].isChecked()
    assert not panel.ff_buttons[2].isChecked()
    panel.ff_buttons[2].click()
    assert panel.shuttle.speed == 2 and not panel.ff_buttons[5].isChecked()
    panel.ff_buttons[2].click()
    assert not panel.shuttle.active and not any(b.isChecked() for b in panel.ff_buttons.values())
    # PANIC cancels fast-forward and its resume
    assert window.playback.play()
    panel.ff_buttons[10].click()
    assert panel.shuttle.resume
    window.a_panic.trigger()
    assert not panel.shuttle.busy and not window.playback.playing


# ── media helpers ───────────────────────────────────────────────────────────
def test_find_sibling_funscripts(tmp_path):
    (tmp_path / "Movie.mp4").write_bytes(b"")
    for name in ("Movie.funscript", "movie.roll.funscript", "Movie.part2.x.funscript",
                 "Other.funscript", "Movie.txt"):
        (tmp_path / name).write_text("{}")
    found = sorted(p.split("\\")[-1].split("/")[-1] for p in
                   find_sibling_funscripts(tmp_path / "Movie.mp4"))
    assert found == ["Movie.funscript", "movie.roll.funscript"]


def test_thumbnail_times():
    t = thumbnail_times(10.0, every_s=2.0)
    assert len(t) == 5 and t[0] == pytest.approx(1.0) and t[-1] == pytest.approx(9.0)
    assert len(thumbnail_times(10_000.0)) == 400 and len(thumbnail_times(0)) == 0


@needs_av
def test_probe_and_thumbnails(qapp, video_file):
    from stimstudio.gui.video.media import ThumbnailCache, probe_duration
    assert probe_duration(str(video_file)) == pytest.approx(3.0, abs=0.05)
    cache = ThumbnailCache()
    strip = cache.strip(str(video_file), 3.0)
    assert cache.wait(10000)
    assert strip.ready() == len(strip.times) >= 1
    img = strip.image_at(2.9)
    assert img is not None and img.height() == 36
    assert cache.strip(str(video_file), 3.0) is strip                # cached


@needs_av
def test_open_video_in_window(window, video_file, tmp_path):
    import shutil
    folder = tmp_path / "with_script"
    folder.mkdir()
    video = folder / "clip.mp4"
    shutil.copy(video_file, video)
    (folder / "clip.funscript").write_text(json.dumps(
        {"actions": [{"at": 0, "pos": 0}, {"at": 1000, "pos": 100}]}), encoding="utf-8")
    asked, imported = [], []
    window.import_funscript = lambda paths, start_s=None, **k: imported.append((paths, start_s))
    clip = window.open_video(str(video), ask_funscripts=lambda p: asked.append(p) or True)
    assert clip.length_s == pytest.approx(3.0, abs=0.05)
    assert window.doc.project.tracks[0].type == "video"
    assert asked and imported == [([str(folder / "clip.funscript")], 0.0)]
    panel = window.video
    assert panel.path == str(video) and panel.sync.has_clip
    window.doc.set_cursor(1.0)
    panel.sync.scrub(1.0)
    assert panel.sync.state == INSIDE
    window.editor.refresh()
    assert window.editor.clip_items[clip.id].video


# ── sound lane, view toggles, full screen ───────────────────────────────────
@pytest.fixture(scope="module")
def video_with_sound(tmp_path_factory):
    """3 s video whose sound is a 440 Hz tone for 1.5 s, then silence."""
    if not _have_av():
        pytest.skip("PyAV not installed")
    import av
    path = tmp_path_factory.mktemp("video") / "sound.mp4"
    sr = 44100
    t = np.arange(3 * sr) / sr
    tone = (0.8 * np.sin(2 * np.pi * 440 * t) * (t < 1.5)).astype(np.float32)
    with av.open(str(path), "w") as c:
        vs = c.add_stream("mpeg4", rate=25)
        vs.width, vs.height, vs.pix_fmt = 64, 36, "yuv420p"
        aus = c.add_stream("aac", rate=sr)
        aus.layout = "mono"
        for i in range(75):
            frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), i * 3, np.uint8),
                                               format="rgb24")
            for pk in vs.encode(frame):
                c.mux(pk)
        for pk in vs.encode():
            c.mux(pk)
        for k, start in enumerate(range(0, len(tone), 1024)):
            block = tone[start:start + 1024]
            if len(block) < 1024:
                block = np.pad(block, (0, 1024 - len(block)))
            af = av.AudioFrame.from_ndarray(block[None, :], format="fltp", layout="mono")
            af.sample_rate, af.pts = sr, start
            for pk in aus.encode(af):
                c.mux(pk)
        for pk in aus.encode():
            c.mux(pk)
    return path


@needs_av
def test_audio_peaks_follow_the_sound(video_with_sound, video_file):
    from stimstudio.io.video import PEAK_RATE, has_audio, iter_audio_peaks
    chunks = list(iter_audio_peaks(video_with_sound, chunk_s=0.5))
    assert len(chunks) > 1 and chunks[0][0] == 0
    peaks = np.concatenate([p for _, p in chunks])
    assert len(peaks) == pytest.approx(3 * PEAK_RATE, abs=15)
    assert peaks[20:130].min() > 0.5 and peaks[170:280].max() < 0.05     # tone, then silence
    assert not has_audio(video_file) and list(iter_audio_peaks(video_file)) == []


def test_waveform_columns():
    from stimstudio.gui.video.media import Waveform
    w = Waveform(10.0)
    w.add(0, np.linspace(0, 1, 1001).astype(np.float32))
    np.testing.assert_allclose(w.column_peaks(0, 1.0, 10), np.arange(1, 11) * 0.1 - 0.001,
                               atol=2e-3)                            # max of each second
    assert np.all(w.column_peaks(5.0, 0.001, 4) == pytest.approx(0.5))   # zoomed in: one bin
    assert w.top == 1.0 and w.ready_bins == 1001


@needs_av
def test_sound_lane_and_view_toggles(window, video_with_sound):
    from stimstudio.gui.video.media import waveform_cache
    ed = window.editor
    clip = window.open_video(str(video_with_sound), ask_funscripts=lambda p: False)
    track = window.doc.project.video_track()
    item = ed.clip_items[clip.id]
    assert track.id in ed.expanded and item.audio_lane              # sound lane by default
    assert ed.layout.row(track.id).n_lanes == 1 and item.height > 60
    wave = waveform_cache().waveform(str(video_with_sound), clip.length_s)
    assert waveform_cache().wait() and wave.finished and wave.has_audio
    header = ed.headers.headers[track.id]
    header.audio_btn.setChecked(False)                               # hide the sound lane
    assert track.id not in ed.expanded and not ed.clip_items[clip.id].audio_lane
    assert ed.layout.row(track.id).n_lanes == 0
    header.thumbs_btn.setChecked(False)                              # hide the frames
    assert not ed.video_thumbs and not ed.clip_items[clip.id].show_thumbs
    header.thumbs_btn.setChecked(True)
    header.audio_btn.setChecked(True)
    assert ed.clip_items[clip.id].audio_lane and header.video_frames[0].isVisible()
    window.grab()                                                    # paints filmstrip + wave


@needs_av
def test_fullscreen_window_on_the_dock_screen(window, video_file):
    window.open_video(str(video_file), ask_funscripts=lambda p: False)
    panel = window.video
    screen = panel.target_screen()
    panel.set_fullscreen(True)
    assert panel.is_fullscreen() and panel.player.videoOutput() is panel.fullscreen.view
    assert panel.fullscreen.geometry() == screen.geometry()
    assert panel.view.window() is window                             # docked view stays put
    window.a_panic.trigger()                                         # Esc / PANIC leaves
    assert not panel.is_fullscreen() and panel.player.videoOutput() is panel.view
    panel.toggle_fullscreen()
    panel.toggle_fullscreen()
    assert not panel.is_fullscreen()
