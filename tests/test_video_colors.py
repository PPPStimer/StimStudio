"""Color rhythm: palette, per-colour statistics, fill → pulses → beats, the Tools window."""

import numpy as np
import pytest

from stimstudio.core.beats import detect_peaks
from stimstudio.core.pulse import FS
from stimstudio.io import video_colors as vc

FPS = 25
W, H = 160, 96
DURATION = 5.0
BEATS = (1.0, 2.0, 2.5, 3.5, 4.0)          # the red square flashes here …
FLASH = 0.2                                 # … for this long
TARGET = (100, 20, 50, 60)                  # x, y, w, h in pixels
ROI = (TARGET[0] / W, TARGET[1] / H, TARGET[2] / W, TARGET[3] / H)
RED_SHARE = 30 * 30 / (50 * 60)             # red square / target area


def _frame(t: float) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[30:60, 10:40] = (30, 60, 230)                       # a blue shape outside the target
    x, y, w, h = TARGET
    img[y:y + h, x:x + w] = (90, 90, 90)                    # the grey target
    if any(b <= t < b + FLASH for b in BEATS):
        img[y + 15:y + 45, x + 10:x + 40] = (235, 20, 25)   # the flash
    return img


def _frames():
    for i in range(int(DURATION * FPS)):
        t = i / FPS
        yield t, _frame(t)


def _bin(rgb) -> int:
    return int(vc.color_bins(np.array([[rgb]], np.uint8))[0, 0])


def test_palette():
    red, wrap, blue = _bin((255, 0, 0)), _bin((255, 0, 8)), _bin((30, 60, 230))
    assert red == wrap == _bin((235, 20, 25))               # hue 0° and ~358° share a bin
    assert len({red, blue, _bin((0, 255, 0)), _bin((255, 128, 0)), _bin((128, 128, 128)),
                _bin((0, 0, 0)), _bin((255, 255, 255)), _bin((100, 0, 0))}) == 8
    assert _bin((0, 0, 0)) == 0 and _bin((255, 255, 255)) == vc.GREY_BINS - 1
    assert _bin((20, 0, 0)) == 0                             # very dark → black, not red
    assert all(0 <= _bin(c) < vc.N_BINS for c in np.random.default_rng(1).integers(0, 256, (200, 3)))
    assert "red" in vc.bin_label(red) and "blue" in vc.bin_label(blue)
    assert vc.bin_label(0) == "black"
    r, g, b = vc.bin_rgb(red)
    assert r > 200 and g < 80 and b < 80
    img = _frame(BEATS[0])
    assert vc.dominant_bin(img, 125, 50) == red
    assert vc.dominant_bin(img, 102, 22) == _bin((90, 90, 90))
    mask = vc.bin_mask(img, [red])
    assert mask.sum() == 30 * 30


def test_appearances_hysteresis():
    p = 0.1
    col = np.array([0, .2, .2, .07, .2, 0, 0, .3, .04, .3, 0])     # .07 is not "gone"
    counts = vc.appearance_counts(col[:, None].astype(np.float32), p)
    assert counts.tolist() == [3]
    assert vc.appearance_counts(np.zeros((0, 2), np.float32), p).tolist() == [0, 0]


def test_stats_and_fill():
    an = vc.analyse_color_frames(_frames(), FPS, ROI, DURATION)
    assert an.n == int(DURATION * FPS) and an.frac.shape == (an.n, vc.N_BINS)
    assert an.roi_size == (50, 60)
    assert np.allclose(an.frac.sum(axis=1), 1.0, atol=1e-5)
    red = _bin((235, 20, 25))
    stats = {s.bin: s for s in vc.color_stats(an)}
    assert _bin((30, 60, 230)) not in stats                  # outside the region
    s = stats[red]
    assert s.appearances == len(BEATS)
    assert s.frames_pct == pytest.approx(100 * len(BEATS) * FLASH / DURATION, abs=1.0)
    assert s.screen_pct == pytest.approx(s.frames_pct * RED_SHARE, abs=0.5)
    assert s.peak_pct == pytest.approx(100 * RED_SHARE, abs=0.1)
    assert tuple(s.rgb) == (235, 20, 25) and s.hex == "#eb1419"
    grey = stats[_bin((90, 90, 90))]
    assert list(stats.values()) == sorted(stats.values(), key=lambda x: -x.screen_pct)
    assert grey.frames_pct == 100.0 and grey.appearances == 1
    sig = vc.fill_signal(an, [red])
    assert sig.max() == pytest.approx(RED_SHARE, abs=1e-6) and sig.min() == 0.0
    assert np.allclose(vc.fill_signal(an, [red, grey.bin]), 1.0, atol=1e-5)
    assert not vc.fill_signal(an, []).any()
    whole = vc.analyse_color_frames(_frames(), FPS, None, DURATION)
    assert {s.bin for s in vc.color_stats(whole)} >= {red, _bin((30, 60, 230))}


def test_fill_to_pulses_and_beats():
    an = vc.analyse_color_frames(_frames(), FPS, ROI, DURATION)
    sig = vc.fill_signal(an, [_bin((235, 20, 25))])
    p = vc.fill_to_pulses(sig, an.rate, an.duration_s, freq=0.3)
    assert p.shape == (int(DURATION * FS), 4) and p.dtype == np.float32
    assert p[:, 0].max() == pytest.approx(1.0) and np.array_equal(p[:, 0], p[:, 1])
    assert np.allclose(p[:, 2:], 0.3)
    raw = vc.fill_to_pulses(sig, an.rate, an.duration_s, normalize=False)
    assert raw[:, 0].max() == pytest.approx(RED_SHARE, abs=1e-6)
    ranged = vc.fill_to_pulses(sig, an.rate, an.duration_s, floor=0.2, ceiling=0.8)
    assert ranged[:, 0].min() == pytest.approx(0.2) and ranged[:, 0].max() == pytest.approx(0.8)
    smooth = vc.fill_to_pulses(sig, an.rate, an.duration_s, smooth_s=0.5)
    assert smooth[:, 0].max() == pytest.approx(1.0) and np.diff(smooth[:, 0]).std() < \
        np.diff(p[:, 0]).std()
    idx, _h = detect_peaks(p[:, 0], 0.5, "cross", min_gap_s=0.1)
    assert np.allclose(idx / FS, BEATS, atol=1.0 / FPS + 1.0 / FS)
    assert not vc.fill_to_pulses(np.zeros(10), FPS, 1.0)[:, 0].any()


def test_decode_width():
    assert vc.decode_width(None) == vc.BASE_WIDTH
    assert vc.decode_width((0, 0, 0.1, 0.1)) == 1280
    assert vc.decode_width((0, 0, 0.25, 0.1)) == 512


# ── real files and the window ───────────────────────────────────────────────
def _have_av():
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def flashes_file(tmp_path_factory):
    if not _have_av():
        pytest.skip("PyAV not installed")
    import av
    path = tmp_path_factory.mktemp("vc") / "flashes.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"qscale": "2"}
        for _t, img in _frames():
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return str(path)


def _red_beats(an):
    red = _bin((235, 20, 25))
    p = vc.fill_to_pulses(vc.fill_signal(an, [red]), an.rate, an.duration_s)
    idx, _h = detect_peaks(p[:, 0], 0.5, "cross", min_gap_s=0.1)
    return idx / FS


def test_real_file(flashes_file):
    from stimstudio.io.video import grab_frame
    seen = []
    an = vc.analyse_video_colors(flashes_file, FPS, ROI, progress=seen.append)
    assert an.duration_s == pytest.approx(DURATION, abs=0.1)
    assert seen and seen[-1] == 1.0
    assert an.roi_size[0] >= 50                                # decoded at source width
    assert np.allclose(_red_beats(an), BEATS, atol=0.1)
    frame = grab_frame(flashes_file, BEATS[1] + 0.1, 360)
    assert frame.shape == (H, W, 3)
    assert vc.dominant_bin(frame, 125, 50) == _bin((235, 20, 25))
    off = grab_frame(flashes_file, BEATS[1] - 0.3, 360)
    assert vc.dominant_bin(off, 125, 50) != _bin((235, 20, 25))


def test_tools_menu(window):
    titles = {a.text().replace("&", ""): a for a in window.menuBar().actions()}
    tools = titles["Tools"].menu()
    texts = [a.text() for a in tools.actions() if not a.isSeparator()]
    assert texts[0].replace("&", "").startswith("Color rhythm")
    assert window.a_video_motion in tools.actions() and window.a_video_cuts in tools.actions()
    win = window.show_tool("color_rhythm")
    assert window.show_tool("color_rhythm") is win and win.isVisible()
    assert win.path is None and not win.create_btn.isEnabled()
    win.close()


def test_window_end_to_end(window, flashes_file):
    from stimstudio.gui.tools import color_rhythm as cr
    doc = window.doc
    window.open_video(flashes_file, ask_funscripts=lambda p: False)
    vt = doc.project.video_track()
    doc.set_clip_attrs(vt.clips[0].id, start_s=2.0)
    win = window.show_tool("color_rhythm")
    assert win.path == flashes_file and win.from_project
    win.picker.set_roi(ROI)
    win.rate.setCurrentIndex(cr.RATES.index(25))
    assert win.run_btn.text() == "Analyse"
    assert win.run_analysis()
    assert win.run_btn.text().startswith("Analysed")
    assert cr.cached_analysis(flashes_file, 25, ROI, "ellipse", 0.7) is win.analysis
    assert win.table.rowCount() >= 2

    win.time_slider.setValue(int((BEATS[1] + 0.1) * 1000))
    win._load_frame()
    assert win.measure_key() == "difference"
    red = win.pick_color(125, 50)                             # eyedropper on the flash
    assert red == _bin((235, 20, 25)) and win.selected == {red}
    assert win.measure_key() == "colours"                     # picking switched the measure
    assert win.create_btn.isEnabled()
    from PyQt6.QtCore import Qt
    checked = [win.table.item(r, 0) for r in range(win.table.rowCount())
               if win.table.item(r, 0).checkState() == Qt.CheckState.Checked]
    assert [int(i.data(Qt.ItemDataRole.UserRole)) for i in checked] == [red]
    assert np.array_equal(win.picker.highlight, vc.bin_mask(win.picker.rgb, {red}))

    n_tracks = len(doc.project.tracks)
    clips = win.create_clip()
    assert len(clips) == 1
    track, clip = doc.project.find_clip(clips[0].id)
    assert doc.project.tracks.index(track) == doc.project.tracks.index(vt) + 1
    assert clip.start_s == 2.0 and clip.length_s == pytest.approx(DURATION, abs=0.1)
    assert "red" in track.name.lower() and clip.source.origin.endswith("flashes.mp4")
    marks = window.marks_from_peaks(clip.id, {"level": 0.5})
    found = doc.project.find_clip(marks.id)[1] if marks is not None else None
    assert found is not None and len(found.source.marks) == len(BEATS)
    doc.undo.undo()                                           # the marks
    doc.undo.undo()                                           # the colour clip: one step
    assert len(doc.project.tracks) == n_tracks

    win.separate.setChecked(True)
    win.set_selected([red, _bin((90, 90, 90))])
    data = win.result_data()
    assert [len(c["bins"]) for c in data["clips"]] == [1, 1]
    assert data["clips"][0]["bins"] == [_bin((90, 90, 90))]   # most screen time first
    clips = window.add_color_rhythm(data)
    assert len(clips) == 2 and len(doc.project.tracks) == n_tracks + 2
    doc.undo.undo()
    assert len(doc.project.tracks) == n_tracks
    win.close()


def test_other_file_lands_at_cursor(window):
    doc = window.doc
    doc.set_cursor(3.0)
    pulses = np.full((80, 4), 0.5, np.float32)
    n = len(doc.project.tracks)
    clips = window.add_color_rhythm({"path": "x.mp4", "from_project": False,
                                     "clips": [{"name": "Red", "color": "#ff0000",
                                                "pulses": pulses}]})
    track, clip = doc.project.find_clip(clips[0].id)
    assert clip.start_s == 3.0 and track.color == "#ff0000"
    assert doc.project.tracks[-1] is track and len(doc.project.tracks) == n + 1
    assert window.add_color_rhythm({"clips": []}) == []


# ── difference from the surroundings: a ring target, changing backgrounds ──
RING_C, RING_R = (120.0, 48.0), 20.0
RING_BEATS = (1.0, 1.8, 2.9, 3.6, 4.5)
DISCS = ((255, 255, 255), (230, 40, 40), (40, 200, 60), (255, 255, 255), (60, 90, 240))
SCENES = ((0.0, (10, 10, 10)), (2.5, (76, 31, 15)), (4.0, (124, 124, 124)))
RING_ROI = ((RING_C[0] - 21.5) / W, (RING_C[1] - 21.5) / H, 43 / W, 43 / H)
_YY, _XX = np.mgrid[0:H, 0:W] + 0.5


def _ring_frame(t: float) -> np.ndarray:
    bg = [c for t0, c in SCENES if t >= t0][-1]
    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    img[np.abs(np.hypot(_XX - RING_C[0], _YY - RING_C[1]) - RING_R) <= 1.5] = 255
    for b, colour in zip(RING_BEATS, DISCS):        # a disc flies in from the left …
        if b - 0.4 <= t < b + 0.08:                  # … and vanishes just after the beat
            cx = 40 + (RING_C[0] - 40) * min(1.0, (t - b + 0.4) / 0.4)
            img[np.hypot(_XX - cx, _YY - RING_C[1]) <= 8] = colour
    return img


def _ring_frames():
    for i in range(int(DURATION * FPS)):
        yield i / FPS, _ring_frame(i / FPS)


def test_zone_geometry():
    ys, xs, measure, band = vc.zone_geometry((H, W), vc.Zone(ROI))
    assert measure.sum() == 50 * 60 and not (measure & band).any()
    assert band.any()
    ys, xs, measure, band = vc.zone_geometry((H, W), vc.Zone(RING_ROI, "ellipse", 0.7))
    area = np.pi * (21.5 * 0.7) ** 2
    assert measure.sum() == pytest.approx(area, rel=0.08)
    ys, xs, measure, band = vc.zone_geometry((H, W), vc.Zone(None))
    assert measure.all() and not band.any()                  # whole frame: no surroundings


def test_difference_ignores_ring_and_background():
    zone = vc.Zone(RING_ROI, "ellipse", 0.7)
    for t in (0.3, 2.6, 4.2):                                # empty target, three backgrounds
        assert not vc.difference_mask(_ring_frame(t), zone).any()
    full = vc.difference_mask(_ring_frame(RING_BEATS[0] - 0.01), zone)
    assert full.sum() == pytest.approx(np.pi * 8 ** 2, rel=0.15)
    an = vc.analyse_color_frames(_ring_frames(), FPS, RING_ROI, DURATION, "ellipse", 0.7)
    assert an.dist.shape == (an.n, vc.DIST_BINS) and an.bg_rgb.shape == (an.n, 3)
    assert tuple(an.bg_rgb[int(3.0 * FPS)]) == (76, 31, 15)
    sig = vc.difference_signal(an)
    assert sig[int(0.3 * FPS)] == 0 and sig[int(2.6 * FPS)] == 0 and sig[int(4.2 * FPS)] == 0
    p = vc.fill_to_pulses(sig, an.rate, an.duration_s)
    idx, _h = detect_peaks(p[:, 0], 0.5, "cross", min_gap_s=0.2)
    beats = idx / FS
    assert len(beats) == len(RING_BEATS)                     # every disc, whatever its colour
    assert np.all(np.abs(beats - RING_BEATS) < 0.15)


@pytest.fixture(scope="module")
def ring_file(tmp_path_factory):
    if not _have_av():
        pytest.skip("PyAV not installed")
    import av
    path = tmp_path_factory.mktemp("vc") / "ring.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"qscale": "2"}
        for _t, img in _ring_frames():
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return str(path)


def test_window_difference_measure(window, ring_file):
    doc = window.doc
    window.open_video(ring_file, ask_funscripts=lambda p: False)
    win = window.show_tool("color_rhythm")
    assert win.measure_key() == "difference" and win.create_btn.isEnabled()
    assert win.zone().shape == "ellipse" and win.zone().inner == pytest.approx(0.7)
    win.picker.set_roi(RING_ROI)
    win.time_slider.setValue(int((RING_BEATS[0] - 0.02) * 1000))
    win._load_frame()
    assert win.picker.highlight is not None and win.picker.highlight.sum() > 100
    assert win.run_analysis()
    clips = win.create_clip()
    track, clip = doc.project.find_clip(clips[0].id)
    assert track.name == "Target fill" and clip.source.origin.endswith("ring.mp4")
    marks = window.marks_from_peaks(clip.id, {"level": 0.5, "min_gap_s": 0.2})
    found = doc.project.find_clip(marks.id)[1]
    assert np.all(np.abs(np.array([m.t_s for m in found.source.marks]) - RING_BEATS) < 0.15)
    win.inner.setValue(60)                                   # another zone → analyse again
    assert win.run_btn.text() == "Analyse"
    win.inner.setValue(70)                                   # back → from the cache
    assert win.run_btn.text().startswith("Analysed")
    win.close()


# ── fast rhythms: up to 10 shapes a second ─────────────────────────────────
def test_resample_keeps_one_frame_spikes():
    x = np.zeros(120)                                        # 2 s at 60 fps
    x[[31, 50, 77]] = 1.0                                    # single-frame hits
    out = vc.resample_peaks(x, 60.0, 80)
    assert out.shape == (80,) and out.dtype == np.float32
    assert np.flatnonzero(out == 1.0).tolist() == [20, 33, 51]
    slow = vc.resample_peaks(np.array([0.0, 1.0]), 10.0, 5)  # slower than 40 Hz: interpolated
    assert np.allclose(slow, [0, 0.25, 0.5, 0.75, 1.0])


def test_shape_signal():
    fill = np.array([0, .2, .6, .6, 0, 0, .5, .1])
    assert np.array_equal(vc.shape_signal(fill, "fill"), fill)
    assert np.allclose(vc.shape_signal(fill, "rises"), [0, .2, .4, 0, 0, 0, .5, 0])
    assert np.allclose(vc.shape_signal(fill, "drops"), [0, 0, 0, 0, .6, 0, 0, .4])
    with pytest.raises(ValueError):
        vc.shape_signal(fill, "nope")


FAST_FPS = 60
FAST_BEATS = tuple(np.round(np.concatenate([0.5 + 0.1 * np.arange(15),       # 10 a second
                                            2.1 + 0.15 * np.arange(5)]), 3))


def _fast_frame(t: float) -> np.ndarray:
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (10, 10, 10) if t < 1.2 else (76, 31, 15)
    img[np.abs(np.hypot(_XX - RING_C[0], _YY - RING_C[1]) - RING_R) <= 1.5] = 255
    for k, b in enumerate(FAST_BEATS):               # several discs in flight at once
        if b - 0.3 <= t < b:
            cx = 40 + (RING_C[0] - 40) * (t - b + 0.3) / 0.3
            img[np.hypot(_XX - cx, _YY - RING_C[1]) <= 6] = DISCS[k % len(DISCS)]
    return img


@pytest.mark.parametrize("mode,method", [("drops", "max"), ("fill", "cross")])
def test_ten_shapes_a_second(mode, method):
    frames = ((i / FAST_FPS, _fast_frame(i / FAST_FPS)) for i in range(int(3.0 * FAST_FPS)))
    an = vc.analyse_color_frames(frames, FAST_FPS, RING_ROI, 3.0, "ellipse", 0.7)
    p = vc.fill_to_pulses(vc.difference_signal(an), an.rate, an.duration_s, mode=mode)
    idx, _h = detect_peaks(p[:, 0], 0.5, method, min_gap_s=0.05)
    beats = idx / FS
    assert len(beats) == len(FAST_BEATS)
    assert np.all(np.abs(beats - np.asarray(FAST_BEATS)) <= 0.06)


def test_picker_zoom(qapp):
    from PyQt6.QtCore import QPointF
    from stimstudio.gui.tools.color_rhythm import ColorPicker
    pk = ColorPicker()
    pk.resize(640, 384)
    pk.set_frame(_ring_frame(0.3))                           # 160 × 96: fills the widget
    anchor = QPointF(480, 192)
    before = pk._to_frac(anchor)
    pk.set_zoom(4.0, anchor)
    after = pk._to_frac(anchor)
    assert pk.zoom == 4.0
    assert (after.x(), after.y()) == pytest.approx((before.x(), before.y()), abs=1e-6)
    r = pk._frame_rect()
    assert r.width() == pytest.approx(640 * 4) and r.left() <= 0 and r.right() >= 640
    pk.set_zoom(4.0, QPointF(0, 0))                          # never pans past the frame
    assert pk._frame_rect().left() <= 0 and pk._frame_rect().top() <= 0
    pk.set_zoom(1000)
    assert pk.zoom == 32.0
    pk.fit()
    assert pk.zoom == 1.0 and pk._frame_rect().width() == pytest.approx(640)


def test_picker_small_target_and_pan(qapp):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtTest import QTest
    from stimstudio.gui.tools.color_rhythm import ColorPicker
    pk = ColorPicker()
    pk.resize(640, 384)
    pk.set_frame(_ring_frame(0.3))
    pk.set_zoom(8.0, QPointF(320, 192))
    got = []
    pk.roiChanged.connect(got.append)
    QTest.mousePress(pk, Qt.MouseButton.LeftButton, pos=QPoint(300, 170))
    QTest.mouseMove(pk, QPoint(340, 214))
    QTest.mouseRelease(pk, Qt.MouseButton.LeftButton, pos=QPoint(340, 214))
    assert got[-1] is None                                   # 40 screen px at ×8 = 1.25 px: no
    QTest.mousePress(pk, Qt.MouseButton.LeftButton, pos=QPoint(300, 170))
    QTest.mouseMove(pk, QPoint(400, 280))
    QTest.mouseRelease(pk, Qt.MouseButton.LeftButton, pos=QPoint(400, 280))
    x, y, w, h = got[-1]
    assert w * W == pytest.approx(100 / 32, abs=0.01)        # 3+ frame px: a target
    c0 = pk._to_frac(QPointF(320, 192))
    QTest.mousePress(pk, Qt.MouseButton.RightButton, pos=QPoint(320, 192))
    QTest.mouseMove(pk, QPoint(352, 192))
    QTest.mouseRelease(pk, Qt.MouseButton.RightButton, pos=QPoint(352, 192))
    c1 = pk._to_frac(QPointF(320, 192))
    assert c1.x() == pytest.approx(c0.x() - 32 / (640 * 8), abs=1e-6)
    assert got[-1] == (x, y, w, h)                           # panning leaves the target alone


def test_window_fps_and_frame_steps(window, ring_file):
    from stimstudio.io.video import probe_fps
    assert probe_fps(ring_file) == pytest.approx(FPS)
    window.open_video(ring_file, ask_funscripts=lambda p: False)
    win = window.show_tool("color_rhythm")
    assert win.fps == pytest.approx(FPS) and win.rate.itemText(0) == f"All frames ({FPS} / s)"
    assert win.analysis_rate() == pytest.approx(FPS)
    win.time_slider.setValue(1000)
    win.step_frame(+1)
    assert win.frame_time() == pytest.approx(1.04)
    win.step_frame(-3)
    assert win.frame_time() == pytest.approx(0.92)
    win.picker.set_roi(RING_ROI)
    assert win.run_analysis()
    win.signal.setCurrentIndex(win.signal.findData("drops"))
    data = win.result_data()
    assert data["signal"] == "drops" and data["clips"][0]["name"] == "Target vanishing"
    win.close()
