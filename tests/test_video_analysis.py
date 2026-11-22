"""Video analysis: cut detection, activity, ROI, motion direction, pulses."""

import numpy as np
import pytest

from stimstudio.core.pulse import FS
from stimstudio.io.video_analysis import (VideoAnalysis, analyse_frames, analyse_video, chi2,
                                          detect_cuts, grey_histogram, motion_to_pulses,
                                          phase_shift, roi_slice)

FPS = 25
CUT_AT = 3.0           # scene change
STILL_UNTIL = 1.0      # nothing moves before this


def _frame(t: float) -> np.ndarray:
    """Grey 160×96 frame: background by scene, a square bobbing in the left half
    (from STILL_UNTIL on), a static block on the right."""
    img = np.full((96, 160), 60 if t < CUT_AT else 190, np.uint8)
    img[20:76, 120:150] = 120                          # static, right half
    y = 30 if t < STILL_UNTIL else int(30 + 20 * np.sin(2 * np.pi * 0.5 * (t - STILL_UNTIL)))
    img[y:y + 24, 30:60] = 250 if t < CUT_AT else 20
    return img


def _frames(duration=6.0):
    for i in range(int(duration * FPS)):
        t = i / FPS
        yield t, _frame(t)


def test_helpers():
    assert roi_slice((96, 160), None) == (slice(0, 96), slice(0, 160))
    ys, xs = roi_slice((100, 200), (0.5, 0.25, 0.5, 0.5))
    assert (ys.start, ys.stop, xs.start, xs.stop) == (25, 75, 100, 200)
    h = grey_histogram(np.zeros((4, 4), np.uint8))
    assert chi2(h, h) == 0.0
    assert chi2(h, grey_histogram(np.full((4, 4), 255, np.uint8))) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    b = np.roll(a, 5, axis=0)                           # content moved down by 5 px
    dy, dx = phase_shift(a, b)
    assert dy == pytest.approx(5, abs=0.3) and dx == pytest.approx(0, abs=0.3)


def test_cut_activity_and_roi():
    an = analyse_frames(_frames(), rate=20)
    assert an.rate == 20 and an.n == pytest.approx(120, abs=1)
    cuts = detect_cuts(an)
    assert len(cuts) == 1 and abs(cuts[0] - CUT_AT) <= 1 / FPS + 1 / 20
    t = an.times
    still = an.activity[(t > 0.2) & (t < STILL_UNTIL - 0.1)]
    moving = an.activity[(t > STILL_UNTIL + 0.3) & (t < CUT_AT - 0.2)]
    assert still.max() < 1e-6 < moving.mean()
    right = analyse_frames(_frames(), rate=20, roi=(0.7, 0.0, 0.3, 1.0))
    seg = (t > STILL_UNTIL + 0.3) & (t < CUT_AT - 0.2)
    assert right.activity[seg].max() < 1e-6            # nothing moves in the right half


def test_motion_direction_follows_the_square():
    an = analyse_frames(_frames(), rate=20, roi=(0.1, 0.0, 0.35, 1.0))
    t = an.times
    # y = 30 + 20 sin(pi (t-1)): moving down (positive) for t in (1, 1.5), up for (2, 3)
    down = an.motion_y[(t > 1.15) & (t < 1.4)].mean()
    up = an.motion_y[(t > 2.15) & (t < 2.8)].mean()
    assert down > 0 > up


def test_brightness_steps_and_motion_are_not_cuts():
    """A flat background whose brightness moves in coarse steps (as heavy
    compression does) plus a fast-moving bar: no cuts."""
    def frames():
        for i in range(25 * 12):
            t = i / 25
            img = np.full((90, 160), 70 + 3 * int(t / 0.3), np.uint8)
            x = int((t * 120) % 160)
            img[30:60, max(0, x - 10):x + 10] = 240
            yield t, img
    an = analyse_frames(frames(), rate=20)
    assert detect_cuts(an) == []
    assert detect_cuts(an, 0.8) == []


def test_sensitivity_and_gap():
    score = np.zeros(200, np.float32)
    score[[50, 52, 120]] = [0.6, 0.9, 0.25]
    an = VideoAnalysis(20.0, np.zeros(200), score, np.zeros(200), 10.0)
    assert detect_cuts(an, 0.5) == [52 / 20]           # the weaker 0.25 is ignored, 50/52 merge
    assert detect_cuts(an, 1.0) == [52 / 20, 120 / 20]  # more sensitive: picks up 0.25


def test_motion_to_pulses_shapes():
    an = analyse_frames(_frames(), rate=20, roi=(0.1, 0.0, 0.35, 1.0))
    p = motion_to_pulses(an, floor=0.1, ceiling=0.9, freq=0.3)
    assert p.shape == (round(an.duration_s * FS), 4) and p.dtype == np.float32
    assert np.all(p[:, 0] == p[:, 1]) and np.all(np.abs(p[:, 2] - 0.3) < 1e-6)
    assert p[:, 0].min() >= 0.1 - 1e-6 and p[:, 0].max() <= 0.9 + 1e-6
    assert p[: int(0.8 * FS), 0].max() < 0.11           # still at the start → floor
    tri = motion_to_pulses(an, triphase=True)
    assert tri.shape == p.shape
    assert np.all(tri[:, 1] == 0.5)
    assert 0.0 <= tri[:, 2].min() and tri[:, 2].max() <= 1.0
    assert tri[:, 2].std() > 0.05                       # the stroke moves the position
    speed = motion_to_pulses(an, freq_mode="speed")
    assert speed[:, 2].std() > 0.01


def _have_av():
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _have_av(), reason="PyAV not installed")
def test_analyse_real_file(tmp_path):
    import av
    path = tmp_path / "scenes.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = 160, 96, "yuv420p"
        s.options = {"qscale": "3"}
        for t, g in _frames():
            rgb = np.repeat(g[:, :, None], 3, axis=2)
            for pk in s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    seen = []
    an = analyse_video(path, rate=10, progress=seen.append)
    assert an.duration_s == pytest.approx(6.0, abs=0.15)
    assert seen and seen[-1] == 1.0
    cuts = detect_cuts(an)
    assert len(cuts) == 1 and abs(cuts[0] - CUT_AT) <= 0.15


@pytest.fixture(scope="module")
def scenes_file(tmp_path_factory):
    if not _have_av():
        pytest.skip("PyAV not installed")
    import av
    path = tmp_path_factory.mktemp("va") / "scenes.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = 160, 96, "yuv420p"
        s.options = {"qscale": "3"}
        for _t, g in _frames():
            rgb = np.repeat(g[:, :, None], 3, axis=2)
            for pk in s.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    return str(path)


def test_apply_result_one_undo(window, scenes_file):
    doc = window.doc
    window.open_video(scenes_file, ask_funscripts=lambda p: False)
    vclip = doc.project.video_track().clips[0]
    doc.set_clip_attrs(vclip.id, start_s=2.0)
    n_tracks, n_markers = len(doc.project.tracks), len(doc.project.markers)
    pulses = np.full((240, 4), 0.5, np.float32)
    clip = window.analyse_video(result={"pulses": pulses, "triphase": True, "cuts": [3.0, 4.5]})
    track, found = doc.project.find_clip(clip.id)
    assert track.type == "triphase" and doc.project.tracks.index(track) == 1
    assert found.start_s == 2.0 and found.length_s == pytest.approx(6.0)
    assert [m.t_s for m in doc.project.markers][-2:] == [5.0, 6.5]
    assert doc.project.markers[-1].name == "Cut 2"
    doc.undo.undo()
    assert len(doc.project.tracks) == n_tracks and len(doc.project.markers) == n_markers


def test_dialog_analyses_and_caches(qapp, scenes_file):
    from stimstudio.gui import video_analysis as gva
    dlg = gva.VideoAnalysisDialog(scenes_file, 6.0, motion=True, cuts=True)
    dlg.picker.set_roi((0.1, 0.0, 0.35, 1.0))
    assert dlg.run_btn.text() == "Analyse"
    assert dlg.run_analysis()
    assert dlg.run_btn.text().startswith("Analysed")
    data = dlg.result_data()
    assert data["pulses"].shape[1] == 4 and data["pulses"].shape[0] > 200
    assert len(data["cuts"]) == 1 and abs(data["cuts"][0] - CUT_AT) < 0.2
    assert data["roi"] == (0.1, 0.0, 0.35, 1.0)
    assert gva.cached_analysis(scenes_file, 20, (0.1, 0.0, 0.35, 1.0), False) is dlg.analysis
    dlg.picker.set_roi(None)                     # new settings → needs a new pass
    assert dlg.run_btn.text() == "Analyse"
    dlg.picker.set_roi((0.1, 0.0, 0.35, 1.0))     # back → served from the cache
    assert dlg.run_btn.text().startswith("Analysed")
    dlg.cuts_box.setChecked(False)
    assert dlg.result_data()["cuts"] is None
    dlg.deleteLater()
