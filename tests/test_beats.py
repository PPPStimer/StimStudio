import numpy as np
import pytest

from stimstudio.core.beats import detect_peaks, level_signal, marks_from_peaks, quantize_marks
from stimstudio.core.model import Mark


def bumps(centres, heights, n=200, width=3):
    s = np.zeros(n)
    for c, h in zip(centres, heights):
        for k in range(-width, width + 1):
            if 0 <= c + k < n:
                s[c + k] = max(s[c + k], h * (1 - abs(k) / (width + 1)))
    return s


def test_level_signal():
    d = np.array([[0.2, 0.5, 0.9, 0.1]], np.float32)
    assert level_signal(d)[0] == pytest.approx(0.5)
    assert level_signal(d, "triphase")[0] == pytest.approx(0.2)


def test_crossing_marks_attack_and_height():
    s = bumps([20, 60, 100], [0.9, 0.3, 0.7])
    idx, h = detect_peaks(s, 0.5, "cross")
    assert list(idx) == [19, 99] and h == pytest.approx([0.9, 0.7])


def test_crossing_hysteresis_ignores_ripple():
    s = np.zeros(100)
    s[10:40] = 0.6
    s[20] = 0.48                    # dips under the level but not under level − hyst
    idx, _ = detect_peaks(s, 0.5, "cross", hysteresis=0.05, min_gap_s=0)
    assert list(idx) == [10]
    idx, _ = detect_peaks(s, 0.5, "cross", hysteresis=0.0, min_gap_s=0)
    assert list(idx) == [10, 21]


def test_crossing_min_gap():
    s = bumps([20, 26, 80], [0.9, 0.9, 0.9], width=1)
    assert list(detect_peaks(s, 0.5, "cross", min_gap_s=0.5)[0]) == [20, 80]
    assert list(detect_peaks(s, 0.5, "cross", min_gap_s=0.0)[0]) == [20, 26, 80]


def test_maxima_keep_strongest_within_gap():
    s = bumps([20, 26, 80], [0.6, 0.9, 0.4], width=2)
    idx, h = detect_peaks(s, 0.5, "max", min_gap_s=0.5)
    assert list(idx) == [26] and h == pytest.approx([0.9])
    idx, _ = detect_peaks(s, 0.3, "max", min_gap_s=0.0)
    assert list(idx) == [20, 26, 80]


def test_marks_from_peaks():
    marks = marks_from_peaks([0, 40], [0.5, 1.4])
    assert [(m.t_s, m.velocity) for m in marks] == [(0.0, 0.5), (1.0, 1.0)]


def test_quantize_strength_offset_and_merge():
    marks = [Mark(0.1, 0.5), Mark(0.45), Mark(0.55, 0.2)]
    full = quantize_marks(marks, 0.5, 1.0)
    assert [m.t_s for m in full] == pytest.approx([0.0, 0.5])
    assert full[1].velocity == 1.0               # merged: the louder stays
    half = quantize_marks(marks, 0.5, 0.5)
    assert [m.t_s for m in half] == pytest.approx([0.05, 0.475, 0.525])
    shifted = quantize_marks([Mark(0.1)], 0.5, 1.0, offset_s=0.3)   # timeline 0.4 → 0.5
    assert shifted[0].t_s == pytest.approx(0.2)
