"""Research-based generators: each property cited in the module docstrings is checked."""

import math

import numpy as np
import pytest

from stimstudio.core.pulse import FS
from stimstudio.generators import get_generator, list_generators

NEW_MODULES = {"stim", "perceptual", "complexity", "longform"}
NEW = [g for g in list_generators() if g.__module__.rsplit(".", 1)[-1] in NEW_MODULES
       or g.type_id in ("tri_lissajous", "tri_rabbit", "tri_attractor")]


def render(type_id, seconds=20.0, seed=1, **params):
    return get_generator(type_id).render(round(seconds * FS), params, seed).to_array()


def onsets(x, thr=0.5):
    """Indices where *x* rises above *thr* (a value above it at index 0 counts)."""
    on = x > thr
    return np.flatnonzero(on & ~np.r_[False, on[:-1]])


def test_all_new_generators_registered():
    ids = {g.type_id for g in NEW}
    assert ids == {"burst", "dense_disperse", "anti_habit", "beat_sweep", "iso_sweep", "phantom",
                   "rabbit", "touch", "colored_noise", "attractor", "groove", "polyrhythm",
                   "breath", "heartbeat", "risset", "tri_lissajous", "tri_rabbit",
                   "tri_attractor"}


@pytest.mark.parametrize("case", [(g, s.key, c) for g in NEW for s in g.param_specs()
                                  if s.kind == "choice" for c in s.choices],
                         ids=lambda c: f"{c[0].type_id}-{c[1]}={c[2]}")
def test_every_choice_renders(case):
    gen_cls, key, choice = case
    for n in (0, 1, 7, 400):
        arr = gen_cls().render(n, {key: choice, "_bpm": 100.0}, 3).clamp().to_array()
        assert arr.shape == (n, 4) and np.isfinite(arr).all()


@pytest.mark.parametrize("gen_cls", [g for g in NEW if g.tempo_synced], ids=lambda g: g.type_id)
def test_tempo_synced_follow_bpm(gen_cls):
    a = gen_cls().render(400, {"_bpm": 90.0}, 2).to_array()
    b = gen_cls().render(400, {"_bpm": 140.0}, 2).to_array()
    assert not np.array_equal(a, b)


# ── electrotherapy ──────────────────────────────────────────────────────────
def test_burst_duty_rate_and_frequency():
    p = render("burst", 10.0, rate=2.0, duty=0.4, attack=0.0, release=0.0, amp_base=0.0)
    assert len(onsets(p[:, 0], 0.01)) in (19, 20)                    # 2 bursts / s
    assert np.mean(p[:, 0] > 0.01) == pytest.approx(0.4, abs=0.03)
    assert np.all(p[p[:, 0] > 0.01, 2] == pytest.approx(0.9))       # high freq inside
    b = render("burst", 10.0, rate=2.0, b_offset=0.5, attack=0.0, release=0.0)
    assert np.mean((b[:, 0] > 0.01) & (b[:, 1] > 0.01)) < 0.01       # B interleaved


def test_dense_disperse_alternates_with_compensation():
    p = render("dense_disperse", 12.0, low_s=3.0, high_s=3.0, transition_s=0.0, compensate=0.5)
    assert p[40, 2] == pytest.approx(0.05) and p[160, 2] == pytest.approx(0.95)
    assert p[160, 0] < p[40, 0]                                      # less level when high
    o = render("dense_disperse", 12.0, b_mode="opposite", transition_s=0.0)
    np.testing.assert_allclose(o[:, 2] + o[:, 3], 1.0, atol=1e-6)


def test_anti_habituation_is_irregular_and_has_novelty():
    def hold_lengths(x):       # pulses between glide starts
        moving = np.abs(np.diff(x)) > 1e-6
        starts = np.flatnonzero(moving[1:] & ~moving[:-1])
        return np.diff(starts)

    regular = render("anti_habit", 120.0, unpredictability=0.0, novelty_every=0.0)
    random = render("anti_habit", 120.0, unpredictability=1.0, novelty_every=0.0)
    assert np.ptp(hold_lengths(regular[:, 2])) == 0                   # fixed rhythm
    assert hold_lengths(random[:, 2]).std() > 10                      # irregular holds
    with_nov = render("anti_habit", 120.0, novelty_every=10.0, seed=4)
    without = render("anti_habit", 120.0, novelty_every=0.0, seed=4)
    assert not np.array_equal(with_nov, without)


def test_beat_sweep_beat_rate():
    p = render("beat_sweep", 20.0, beat_min=4.0, beat_max=4.0, depth=1.0)
    assert len(onsets(p[:, 0], 0.5)) == pytest.approx(80, abs=2)     # 4 Hz × 20 s
    assert np.corrcoef(p[:, 0], p[:, 1])[0, 1] < -0.9                # B in antiphase


def test_iso_sweep_level_drops_with_frequency():
    p = render("iso_sweep", 10.0, compensation=0.5)
    assert np.corrcoef(p[:, 0], p[:, 2])[0, 1] < -0.99
    flat = render("iso_sweep", 10.0, compensation=0.0)
    assert np.ptp(flat[:, 0]) < 1e-6


# ── tactile illusions ───────────────────────────────────────────────────────
@pytest.mark.parametrize("motion", ["sweep", "ease", "bounce", "random"])
def test_phantom_energy_is_constant(motion):
    p = render("phantom", 20.0, motion=motion, intensity=0.8, law="energy")
    np.testing.assert_allclose(p[:, 0] ** 2 + p[:, 1] ** 2, 0.64, atol=1e-5)
    assert p[:, 0].max() > 0.75 and p[:, 1].max() > 0.75             # reaches both ends


def test_phantom_steps_use_tactile_brush_soa():
    from stimstudio.generators.perceptual import apparent_motion_soa
    assert apparent_motion_soa(0.1) == pytest.approx(0.0793)
    p = render("phantom", 5.0, motion="steps", points=2, step_ms=100.0)
    a = onsets(p[:, 0], 0.4)
    assert len(a) >= 3


def test_rabbit_taps_hop_from_a_to_b():
    p = render("rabbit", 3.0, taps=3, isi_ms=50.0, tap_ms=25.0, mid_site=False,
               direction="A → B", pause_s=1.0)
    a, b = onsets(p[:, 0], 0.5), onsets(p[:, 1], 0.5)
    first_a, first_b = a[a < 20], b[b < 20]
    assert list(np.diff(first_a)) == [2, 2] and list(np.diff(first_b)) == [2, 2]   # 50 ms
    assert first_b[0] - first_a[-1] == 2 and first_a[0] == 0


def test_touch_transients_stronger_than_hold():
    p = render("touch", 60.0, rate=10.0, hold_s=2.0, jitter=0.0, sustain=0.3, onset_gain=1.0)
    amp, freq = p[:, 0], p[:, 2]
    assert amp.max() > 0.6
    held = (amp > 0.05) & (np.abs(np.gradient(amp)) < 1e-4)
    assert held.any() and amp[held].max() < 0.35                      # SA part is weak
    assert freq[amp > 0.6].mean() > freq[held].mean()                 # transients higher


# ── noise / chaos / rhythm ──────────────────────────────────────────────────
@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_colored_noise_spectral_slope(beta):
    from stimstudio.generators.complexity import colored_noise
    w = np.random.default_rng(0).standard_normal((8, 2 ** 15))
    x = colored_noise(w, beta, max_hz=20.0)
    f = np.fft.rfftfreq(w.shape[1], 1 / FS)
    psd = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0)
    band = (f > 0.05) & (f < 5.0)
    slope = np.polyfit(np.log(f[band]), np.log(psd[band]), 1)[0]
    assert slope == pytest.approx(-beta, abs=0.15)


def test_colored_noise_correlation():
    p = render("colored_noise", 120.0, correlation=1.0)
    np.testing.assert_allclose(p[:, 0], p[:, 1])
    q = render("colored_noise", 120.0, correlation=0.0)
    assert abs(np.corrcoef(q[:, 0], q[:, 1])[0, 1]) < 0.3


@pytest.mark.parametrize("system", ["lorenz", "rossler", "logistic"])
def test_attractor_bounded_and_aperiodic(system):
    from stimstudio.generators.complexity import attractor_path
    a = attractor_path(system, 4000, 1.0, 1.0, 3)
    assert a.min() >= 0 and a.max() <= 1 and a.std(axis=1).min() > 0.05
    half = 2000
    assert not np.allclose(a[:, :half], a[:, half:], atol=1e-3)
    # independent of clip length: the first seconds are the same for a shorter render
    np.testing.assert_allclose(attractor_path(system, 400, 1.0, 1.0, 3), a[:, :400])


def test_groove_syncopation_rises_with_parameter():
    from stimstudio.generators.complexity import groove_pattern, lhl_syncopation
    assert lhl_syncopation(np.array([1, 0, 0, 0] * 4, bool)) == 0
    assert lhl_syncopation(np.array([0, 0, 1, 0] * 4, bool)) > 0

    def mean_sync(s):
        return np.mean([lhl_syncopation(groove_pattern(np.random.default_rng(i), 0.4, s))
                        for i in range(200)])
    lo, mid, hi = mean_sync(0.05), mean_sync(0.5), mean_sync(0.95)
    assert lo < mid < hi


def test_groove_varies_every_n_bars():
    bpm = 120.0        # a bar = 2 s = 80 pulses
    p = get_generator("groove").render(80 * 8, {"_bpm": bpm, "vary_bars": 2}, 5).to_array()
    bars = p[:, 0].reshape(8, 80)
    np.testing.assert_array_equal(bars[0], bars[1])
    assert not all(np.array_equal(bars[0], bars[k]) for k in range(2, 8, 2))


@pytest.mark.parametrize("m,n", [(3, 2), (4, 3), (5, 4)])
def test_polyrhythm_beats_per_cycle(m, n):
    bpm = 120.0        # 4 beats = 2 s = 80 pulses
    p = get_generator("polyrhythm").render(80, {"_bpm": bpm, "a_beats": m, "b_beats": n,
                                                "accent": 0.0}, 0).to_array()
    assert len(onsets(p[:, 0], 0.5)) == m
    assert len(onsets(p[:, 1], 0.5)) == n


# ── long forms ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rate", [6.0, 10.0])
def test_breath_period(rate):
    p = render("breath", 120.0, rate=rate, drift=0.0)
    peaks = onsets(p[:, 0], 0.7)
    assert np.diff(peaks).mean() / FS == pytest.approx(60.0 / rate, rel=0.02)


def test_breath_box_has_holds():
    p = render("breath", 20.0, rate=6.0, pattern="box 4-4-4-4", amp_min=0.0, amp_max=1.0)
    cycle = p[:400, 0]
    assert np.mean(cycle > 0.999) == pytest.approx(0.25, abs=0.03)
    assert np.mean(cycle < 0.001) == pytest.approx(0.25, abs=0.03)


def test_heartbeat_rate_and_lub_dub():
    p = render("heartbeat", 60.0, bpm_start=60.0, bpm_end=60.0, hrv=0.0, jitter=0.0)
    lub = onsets(p[:, 0], 0.9)
    assert len(lub) == 60
    q = render("heartbeat", 60.0, hrv=0.0, jitter=0.0, b_mode="lub A / dub B")
    assert onsets(q[:, 1], 0.3)[0] > onsets(q[:, 0], 0.3)[0]          # dub after lub
    up = render("heartbeat", 60.0, bpm_start=60.0, bpm_end=120.0, hrv=0.0, jitter=0.0)
    beats = onsets(up[:, 0], 0.9)
    assert np.diff(beats[:5]).mean() > np.diff(beats[-5:]).mean()      # speeds up


def test_risset_layers_hand_over_seamlessly():
    from stimstudio.generators.longform import risset_onsets
    cycle = 16 * math.log(2) / 2.0
    t, dt, layer, x, w = risset_onsets(16, 3, cycle)
    for k in range(2):
        before = dt[layer == k][np.argsort(t[layer == k])][-2:]       # end of layer k …
        after = dt[layer == k + 1][np.argsort(t[layer == k + 1])][:1]  # … continues as k+1
        # the accelerating stream keeps its geometric interval ratio across the seam
        assert before[1] / after[0] == pytest.approx(before[0] / before[1], rel=0.01)
    assert w[x < 0.02].max() < 0.01 and w[x > 0.98].max() < 0.01       # faded at both ends
    assert np.all(np.diff(t) >= 0) and t.max() < cycle
    r = risset_onsets(16, 3, cycle, "ritardando")
    np.testing.assert_allclose(np.sort(cycle - r[0]), np.sort(t + dt), atol=1e-9)


def test_risset_opposite_channel():
    p = render("risset", 30.0, b_mode="opposite")
    assert not np.array_equal(p[:, 0], p[:, 1]) and p[:, 0].max() > 0.5


# ── triphase ────────────────────────────────────────────────────────────────
def test_triphase_generators_stay_in_disc():
    from stimstudio.core.triphase import unit_to_xy
    for tid in ("tri_lissajous", "tri_rabbit", "tri_attractor"):
        assert get_generator(tid).domain == "triphase"
        p = render(tid, 10.0)
        x, y = unit_to_xy(p[:, 1]), unit_to_xy(p[:, 2])
        assert np.all(np.hypot(x, y) <= 1.0 + 1e-5)


def test_tri_rabbit_hops_between_electrodes():
    from stimstudio.core.triphase import unit_to_xy
    p = render("tri_rabbit", 2.0, taps=2, isi_ms=50.0, pause_s=0.0, sequence="L → R")
    x = unit_to_xy(p[:, 1])
    assert x[0] < -0.5 and x[4] > 0.5                                  # L (150°) then R (30°)
    assert p[0, 0] > 0 and p[1, 0] == 0                                 # tap, then gap
