"""Effects: HWL Viewer golden reference + the new StimStudio effects."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from stimstudio.effects import LEGACY_EFFECTS, apply_effects, get_effect, list_effects

REF = Path(__file__).parent / "data" / "golden.npz"
TOLERANCE = 1e-6

# ── golden (copied from HWL Viewer's tests/test_golden.py) ─────────────────
_rng = np.random.default_rng(7)
BASE = _rng.random((733, 4))
BASE[0] = [0.0, 0.0, 0.0, 0.0]
BASE[1] = [1.0, 1.0, 1.0, 1.0]
BASE[2] = [0.0005, 0.9995, 0.0, 1.0]

VARIANTS = {
    "phase_shift":            [{"us": 0}, {"us": 250_000}, {"us": 9_975_000}],
    "frequency_adjust":       [{"amount": 0.3}, {"amount": -0.4}],
    "scale_amplitude":        [{"value": 0.25}, {"value": 1.0}],
    "scale_frequency":        [{"value": 0.5}, {"value": 1.7}],
    "amplitude_feel":         [{"feel": 0.4}, {"feel": 2.5}],
    "frequency_feel":         [{"feel": 0.4}, {"feel": 2.5}],
    "random_amplitude_noise": [
        {"amount": 0.2, "speed": 4.0, "skip_zeros": False, "threshold": 0.001},
        {"amount": 0.3, "speed": 4.0, "skip_zeros": True,  "threshold": 0.01},
        {"amount": 0.3, "speed": 0.0, "skip_zeros": False, "threshold": 0.001},
    ],
    "random_frequency_noise": [{"amount": 0.2, "speed": 4.0}, {"amount": 0.5, "speed": 0.0}],
    "smooth_amplitude":       [{"window": 0.1}, {"window": 0.01}, {"window": 1.5}],
    "smooth_frequency":       [{"window": 0.1}, {"window": 2.0}],
}
CHANNEL_SETS = [["A", "B"], ["A"], ["B"], []]


def _dp(t):
    return get_effect(t).default_params()


def golden_cases():
    for etype in LEGACY_EFFECTS:
        for vi, extra in enumerate(VARIANTS.get(etype, [{}])):
            for chans in CHANNEL_SETS:
                eff = {"type": etype, **_dp(etype), **extra, "channels": list(chans)}
                yield f"{etype}|v{vi}|{''.join(chans) or 'none'}", [eff]
    full = [{"type": t, **_dp(t), **VARIANTS.get(t, [{}])[0], "channels": ["A", "B"]}
            for t in LEGACY_EFFECTS]
    yield "chain|all|AB", full
    yield "chain|all|A", [dict(e, channels=["A"]) for e in full]
    for pair in itertools.combinations(
            ("channel_swap", "scale_amplitude", "amplitude_feel",
             "smooth_amplitude", "random_amplitude_noise"), 2):
        chain = [{"type": t, **_dp(t), **VARIANTS.get(t, [{}])[0], "channels": ["A", "B"]}
                 for t in pair]
        yield "chain|" + "+".join(pair), chain
    yield "empty|no-effects", []


CASES = list(golden_cases())


def test_golden_case_set_matches():
    ref = np.load(REF)
    assert {name for name, _ in CASES} == set(ref.files)


@pytest.mark.parametrize("name,effects", CASES, ids=[c[0] for c in CASES])
def test_golden(name, effects):
    want = np.load(REF)[name]
    got = np.asarray(apply_effects(BASE.astype(np.float32).copy(), effects), dtype=np.float64)
    assert got.shape == want.shape
    assert float(np.max(np.abs(got - want))) <= TOLERANCE


# ── chain behaviour ─────────────────────────────────────────────────────────
def _const(n=400, amp=0.8, freq=0.5):
    p = np.zeros((n, 4), dtype=np.float32)
    p[:, :2] = amp
    p[:, 2:] = freq
    return p


def test_input_is_never_modified():
    p = _const()
    before = p.copy()
    apply_effects(p, [get_effect("tremolo").make(), get_effect("slew_limit").make()])
    np.testing.assert_array_equal(p, before)


def test_disabled_and_unknown_effects_are_skipped():
    p = _const()
    out = apply_effects(p, [dict(get_effect("scale_amplitude").make(value=0.1), enabled=False),
                            {"type": "does_not_exist"}])
    np.testing.assert_array_equal(out, p)


def test_order_matters():
    p = _const(amp=0.5)
    a = apply_effects(p, [get_effect("scale_amplitude").make(value=2.0),
                          get_effect("amplitude_feel").make(feel=2.0)])
    b = apply_effects(p, [get_effect("amplitude_feel").make(feel=2.0),
                          get_effect("scale_amplitude").make(value=2.0)])
    assert a[0, 0] == pytest.approx(1.0) and b[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("definition", list_effects(), ids=lambda d: d.type_id)
def test_every_effect_stays_in_range(definition):
    p = np.random.default_rng(1).random((300, 4)).astype(np.float32)
    out = apply_effects(p, [definition.make()])
    assert out.shape == p.shape and out.dtype == np.float32
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


# ── new effects ─────────────────────────────────────────────────────────────
def test_tremolo_depth_and_channel_scope():
    p = _const(amp=1.0)
    out = apply_effects(p, [get_effect("tremolo").make(channels=["A"], depth=0.5, rate=1.0)])
    assert out[:, 0].min() == pytest.approx(0.5, abs=1e-3)
    assert out[:, 0].max() == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_array_equal(out[:, 1], p[:, 1])  # B untouched


def test_gate_follows_tempo():
    p = _const(n=40 * 4, amp=1.0)  # 4 s
    out = apply_effects(p, [get_effect("gate").make(division="1/4", duty=0.5, edge=0.0)],
                        bpm=60.0)  # one step per second, open for 0.5 s
    a = out[:, 0]
    assert a[5] == pytest.approx(1.0) and a[30] == pytest.approx(0.0)
    assert a[45] == pytest.approx(1.0) and a[70] == pytest.approx(0.0)


def test_slew_limit_ramps_steps():
    p = _const(n=80, amp=0.0)
    p[20:, :2] = 1.0
    out = apply_effects(p, [get_effect("slew_limit").make(rise=2.0)])
    rise = np.diff(out[:, 0])
    assert rise.max() <= 2.0 / 40 + 1e-6
    assert out[-1, 0] == pytest.approx(1.0)       # reaches the target (0.5 s ramp)
    np.testing.assert_array_equal(out[:, 2:], p[:, 2:])


def test_slew_limit_starts_from_silence():
    out = apply_effects(_const(n=40, amp=1.0), [get_effect("slew_limit").make(rise=4.0)])
    assert out[0, 0] == pytest.approx(0.1)


def test_freq_range_remaps():
    p = _const(freq=0.5)
    out = apply_effects(p, [get_effect("freq_range").make(low=0.2, high=0.4)])
    assert out[0, 2] == pytest.approx(0.3)
