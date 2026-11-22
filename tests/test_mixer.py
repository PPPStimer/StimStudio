import numpy as np
import pytest

from stimstudio.core.mixer import DEFAULT_FREQ, Accumulator, hold_frequency


def layer(n, amp, freq):
    p = np.zeros((n, 4), dtype=np.float32)
    p[:, :2] = amp
    p[:, 2:] = freq
    return p


def test_single_layer_passes_through():
    p = np.random.default_rng(0).random((50, 4)).astype(np.float32)
    acc = Accumulator(50)
    acc.add(p)
    out, active = acc.result()
    np.testing.assert_allclose(out, p, atol=1e-5)
    assert active.all()


def test_add_sums_amplitude_and_weights_frequency():
    acc = Accumulator(10)
    acc.add(layer(10, 0.3, 0.2))
    acc.add(layer(10, 0.6, 0.8))
    out, _ = acc.result()
    assert out[0, 0] == pytest.approx(0.9)
    assert out[0, 2] == pytest.approx((0.3 * 0.2 + 0.6 * 0.8) / 0.9, abs=1e-4)


def test_add_clamps():
    acc = Accumulator(4)
    acc.add(layer(4, 0.8, 0.5))
    acc.add(layer(4, 0.8, 0.5))
    assert acc.result()[0][0, 0] == pytest.approx(1.0)


def test_max_multiply_replace():
    acc = Accumulator(4)
    acc.add(layer(4, 0.4, 0.2))
    acc.add(layer(4, 0.7, 0.9), blend="max")
    assert acc.result()[0][0, 0] == pytest.approx(0.7)

    acc.add(layer(4, 0.5, 0.0), blend="multiply")
    out, _ = acc.result()
    assert out[0, 0] == pytest.approx(0.35)

    acc.add(layer(4, 0.1, 0.33), blend="replace")
    out, _ = acc.result()
    assert out[0, 0] == pytest.approx(0.1) and out[0, 2] == pytest.approx(0.33, abs=1e-5)


def test_gain_and_offset_placement():
    acc = Accumulator(10)
    acc.add(layer(4, 1.0, 0.5), start=8, gain=0.5)   # partly off the end
    acc.add(layer(4, 1.0, 0.5), start=-2)            # partly before the start
    out, active = acc.result()
    assert active.tolist() == [True, True] + [False] * 6 + [True, True]
    assert out[9, 0] == pytest.approx(0.5) and out[0, 0] == pytest.approx(1.0)
    assert out[5, 0] == 0.0


def test_frequency_is_held_through_silence():
    acc = Accumulator(10)
    acc.add(layer(3, 0.5, 0.7), start=2)
    out, _ = acc.result()
    assert out[0, 2] == pytest.approx(0.7, abs=1e-5)   # back-filled before first
    assert out[9, 2] == pytest.approx(0.7, abs=1e-5)   # held after last


def test_hold_frequency_without_any_signal():
    out = hold_frequency(np.zeros(5), np.zeros(5))
    assert np.all(out == np.float32(DEFAULT_FREQ))


def test_active_mask_limits_contribution():
    acc = Accumulator(6)
    mask = np.array([True, False, True, False, True, False])
    acc.add(layer(6, 0.5, 0.5), active=mask)
    out, active = acc.result()
    np.testing.assert_array_equal(active, mask)
    assert out[1, 0] == 0.0 and out[0, 0] == pytest.approx(0.5)


def test_amp_and_freq_can_come_from_different_layers():
    acc = Accumulator(4)
    acc.add(layer(4, 0.6, 0.2), freq_on=(False, True))         # amplitude, freq only on B
    acc.add(layer(4, 0.0, 0.9), amp_on=(False, False),          # frequency only
            freq_weight=np.full((4, 2), 0.5))
    out, _ = acc.result()
    assert out[0, 0] == pytest.approx(0.6) and out[0, 1] == pytest.approx(0.6)
    assert out[0, 2] == pytest.approx(0.9, abs=1e-5)            # A: the second layer only
    assert out[0, 3] == pytest.approx((0.6 * 0.2 + 0.5 * 0.9) / 1.1, abs=1e-4)


def test_muted_amplitude_does_not_gate_or_replace():
    acc = Accumulator(4)
    acc.add(layer(4, 0.5, 0.3))
    acc.add(layer(4, 0.0, 0.7), blend="multiply", amp_on=(False, False))
    acc.add(layer(4, 0.1, 0.7), blend="replace", amp_on=(False, True))
    out, _ = acc.result()
    assert out[0, 0] == pytest.approx(0.5)                      # untouched
    assert out[0, 1] == pytest.approx(0.1)
    assert out[0, 2] == pytest.approx(0.7, abs=1e-5)            # frequency still replaced
