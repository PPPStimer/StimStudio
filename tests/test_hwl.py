import numpy as np
import pytest

from stimstudio.core.pulse import PulseTrack, array_from_bytes, array_to_bytes
from stimstudio.io.hwl import HWL_MAGIC, decode_hwl, encode_hwl, read_hwl, write_hwl


def _pulses(n=100, seed=0):
    return np.random.default_rng(seed).random((n, 4)).astype(np.float32)


def test_roundtrip(tmp_path):
    p = _pulses()
    path = tmp_path / "x.hwl"
    size = write_hwl(path, p)
    assert size == 8 + 16 * len(p)
    assert path.read_bytes()[:8] == HWL_MAGIC
    np.testing.assert_array_equal(read_hwl(path), p)


def test_layout_is_interleaved_little_endian():
    p = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    assert encode_hwl(p)[8:] == np.array([0.1, 0.2, 0.3, 0.4], dtype="<f4").tobytes()


def test_values_are_clamped():
    p = np.array([[-1.0, 2.0, 0.5, 1.5]], dtype=np.float32)
    np.testing.assert_array_equal(decode_hwl(encode_hwl(p)), [[0.0, 1.0, 0.5, 1.0]])


def test_bad_header():
    with pytest.raises(ValueError):
        decode_hwl(b"NOTHOWL!" + b"\0" * 16)


def test_bad_length():
    with pytest.raises(ValueError):
        decode_hwl(HWL_MAGIC + b"\0" * 15)


def test_pulsetrack_bridge():
    p = _pulses(10)
    t = PulseTrack.from_array(p)
    np.testing.assert_array_equal(t.l_freq, p[:, 2])
    np.testing.assert_array_equal(t.to_array(), p)
    np.testing.assert_array_equal(array_from_bytes(array_to_bytes(p)), p)
    assert t.to_bytes() == array_to_bytes(p)
    # write_hwl accepts a PulseTrack too
    assert encode_hwl(t) == encode_hwl(p)
