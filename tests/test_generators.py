from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from stimstudio.generators import get_generator, list_generators

GENERATORS = list_generators()


@pytest.mark.parametrize("gen_cls", GENERATORS, ids=lambda c: c.type_id)
def test_generator_produces_valid_track(gen_cls):
    n = 160
    track = gen_cls().render(n, gen_cls.default_params(), 1)
    assert track.n == n
    for arr in track.channels:
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))
    arr = track.clamp().to_array()
    assert arr.min() >= 0.0 and arr.max() <= 1.0


@pytest.mark.parametrize("gen_cls", [g for g in GENERATORS if g.stochastic],
                         ids=lambda c: c.type_id)
def test_stochastic_generators_are_deterministic(gen_cls):
    a = gen_cls().render(200, {}, 5).to_array()
    b = gen_cls().render(200, {}, 5).to_array()
    np.testing.assert_array_equal(a, b)


def test_parallel_renders_match_sequential():
    """The engine RNG is thread-local: parallel activity renders must not interfere."""
    jobs = [("licks", 1), ("chaos", 2), ("succubus", 3), ("milkmaster", 4)] * 2

    def run(job):
        type_id, seed = job
        return get_generator(type_id).render(400, {}, seed).to_array()

    sequential = [run(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(run, jobs))
    for s, p in zip(sequential, parallel):
        np.testing.assert_array_equal(s, p)


def test_segment_boundaries_for_staged_activity():
    gen = get_generator("milkmaster")
    bounds = gen.segment_boundaries(60 * 40, {}, 1)
    assert bounds and bounds == sorted(bounds)
    assert bounds == gen.segment_boundaries(60 * 40, {}, 1)


@pytest.mark.parametrize("type_id", ["licks", "luxury_hj", "penetration", "vibrator", "chaos",
                                     "succubus"])
def test_activities_expose_stages(type_id):
    """Activities with discrete internal changes report stage boundaries."""
    bounds = get_generator(type_id).segment_boundaries(600 * 40, {}, 3)
    assert len(bounds) > 3 and bounds == sorted(bounds)


def test_luxury_hj_bonus_freezes_as_manual():
    from stimstudio.core.render import manual_params_at, segment_stages
    snaps = [manual_params_at("luxury_hj", {}, 0, (a + b) / 2)
             for a, b in segment_stages("luxury_hj", {}, 0, 600.0)]
    assert any(s["bonus"] in ("A", "B") for s in snaps)
    assert all(s["manual"] is True for s in snaps)


def test_manual_params_snapshot():
    snap = get_generator("licks").manual_params_at(12 * 40, {}, 2)
    assert snap.get("manual") is True and "speed" in snap


#: Generators that are new in StimStudio (no WaveHowl counterpart).
NEW_GENERATORS = {"noise", "euclid", "ramp", "tri_point", "tri_circle", "tri_pendulum",
                  "tri_spiral", "tri_wander", "tri_electrodes", "tri_lissajous", "tri_rabbit",
                  "tri_attractor"}
NEW_GENERATORS |= {g.type_id for g in GENERATORS
                   if g.__module__.rsplit(".", 1)[-1] in {"stim", "perceptual", "complexity",
                                                          "longform"}}


@pytest.mark.parametrize("type_id", [g.type_id for g in GENERATORS
                                     if g.type_id not in NEW_GENERATORS])
def test_parity_with_wavehowl(wavehowl, type_id):
    """Every generator ported from WaveHowl renders bit-identically."""
    from wavehowl.generators import get_generator as wh_get
    params = get_generator(type_id).default_params()
    ours = get_generator(type_id).render(300, params, 11).to_array()
    theirs = wh_get(type_id).render(300, params, 11)
    theirs = np.stack(theirs.channels, axis=1)
    np.testing.assert_array_equal(ours, theirs)
