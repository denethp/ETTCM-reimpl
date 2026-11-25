"""Leaky-firing event sampling (supplementary Table 6) and the reduced camera."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.calib import make_camera  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.sampling import leaky_sampling  # noqa: E402


def make_events(xs, ys, ps):
    ev = np.zeros(len(xs), EVENT_DTYPE)
    ev["t"] = np.arange(len(xs))
    ev["x"], ev["y"], ev["p"] = xs, ys, ps
    return ev


def test_sampling():
    # r = 1: unchanged
    ev = make_events([0, 1, 2], [0, 0, 0], [1, 1, 1])
    assert leaky_sampling(ev, 4, 4, 1) is ev

    # r = 2: the 4 pixels (0..1, 0..1) share one ON counter, threshold 4:
    # the 4th and 8th ON events fire, at reduced coordinates (0, 0)
    xs = [0, 1, 0, 1, 0, 1, 0, 1, 3]
    ys = [0, 0, 1, 1, 0, 0, 1, 1, 3]
    ps = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    out = leaky_sampling(make_events(xs, ys, ps), 4, 4, 2)
    assert list(out["t"]) == [3, 7], out["t"]
    assert list(out["x"]) == [0, 0] and list(out["y"]) == [0, 0]

    # ON and OFF have separate counters
    out = leaky_sampling(make_events([0] * 4, [0] * 4, [1, 0, 1, 0]), 4, 4, 2)
    assert len(out) == 0
    print("PASS test_sampling")


def test_reduced_camera():
    full = make_camera(fx=200.0, fy=210.0, cx=120.0, cy=90.0, width=241, height=181)
    half = make_camera(fx=200.0, fy=210.0, cx=120.0, cy=90.0, width=241, height=181, sampling=2)
    assert (half.width, half.height) == (121, 91)          # ceil, as the official code
    assert np.allclose([half.fx, half.fy, half.cx, half.cy],
                       [full.fx / 2, full.fy / 2, full.cx / 2, full.cy / 2])
    assert half.map.shape == (91, 121, 2)
    print("PASS test_reduced_camera")


if __name__ == "__main__":
    test_sampling()
    test_reduced_camera()
