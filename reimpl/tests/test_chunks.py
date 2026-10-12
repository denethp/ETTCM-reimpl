"""Processing an event stream in chunks (as run_mvsec.py does) gives exactly the
same result as one run, for every model."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.calib import make_camera  # noqa: E402
from ettcm.estimator import EttcmState, run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402


def moving_edges(n=3000, w=40, h=30, seed=0):
    """Vertical edges sliding right: events with consistent motion."""
    rng = np.random.default_rng(seed)
    ev = np.empty(n, EVENT_DTYPE)
    t = np.sort(rng.integers(0, 200_000, n)).astype(np.uint64)
    ev["t"] = t
    ev["x"] = ((t // 2000) + rng.integers(0, 3, n) * 10) % w
    ev["y"] = rng.integers(0, h, n)
    ev["p"] = 1
    return ev


def test_chunks():
    cam = make_camera(fx=40.0, fy=40.0, cx=19.5, cy=14.5, width=40, height=30)
    ev = moving_edges()
    xr, yr = ev["x"].astype(np.float32), ev["y"].astype(np.float32)
    for name, mid in MODEL_IDS.items():
        a = run_ettcm(ev, xr, yr, cam, mid, s=2)
        st = EttcmState(cam.width, cam.height)
        parts = [run_ettcm(ev[i:j], xr[i:j], yr[i:j], cam, mid, s=2, state=st)
                 for i, j in [(0, 1000), (1000, 1001), (1001, 3000)]]
        for key in ("rho", "vx", "vy", "div", "theta"):
            assert np.array_equal(a[key], np.concatenate([p[key] for p in parts])), (name, key)


if __name__ == "__main__":
    test_chunks()
    print("PASS test_chunks")
