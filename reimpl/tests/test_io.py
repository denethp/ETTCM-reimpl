"""Round-trip test for the .es reader/writer."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.io_es import EVENT_DTYPE, read_es, write_es  # noqa: E402


def test_roundtrip():
    rng = np.random.default_rng(0)
    n = 5000
    ev = np.empty(n, EVENT_DTYPE)
    ev["t"] = np.cumsum(rng.integers(0, 600, n))  # includes gaps > 127 us (overflow bytes)
    ev["x"] = rng.integers(0, 1280, n)
    ev["y"] = rng.integers(0, 720, n)
    ev["p"] = rng.integers(0, 2, n)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.es")
        write_es(path, ev, 1280, 720)
        ev2, w, h = read_es(path)
    assert (w, h) == (1280, 720)
    assert np.array_equal(ev, ev2)


if __name__ == "__main__":
    test_roundtrip()
    print("PASS test_roundtrip")
