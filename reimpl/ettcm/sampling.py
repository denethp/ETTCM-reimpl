"""Event sampling by "leaky firing" (supplementary material, Sec. B, Fig. 7, Table 6).

Mirrors the official include/ETTCM/leaky_sampling.hpp with width and height sampling factors
both equal to r and firing threshold r * r:
  - each event's coordinates are divided by r and truncated: (x // r, y // r);
  - one counter per reduced pixel and polarity (ON / OFF) is incremented;
  - when it reaches r * r, r * r is subtracted and the event is passed on, with the
    reduced coordinates (the rest of the pipeline then runs at the reduced resolution,
    with the camera from calib.make_camera(..., sampling=r)).
r = 1 keeps every event unchanged (Table 2).
"""
import math

import numpy as np

from ._jit import njit


@njit
def _leaky(x, y, p, r, width, height, keep, xs, ys):
    w = int(math.ceil(width / r))
    h = int(math.ceil(height / r))
    ctx = np.zeros((2, h, w))
    thr = float(r * r)
    for k in range(x.shape[0]):
        xi = int(x[k] / r)
        yi = int(y[k] / r)
        c = 1 if p[k] else 0
        ctx[c, yi, xi] += 1.0
        if ctx[c, yi, xi] >= thr:
            ctx[c, yi, xi] -= thr
            keep[k] = True
            xs[k] = xi
            ys[k] = yi


def leaky_sampling(events, width, height, r):
    """Events that fire, with coordinates on the grid reduced by r (structured array copy)."""
    if r == 1:
        return events
    n = len(events)
    keep = np.zeros(n, np.bool_)
    xs = np.zeros(n, np.int64)
    ys = np.zeros(n, np.int64)
    _leaky(events["x"].astype(np.int64), events["y"].astype(np.int64),
           events["p"].astype(np.bool_), float(r), int(width), int(height), keep, xs, ys)
    out = events[keep].copy()
    out["x"] = xs[keep]
    out["y"] = ys[keep]
    return out
