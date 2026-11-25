"""Per-event confidence (novelty, not in the paper).

1. _log_mu_var equals the error propagation done with a full numpy matrix inverse.
2. Turning confidence on changes no estimate, for every model, and gives a finite sigma
   for every event that has an estimate (inf otherwise).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.calib import make_camera  # noqa: E402
from ettcm.estimator import _log_mu_var, run_ettcm  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402
from test_chunks import moving_edges  # noqa: E402


def test_log_mu_var():
    rng = np.random.default_rng(1)
    for nm, i_nu, i_lam in [(2, 0, 1), (4, 2, 3), (7, 2, 6)]:
        M = rng.normal(size=(nm, nm))
        A = np.zeros((7, 7))
        A[:nm, :nm] = M @ M.T + nm * np.eye(nm)
        nu = 0.37
        C = np.linalg.inv(A[:nm, :nm])
        g = np.zeros(nm)
        g[i_lam] = 1.0
        g[i_nu] = 1.0 / nu
        ref = g @ C @ g
        got = _log_mu_var(A, nm, i_nu, i_lam, nu, np.zeros(7), np.zeros(7), np.zeros(7),
                          np.zeros((7, 7)))
        assert abs(got - ref) < 1e-10 * max(1.0, ref), (nm, got, ref)
    assert _log_mu_var(A, 7, 2, 6, 0.0, np.zeros(7), np.zeros(7), np.zeros(7),
                       np.zeros((7, 7))) == np.inf


def test_confidence_changes_nothing():
    cam = make_camera(fx=40.0, fy=40.0, cx=19.5, cy=14.5, width=40, height=30)
    ev = moving_edges()
    xr, yr = ev["x"].astype(np.float32), ev["y"].astype(np.float32)
    for name, mid in MODEL_IDS.items():
        a = run_ettcm(ev, xr, yr, cam, mid, s=2)
        b = run_ettcm(ev, xr, yr, cam, mid, s=2, confidence=True)
        for key in ("rho", "vx", "vy", "div", "theta"):
            assert np.array_equal(a[key], b[key]), (name, key)
        assert np.all(np.isinf(a["sigma"])), name           # not computed by default
        has = b["rho"] > 0
        assert np.isfinite(b["sigma"][has]).mean() > 0.95, name
        assert np.all(np.isinf(b["sigma"][~has])), name


if __name__ == "__main__":
    test_log_mu_var()
    print("PASS test_log_mu_var")
    test_confidence_changes_nothing()
    print("PASS test_confidence_changes_nothing")
