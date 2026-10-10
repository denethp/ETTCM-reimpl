"""Checks your B matrix (Eq. 9) against brute-force projection of a moving 3D point.

Run:  python -m pytest -q tests    (or: python tests/test_models.py)

Physics: camera with linear velocity nu and angular velocity omega (camera frame).
A static world point P (camera coordinates) moves as  dP/dt = -nu - omega x P.
Its pixel x = fx*X/Z + cx, y = fy*Y/Z + cy. The finite-difference pixel velocity
must equal  B(p, rho=1/Z) @ theta  with theta = (nu, omega).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.models import SCALING, SIX_DOF, TRANSLATION, b_matrix, n_vars  # noqa: E402

FX, FY, CX, CY = 320.0, 300.0, 160.0, 120.0


def project(P):
    return np.array([FX * P[0] / P[2] + CX, FY * P[1] / P[2] + CY])


def numeric_flow(P, nu, om, h=1e-6):
    dP = -nu - np.cross(om, P)
    return (project(P + h * dP) - project(P - h * dP)) / (2 * h)


def model_flow(model, P, theta):
    x, y = project(P)
    B = np.zeros((2, 7))
    b_matrix(model, x - CX, y - CY, FX, FY, 1.0 / P[2], B)
    return B[:, :n_vars(model)] @ theta


def test_scaling():
    rng = np.random.default_rng(0)
    for _ in range(20):
        P = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])
        nu = np.array([0.0, 0.0, rng.uniform(-1, 1)])
        np.testing.assert_allclose(model_flow(SCALING, P, nu[2:]),
                                   numeric_flow(P, nu, np.zeros(3)), rtol=1e-5, atol=1e-6)


def test_translation():
    rng = np.random.default_rng(1)
    for _ in range(20):
        P = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])
        nu = rng.uniform(-1, 1, 3)
        np.testing.assert_allclose(model_flow(TRANSLATION, P, nu),
                                   numeric_flow(P, nu, np.zeros(3)), rtol=1e-5, atol=1e-6)


def test_six_dof():
    rng = np.random.default_rng(2)
    for _ in range(20):
        P = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])
        nu, om = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        np.testing.assert_allclose(model_flow(SIX_DOF, P, np.r_[nu, om]),
                                   numeric_flow(P, nu, om), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    for f in (test_scaling, test_translation, test_six_dof):
        try:
            f()
            print("PASS", f.__name__)
        except NotImplementedError as e:
            print("TODO", f.__name__, "-", e)
