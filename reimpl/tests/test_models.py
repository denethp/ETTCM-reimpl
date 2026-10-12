"""Checks the B matrices (Eq. 9) against brute-force projection of a moving 3D point.

Run:  python tests/test_models.py   (or: python -m pytest -q tests)

Physics: camera with linear velocity nu and angular velocity omega (camera frame).
A static world point P (camera coordinates) moves as  dP/dt = -nu - omega x P.
Its pixel x = fx*X/Z + cx, y = fy*Y/Z + cy. The finite-difference pixel velocity
must equal  B(p, rho=1/Z) @ theta  with theta = (nu, omega).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.models import DRIVING, SCALING, TRANSLATION, b_matrix, b_matrix_param, n_vars  # noqa: E402

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
    """Official re-parameterised 6-DOF (code sign: theta_code = -theta_paper).

    Unknowns m = (rho*nu_x + w_y, rho*nu_y - w_x, nu_z, w_x, w_y, w_z) in code
    sign; B_param(p, rho) @ m must give the same image velocity as the physics.
    """
    rng = np.random.default_rng(2)
    for _ in range(20):
        P = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])
        nu, om = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        rho = 1.0 / P[2]
        nc, wc = -nu, -om                      # code sign
        m = np.array([rho * nc[0] + wc[1], rho * nc[1] - wc[0], nc[2], wc[0], wc[1], wc[2]])
        x, y = project(P)
        B = np.zeros((2, 7))
        b_matrix_param(x - CX, y - CY, FX, FY, rho, B)
        np.testing.assert_allclose(B[:, :6] @ m, numeric_flow(P, nu, om),
                                   rtol=1e-5, atol=1e-5)


def test_driving():
    """Driving: forward speed nu_z and yaw rate w_y only (Sec. 4.2)."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        P = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 5)])
        nu = np.array([0.0, 0.0, rng.uniform(-1, 1)])
        om = np.array([0.0, rng.uniform(-1, 1), 0.0])
        np.testing.assert_allclose(model_flow(DRIVING, P, np.r_[nu[2], om[1]]),
                                   numeric_flow(P, nu, om), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    for f in (test_scaling, test_translation, test_six_dof, test_driving):
        f()
        print("PASS", f.__name__)
