"""ETTCM per-event estimator (milestones M1-M3).

Mirrors the official C++ core (include/ETTCM/global_motion_inverse_depth.hpp)
in the paper's sign convention.

State kept between events
-------------------------
theta              global motion, paper convention (nu, omega); starts at 0
rho_map[y, x]      last inverse depth estimated at raw pixel (x, y); starts at 1
ctx_t[c, y, x]     timestamp (s) of the last event at (x, y) with polarity c; -1 = none
ctx_w[c, y, x]     Eq. (11) weight of that event, refreshed when it is a neighbour
a, t_last          global-decay state (decay.py): a = "global activity" a(t)
div_num, div_den   running global divergence (decayed average)

Per event i (t in seconds, raw integer pixel (x_raw, y_raw), rectified (xr, yr)):
  1. predict flow direction, shift the (2s+1)^2 window back along it
  2. initialise rho_i from the neighbours (Eq. 11)
  3. Gauss-Newton on gamma = (theta, lambda = log rho) with priors (Eq. 8, Sec. 3.6)
  4. per-event flow (Eq. 9) + store the event in the context
  5. global divergence = decayed mean of rho_i * nu_z

Hyper-parameters (Table 1 + official scripts): s = 2 or 3, w_thresh = 0.01,
l_theta = 1000 (VL) / 100 (MVSEC), l_rho = 1, n_iter = 2, d_min = 1e-4.
"""
import math

import numpy as np

from ._jit import njit
from .decay import decay_step
from .linalg import cholesky_solve
from .models import (SIX_DOF, b_matrix, b_matrix_code6, b_matrix_param,
                     n_v, n_vars)

_TWO52 = 4503599627370496.0  # 2**52


@njit
def fast_exp(x):
    """Schraudolph's fast exp, bit-for-bit the official compute_exp() with
    ETTCM_FAST_EXP (ON by default in the official CMake build, so the paper's
    numbers use it). Builds the IEEE-754 double whose bits are
    trunc(2^52/ln2 * x + 2^52 * 1023): a piecewise-linear 2^(x/ln2), up to ~6 % off.
    """
    if x < -600.0:
        return 0.0
    bits = int((_TWO52 / 0.69314718056) * x + _TWO52 * 1023.0)
    e = bits >> 52
    mant = bits & 4503599627370495
    return (1.0 + mant / _TWO52) * 2.0 ** (e - 1023)


@njit
def _kernel(x, use_fast_exp):
    if use_fast_exp:
        return fast_exp(x)
    return math.exp(x)


@njit
def _run(t_us, x_raw, y_raw, pol, xr, yr, cam_map, width, height,
         fx, fy, cx, cy, model, s, w_thresh, l_theta, l_rho, n_iter, d_min,
         use_fast_exp, out_rho, out_vx, out_vy, out_div, out_theta):
    nv = n_v(model)        # number of linear-velocity params; nu_z = theta[nv - 1]
    nvar = n_vars(model)   # number of motion params
    nm = nvar + 1          # + lambda (log inverse depth)

    ctx_t = np.full((2, height, width), -1.0)
    ctx_w = np.zeros((2, height, width))
    rho_map = np.ones((height, width))

    theta = np.zeros(nvar)
    gamma = np.zeros(nm)     # current estimate (theta, lambda)
    gamma0 = np.zeros(nm)    # value at the start of this event (prior centre)
    B = np.zeros((2, 7))
    J = np.zeros((2, 7))
    Psi = np.zeros((7, 7))   # sum w J^T J
    psi = np.zeros(7)        # sum w J^T r
    A = np.zeros((7, 7))
    rhs = np.zeros(7)
    delta = np.zeros(7)
    Lw = np.zeros((7, 7))
    prior = np.full(nm, l_theta)  # L = diag(l_theta, ..., l_theta, l_rho)
    prior[nvar] = l_rho

    t_last = 0
    a = 0.0
    div_num = 0.0
    div_den = 0.0

    for k in range(t_us.shape[0]):
        decay, t_last, a = decay_step(t_us[k], t_last, a)
        t = 1e-6 * float(t_us[k])
        c = 1 if pol[k] else 0          # separate ON / OFF contexts
        px = xr[k] - cx
        py = yr[k] - cy

        # ---- 1. Predicted flow direction and shifted window ------------------
        # Direction the scene is sliding at this pixel (rho = 1, only the
        # direction matters). The edge came from "behind" it, so the window
        # is centred s pixels back along that direction.
        b_matrix(model, px, py, fx, fy, 1.0, B)
        vx = 0.0
        vy = 0.0
        for m in range(nvar):
            vx += B[0, m] * theta[m]
            vy += B[1, m] * theta[m]
        nrm = math.sqrt(vx * vx + vy * vy)
        if nrm > 0:
            vx /= nrm
            vy /= nrm
        xc = int(x_raw[k]) - int(s * vx)     # int() truncates like the C++ cast
        yc = int(y_raw[k]) - int(s * vy)
        x0 = max(xc - s, 0)
        x1 = min(xc + s, width - 1)
        y0 = max(yc - s, 0)
        y1 = min(yc + s, height - 1)
        if xc < 0 or yc < 0:
            # The C++ centre is a uint16: a negative value wraps to ~65535,
            # which makes the window empty. Mirror that.
            x1 = -1

        rho = 0.0   # stays 0 if the window is empty -> event ignored for divergence
        if x1 >= x0 and y1 >= y0:
            # ---- 2. Inverse-depth initialisation (Sec. 3.5, Eq. 11) ----------
            # Weighted mean of the neighbours' rho; recent neighbours weigh more.
            num = 0.0
            den = 0.0
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    tj = ctx_t[c, yy, xx]
                    if tj >= 0:
                        w = 1.0 / ((t - tj) * a + 1.0)
                        ctx_w[c, yy, xx] = w
                        num += w * rho_map[yy, xx]
                        den += w
            rho = num / den if den > 0 else 1.0

            for m in range(nvar):
                gamma[m] = theta[m]
            gamma[nvar] = math.log(rho)
            for m in range(nm):
                gamma0[m] = gamma[m]

            # ---- 3. Gauss-Newton on gamma = (theta, log rho), Eq. (8) --------
            for it in range(n_iter):
                for i in range(nm):
                    psi[i] = 0.0
                    for j in range(nm):
                        Psi[i, j] = 0.0
                w_sum = 0.0
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        dt = t - ctx_t[c, yy, xx]
                        if dt > 0 and ctx_w[c, yy, xx] >= w_thresh:
                            uj = cam_map[yy, xx, 0]
                            vj = cam_map[yy, xx, 1]
                            # Eq. (10): neighbour position, CURRENT event's rho
                            b_matrix(model, uj - cx, vj - cy, fx, fy, rho, B)
                            fvx = 0.0
                            fvy = 0.0
                            tvx = 0.0
                            tvy = 0.0
                            for m in range(nvar):
                                fvx += B[0, m] * gamma[m]
                                fvy += B[1, m] * gamma[m]
                                if m < nv:
                                    tvx += B[0, m] * gamma[m]
                                    tvy += B[1, m] * gamma[m]
                            # residual: where the event is minus where the
                            # neighbour's edge should have moved to by now
                            r0 = (xr[k] - uj) - dt * fvx
                            r1 = (yr[k] - vj) - dt * fvy
                            for m in range(nvar):
                                J[0, m] = -dt * B[0, m]
                                J[1, m] = -dt * B[1, m]
                            # d(rho V nu)/d lambda = rho V nu  (rho = exp(lambda))
                            J[0, nvar] = -dt * tvx
                            J[1, nvar] = -dt * tvy
                            w = ctx_w[c, yy, xx] * _kernel(-0.5 * (r0 * r0 + r1 * r1), use_fast_exp)
                            w_sum += w
                            for i in range(nm):
                                psi[i] += w * (J[0, i] * r0 + J[1, i] * r1)
                                for j in range(nm):
                                    Psi[i, j] += w * (J[0, i] * J[0, j] + J[1, i] * J[1, j])
                if w_sum <= 0:
                    break

                # ---- 4. Solve (Psi + L) delta = -psi + L (gamma0 - gamma) ----
                for i in range(nm):
                    for j in range(nm):
                        A[i, j] = Psi[i, j]
                    A[i, i] += prior[i]
                    rhs[i] = -psi[i] + prior[i] * (gamma0[i] - gamma[i])
                if not cholesky_solve(A, rhs, nm, delta, Lw):
                    break
                dn = 0.0
                for i in range(nm):
                    gamma[i] += delta[i]
                    dn += delta[i] * delta[i]
                rho = math.exp(gamma[nvar])
                # |nu| <= 1 (Sec. 3.6): only the direction/scale-free part is kept
                vn = 0.0
                for m in range(nv):
                    vn += gamma[m] * gamma[m]
                vn = math.sqrt(vn)
                if vn > 1.0:
                    for m in range(nv):
                        gamma[m] /= vn
                if math.sqrt(dn) < d_min:
                    break
            for m in range(nvar):
                theta[m] = gamma[m]
            rho_map[y_raw[k], x_raw[k]] = rho

        # ---- Per-event flow (Eq. 9) and context update -----------------------
        b_matrix(model, px, py, fx, fy, rho, B)
        vx = 0.0
        vy = 0.0
        for m in range(nvar):
            vx += B[0, m] * theta[m]
            vy += B[1, m] * theta[m]
        ctx_t[c, y_raw[k], x_raw[k]] = t
        ctx_w[c, y_raw[k], x_raw[k]] = 1.0

        # ---- 5. Global divergence = decayed mean of rho_i * nu_z -------------
        if rho > 0:
            div_num = decay * div_num + rho * theta[nv - 1]
            div_den = decay * div_den + 1.0

        out_rho[k] = rho
        out_vx[k] = vx
        out_vy[k] = vy
        out_div[k] = div_num / div_den if div_den > 0 else 0.0
        for m in range(nvar):
            out_theta[k, m] = theta[m]


@njit
def _run_param(t_us, x_raw, y_raw, pol, xr, yr, cam_map, width, height,
               fx, fy, cx, cy, s, w_thresh, l_theta, l_rho, n_iter, d_min,
               use_fast_exp, out_rho, out_vx, out_vy, out_div, out_theta):
    """6-DOF with the official re-parameterisation (ParameterizedSixDOF).

    Same five steps as _run, written in the official code's sign convention
    (see models.b_matrix_param). The Gauss-Newton Jacobian is copied exactly
    from the C++ (ParameterizedSixDOF::jacobian and the d/d(lambda) column in
    global_motion_inverse_depth.hpp), even though it does not match the model's
    own derivative (e.g. the lambda column is not multiplied by -dt): the
    paper's numbers were produced with it. Outputs are converted to the paper
    convention.
    """
    nm = 7
    ctx_t = np.full((2, height, width), -1.0)
    ctx_w = np.zeros((2, height, width))
    rho_map = np.ones((height, width))
    mv = np.zeros(nm)        # (m0..m5, lambda), code convention
    mv0 = np.zeros(nm)
    v3 = np.zeros(3)
    B = np.zeros((2, 7))
    Bc = np.zeros((2, 7))
    J = np.zeros((2, 7))
    Psi = np.zeros((7, 7))
    psi = np.zeros(7)
    A = np.zeros((7, 7))
    rhs = np.zeros(7)
    delta = np.zeros(7)
    Lw = np.zeros((7, 7))
    prior = np.full(nm, l_theta)
    prior[6] = l_rho
    t_last = 0
    a = 0.0
    div_num = 0.0
    div_den = 0.0

    for k in range(t_us.shape[0]):
        decay, t_last, a = decay_step(t_us[k], t_last, a)
        t = 1e-6 * float(t_us[k])
        c = 1 if pol[k] else 0
        px = xr[k] - cx
        py = yr[k] - cy

        b_matrix_param(px, py, fx, fy, 1.0, B)
        vx = 0.0
        vy = 0.0
        for m in range(6):
            vx += B[0, m] * mv[m]
            vy += B[1, m] * mv[m]
        nrm = math.sqrt(vx * vx + vy * vy)
        if nrm > 0:
            vx /= nrm
            vy /= nrm
        xc = int(x_raw[k]) - int(s * vx)
        yc = int(y_raw[k]) - int(s * vy)
        x0 = max(xc - s, 0)
        x1 = min(xc + s, width - 1)
        y0 = max(yc - s, 0)
        y1 = min(yc + s, height - 1)
        if xc < 0 or yc < 0:
            x1 = -1

        rho = 0.0
        if x1 >= x0 and y1 >= y0:
            num = 0.0
            den = 0.0
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    tj = ctx_t[c, yy, xx]
                    if tj >= 0:
                        w = 1.0 / ((t - tj) * a + 1.0)
                        ctx_w[c, yy, xx] = w
                        num += w * rho_map[yy, xx]
                        den += w
            rho = num / den if den > 0 else 1.0
            mv[6] = math.log(rho)
            v3[0] = (mv[0] - mv[4]) / rho
            v3[1] = (mv[1] + mv[3]) / rho
            v3[2] = mv[2]
            for m in range(nm):
                mv0[m] = mv[m]

            for it in range(n_iter):
                for i in range(nm):
                    psi[i] = 0.0
                    for j in range(nm):
                        Psi[i, j] = 0.0
                w_sum = 0.0
                for yy in range(y0, y1 + 1):
                    for xx in range(x0, x1 + 1):
                        dt = t - ctx_t[c, yy, xx]
                        if dt > 0 and ctx_w[c, yy, xx] >= w_thresh:
                            uj = cam_map[yy, xx, 0]
                            vj = cam_map[yy, xx, 1]
                            pcx = uj - cx
                            pcy = vj - cy
                            b_matrix_code6(pcx, pcy, fx, fy, rho, Bc)
                            # Official ParameterizedSixDOF::jacobian = -dt * B_code6
                            # but with fx, fy (NOT rho*fx, rho*fy) in columns 0, 1:
                            # d(B_param m)/dm0 = fx, d/dm1 = fy.
                            for m in range(6):
                                J[0, m] = -dt * Bc[0, m]
                                J[1, m] = -dt * Bc[1, m]
                            J[0, 0] = -dt * fx
                            J[1, 1] = -dt * fy
                            J[0, 6] = Bc[0, 0] * v3[0] + Bc[0, 1] * v3[1] + Bc[0, 2] * v3[2]
                            J[1, 6] = Bc[1, 0] * v3[0] + Bc[1, 1] * v3[1] + Bc[1, 2] * v3[2]
                            b_matrix_param(pcx, pcy, fx, fy, rho, B)
                            fvx = 0.0
                            fvy = 0.0
                            for m in range(6):
                                fvx += B[0, m] * mv[m]
                                fvy += B[1, m] * mv[m]
                            r0 = (xr[k] - uj) - dt * fvx
                            r1 = (yr[k] - vj) - dt * fvy
                            w = ctx_w[c, yy, xx] * _kernel(-0.5 * (r0 * r0 + r1 * r1), use_fast_exp)
                            w_sum += w
                            for i in range(nm):
                                psi[i] += w * (J[0, i] * r0 + J[1, i] * r1)
                                for j in range(nm):
                                    Psi[i, j] += w * (J[0, i] * J[0, j] + J[1, i] * J[1, j])
                if w_sum <= 0:
                    break
                for i in range(nm):
                    for j in range(nm):
                        A[i, j] = Psi[i, j]
                    A[i, i] += prior[i]
                    rhs[i] = -psi[i] + prior[i] * (mv0[i] - mv[i])
                if not cholesky_solve(A, rhs, nm, delta, Lw):
                    break
                dn = 0.0
                for i in range(nm):
                    mv[i] += delta[i]
                    dn += delta[i] * delta[i]
                rho = math.exp(mv[6])
                v3[0] = (mv[0] - mv[4]) / rho
                v3[1] = (mv[1] + mv[3]) / rho
                v3[2] = mv[2]
                vn = math.sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])
                if vn > 1.0:
                    v3[0] /= vn
                    v3[1] /= vn
                    v3[2] /= vn
                    mv[0] = rho * v3[0] + mv[4]
                    mv[1] = rho * v3[1] - mv[3]
                    mv[2] = v3[2]
                if math.sqrt(dn) < d_min:
                    break
            rho_map[y_raw[k], x_raw[k]] = rho

        b_matrix_param(px, py, fx, fy, rho, B)
        vx = 0.0
        vy = 0.0
        for m in range(6):
            vx += B[0, m] * mv[m]
            vy += B[1, m] * mv[m]
        ctx_t[c, y_raw[k], x_raw[k]] = t
        ctx_w[c, y_raw[k], x_raw[k]] = 1.0

        nu_z = -mv[2]                      # paper sign
        if rho > 0:
            div_num = decay * div_num + rho * nu_z
            div_den = decay * div_den + 1.0
        out_rho[k] = rho
        out_vx[k] = vx
        out_vy[k] = vy
        out_div[k] = div_num / div_den if div_den > 0 else 0.0
        r = rho if rho > 0 else 1.0
        out_theta[k, 0] = -(mv[0] - mv[4]) / r
        out_theta[k, 1] = -(mv[1] + mv[3]) / r
        out_theta[k, 2] = nu_z
        out_theta[k, 3] = -mv[3]
        out_theta[k, 4] = -mv[4]
        out_theta[k, 5] = -mv[5]


def run_ettcm(events, xr, yr, cam, model, s=2, w_thresh=0.01, l_theta=1000.0,
              l_rho=1.0, n_iter=2, d_min=1e-4, fast_exp=True):
    """Run the estimator on a whole (rectified) event stream.

    fast_exp=True uses the official build's approximate exp() for the residual
    kernel (needed to reproduce the paper's numbers exactly, mostly for 6-DOF);
    fast_exp=False uses the exact exp().

    Returns a dict of per-event arrays: rho, vx, vy (px/s), div (global divergence
    1/s, paper sign), ttc = 1/(rho*nu_z) per event, theta (N, n_vars).
    """
    n = len(events)
    out_rho = np.zeros(n)
    out_vx = np.zeros(n)
    out_vy = np.zeros(n)
    out_div = np.zeros(n)
    out_theta = np.zeros((n, n_vars(model)))
    args = (events["t"].astype(np.int64), events["x"].astype(np.int64),
            events["y"].astype(np.int64), events["p"].astype(np.bool_),
            xr.astype(np.float64), yr.astype(np.float64), cam.map.astype(np.float64),
            cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
    hyper = (int(s), float(w_thresh), float(l_theta), float(l_rho), int(n_iter),
             float(d_min), bool(fast_exp), out_rho, out_vx, out_vy, out_div, out_theta)
    if model == SIX_DOF:
        _run_param(*args, *hyper)
    else:
        _run(*args, model, *hyper)
    nu_z = out_theta[:, n_v(model) - 1]
    with np.errstate(divide="ignore"):
        ttc = 1.0 / (out_rho * nu_z)
    return dict(rho=out_rho, vx=out_vx, vy=out_vy, div=out_div, ttc=ttc,
                theta=out_theta)
