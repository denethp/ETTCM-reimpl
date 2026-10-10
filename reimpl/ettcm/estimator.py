"""ETTCM per-event estimator -- YOUR re-implementation goes here (milestones M1-M3).

Everything around the maths (buffers, loops, outputs) is set up. Fill the five
TODO blocks. Each block says which equation it is and what the official C++
(include/ETTCM/global_motion_inverse_depth.hpp) does there.

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
from .models import b_matrix, n_v, n_vars


@njit
def _run(t_us, x_raw, y_raw, pol, xr, yr, cam_map, width, height,
         fx, fy, cx, cy, model, s, w_thresh, l_theta, l_rho, n_iter, d_min,
         out_rho, out_vx, out_vy, out_div, out_theta):
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

        # ------------------------------------------------------------ TODO 1
        # Predicted flow direction and shifted window.
        #   v = B(p_i, rho=1) @ theta, normalise it (skip if its norm is 0)
        #   centre  xc = x_raw - int(s * v_x),  yc = y_raw - int(s * v_y)
        #   window  x0 = max(xc - s, 0) ... x1 = min(xc + s, width - 1), same for y
        # (The C++ casts s*v to uint16 -- for negative values use int(), i.e. trunc.)
        x0 = x1 = y0 = y1 = 0
        raise NotImplementedError("TODO 1: window")

        rho = 0.0   # stays 0 if the window is empty -> event ignored for divergence
        if x1 >= x0 and y1 >= y0:
            # -------------------------------------------------------- TODO 2
            # Inverse-depth initialisation (Sec. 3.5, Eq. 11).
            # For every window cell with ctx_t >= 0:
            #     w = 1 / ((t - ctx_t) * a + 1);  ctx_w[c, y, x] = w   (store it!)
            #     accumulate w * rho_map[y, x] and w
            # rho = weighted mean, or 1.0 if no neighbour.
            # Then gamma = (theta, log(rho)), gamma0 = copy of gamma.

            # -------------------------------------------------------- TODO 3
            # Gauss-Newton, at most n_iter iterations. Each iteration:
            #   zero Psi, psi, w_sum
            #   for every window cell with dt = t - ctx_t > 0 and ctx_w >= w_thresh:
            #       (uj, vj) = cam_map[y, x]        rectified position of that pixel
            #       B = b_matrix(model, uj - cx, vj - cy, fx, fy, rho)   # Eq. (10):
            #           neighbour's position but the CURRENT event's rho
            #       v_full  = B @ gamma[:nvar]       ;  v_trans = B[:, :nv] @ gamma[:nv]
            #       r       = (xr_i - uj, yr_i - vj) - dt * v_full    # r = x_i - x_j'
            #       J[:, :nvar] = -dt * B
            #       J[:, nvar]  = -dt * v_trans      # d r / d lambda (rho = e^lambda)
            #       w = ctx_w * exp(-0.5 * |r|^2)    # Gaussian kernel, sigma = 1 px
            #       Psi += w J^T J ;  psi += w J^T r ;  w_sum += w
            #   if w_sum <= 0: break
            #   then TODO 4.

            # -------------------------------------------------------- TODO 4
            #   Solve (Psi + L) delta = -psi + L (gamma0 - gamma)   [cholesky_solve]
            #   gamma += delta ;  rho = exp(gamma[nvar])
            #   if |gamma[:nv]| > 1: gamma[:nv] /= |gamma[:nv]|     (|nu| <= 1, Sec. 3.6)
            #   if |delta| < d_min: break
            # After the loop: theta = gamma[:nvar];  rho_map[y_raw, x_raw] = rho
            raise NotImplementedError("TODO 2-4: estimation")

        # ------------------------------------------------------------ given
        # Per-event flow (Eq. 9) and context update.
        b_matrix(model, px, py, fx, fy, rho, B)
        vx = 0.0
        vy = 0.0
        for m in range(nvar):
            vx += B[0, m] * theta[m]
            vy += B[1, m] * theta[m]
        ctx_t[c, y_raw[k], x_raw[k]] = t
        ctx_w[c, y_raw[k], x_raw[k]] = 1.0

        # ------------------------------------------------------------ TODO 5
        # Global divergence (what Table 2 evaluates). Only if rho > 0:
        #     div_num = decay * div_num + rho * nu_z      (nu_z = theta[nv - 1])
        #     div_den = decay * div_den + 1

        out_rho[k] = rho
        out_vx[k] = vx
        out_vy[k] = vy
        out_div[k] = div_num / div_den if div_den > 0 else 0.0
        for m in range(nvar):
            out_theta[k, m] = theta[m]


def run_ettcm(events, xr, yr, cam, model, s=2, w_thresh=0.01, l_theta=1000.0,
              l_rho=1.0, n_iter=2, d_min=1e-4):
    """Run the estimator on a whole (rectified) event stream.

    Returns a dict of per-event arrays: rho, vx, vy (px/s), div (global divergence
    1/s, paper sign), ttc = 1/(rho*nu_z) per event, theta (N, n_vars).
    """
    n = len(events)
    out_rho = np.zeros(n)
    out_vx = np.zeros(n)
    out_vy = np.zeros(n)
    out_div = np.zeros(n)
    out_theta = np.zeros((n, n_vars(model)))
    _run(events["t"].astype(np.int64), events["x"].astype(np.int64),
         events["y"].astype(np.int64), events["p"].astype(np.bool_),
         xr.astype(np.float64), yr.astype(np.float64), cam.map.astype(np.float64),
         cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, model, int(s),
         float(w_thresh), float(l_theta), float(l_rho), int(n_iter), float(d_min),
         out_rho, out_vx, out_vy, out_div, out_theta)
    nu_z = out_theta[:, n_v(model) - 1]
    with np.errstate(divide="ignore"):
        ttc = 1.0 / (out_rho * nu_z)
    return dict(rho=out_rho, vx=out_vx, vy=out_vy, div=out_div, ttc=ttc,
                theta=out_theta)
