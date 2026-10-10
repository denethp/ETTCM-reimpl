"""Synthetic events with known ground truth (milestone M1 test data).

Scene: an infinite textured plane, fronto-parallel, at depth Z(t) = Z0 - nu_z * t.
The camera translates with constant velocity nu = (nu_x, nu_y, nu_z) (paper
convention: camera frame, nu_z > 0 means approaching). No rotation, no distortion.

Ground truth per time t:
    inverse depth rho(t) = 1 / Z(t)       (same for every pixel: planar scene)
    divergence     mu(t) = nu_z / Z(t)
    TTC           tau(t) = Z(t) / nu_z
    flow at pixel p      = rho * V(p) nu    (Eq. 9)

Events come from an idealised event camera: log-intensity is rendered at a high
frame rate and an event is emitted every time it crosses +/- C since the last
event at that pixel, with timestamps linearly interpolated between frames.
"""
import numpy as np

from .io_es import EVENT_DTYPE


def _texture(X, Y, comps, sharpness):
    """Random texture, strictly positive. sharpness=0: smooth; >0: blobs with edges."""
    S = np.zeros_like(X)
    for a, kx, ky, ph in comps:
        S += a * np.cos(kx * X + ky * Y + ph)
    if sharpness > 0:
        return 1.0 + 0.8 * np.tanh(sharpness * S)
    return 1.0 + S


def make_plane_sequence(width=120, height=90, f=100.0, Z0=2.0,
                        nu=(0.0, 0.0, 1.0), duration=0.5, frame_dt=2e-4,
                        C=0.6, n_comp=24, sharpness=30.0, seed=0):
    """Returns (events, gt) where gt is a dict of callables/arrays (see module doc)."""
    rng = np.random.default_rng(seed)
    nu = np.asarray(nu, float)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    # random spatial frequencies (rad / m) chosen so features are a few px wide at Z0
    k = rng.uniform(15.0, 60.0, n_comp)
    ang = rng.uniform(0, 2 * np.pi, n_comp)
    comps = [(0.6 / np.sqrt(n_comp) * rng.uniform(0.5, 1.5), k[i] * np.cos(ang[i]),
              k[i] * np.sin(ang[i]), rng.uniform(0, 2 * np.pi)) for i in range(n_comp)]

    xs, ys = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    rx, ry = (xs - cx) / f, (ys - cy) / f  # normalised ray directions

    def log_frame(t):
        Z = Z0 - nu[2] * t
        Xc, Yc = nu[0] * t, nu[1] * t      # camera position on the plane
        I = _texture(rx * Z + Xc, ry * Z + Yc, comps, sharpness)
        return np.log(np.clip(I, 1e-3, None))

    n_frames = int(round(duration / frame_dt))
    L_ref = log_frame(0.0)
    L_prev = L_ref.copy()
    chunks = []
    for i in range(1, n_frames + 1):
        t0, t1 = (i - 1) * frame_dt, i * frame_dt
        L_new = log_frame(t1)
        diff = L_new - L_ref
        n_cross = np.floor(np.abs(diff) / C).astype(int)
        yy, xx = np.nonzero(n_cross)
        if yy.size:
            sgn = np.sign(diff[yy, xx])
            dL = L_new[yy, xx] - L_prev[yy, xx]
            for m in range(1, n_cross.max() + 1):
                sel = n_cross[yy, xx] >= m
                level = L_ref[yy[sel], xx[sel]] + sgn[sel] * m * C
                frac = (level - L_prev[yy[sel], xx[sel]]) / np.where(dL[sel] == 0, 1, dL[sel])
                frac = np.clip(frac, 0.0, 1.0)
                ev = np.empty(sel.sum(), EVENT_DTYPE)
                ev["t"] = np.round(1e6 * (t0 + frac * frame_dt)).astype(np.uint64)
                ev["x"], ev["y"] = xx[sel], yy[sel]
                ev["p"] = (sgn[sel] > 0).astype(np.uint8)
                chunks.append(ev)
            L_ref[yy, xx] += sgn * n_cross[yy, xx] * C
        L_prev = L_new

    events = np.concatenate(chunks) if chunks else np.empty(0, EVENT_DTYPE)
    events = events[np.argsort(events["t"], kind="stable")]

    gt = dict(width=width, height=height, fx=f, fy=f, cx=cx, cy=cy, Z0=Z0, nu=nu,
              depth=lambda t: Z0 - nu[2] * np.asarray(t),
              divergence=lambda t: nu[2] / (Z0 - nu[2] * np.asarray(t)),
              ttc=lambda t: (Z0 - nu[2] * np.asarray(t)) / nu[2])
    return events, gt
