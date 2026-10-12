"""Diagnostics for the MVSEC divergence (Table 3): where does a scale error come from?

    python scripts/diag_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --start 20 --seconds 10

1. GT camera velocity two independent ways: (a) least squares from the GT flow + depth
   (what DivergenceGT uses), (b) from MVSEC's odometry poses (4x4, gt.hdf5). If (a) and (b)
   disagree, our GT divergence is wrong.
2. Our estimate per component: rho * nu_x, rho * nu_y, rho * nu_z (velocity / depth, 1/s)
   against the GT V_x / Z, V_y / Z, V_z / Z at the same pixel. A component-wise median
   ratio shows whether only the forward (z) part is off.
3. Flow magnitude ratio |estimated| / |GT| per event.
4. A time series (every 1 s) of median estimated vs GT divergence.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm import mvsec as M  # noqa: E402
from ettcm.estimator import EttcmState, run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402


def odom_velocity(h5, k_time):
    """Camera velocity at time t from consecutive odometry poses (MVSEC compute_flow.py)."""
    g = h5["davis"]["left"]
    ts = np.asarray(g["odometry_ts"][:], dtype=np.float64)
    out = []
    for t in k_time:
        j = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
        H0 = np.asarray(g["odometry"][j - 1], dtype=np.float64)
        H1 = np.asarray(g["odometry"][j], dtype=np.float64)
        H01 = np.linalg.inv(H0) @ H1
        out.append(H01[:3, 3] / (ts[j] - ts[j - 1]))
    return np.array(out)


def main():
    from ettcm.tee import tee_output
    tee_output()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default="indoor_flying1")
    ap.add_argument("--model", default="translation", choices=["translation", "six_dof"])
    ap.add_argument("--start", type=float, default=20.0, help="seconds after the window start")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    args = ap.parse_args()

    root = args.data
    cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(args.seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(root, f"{args.seq}_data.hdf5"))
    flow = M.GTFlow(M.find_file(root, f"{args.seq}_gt_flow_dist.npz"), args.seq)
    div = M.DivergenceGT(M.find_file(root, f"{args.seq}_gt.hdf5"), flow, cam)
    w0, _ = flow.window("paper", evf.time(0))
    ta, tb = w0 + args.start, w0 + args.start + args.seconds

    # ---- 1. GT velocity, two ways
    ks = np.unique(div.frame_of(np.linspace(ta, tb, 12)))
    v_lsq = np.array([div.velocity(int(k))[0] if div.velocity(int(k)) else [np.nan] * 3 for k in ks])
    v_odo = odom_velocity(div.h5, div.dts[ks])
    print("1) GT camera velocity [m/s] at a few depth frames: least squares from GT flow | odometry")
    for k, a, b in zip(ks, v_lsq, v_odo):
        print(f"   frame {k:5d}  lsq ({a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f})   odom ({b[0]:+.3f} {b[1]:+.3f} {b[2]:+.3f})")
    ok = np.isfinite(v_lsq).all(1)
    print(f"   median |lsq| / |odom| = {np.median(np.linalg.norm(v_lsq[ok], axis=1) / np.linalg.norm(v_odo[ok], axis=1)):.3f}"
          "  (the odometry is unsmoothed; MVSEC's flow uses a +-10 frame average, so expect"
          " similar direction and size, not equality)")

    # ---- run the estimator (1 s warm-up)
    i0, i1 = evf.index_of_time(ta - 1.0), evf.index_of_time(tb)
    x, y, t, p = evf.read(i0, i1)
    keep, xr, yr = M.rectify(x, y, cam)
    x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
    ev = np.empty(len(t), EVENT_DTYPE)
    ev["t"] = np.round((t - t[0]) * 1e6).astype(np.int64)
    ev["x"], ev["y"], ev["p"] = x, y, p
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, w_thresh=0.01,
                    l_theta=args.l_theta, l_rho=1.0, state=EttcmState(cam.width, cam.height))
    sel = (t >= ta) & (out["rho"] > 0)
    xi = np.clip(np.round(xr).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr).astype(int), 0, cam.height - 1)
    idx = np.nonzero(sel)[0][::20]                       # every 20th event is plenty
    kd = div.frame_of(t[idx])

    est = out["rho"][idx, None] * out["theta"][idx, :3]  # rho * nu  [1/s]
    gt = np.full((len(idx), 3), np.nan)
    vcache = {}
    for k in np.unique(kd):
        if k not in vcache:
            vel = div.velocity(int(k))
            vcache[k] = vel[0] if vel else np.full(3, np.nan)
        Z = np.asarray(div.depth[int(k)], dtype=np.float64)
        m = kd == k
        z = Z[yi[idx[m]], xi[idx[m]]]
        gt[m] = vcache[k][None, :] / z[:, None]
    good = np.isfinite(gt).all(1) & (np.abs(gt) > 0).all(1)
    est, gt, tt = est[good], gt[good], t[idx][good]

    print("\n2) per component, estimate (rho*nu) vs GT (V/Z), events with valid depth:")
    for c, name in enumerate(["x (sideways)", "y (vertical)", "z (forward) = divergence"]):
        r = est[:, c] / gt[:, c]
        same = 100 * np.mean(np.sign(est[:, c]) == np.sign(gt[:, c]))
        big = np.abs(gt[:, c]) > np.percentile(np.abs(gt[:, c]), 50)   # ignore near-zero GT
        print(f"   {name:26s} median est/GT {np.median(r):+.3f}  (|GT| above median: {np.median(r[big]):+.3f})"
              f"   same sign {same:.1f} %   median |est| {np.median(np.abs(est[:, c])):.4f}"
              f"  |GT| {np.median(np.abs(gt[:, c])):.4f}")

    # ---- 3. flow magnitude
    dts = 1.0 / 31.4
    pred = np.stack([out["vx"][idx], out["vy"][idx]], 1)[good] * dts
    gu, gv = flow.displacement(tt[0], tt[0] + dts)
    print(f"\n3) median estimated flow speed {np.median(np.hypot(pred[:, 0], pred[:, 1])):.3f} px/frame;"
          f" GT at that time {np.median(np.hypot(gu, gv)[np.hypot(gu, gv) > 0]):.3f} px/frame")

    # ---- 4. time series
    print("\n4) every 1 s: median divergence estimate vs GT [1/s]")
    for a in np.arange(ta, tb, 1.0):
        m = (tt >= a) & (tt < a + 1.0)
        if m.sum() > 10:
            print(f"   t={a - w0:6.1f} s  est {np.median(est[m, 2]):+.4f}  GT {np.median(gt[m, 2]):+.4f}"
                  f"  ratio {np.median(est[m, 2] / gt[m, 2]):+.2f}")
    div.close()
    evf.close()


if __name__ == "__main__":
    main()
