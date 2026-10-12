"""Qualitative figures like the paper's Fig. 5/6: events, optical flow and TTC map.

MVSEC (with GT flow and GT TTC panels if the files are there):
    python scripts/figures.py --dataset mvsec --data ../datasets/MVSEC \\
        --seq indoor_flying1 --model translation --t 20 --out ../figures/if1_t20.png
VL (no GT maps):
    python scripts/figures.py --dataset vl --data ../datasets/VL --seq 2D-1 \\
        --model translation -s 2 --l-theta 1000 --t 8 --out ../figures/vl_2d1.png

--t is the time (s) from the start of the evaluated window (MVSEC; outdoor_day1: 222.4 s
into the recording) or of the sequence (VL);
events in [t, t + window] are drawn, the estimator runs from t - warmup.
Colours: flow = hue for direction, brightness for speed; TTC = 1 s (red) .. 10 s (violet),
as in the paper's colour bar; grey = no estimate or moving away.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.estimator import EttcmState, run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402


def flow_rgb(u, v, mask, scale=None):
    import matplotlib.colors as mc

    mag = np.hypot(u, v)
    if scale is None:
        scale = np.percentile(mag[mask], 95) if mask.any() else 1.0
    hsv = np.zeros(u.shape + (3,))
    hsv[..., 0] = (np.arctan2(v, u) / (2 * np.pi)) % 1.0
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / max(scale, 1e-9), 0, 1)
    rgb = mc.hsv_to_rgb(hsv)
    rgb[~mask] = 0
    return rgb, scale


def ttc_rgb(ttc, mask):
    import matplotlib.pyplot as plt

    rgb = np.zeros(ttc.shape + (3,))
    good = mask & np.isfinite(ttc) & (ttc > 0)
    x = np.clip(np.log10(np.where(good, ttc, 1.0)), 0, 1)        # 1 s .. 10 s
    rgb[good] = plt.get_cmap("rainbow_r")(x[good])[:, :3]
    rgb[mask & ~good] = 0.4
    return rgb


def last_per_pixel(H, W, xi, yi, vals):
    out = np.full((H, W) + vals.shape[1:], np.nan)
    out[yi, xi] = vals                  # later events overwrite earlier ones
    return out


def run_mvsec(args):
    from ettcm import mvsec as M

    cam, _ = M.load_calib(M.find_file(args.data, f"{M.scene_of(args.seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(args.data, f"{args.seq}_data.hdf5"))
    flow = None
    try:
        flow = M.GTFlow(M.find_file(args.data, f"{args.seq}_gt_flow_dist.npz"), args.seq)
        t_ref = flow.window(evf.time(0))[0]
    except FileNotFoundError:
        t_ref = evf.time(0)
    ta, tb = t_ref + args.t, t_ref + args.t + args.window
    i0, i1 = evf.index_of_time(ta - args.warmup), evf.index_of_time(tb)
    x, y, t, p = evf.read(i0, i1)
    keep, xr, yr = M.rectify(x, y, cam)
    x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
    ev = np.empty(len(t), EVENT_DTYPE)
    ev["t"] = np.round((t - t[0]) * 1e6).astype(np.int64)
    ev["x"], ev["y"], ev["p"] = x, y, p
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, w_thresh=0.01,
                    l_theta=args.l_theta, l_rho=1.0, state=EttcmState(cam.width, cam.height))
    sel = t >= ta
    xi = np.clip(np.round(xr[sel]).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr[sel]).astype(int), 0, cam.height - 1)
    gt_flow = gt_ttc = None
    if flow is not None:
        gu, gv = flow.displacement(ta, tb)
        gt_flow = (gu / args.window, gv / args.window)      # px/s
    try:
        try:
            odom = M.find_file(args.data, f"{args.seq}_odom.npz")
        except FileNotFoundError:
            odom = None
        div = M.DivergenceGT(M.find_file(args.data, f"{args.seq}_gt.hdf5"), flow, cam, odom)
        with np.errstate(divide="ignore"):
            gt_ttc = 1.0 / div.mu_map(int(div.frame_of(ta)))
    except (FileNotFoundError, TypeError, AttributeError):
        pass
    return cam.width, cam.height, xi, yi, {k: v[sel] for k, v in out.items()}, gt_flow, gt_ttc


def run_vl(args):
    from ettcm.calib import make_camera, rectify_events
    from ettcm.evaluate import read_vl_sequence

    cam = make_camera(os.path.join(args.data, "calib.txt"))
    ev, w, h, ts, gt = read_vl_sequence(os.path.join(args.data, args.seq))
    ev, xr, yr = rectify_events(ev, cam)
    t = ev["t"].astype(np.float64) * 1e-6
    ta = t[0] + args.t
    sel_run = (t >= ta - args.warmup) & (t < ta + args.window)
    ev, xr, yr, t = ev[sel_run], xr[sel_run], yr[sel_run], t[sel_run]
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, l_theta=args.l_theta)
    sel = t >= ta
    xi = np.clip(np.round(xr[sel]).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr[sel]).astype(int), 0, cam.height - 1)
    return cam.width, cam.height, xi, yi, {k: v[sel] for k, v in out.items()}, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvsec", "vl"], default="mvsec")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="translation", choices=list(MODEL_IDS))
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--t", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--out", default="maps.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H, xi, yi, o, gt_flow, gt_ttc = (run_mvsec if args.dataset == "mvsec" else run_vl)(args)
    has = o["rho"] > 0
    counts = np.zeros((H, W))
    np.add.at(counts, (yi, xi), 1)
    mask = counts > 0
    est = last_per_pixel(H, W, xi[has], yi[has], np.stack([o["vx"][has], o["vy"][has]], 1))
    emask = np.isfinite(est[..., 0])
    ttc = last_per_pixel(H, W, xi[has], yi[has], o["ttc"][has])

    panels = [("Accumulated events", np.clip(counts / max(np.percentile(counts[mask], 99), 1), 0, 1), "gray")]
    rgb, scale = flow_rgb(np.nan_to_num(est[..., 0]), np.nan_to_num(est[..., 1]), emask)
    if gt_flow is not None:
        grgb, _ = flow_rgb(gt_flow[0], gt_flow[1], mask & (np.hypot(*gt_flow) > 0), scale)
        panels.append(("Optical flow GT", grgb, None))
    panels.append((f"Optical flow ({args.model})", rgb, None))
    if gt_ttc is not None:
        panels.append(("TTCM GT", ttc_rgb(gt_ttc, mask), None))
    panels.append((f"TTCM ({args.model})", ttc_rgb(ttc, emask), None))

    fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8))
    for ax, (title, img, cmap) in zip(axs, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{args.seq}  t = {args.t:.2f} s  window {1000 * args.window:.0f} ms", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
