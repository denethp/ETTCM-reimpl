"""Qualitative figures like the paper's Fig. 5/6: events, optical flow and TTC map.

MVSEC (with GT flow and GT TTC panels if the files are there):
    python scripts/figures.py --dataset mvsec --data ../datasets/MVSEC \\
        --seq indoor_flying1 --model translation --t 20 --out ../figures/if1_t20.png
Fig. 6 (one row per sequence; --t takes one time per sequence, or one for all):
    python scripts/figures.py --data ../datasets/MVSEC --model translation \\
        --seq indoor_flying1,indoor_flying2,indoor_flying3,outdoor_day1 --t 20,20,20,8 \\
        --out ../figures/fig6_translation.png
ECD slider_depth (Fig. 4: events, flow, depth; https://rpg.ifi.uzh.ch/datasets/davis/slider_depth.zip):
    python scripts/figures.py --dataset ecd --data ../datasets/ECD --seq slider_depth \\
        --model translation --t 1.5 --out ../figures/fig4_slider_depth.png
VL (no GT maps):
    python scripts/figures.py --dataset vl --data ../datasets/VL --seq 2D-1 \\
        --model translation -s 2 --l-theta 1000 --t 8 --out ../figures/vl_2d1.png

--t is the time (s) from the start of the evaluated window (MVSEC; outdoor_day1: 222.4 s
into the recording) or of the sequence (VL);
events in [t, t + window] are drawn, the estimator runs from t - warmup.
Colours: flow = hue for direction, brightness for speed; TTC on a log scale
as in the paper's colour bar (1 s magenta .. red .. yellow .. green .. 10 s cyan);
grey = moving away (TTC < 0).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.estimator import run_ettcm  # noqa: E402
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


def paper_scale(x):
    """The paper's TTC / depth colour bar (Figs. 4-6): x = 0 magenta -> red -> yellow ->
    green -> x = 1 cyan, i.e. hue 300 deg -> 540 (= 180) deg."""
    import matplotlib.colors as mc

    hsv = np.ones(x.shape + (3,))
    hsv[..., 0] = ((300.0 + 240.0 * x) / 360.0) % 1.0
    return mc.hsv_to_rgb(hsv)


def ttc_rgb(ttc, mask):
    rgb = np.zeros(ttc.shape + (3,))
    good = mask & np.isfinite(ttc) & (ttc > 0)
    x = np.clip(np.log10(np.where(good, ttc, 1.0)), 0, 1)        # 1 s .. 10 s
    rgb[good] = paper_scale(x[good])
    rgb[mask & ~good] = 0.4
    return rgb


def last_per_pixel(H, W, xi, yi, vals):
    out = np.full((H, W) + vals.shape[1:], np.nan)
    out[yi, xi] = vals                  # later events overwrite earlier ones
    return out


def run_mvsec(args, seq, t_rel):
    from ettcm import mvsec as M

    cam, _ = M.load_calib(M.find_file(args.data, f"{M.scene_of(seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(args.data, f"{seq}_data.hdf5"))
    flow = None
    try:
        flow = M.GTFlow(M.find_file(args.data, f"{seq}_gt_flow_dist.npz"), seq)
        t_ref = flow.window(evf.time(0))[0]
    except FileNotFoundError:
        t_ref = evf.time(0)
    ta, tb = t_ref + t_rel, t_ref + t_rel + args.window
    i0, i1 = evf.index_of_time(ta - args.warmup), evf.index_of_time(tb)
    x, y, t, p = evf.read(i0, i1)
    keep, xr, yr = M.rectify(x, y, cam)
    x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
    ev = np.empty(len(t), EVENT_DTYPE)
    ev["t"] = np.round((t - t[0]) * 1e6).astype(np.int64)
    ev["x"], ev["y"], ev["p"] = x, y, p
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, w_thresh=0.01,
                    l_theta=args.l_theta, l_rho=1.0,
                    state=M.initial_state(cam, MODEL_IDS[args.model], seq))
    sel = t >= ta
    xi = np.clip(np.round(xr[sel]).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr[sel]).astype(int), 0, cam.height - 1)
    gt_flow = gt_ttc = None
    if flow is not None:
        gu, gv = flow.displacement(ta, tb)
        gt_flow = (gu / args.window, gv / args.window)      # px/s
    try:
        try:
            odom = M.find_file(args.data, f"{seq}_odom.npz")
        except FileNotFoundError:
            odom = None
        div = M.DivergenceGT(M.find_file(args.data, f"{seq}_gt.hdf5"), flow, cam, odom)
        with np.errstate(divide="ignore"):
            gt_ttc = 1.0 / div.mu_map(int(div.frame_of(ta)))
    except (FileNotFoundError, TypeError, AttributeError):
        pass
    return cam.width, cam.height, xi, yi, {k: v[sel] for k, v in out.items()}, gt_flow, gt_ttc


def run_ecd(args, seq, t_rel):
    """Event Camera Dataset (Mueggler et al.) text format: events.txt (t x y p), calib.txt
    (fx fy cx cy k1 k2 p1 p2 k3), DAVIS240C 240 x 180."""
    from ettcm.calib import make_camera, rectify_events
    from ettcm.io_es import EVENT_DTYPE

    folder = os.path.join(args.data, seq)
    c = np.loadtxt(os.path.join(folder, "calib.txt")).ravel()
    cam = make_camera(fx=c[0], fy=c[1], cx=c[2], cy=c[3], dist=c[4:9], width=240, height=180)
    raw = np.loadtxt(os.path.join(folder, "events.txt"))
    t = raw[:, 0]
    ta = t[0] + t_rel
    sel_run = (t >= ta - args.warmup) & (t < ta + args.window)
    raw, t = raw[sel_run], t[sel_run]
    ev = np.empty(len(t), EVENT_DTYPE)
    ev["t"] = np.round((t - t[0]) * 1e6).astype(np.int64)
    ev["x"], ev["y"], ev["p"] = raw[:, 1], raw[:, 2], raw[:, 3]
    ev, xr, yr = rectify_events(ev, cam)
    t = t[0] + ev["t"] * 1e-6
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, l_theta=args.l_theta)
    sel = t >= ta
    xi = np.clip(np.round(xr[sel]).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr[sel]).astype(int), 0, cam.height - 1)
    return cam.width, cam.height, xi, yi, {k: v[sel] for k, v in out.items()}, None, None


def depth_rgb(rho, mask):
    """Up-to-scale depth 1/rho on the paper's bar (Fig. 4): near magenta .. far cyan,
    log scale between the 5th and 95th percentiles."""
    rgb = np.zeros(rho.shape + (3,))
    good = mask & np.isfinite(rho) & (rho > 0)
    if not good.any():
        return rgb
    z = np.log(1.0 / rho[good])
    lo, hi = np.percentile(z, [5, 95])
    rgb[good] = paper_scale(np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1))
    return rgb


def run_vl(args, seq, t_rel):
    from ettcm.calib import make_camera, rectify_events
    from ettcm.evaluate import read_vl_sequence

    cam = make_camera(os.path.join(args.data, "calib.txt"))
    ev, w, h, ts, gt = read_vl_sequence(os.path.join(args.data, seq))
    ev, xr, yr = rectify_events(ev, cam)
    t = ev["t"].astype(np.float64) * 1e-6
    ta = t[0] + t_rel
    sel_run = (t >= ta - args.warmup) & (t < ta + args.window)
    ev, xr, yr, t = ev[sel_run], xr[sel_run], yr[sel_run], t[sel_run]
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s, l_theta=args.l_theta)
    sel = t >= ta
    xi = np.clip(np.round(xr[sel]).astype(int), 0, cam.width - 1)
    yi = np.clip(np.round(yr[sel]).astype(int), 0, cam.height - 1)
    return cam.width, cam.height, xi, yi, {k: v[sel] for k, v in out.items()}, None, None


def make_panels(args, seq, t_rel):
    W, H, xi, yi, o, gt_flow, gt_ttc = {"mvsec": run_mvsec, "vl": run_vl, "ecd": run_ecd}[args.dataset](args, seq, t_rel)
    has = o["rho"] > 0
    counts = np.zeros((H, W))
    np.add.at(counts, (yi, xi), 1)
    mask = counts > 0
    est = last_per_pixel(H, W, xi[has], yi[has], np.stack([o["vx"][has], o["vy"][has]], 1))
    emask = np.isfinite(est[..., 0])
    ttc = last_per_pixel(H, W, xi[has], yi[has], o["ttc"][has])
    rho = last_per_pixel(H, W, xi[has], yi[has], o["rho"][has])

    panels = [("Accumulated events", np.clip(counts / max(np.percentile(counts[mask], 99), 1), 0, 1), "gray")]
    rgb, scale = flow_rgb(np.nan_to_num(est[..., 0]), np.nan_to_num(est[..., 1]), emask)
    if gt_flow is not None and not args.no_gt:
        grgb, _ = flow_rgb(gt_flow[0], gt_flow[1], mask & (np.hypot(*gt_flow) > 0), scale)
        panels.append(("Optical flow GT", grgb, None))
    panels.append(("Optical flow", rgb, None))
    if gt_ttc is not None and not args.no_gt:
        panels.append(("TTCM GT", ttc_rgb(gt_ttc, mask), None))
    if args.dataset == "ecd":
        panels.append(("Depth (near magenta .. far cyan)", depth_rgb(rho, emask), None))
    else:
        panels.append(("TTCM", ttc_rgb(ttc, emask), None))
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvsec", "vl", "ecd"], default="mvsec")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", required=True, help="one sequence, or a comma list (one row each)")
    ap.add_argument("--model", default="translation", choices=list(MODEL_IDS))
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--t", default="10", help="seconds; comma list = one per --seq")
    ap.add_argument("--window", type=float, default=0.03)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--no-gt", action="store_true", help="leave out the GT panels (Fig. 5)")
    ap.add_argument("--out", default="maps.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = args.seq.split(",")
    ts = [float(v) for v in str(args.t).split(",")]
    ts = ts * len(seqs) if len(ts) == 1 else ts
    if len(ts) != len(seqs):
        raise SystemExit("--t needs one value, or one per sequence")
    rows = []
    for seq, t_rel in zip(seqs, ts):
        print(f"  {seq} t = {t_rel} s")
        rows.append((seq, t_rel, make_panels(args, seq, t_rel)))
    ncol = max(len(r[2]) for r in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(3.2 * ncol, 2.6 * len(rows)), squeeze=False)
    for r, (seq, t_rel, panels) in enumerate(rows):
        for c in range(ncol):
            ax = axs[r, c]
            ax.axis("off")
            if c < len(panels):
                title, img, cmap = panels[c]
                ax.imshow(img, cmap=cmap)
                if r == 0 or len(rows) == 1:
                    ax.set_title(title, fontsize=9)
        axs[r, 0].text(-0.04, 0.5, f"{seq}\nt = {t_rel:g} s", transform=axs[r, 0].transAxes,
                       rotation=90, ha="right", va="center", fontsize=8)
    scale = ("depth up to scale" if args.dataset == "ecd" else "TTC 1 s (magenta) .. 10 s (cyan)")
    fig.suptitle(f"{args.model}, window {1000 * args.window:.0f} ms; {scale}",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
