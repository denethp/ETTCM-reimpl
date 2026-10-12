"""Tables 3-5 of the paper on MVSEC: optical flow (AEE, Out, AREE, AAE) and divergence.

Expects the MVSEC files somewhere under --data (sub-folders are fine):
    <seq>_data.hdf5, <seq>_gt_flow_dist.npz, <seq>_gt.hdf5 (only for --table3),
    indoor_flying_calib.zip, outdoor_day_calib.zip
for seq in indoor_flying1, indoor_flying2, indoor_flying3, outdoor_day1.

    python scripts/run_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
    python scripts/run_mvsec.py --data ../datasets/MVSEC --seq all --model all --table3 \\
        --csv ../results_python_mvsec.csv
    python scripts/run_mvsec.py ... --max-seconds 5        # quick try on the first 5 s

Settings (paper, Table 1 + Sec. 4.1): s = 3, w_thresh = 0.01, l_theta = 100, l_rho = 1.
Each run prints our numbers next to the paper's (Tables 3, 4, 5).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.estimator import EttcmState, run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402
from ettcm import mvsec as M  # noqa: E402

SEQS = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
MODELS = ["driving", "translation", "six_dof"]


class Interval:
    """Per-event and per-pixel accumulators of one grayscale-frame interval."""

    def __init__(self, H, W):
        self.pred, self.gt = [], []
        self.sum = np.zeros((H, W, 2))
        self.cnt = np.zeros((H, W))


def evaluate(args, seq, model):
    root = args.data
    cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(root, f"{seq}_data.hdf5"))
    flow = M.GTFlow(M.find_file(root, f"{seq}_gt_flow_dist.npz"), seq)
    rows = M.OUTDOOR_MAX_ROW if seq.startswith("outdoor") else None
    H, W = cam.height, cam.width

    # evaluation intervals: grayscale frames inside the evaluated window, every dt frames
    t_start, t_end = flow.window(args.protocol, evf.time(0))
    t_start += args.start_seconds
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    gray = evf.gray_ts[(evf.gray_ts > t_start) & (evf.gray_ts < t_end)]
    edges = gray[::args.dt]
    if len(edges) < 2:
        raise RuntimeError(f"{seq}: fewer than 2 grayscale frames in the evaluation window")

    div = None
    if args.table3:
        try:
            div = M.DivergenceGT(M.find_file(root, f"{seq}_gt.hdf5"), flow, cam)
        except FileNotFoundError as e:
            print("  (Table 3 skipped:", e, ")")

    i0 = evf.index_of_time(edges[0] - args.warmup)
    i1 = evf.index_of_time(edges[-1])
    t0 = evf.time(i0)
    state = EttcmState(W, H)
    mid = MODEL_IDS[model]

    m_event, m_pixel = M.FlowMetrics(), M.FlowMetrics()
    pending = {}
    disp_cache = {}
    srcs = M.DivergenceGT.VEL_SOURCES
    # per GT velocity source: n, sum rel. error, n tried, n same sign, sum mu_hat, sum mu_gt
    d3 = {src: np.zeros(6) for src in srcs}
    d3_sample = {src: [] for src in srcs}   # every 50th (mu_hat, mu_gt) pair, for medians
    gt_speed = []           # median GT flow magnitude per interval (px/frame)
    n_ev = n_est = 0
    t_run = 0.0

    def gt_disp(k):
        if k not in disp_cache:
            if len(disp_cache) > 4:
                disp_cache.clear()
            gu, gv = flow.displacement(edges[k], edges[k + 1])
            disp_cache[k] = (gu, gv, M.gt_mask(gu, gv, rows))
        return disp_cache[k]

    def flush(k):
        it = pending.pop(k)
        gu, gv, ok = gt_disp(k)
        if ok.any():
            gt_speed.append(np.median(np.hypot(gu[ok], gv[ok])))
        if it.pred:
            m_event.add(np.concatenate(it.pred), np.concatenate(it.gt))
        sel = (it.cnt > 0) & ok
        if sel.any():
            mean = it.sum[sel] / it.cnt[sel][:, None]
            m_pixel.add(mean, np.stack([gu[sel], gv[sel]], axis=1))

    for a in range(i0, i1, args.chunk):
        b = min(a + args.chunk, i1)
        x, y, t, p = evf.read(a, b)
        keep, xr, yr = M.rectify(x, y, cam)
        x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
        ev = np.empty(len(t), EVENT_DTYPE)
        ev["t"] = np.round((t - t0) * 1e6).astype(np.int64)
        ev["x"], ev["y"], ev["p"] = x, y, p
        tic = time.perf_counter()
        out = run_ettcm(ev, xr, yr, cam, mid, s=args.s, w_thresh=0.01,
                        l_theta=args.l_theta, l_rho=1.0, state=state)
        t_run += time.perf_counter() - tic

        # ---- optical flow (Tables 4, 5)
        k_ev = np.searchsorted(edges, t, side="right") - 1
        inwin = (k_ev >= 0) & (k_ev < len(edges) - 1)
        est = out["rho"] > 0
        n_ev += int(inwin.sum())
        n_est += int((inwin & est).sum())
        if args.loc == "rect":
            xi = np.clip(np.round(xr).astype(np.int64), 0, W - 1)
            yi = np.clip(np.round(yr).astype(np.int64), 0, H - 1)
        else:
            xi, yi = x, y
        use = inwin & est
        for k in np.unique(k_ev[use]):
            sel = use & (k_ev == k)
            gu, gv, ok = gt_disp(k)
            dtf = edges[k + 1] - edges[k]
            pred = np.stack([out["vx"][sel], out["vy"][sel]], axis=1) * dtf
            px, py = xi[sel], yi[sel]
            it = pending.setdefault(int(k), Interval(H, W))
            g = ok[py, px]
            it.pred.append(pred[g])
            it.gt.append(np.stack([gu[py[g], px[g]], gv[py[g], px[g]]], axis=1))
            np.add.at(it.sum, (py, px), pred)
            np.add.at(it.cnt, (py, px), 1)
        if len(k_ev[use]):
            for k in [k for k in pending if k < k_ev[use].max()]:
                flush(k)

        # ---- divergence (Table 3)
        if div is not None:
            ii = np.nonzero(use)[0]
            kd = div.frame_of(t[ii])
            for k in np.unique(kd):
                s = ii[kd == k]
                for src in srcs:
                    mu_gt = div.mu_map(int(k), src)[yi[s], xi[s]]
                    g = np.isfinite(mu_gt) & (np.abs(mu_gt) > 1e-9)
                    mh, mg = out["mu"][s][g], mu_gt[g]
                    d3[src] += [g.sum(), (np.abs(mh - mg) / np.abs(mg)).sum(), len(s),
                                (np.sign(mh) == np.sign(mg)).sum(), mh.sum(), mg.sum()]
                    d3_sample[src].append(np.stack([mh[::50], mg[::50]], axis=1))

        done = (b - i0) / max(i1 - i0, 1)
        print(f"\r  {seq} {model}: {100 * done:5.1f} %", end="", flush=True)
    for k in sorted(pending):
        flush(k)
    print()
    evf.close()
    if div is not None:
        div.close()

    res = {"seq": seq, "model": model, "s": args.s, "l_theta": args.l_theta,
           "dt": args.dt, "loc": args.loc, "protocol": args.protocol,
           "seconds": float(edges[-1] - edges[0]), "events": n_ev,
           "coverage": 100.0 * n_est / max(n_ev, 1),
           "us_per_event": 1e6 * t_run / max(i1 - i0, 1)}
    res.update({k: v for k, v in m_event.result().items()})
    res.update({"pix_" + k: v for k, v in m_pixel.result().items()})
    res["gt_speed_median"] = float(np.median(gt_speed)) if gt_speed else 0.0
    if div is not None:
        for src in srcs:
            d, n3 = d3[src], max(d3[src][0], 1)
            pre = "div_" if src == args.gt_vel else f"div_{src}_"
            res[pre + "AREE"] = 100 * d[1] / n3
            res[pre + "n"] = int(d[0])
            res[pre + "same_sign"] = 100 * d[3] / n3
            res[pre + "mean_est"] = d[4] / n3
            res[pre + "mean_gt"] = d[5] / n3
            if d3_sample[src]:
                smp = np.concatenate(d3_sample[src])
                res[pre + "median_rel"] = 100 * float(np.median(np.abs(smp[:, 0] - smp[:, 1]) / np.abs(smp[:, 1])))
                res[pre + "median_ratio"] = float(np.median(smp[:, 0] / smp[:, 1]))
        res["gt_vel"] = args.gt_vel
    return res


def report(r):
    seq, model = r["seq"], r["model"]
    p4 = M.PAPER_TABLE4.get(model, {}).get(seq)
    p5 = M.PAPER_TABLE5.get(model, {}).get(seq)
    p3 = M.PAPER_TABLE3.get(model, {}).get(seq)
    f = lambda v: f"{v:8.2f}"  # noqa: E731
    print(f"  evaluated {r['seconds']:.1f} s, {r['events']} events, {r['coverage']:.1f} % with an "
          f"estimate, {r['us_per_event']:.2f} us/event, {r['frames']} intervals (dt={r['dt']}), "
          f"median GT flow {r['gt_speed_median']:.2f} px/frame")
    print(f"  {'':22s} {'AEE':>8s} {'Out%':>8s} {'AREE%':>8s} {'AAE':>8s}")
    print(f"  {'ours, per event':22s} {f(r['AEE'])} {f(r['Out'])} {f(r['AREE'])} {f(r['AAE'])}")
    print(f"  {'ours, per pixel':22s} {f(r['pix_AEE'])} {f(r['pix_Out'])} {f(r['pix_AREE'])} {f(r['pix_AAE'])}")
    print(f"  {'ours, per-frame mean':22s} {f(r['AEE_frame'])} {f(r['Out_frame'])} {f(r['AREE_frame'])} {f(r['AAE_frame'])}")
    pa = (f(p4[0]) + f(p4[1])) if p4 else "       -       -"
    pb = (f(p5[0]) + f(p5[1])) if p5 else "       -       -"
    print(f"  {'paper (Tab. 4 / 5)':22s} {pa} {pb}")
    if "div_AREE" in r:
        print(f"  divergence AREE (Tab. 3), paper {p3 if p3 is not None else '-'}; ours with GT camera velocity from:")
        for src in M.DivergenceGT.VEL_SOURCES:
            pre = "div_" if src == r["gt_vel"] else f"div_{src}_"
            if pre + "AREE" not in r or r[pre + "n"] == 0:
                print(f"    {src:12s} not available")
                continue
            print(f"    {src:12s}{' (default)' if src == r['gt_vel'] else '          '} AREE {r[pre + 'AREE']:8.2f} %"
                  f"  median rel. err {r.get(pre + 'median_rel', float('nan')):7.1f} %"
                  f"  median est/GT {r.get(pre + 'median_ratio', float('nan')):6.3f}"
                  f"  same sign {r[pre + 'same_sign']:5.1f} %  ({r[pre + 'n']} events)")


CSV_KEYS = ["seq", "model", "s", "l_theta", "dt", "loc", "protocol", "seconds", "events", "coverage",
            "us_per_event", "AEE", "Out", "AREE", "AAE", "pix_AEE", "pix_Out", "pix_AREE",
            "pix_AAE", "AEE_frame", "Out_frame", "AREE_frame", "AAE_frame", "gt_speed_median",
            "gt_vel", "div_AREE", "div_same_sign", "div_median_rel", "div_median_ratio",
            "div_odom_mvsec_AREE", "div_odom_smooth_AREE", "div_odom_AREE", "div_flow_AREE"]


def main():
    from ettcm.tee import tee_output
    tee_output()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing the MVSEC files")
    ap.add_argument("--seq", default="indoor_flying1", choices=SEQS + ["all"])
    ap.add_argument("--model", default="translation", choices=MODELS + ["all"])
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--dt", type=int, default=1, help="grayscale frames per interval (1 or 4)")
    ap.add_argument("--loc", default="rect", choices=["rect", "raw"],
                    help="pixel used to read the GT: rectified (default) or raw event position")
    ap.add_argument("--protocol", default="paper", choices=["paper", "secrets"],
                    help="paper: whole indoor sequences + outdoor_day1 222.4-240.4 s (~266 s, as "
                         "in the paper); secrets: tub-rip valid GT frames (~463 s)")
    ap.add_argument("--table3", action="store_true", help="also divergence AREE (needs _gt.hdf5)")
    ap.add_argument("--gt-vel", default="odom_mvsec", choices=["odom_mvsec", "odom_smooth", "odom", "flow"],
                    help="camera velocity for the Table 3 GT reported in the CSV (all three are printed)")
    ap.add_argument("--warmup", type=float, default=0.0, help="seconds of events before the window")
    ap.add_argument("--start-seconds", type=float, default=0.0,
                    help="skip the first N seconds of the window (quick tests after take-off)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="only N seconds")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="events per chunk")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    seqs = SEQS if args.seq == "all" else [args.seq]
    models = MODELS if args.model == "all" else [args.model]
    for seq in seqs:
        for model in models:
            print(f"== {seq}  model={model}  s={args.s}  l_theta={args.l_theta}")
            r = evaluate(args, seq, model)
            report(r)
            if args.csv:
                new = not os.path.exists(args.csv)
                with open(args.csv, "a") as fcsv:
                    if new:
                        fcsv.write(",".join(CSV_KEYS) + "\n")
                    fcsv.write(",".join(str(r.get(k, "")) for k in CSV_KEYS) + "\n")


if __name__ == "__main__":
    main()
