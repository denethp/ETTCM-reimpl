"""Tables 4 and 5 of the paper on MVSEC: optical flow AEE, Out, AREE, AAE.

Expects the MVSEC files somewhere under --data (sub-folders are fine):
    <seq>_data.hdf5, <seq>_gt_flow_dist.npz, indoor_flying_calib.zip, outdoor_day_calib.zip
for seq in indoor_flying1, indoor_flying2, indoor_flying3, outdoor_day1.

    python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq all --model all
    python scripts/tables45_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
    python scripts/tables45_mvsec.py ... --max-seconds 5        # quick try on the first 5 s

Settings (paper, Table 1 + Sec. 4.1): s = 3, w_thresh = 0.01, l_theta = 100, l_rho = 1.
Protocol (EV-FlowNet): the whole indoor_flying sequences, outdoor_day1 from 222.4 to 240.4 s,
intervals between consecutive grayscale frames, outdoor_day1 rows < 190 only. Each event with
an estimate gives m_i = dt_frame * v_i, compared with the GT displacement at its rectified pixel
(Eqs. 13-15, N = number of such events).
For outdoor_day1 the estimator starts at the beginning of the recording (warm start) and only
the window is scored; starting cold at 222.4 s can leave Driving on the wrong sign of nu_z.
Table 3 (divergence) is computed by table3_mvsec.py.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm import mvsec as M  # noqa: E402
from ettcm.estimator import EttcmState, run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402

SEQS = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
MODELS = ["driving", "translation", "six_dof"]


def evaluate(args, seq, model):
    root = args.data
    cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(root, f"{seq}_data.hdf5"))
    flow = M.GTFlow(M.find_file(root, f"{seq}_gt_flow_dist.npz"), seq)
    rows = M.OUTDOOR_MAX_ROW if seq.startswith("outdoor") else None
    H, W = cam.height, cam.width

    # evaluation intervals: consecutive grayscale frames inside the evaluated window
    t_rec0 = evf.time(0)
    t_start, t_end = flow.window(t_rec0)
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    edges = evf.gray_ts[(evf.gray_ts > t_start) & (evf.gray_ts < t_end)]
    if len(edges) < 2:
        raise RuntimeError(f"{seq}: fewer than 2 grayscale frames in the evaluation window")

    # warm start: outdoor_day1 from the start of the recording, indoor sequences are whole
    i0 = 0 if (seq == "outdoor_day1" and not args.cold) else evf.index_of_time(edges[0])
    i1 = evf.index_of_time(edges[-1])
    t0 = evf.time(i0)
    state = EttcmState(W, H)
    mid = MODEL_IDS[model]

    metrics = M.FlowMetrics()
    pending = {}          # interval k -> (list of pred, list of gt)
    disp_cache = {}
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
        preds, gts = pending.pop(k)
        if preds:
            metrics.add(np.concatenate(preds), np.concatenate(gts))

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

        k_ev = np.searchsorted(edges, t, side="right") - 1
        inwin = (k_ev >= 0) & (k_ev < len(edges) - 1)
        use = inwin & (out["rho"] > 0)
        n_ev += int(inwin.sum())
        n_est += int(use.sum())
        xi = np.clip(np.round(xr).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(yr).astype(np.int64), 0, H - 1)
        for k in np.unique(k_ev[use]):
            sel = use & (k_ev == k)
            gu, gv, ok = gt_disp(k)
            pred = np.stack([out["vx"][sel], out["vy"][sel]], axis=1) * (edges[k + 1] - edges[k])
            px, py = xi[sel], yi[sel]
            g = ok[py, px]
            preds, gts = pending.setdefault(int(k), ([], []))
            preds.append(pred[g])
            gts.append(np.stack([gu[py[g], px[g]], gv[py[g], px[g]]], axis=1))
        if use.any():
            for k in [k for k in pending if k < k_ev[use].max()]:
                flush(k)
        print(f"\r  {seq} {model}: {100 * (b - i0) / max(i1 - i0, 1):5.1f} %", end="", flush=True)
    for k in sorted(pending):
        flush(k)
    print()
    evf.close()

    res = {"seq": seq, "model": model, "s": args.s, "l_theta": args.l_theta,
           "seconds": float(edges[-1] - edges[0]), "events": n_ev,
           "coverage": 100.0 * n_est / max(n_ev, 1),
           "us_per_event": 1e6 * t_run / max(i1 - i0, 1)}
    res.update(metrics.result())
    return res


def report(r):
    seq, model = r["seq"], r["model"]
    p4 = M.PAPER_TABLE4.get(model, {}).get(seq)
    p5 = M.PAPER_TABLE5.get(model, {}).get(seq)
    f = lambda v: f"{v:8.2f}"  # noqa: E731
    print(f"  evaluated {r['seconds']:.1f} s, {r['events']} events, {r['coverage']:.1f} % with an "
          f"estimate, {r['us_per_event']:.2f} us/event, {r['frames']} intervals")
    print(f"  {'':10s} {'AEE':>8s} {'Out%':>8s} {'AREE%':>8s} {'AAE':>8s}")
    print(f"  {'ours':10s} {f(r['AEE'])} {f(r['Out'])} {f(r['AREE'])} {f(r['AAE'])}")
    pa = (f(p4[0]) + " " + f(p4[1])) if p4 else f"{'-':>8s} {'-':>8s}"
    pb = (f(p5[0]) + " " + f(p5[1])) if p5 else f"{'-':>8s} {'-':>8s}"
    print(f"  {'paper':10s} {pa} {pb}")


CSV_KEYS = ["seq", "model", "s", "l_theta", "seconds", "events", "coverage", "us_per_event",
            "AEE", "Out", "AREE", "AAE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing the MVSEC files")
    ap.add_argument("--seq", default="indoor_flying1", choices=SEQS + ["all"])
    ap.add_argument("--model", default="translation", choices=MODELS + ["all"])
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--cold", action="store_true",
                    help="outdoor_day1: start the estimator at 222.4 s instead of the recording start")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="only the first N seconds (tests)")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="events per chunk")
    ap.add_argument("--csv", default=None, help="append results to this CSV file")
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
