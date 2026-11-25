"""NOVELTY (not in the paper): per-event confidence + fairer TTC metrics, on MVSEC and VL.

    python scripts/novelty_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
    python scripts/novelty_mvsec.py --data ../datasets/MVSEC          # Table 3 columns x 3 models
    python scripts/novelty_mvsec.py --dataset vl --data ../datasets/VL --model translation

1. Confidence. run_ettcm(..., confidence=True) returns sigma, the standard deviation of
   log|mu| propagated from the Gauss-Newton information matrix (Psi + L) of each event, and
   res2, the weighted mean squared residual of its neighbours. The uncertainty used here is
   u = sigma * sqrt(res2) (covariance scaled by the residual variance, as in standard
   least-squares). It needs no ground truth, so it can be used online to drop unreliable events;
   the rows "most confident X %" keep the X % of events with the smallest u.
2. Metrics. The paper's AREE |mu_est - mu_gt| / |mu_gt| explodes when the camera barely moves
   forward (mu_gt ~ 0), which is why our Table 3 numbers are 2-3x the paper's. We add:
     sign acc.   % of events whose estimate has the right sign (approaching vs receding)
     LTE         log-TTC error |ln(tau_est / tau_gt)| = |ln(mu_est / mu_gt)| (same-sign events):
                 symmetric, scale-free, grows only logarithmically when mu_gt ~ 0
     delta<1.25, delta<2   % of ALL events with the right sign and within a factor 1.25 / 2 of
                 the GT TTC (the standard accuracy metric of depth estimation)
     danger zone the same metrics only on events whose GT TTC is in (0, --danger] s, i.e. the
                 obstacles that matter for collision avoidance
   and report each with no filter, keeping only the most confident events (no GT used), and,
   for comparison, with the GT-based exclusion |mu_gt| >= 0.2 /s (needs GT, not usable online).

VL (--dataset vl): the 7 planar 2D sequences, where the camera descends onto flat ground, so
every pixel has the same divergence: per-event GT = the sequence's GT divergence interpolated
in time (3D is left out: its GT is only a global value). Settings as Table 2 (l_theta = 1000).

Data, settings and GT exactly as table3_mvsec.py (s = 3, l_theta = 100, GT velocity from
<seq>_odom.npz; outdoor_day1 scored on 222.4-240.4 s with a warm start). --cache DIR keeps the
per-event (mu_est, mu_gt, sigma) arrays so the metrics can be recomputed without re-running.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm import mvsec as M  # noqa: E402
from ettcm.estimator import run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402

SEQS = ["indoor_flying1", "indoor_flying2", "outdoor_day1"]
VL_SEQS = [f"2D-{i}" for i in range(1, 8)]
MODELS = ["driving", "translation", "six_dof"]


def collect_vl(args, seq, model):
    """VL planar sequence: return float32 mu_est, mu_gt, u for events inside the GT time range."""
    from ettcm.calib import make_camera, rectify_events
    from ettcm.evaluate import read_vl_sequence
    ev, _, _, ts, dg = read_vl_sequence(os.path.join(args.data, seq))
    cam = make_camera(os.path.join(args.data, "calib.txt"))
    ev, xr, yr = rectify_events(ev, cam)
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=args.s, w_thresh=0.01,
                    l_theta=args.l_theta, l_rho=1.0, confidence=True)
    t = ev["t"].astype(np.float64)
    m = (out["rho"] > 0) & (t >= ts[0]) & (t <= ts[-1])
    g = np.interp(t[m], ts.astype(np.float64), dg)
    u = out["sigma"][m] * np.sqrt(out["res2"][m])
    return out["mu"][m].astype(np.float32), g.astype(np.float32), u.astype(np.float32)


def collect(args, seq, model):
    """Run the estimator with confidence; return float32 arrays mu_est, mu_gt, u."""
    root = args.data
    cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(root, f"{seq}_data.hdf5"))
    flow = M.GTFlow(M.find_file(root, f"{seq}_gt_flow_dist.npz"), seq)
    try:
        odom = M.find_file(root, f"{seq}_odom.npz")
    except FileNotFoundError:
        odom = None
    div = M.DivergenceGT(M.find_file(root, f"{seq}_gt.hdf5"), flow, cam, odom)
    src = div.default_source
    t_start, t_end = flow.window(evf.time(0))
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    i0 = evf.index_of_time(evf.time(0)) if seq == "outdoor_day1" else evf.index_of_time(t_start)
    i1 = evf.index_of_time(t_end)
    t0 = evf.time(i0)
    state = M.initial_state(cam, MODEL_IDS[model], seq)
    H, W = cam.height, cam.width
    out_e, out_g, out_s = [], [], []
    tic = time.perf_counter()
    for a in range(i0, i1, args.chunk):
        b = min(a + args.chunk, i1)
        x, y, t, p = evf.read(a, b)
        keep, xr, yr = M.rectify(x, y, cam)
        x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
        ev = np.empty(len(t), EVENT_DTYPE)
        ev["t"] = np.round((t - t0) * 1e6).astype(np.int64)
        ev["x"], ev["y"], ev["p"] = x, y, p
        out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=args.s, w_thresh=0.01,
                        l_theta=args.l_theta, l_rho=1.0, state=state, confidence=True)
        xi = np.clip(np.round(xr).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(yr).astype(np.int64), 0, H - 1)
        ii = np.nonzero((out["rho"] > 0) & (t >= t_start))[0]
        kd = div.frame_of(t[ii])
        for k in np.unique(kd):
            sl = ii[kd == k]
            g = div.mu_map(int(k), src)[yi[sl], xi[sl]]
            ok = np.isfinite(g) & (np.abs(g) > 1e-9)
            out_e.append(out["mu"][sl][ok].astype(np.float32))
            out_g.append(g[ok].astype(np.float32))
            u = out["sigma"][sl] * np.sqrt(out["res2"][sl])
            out_s.append(u[ok].astype(np.float32))
        print(f"\r    {seq} {model}: {100 * (b - i0) / max(i1 - i0, 1):5.1f} %", end="", flush=True)
    print(f"  ({time.perf_counter() - tic:.0f} s)")
    div.close()
    evf.close()
    return np.concatenate(out_e), np.concatenate(out_g), np.concatenate(out_s)


def metrics(e, g, sel=None):
    """All metrics on the events in sel (bool mask). Percentages."""
    if sel is not None:
        e, g = e[sel], g[sel]
    n = len(e)
    if n == 0:
        return dict(n=0, aree=np.nan, med=np.nan, sign=np.nan, lte=np.nan, lte_med=np.nan,
                    d125=np.nan, d2=np.nan)
    e = e.astype(np.float64)
    g = g.astype(np.float64)
    rel = np.abs(e - g) / np.abs(g)
    same = np.sign(e) == np.sign(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(e[same] / g[same]))
    ratio_ok = np.zeros(n, bool)
    ratio_ok[same] = True
    d125 = np.zeros(n, bool)
    d2 = np.zeros(n, bool)
    d125[same] = lr < np.log(1.25)
    d2[same] = lr < np.log(2.0)
    return dict(n=n, aree=100 * rel.mean(), med=100 * np.median(rel), sign=100 * same.mean(),
                lte=lr.mean() if lr.size else np.nan,
                lte_med=np.median(lr) if lr.size else np.nan,
                d125=100 * d125.mean(), d2=100 * d2.mean())


def spearman(a, b, n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a), min(n, len(a)), replace=False)
    ra = np.argsort(np.argsort(a[idx]))
    rb = np.argsort(np.argsort(b[idx]))
    return float(np.corrcoef(ra, rb)[0, 1])


def report(seq, model, e, g, sg, args):
    fin = np.isfinite(sg)
    print(f"\n== {seq} / {model}: {len(e):,} scored events (u finite for {100 * fin.mean():.1f} %)")
    paper = M.PAPER_TABLE3.get(model, {}).get(seq, float("nan"))
    if np.isfinite(paper):
        print(f"   paper's Table 3 AREE: {paper:.2f} %")
    # how well sigma predicts the error
    same = (np.sign(e) == np.sign(g)) & fin
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.abs(np.log(e[same].astype(np.float64) / g[same]))
    print(f"   rank correlation of u with |log TTC error|: {spearman(sg[same], lr):+.3f}   "
          f"with 1/|mu_gt| (near-zero GT): {spearman(sg[fin], -np.abs(g[fin])):+.3f}   (+1 = perfect)")
    q = np.nanpercentile(sg[fin], [25, 50, 75])

    danger = (g > 0) & (1.0 / np.where(g > 0, g, 1) <= args.danger)
    rows = [("no filter (all events)", None)]
    rows += [(f"confidence: most confident {p} %", fin & (sg <= qq))
             for p, qq in zip((75, 50, 25), q[::-1])]
    rows += [("GT-based: |mu_gt| >= 0.2 /s  (needs GT)", np.abs(g) >= 0.2)]
    hdr = (f"   {'':44s}{'kept':>7s}{'AREE':>9s}{'median':>8s}{'sign':>7s}{'LTE':>7s}"
           f"{'LTEmed':>8s}{'d<1.25':>8s}{'d<2':>7s}")
    for zone, zmask in (("all events", None), (f"danger zone: GT TTC in (0, {args.danger:g}] s "
                                               f"({100 * danger.mean():.1f} % of events)", danger)):
        print(f"   -- {zone}")
        print(hdr)
        base_n = len(e) if zmask is None else int(zmask.sum())
        for name, sel in rows:
            m = sel if zmask is None else (zmask if sel is None else (zmask & sel))
            r = metrics(e, g, m)
            kept = 100 * r["n"] / max(base_n, 1)
            print(f"   {name:44s}{kept:6.1f}%{r['aree']:9.2f}{r['med']:8.2f}{r['sign']:6.1f}%"
                  f"{r['lte']:7.3f}{r['lte_med']:8.3f}{r['d125']:7.1f}%{r['d2']:6.1f}%")
    sel = fin & (sg <= q[1])
    return {"paper": paper, "all": metrics(e, g), "conf": metrics(e, g, sel),
            "gt": metrics(e, g, np.abs(g) >= 0.2), "kept_conf": 100 * np.mean(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mvsec", choices=["mvsec", "vl"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default="all", choices=SEQS + ["indoor_flying3"] + VL_SEQS + ["all"])
    ap.add_argument("--model", default="all", choices=MODELS + ["scaling", "all"])
    ap.add_argument("-s", type=int, default=None, help="default 3 (MVSEC) / 2 (VL)")
    ap.add_argument("--l-theta", type=float, default=None, help="default 100 (MVSEC) / 1000 (VL)")
    ap.add_argument("--danger", type=float, default=5.0, help="danger zone: GT TTC <= this (s)")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="only the first N seconds (tests)")
    ap.add_argument("--cache", default="", help="folder to keep/reuse the per-event arrays")
    args = ap.parse_args()
    vl = args.dataset == "vl"
    args.s = args.s or (2 if vl else 3)
    args.l_theta = args.l_theta or (1000.0 if vl else 100.0)
    all_seqs = VL_SEQS if vl else SEQS
    all_models = ["scaling", "translation", "six_dof"] if vl else MODELS
    seqs = all_seqs if args.seq == "all" else [args.seq]
    models = all_models if args.model == "all" else [args.model]
    summary = {}
    for seq in seqs:
        for model in models:
            path = (os.path.join(args.cache, f"{args.dataset}_{seq}_{model}_s{args.s}.npz")
                    if args.cache else "")
            if path and os.path.exists(path) and not args.max_seconds:
                d = np.load(path)
                e, g, sg = d["mu_est"], d["mu_gt"], d["u"]
                print(f"   (cached {path})")
            else:
                e, g, sg = (collect_vl if vl else collect)(args, seq, model)
                if path and not args.max_seconds:
                    os.makedirs(args.cache, exist_ok=True)
                    np.savez(path, mu_est=e, mu_gt=g, u=sg)
            summary[(seq, model)] = report(seq, model, e, g, sg, args)

    print("\n== Summary (all events vs the most confident 50 %, chosen without GT)")
    print(f"   {'':26s}{'paper':>8s}{'AREE all':>10s}{'AREE conf':>11s}{'AREE GT>=0.2':>14s}"
          f"{'LTE all':>9s}{'LTE conf':>10s}{'d<1.25 all':>12s}{'d<1.25 conf':>13s}")
    def line(name, r):
        print(f"   {name:26s}{r['paper']:8.2f}{r['all']['aree']:10.2f}{r['conf']['aree']:11.2f}"
              f"{r['gt']['aree']:14.2f}{r['all']['lte']:9.3f}{r['conf']['lte']:10.3f}"
              f"{r['all']['d125']:11.1f}%{r['conf']['d125']:12.1f}%")

    for (seq, model), r in summary.items():
        line(model[:11] + " / " + seq.replace("indoor_flying", "if").replace("outdoor_day", "od"), r)
    if len(seqs) > 1:
        print("   -- average over the sequences")
        for model in models:
            rs = [r for (q_, m), r in summary.items() if m == model]
            avg = {"paper": np.nanmean([r["paper"] for r in rs]) if any(np.isfinite(r["paper"]) for r in rs)
                   else np.nan}
            for k in ("all", "conf", "gt"):
                avg[k] = {f: float(np.nanmean([r[k][f] for r in rs])) if any(np.isfinite(r[k][f]) for r in rs)
                          else np.nan for f in ("aree", "lte", "d125")}
            line(model, avg)


if __name__ == "__main__":
    main()
