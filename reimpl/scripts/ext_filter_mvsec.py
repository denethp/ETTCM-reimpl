"""EXTENSIONS (not in the paper) on MVSEC:
  1. a causal spatio-temporal median filter on the per-event divergence and flow;
  2. more Gauss-Newton iterations per event (paper: 2).

    python scripts/ext_filter_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation --iters 2
    python scripts/ext_filter_mvsec.py --summary ../ext_filter_results.jsonl     # print the tables

1. Median filter. Each event with an estimate stores (t, mu, vx, vy) in a small ring buffer at
   its pixel (the last K = 4). Its filtered estimate is the component-wise median of every stored
   estimate within a (2R+1)^2 window (R = 2, i.e. 5 x 5 px) that is at most DT = 20 ms old,
   itself included. Only past and current events are used, so it works online. The estimator is
   not touched: the raw estimates are exactly those of tables45_mvsec.py / table3_mvsec.py.
   R, DT and K were fixed before running, not tuned on these results.
2. Iterations: --iters 3 or 5 instead of the paper's 2 (Table 1), nothing else changed.

One estimator run per (sequence, model, iterations), scored twice (raw and filtered) with:
  Table 3  divergence AREE (per event, Eq. 14) and its median, sequences indoor_flying1/2 and
           outdoor_day1, exactly as table3_mvsec.py "per event, all";
  Tables 4/5  flow AEE, Out, AREE, AAE over consecutive grayscale frames (outdoor rows < 190),
           as tables45_mvsec.py. The estimator starts at the start of the scored window (outdoor:
           at the start of the recording, warm start), as both scripts do.
Settings as the paper's MVSEC experiments: s = 3, w_thresh = 0.01, l_theta = 100, l_rho = 1.
Each finished run appends one JSON line to --out.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm import mvsec as M  # noqa: E402
from ettcm._jit import njit  # noqa: E402
from ettcm.estimator import run_ettcm  # noqa: E402
from ettcm.io_es import EVENT_DTYPE  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402

R, DT, K = 2, 0.02, 4
TABLE3_SEQS = ("indoor_flying1", "indoor_flying2", "outdoor_day1")


@njit
def median_filter(t, x, y, has, mu, vx, vy, bt, bmu, bvx, bvy, bpos, r, dt,
                  fmu, fvx, fvy, tmp0, tmp1, tmp2):
    """Causal spatio-temporal median. bt/bmu/bvx/bvy: (H, W, K) ring buffers, bpos: (H, W)."""
    H, W, Kb = bt.shape
    for k in range(t.shape[0]):
        if not has[k]:
            fmu[k] = math.nan
            fvx[k] = math.nan
            fvy[k] = math.nan
            continue
        xx = x[k]
        yy = y[k]
        j = bpos[yy, xx]
        bt[yy, xx, j] = t[k]
        bmu[yy, xx, j] = mu[k]
        bvx[yy, xx, j] = vx[k]
        bvy[yy, xx, j] = vy[k]
        bpos[yy, xx] = (j + 1) % Kb
        n = 0
        for py in range(max(yy - r, 0), min(yy + r, H - 1) + 1):
            for px in range(max(xx - r, 0), min(xx + r, W - 1) + 1):
                for q in range(Kb):
                    tq = bt[py, px, q]
                    if tq >= t[k] - dt and tq <= t[k]:
                        tmp0[n] = bmu[py, px, q]
                        tmp1[n] = bvx[py, px, q]
                        tmp2[n] = bvy[py, px, q]
                        n += 1
        fmu[k] = np.median(tmp0[:n])
        fvx[k] = np.median(tmp1[:n])
        fvy[k] = np.median(tmp2[:n])


class Rel:
    """Mean of relative errors + a subsample for the median."""

    def __init__(self):
        self.s, self.n, self.sample = 0.0, 0, []

    def add(self, rel):
        self.s += float(rel.sum())
        self.n += rel.size
        self.sample.append(rel[::10].astype(np.float32))

    def result(self):
        smp = np.concatenate(self.sample) if self.sample else np.array([np.nan])
        return {"AREE": 100 * self.s / max(self.n, 1), "median": 100 * float(np.median(smp)),
                "n": self.n}


def run(args, seq, model, iters):
    root = args.data
    cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(seq)}_calib.zip"))
    evf = M.EventFile(M.find_file(root, f"{seq}_data.hdf5"))
    flow = M.GTFlow(M.find_file(root, f"{seq}_gt_flow_dist.npz"), seq)
    rows = M.OUTDOOR_MAX_ROW if seq.startswith("outdoor") else None
    do_div = seq in TABLE3_SEQS
    if do_div:
        try:
            odom = M.find_file(root, f"{seq}_odom.npz")
        except FileNotFoundError:
            odom = None
        div = M.DivergenceGT(M.find_file(root, f"{seq}_gt.hdf5"), flow, cam, odom)
        src = div.default_source
    H, W = cam.height, cam.width
    t_rec0 = evf.time(0)
    t_start, t_end = flow.window(t_rec0)
    if args.max_seconds:
        t_end = min(t_end, t_start + args.max_seconds)
    edges = evf.gray_ts[(evf.gray_ts > t_start) & (evf.gray_ts < t_end)]
    i0 = evf.index_of_time(t_rec0) if seq == "outdoor_day1" else evf.index_of_time(t_start)
    i1 = evf.index_of_time(t_end)
    t0 = evf.time(i0)
    mid = MODEL_IDS[model]
    state = M.initial_state(cam, mid, seq)

    bt = np.full((H, W, K), -1e18)
    bmu = np.zeros((H, W, K))
    bvx = np.zeros((H, W, K))
    bvy = np.zeros((H, W, K))
    bpos = np.zeros((H, W), np.int64)
    tmp = [np.zeros((2 * R + 1) ** 2 * K) for _ in range(3)]

    fm = {"raw": M.FlowMetrics(), "filt": M.FlowMetrics()}
    dm = {"raw": Rel(), "filt": Rel()}
    disp_cache = {}

    def gt_disp(k):
        if k not in disp_cache:
            if len(disp_cache) > 4:
                disp_cache.clear()
            gu, gv = flow.displacement(edges[k], edges[k + 1])
            disp_cache[k] = (gu, gv, M.gt_mask(gu, gv, rows))
        return disp_cache[k]

    tic = time.perf_counter()
    t_est = 0.0
    for a in range(i0, i1, args.chunk):
        b = min(a + args.chunk, i1)
        x, y, t, p = evf.read(a, b)
        keep, xr, yr = M.rectify(x, y, cam)
        x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
        ev = np.empty(len(t), EVENT_DTYPE)
        ev["t"] = np.round((t - t0) * 1e6).astype(np.int64)
        ev["x"], ev["y"], ev["p"] = x, y, p
        tt = time.perf_counter()
        out = run_ettcm(ev, xr, yr, cam, mid, s=args.s, w_thresh=0.01, l_theta=args.l_theta,
                        l_rho=1.0, n_iter=iters, state=state)
        t_est += time.perf_counter() - tt
        has = out["rho"] > 0
        fmu = np.empty(len(t))
        fvx = np.empty(len(t))
        fvy = np.empty(len(t))
        median_filter(t.astype(np.float64), x.astype(np.int64), y.astype(np.int64), has,
                      out["mu"], out["vx"], out["vy"], bt, bmu, bvx, bvy, bpos, R, DT,
                      fmu, fvx, fvy, *tmp)
        est = {"raw": (out["mu"], out["vx"], out["vy"]), "filt": (fmu, fvx, fvy)}
        xi = np.clip(np.round(xr).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(yr).astype(np.int64), 0, H - 1)

        # ---- Tables 4/5: flow over grayscale-frame intervals
        k_ev = np.searchsorted(edges, t, side="right") - 1
        use = (k_ev >= 0) & (k_ev < len(edges) - 1) & has
        for k in np.unique(k_ev[use]):
            sel = use & (k_ev == k)
            gu, gv, ok = gt_disp(k)
            px, py = xi[sel], yi[sel]
            g = ok[py, px]
            gt = np.stack([gu[py[g], px[g]], gv[py[g], px[g]]], axis=1)
            dtf = edges[k + 1] - edges[k]
            for name, (_, vxx, vyy) in est.items():
                pred = np.stack([vxx[sel], vyy[sel]], axis=1)[g] * dtf
                fm[name].add(pred, gt)

        # ---- Table 3: per-event divergence vs the GT of the latest depth frame
        if do_div:
            ii = np.nonzero(has & (t >= t_start) & (t <= t_end))[0]
            kd = div.frame_of(t[ii])
            for k in np.unique(kd):
                s_ = ii[kd == k]
                g = div.mu_map(int(k), src)[yi[s_], xi[s_]]
                ok = np.isfinite(g) & (np.abs(g) > 1e-9)
                for name, (mu, _, _) in est.items():
                    dm[name].add(np.abs(mu[s_][ok] - g[ok]) / np.abs(g[ok]))
        print(f"\r  {seq} {model} iters={iters}: {100 * (b - i0) / max(i1 - i0, 1):5.1f} %",
              end="", flush=True)
    print(f"  ({time.perf_counter() - tic:.0f} s)")
    evf.close()
    if do_div:
        div.close()
    res = {"seq": seq, "model": model, "iters": iters, "us_per_event": 1e6 * t_est / max(i1 - i0, 1)}
    for name in ("raw", "filt"):
        f = fm[name].result()
        res[name] = {"AEE": f["AEE"], "Out": f["Out"], "fAREE": f["AREE"], "AAE": f["AAE"],
                     "flow_n": f["n"]}
        if do_div:
            d = dm[name].result()
            res[name].update({"div_AREE": d["AREE"], "div_median": d["median"], "div_n": d["n"]})
    return res


def summary(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by = {(r["seq"], r["model"], r["iters"]): r for r in rows}
    seqs4 = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "outdoor_day1"]
    short = {"indoor_flying1": "if1", "indoor_flying2": "if2", "indoor_flying3": "if3",
             "outdoor_day1": "od1"}

    def cell(v):
        return f"{v:8.2f}" if v is not None else f"{'-':>8s}"

    print("\n== Table 4 (flow), AEE px / Out %   [raw -> median-filtered]")
    for model in ("driving", "translation", "six_dof"):
        for iters in (2, 3, 5):
            line = []
            for q in seqs4:
                r = by.get((q, model, iters))
                if r is None:
                    line.append(f"{'':>30s}")
                    continue
                a, f = r["raw"], r["filt"]
                line.append(f"{a['AEE']:5.2f}/{a['Out']:5.2f} ->{f['AEE']:5.2f}/{f['Out']:5.2f}   ")
            if any((q, model, iters) in by for q in seqs4):
                p4 = M.PAPER_TABLE4.get(model, {})
                print(f"   {model:12s} iters={iters}  " + " | ".join(
                    f"{short[q]} " + s_ for q, s_ in zip(seqs4, line)))
        p4 = M.PAPER_TABLE4.get(model, {})
        print(f"   {'':12s} paper     " + " | ".join(
            f"{short[q]} {p4[q][0]:5.2f}/{p4[q][1]:5.2f}{'':16s}" for q in seqs4 if q in p4))

    print("\n== Table 5 (flow), AREE % / AAE deg   [raw -> median-filtered]")
    for model in ("driving", "translation", "six_dof"):
        for iters in (2, 3, 5):
            parts = []
            for q in ("indoor_flying3", "outdoor_day1"):
                r = by.get((q, model, iters))
                if r:
                    a, f = r["raw"], r["filt"]
                    parts.append(f"{short[q]} {a['fAREE']:6.2f}/{a['AAE']:5.2f} -> "
                                 f"{f['fAREE']:6.2f}/{f['AAE']:5.2f}")
            if parts:
                print(f"   {model:12s} iters={iters}  " + " | ".join(parts))
        p5 = M.PAPER_TABLE5.get(model, {})
        print(f"   {'':12s} paper     " + " | ".join(
            f"{short[q]} {p5[q][0]:6.2f}/{p5[q][1]:5.2f}" for q in ("indoor_flying3", "outdoor_day1")
            if q in p5))

    print("\n== Table 3 (divergence), AREE % per event (median)   [raw -> median-filtered]")
    for model in ("driving", "translation", "six_dof"):
        for iters in (2, 3, 5):
            parts = []
            for q in TABLE3_SEQS:
                r = by.get((q, model, iters))
                if r and "div_AREE" in r["raw"]:
                    a, f = r["raw"], r["filt"]
                    parts.append(f"{short[q]} {a['div_AREE']:7.2f} ({a['div_median']:6.2f}) -> "
                                 f"{f['div_AREE']:7.2f} ({f['div_median']:6.2f})")
            if parts:
                print(f"   {model:12s} iters={iters}  " + " | ".join(parts))
        p3 = M.PAPER_TABLE3.get(model, {})
        print(f"   {'':12s} paper     " + " | ".join(f"{short[q]} {p3[q]:7.2f}" for q in TABLE3_SEQS
                                                     if q in p3))
    print("\n== runtime (estimator only), us/event")
    for (q, m, it), r in sorted(by.items()):
        print(f"   {m:12s} {short[q]} iters={it}: {r['us_per_event']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--seq", choices=["indoor_flying1", "indoor_flying2", "indoor_flying3",
                                      "outdoor_day1"])
    ap.add_argument("--model", choices=["driving", "translation", "six_dof"])
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--out", default="../ext_filter_results.jsonl")
    ap.add_argument("--summary", default="", help="print the tables from a results file")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    res = run(args, args.seq, args.model, args.iters)
    print(json.dumps(res))
    if not args.max_seconds:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
