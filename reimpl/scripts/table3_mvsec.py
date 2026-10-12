"""Table 3 of the paper (divergence on MVSEC), computed every plausible way.

    python scripts/table3_mvsec.py --data ../datasets/MVSEC
    python scripts/table3_mvsec.py --data ../datasets/MVSEC --seq indoor_flying1 --model translation
    python scripts/table3_mvsec.py --data ../datasets/MVSEC --seq indoor_flying3

GT: mu = V_z / Z per pixel, Z = MVSEC's rectified depth map, V_z = MVSEC's camera velocity
(<seq>_odom.npz; if missing, the same velocity rebuilt from the poses in <seq>_gt.hdf5).
Estimate: mu_i = rho_i * nu_z of each event, compared at its rectified pixel with the GT of the
latest depth frame. Settings as Tables 4-5 (s = 3, l_theta = 100); outdoor_day1 is scored on
222.4-240.4 s with the estimator started at the beginning of the recording (warm start).

The paper (Sec. 4.1): "We generate GT TTCM's by applying Eq. (1) given the GT depth maps and
camera velocity. [...] we use the AREE (%) between the divergence estimate mu_i and
corresponding GT mu_gt,i." It does not say which events/pixels, at which times, or whether
anything is excluded. This script prints, for every (sequence, model):

  checks  GT flow in <seq>_gt.hdf5 vs <seq>_gt_flow_dist.npz; forward speed V_z from MVSEC's
          <seq>_odom.npz vs our reconstruction from the poses vs what the GT flow implies
  rows    AREE (Eq. 14) under different protocols:
            per event, all                      (what tables45_mvsec.py reports)
            per event, |mu_gt| >= threshold     (near-zero GT excluded), with % kept
            per event, frames with |V_z| >= v   (frames where the camera barely moves forward excluded)
            per event, each error capped at 100 %
            median per event
            TTC map at the 20 Hz depth frames   (latest estimate per pixel, at most --map-age old)
            each with the GT velocity from MVSEC's odom.npz (fallback: rebuilt from the poses)
          plus "all events" for every GT velocity source.
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

SEQS = ["indoor_flying1", "indoor_flying2", "outdoor_day1"]   # the paper's Table 3 columns
MODELS = ["driving", "translation", "six_dof"]
MU_THR = [0.01, 0.02, 0.05, 0.1, 0.2]       # 1/s
VZ_THR = [0.05, 0.1, 0.2]                   # m/s


class Acc:
    """Mean of relative errors with a count."""

    def __init__(self):
        self.s = 0.0
        self.n = 0

    def add(self, rel):
        self.s += float(rel.sum())
        self.n += int(rel.size)

    def get(self):
        return 100 * self.s / self.n if self.n else float("nan")


def checks(seq, flow, div, t_start, t_end):
    print(f"  -- checks for {seq}")
    # 1. GT flow: gt.hdf5 copy vs npz
    try:
        fd = div.h5["davis"]["left"]["flow_dist"]
        fts = np.asarray(div.h5["davis"]["left"]["flow_dist_ts"][:], dtype=np.float64)
        diffs, ratios = [], []
        for k in np.linspace(len(fts) * 0.2, len(fts) * 0.8, 5).astype(int):
            j = int(np.argmin(np.abs(flow.ts_all - fts[k])))
            a = np.asarray(fd[k], dtype=np.float64)
            b = np.stack([np.asarray(flow.u_all[j]), np.asarray(flow.v_all[j])])
            m = (np.hypot(*b) > 0) & np.isfinite(a).all(0)
            diffs.append(np.abs(a - b)[:, m].max() if m.any() else 0)
            ratios.append(np.median(np.hypot(*a)[m] / np.hypot(*b)[m]) if m.any() else np.nan)
        print(f"     GT flow gt.hdf5 vs npz: max |diff| {max(diffs):.4f} px, median |hdf5|/|npz| "
              f"{np.nanmedian(ratios):.3f}")
    except KeyError:
        print("     (no flow_dist in gt.hdf5)")
    # 2. forward speed from three sources
    ks = np.unique(div.frame_of(np.linspace(t_start, t_end, 60)))
    rows = []
    for k in ks:
        a = div.vz(int(k), "odom_npz")
        b = div.vz(int(k), "poses")
        c = div.vz(int(k), "flow")
        rows.append([np.nan if v is None else v for v in (a, b, c)])
    r = np.array(rows)
    big = np.abs(r[:, 0]) >= np.nanpercentile(np.abs(r[:, 0]), 30)
    print(f"     V_z median |.| [m/s]: odom.npz {np.nanmedian(np.abs(r[:, 0])):.3f}, ours from poses "
          f"{np.nanmedian(np.abs(r[:, 1])):.3f}, implied by GT flow {np.nanmedian(np.abs(r[:, 2])):.3f}")
    print(f"     median ratio (frames with clear forward motion): poses/odom.npz "
          f"{np.nanmedian(r[big, 1] / r[big, 0]):.3f}, flow/odom.npz {np.nanmedian(r[big, 2] / r[big, 0]):.3f}")
    frac = [100 * np.mean(np.abs(r[:, 0]) < v) for v in VZ_THR]
    print("     frames with |V_z| below " + ", ".join(f"{v} m/s: {f:.0f} %" for v, f in zip(VZ_THR, frac)))


def evaluate(args, seq, model, cam, evf, flow, div, t_start, t_end):
    src = div.default_source
    # warm start: outdoor_day1 runs from the start of the recording, only the window is scored
    i0 = 0 if (seq == "outdoor_day1" and not args.cold) else evf.index_of_time(t_start)
    i1 = evf.index_of_time(t_end)
    t0 = evf.time(i0)
    state = EttcmState(cam.width, cam.height)
    H, W = cam.height, cam.width
    acc = {"all": Acc(), "cap": Acc(), "map": Acc()}
    acc.update({f"mu{v}": Acc() for v in MU_THR})
    acc.update({f"vz{v}": Acc() for v in VZ_THR})
    acc_src = {s: Acc() for s in div.VEL_SOURCES}
    n_mu = {v: 0 for v in MU_THR}
    n_vz = {v: 0 for v in VZ_THR}
    n_all_events = 0
    sample = []
    map_mu = np.full((H, W), np.nan)
    map_t = np.full((H, W), -1e18)
    map_frames = []
    vz_frame = {}
    tic = time.perf_counter()

    def frame_vz(k):
        if k not in vz_frame:
            v = div.vz(int(k), src)
            vz_frame[k] = np.nan if v is None else v
        return vz_frame[k]

    for a in range(i0, i1, args.chunk):
        b = min(a + args.chunk, i1)
        x, y, t, p = evf.read(a, b)
        keep, xr, yr = M.rectify(x, y, cam)
        x, y, t, p, xr, yr = x[keep], y[keep], t[keep], p[keep], xr[keep], yr[keep]
        ev = np.empty(len(t), EVENT_DTYPE)
        ev["t"] = np.round((t - t0) * 1e6).astype(np.int64)
        ev["x"], ev["y"], ev["p"] = x, y, p
        out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=args.s, w_thresh=0.01,
                        l_theta=args.l_theta, l_rho=1.0, state=state)
        xi = np.clip(np.round(xr).astype(np.int64), 0, W - 1)
        yi = np.clip(np.round(yr).astype(np.int64), 0, H - 1)
        mu = out["mu"]
        est = (out["rho"] > 0) & (t >= t_start)

        # ---- per event
        ii = np.nonzero(est)[0]
        kd = div.frame_of(t[ii])
        for k in np.unique(kd):
            s = ii[kd == k]
            n_all_events += len(s)
            for sname in div.VEL_SOURCES:
                g = div.mu_map(int(k), sname)[yi[s], xi[s]]
                ok = np.isfinite(g) & (np.abs(g) > 1e-9)
                acc_src[sname].add(np.abs(mu[s][ok] - g[ok]) / np.abs(g[ok]))
            g = div.mu_map(int(k), src)[yi[s], xi[s]]
            ok = np.isfinite(g) & (np.abs(g) > 1e-9)
            mh, mg = mu[s][ok], g[ok]
            rel = np.abs(mh - mg) / np.abs(mg)
            acc["all"].add(rel)
            acc["cap"].add(np.minimum(rel, 1.0))
            sample.append(rel[::25])
            for v in MU_THR:
                sel = np.abs(mg) >= v
                acc[f"mu{v}"].add(rel[sel])
                n_mu[v] += int(sel.sum())
            vzk = frame_vz(k)
            for v in VZ_THR:
                if np.isfinite(vzk) and abs(vzk) >= v:
                    acc[f"vz{v}"].add(rel)
                    n_vz[v] += rel.size

        # ---- TTC map at the depth-frame times
        bounds = np.nonzero((div.dts >= t[0]) & (div.dts < t[-1]) & (div.dts >= t_start)
                            & (div.dts <= t_end))[0]
        start = 0
        for k in bounds:
            stop = np.searchsorted(t, div.dts[k], side="right")
            sl = np.arange(start, stop)
            sl = sl[est[sl]]
            map_mu[yi[sl], xi[sl]] = mu[sl]
            map_t[yi[sl], xi[sl]] = t[sl]
            start = stop
            g = div.mu_map(int(k), src)
            ok = (div.dts[k] - map_t < args.map_age) & np.isfinite(g) & (np.abs(g) > 1e-9)
            if ok.any():
                rel = np.abs(map_mu[ok] - g[ok]) / np.abs(g[ok])
                acc["map"].add(rel)
                map_frames.append(100 * rel.mean())
        sl = np.arange(start, len(t))
        sl = sl[est[sl]]
        map_mu[yi[sl], xi[sl]] = mu[sl]
        map_t[yi[sl], xi[sl]] = t[sl]
        print(f"\r    {seq} {model}: {100 * (b - i0) / max(i1 - i0, 1):5.1f} %", end="", flush=True)
    print()

    smp = np.concatenate(sample) if sample else np.array([np.nan])
    res = {"src": src, "events": n_all_events, "secs": time.perf_counter() - tic,
           "all": acc["all"].get(), "median": 100 * float(np.median(smp)),
           "cap": acc["cap"].get(), "map": acc["map"].get(),
           "map_framemean": float(np.mean(map_frames)) if map_frames else float("nan")}
    n_base = max(acc["all"].n, 1)
    for v in MU_THR:
        res[f"mu{v}"] = (acc[f"mu{v}"].get(), 100 * n_mu[v] / n_base)
    for v in VZ_THR:
        res[f"vz{v}"] = (acc[f"vz{v}"].get(), 100 * n_vz[v] / n_base)
    res["by_src"] = {s: acc_src[s].get() for s in div.VEL_SOURCES}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default="all", choices=SEQS + ["indoor_flying3", "all"])
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("-s", type=int, default=3)
    ap.add_argument("--l-theta", type=float, default=100.0)
    ap.add_argument("--cold", action="store_true",
                    help="outdoor_day1: start the estimator at 222.4 s instead of the recording start")
    ap.add_argument("--map-age", type=float, default=0.05, help="max age (s) of a map pixel")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="only the first N seconds (tests)")
    args = ap.parse_args()

    seqs = SEQS if args.seq == "all" else [args.seq]
    models = MODELS if args.model == "all" else [args.model]
    results = {}
    for seq in seqs:
        root = args.data
        cam, _ = M.load_calib(M.find_file(root, f"{M.scene_of(seq)}_calib.zip"))
        evf = M.EventFile(M.find_file(root, f"{seq}_data.hdf5"))
        flow = M.GTFlow(M.find_file(root, f"{seq}_gt_flow_dist.npz"), seq)
        try:
            odom = M.find_file(root, f"{seq}_odom.npz")
        except FileNotFoundError:
            odom = None
            print(f"  (no {seq}_odom.npz: GT velocity rebuilt from the poses in {seq}_gt.hdf5)")
        div = M.DivergenceGT(M.find_file(root, f"{seq}_gt.hdf5"), flow, cam, odom)
        t_start, t_end = flow.window(evf.time(0))
        if args.max_seconds:
            t_end = min(t_end, t_start + args.max_seconds)
        print(f"== {seq}: {t_end - t_start:.1f} s")
        checks(seq, flow, div, t_start, t_end)
        for model in models:
            results[(seq, model)] = evaluate(args, seq, model, cam, evf, flow, div, t_start, t_end)
        div.close()
        evf.close()

    # ---- the table
    cols = [(q, m) for m in models for q in seqs if (q, m) in results]
    hdr = " ".join(f"{(m[:5] + '/' + q.replace('indoor_flying', 'if').replace('outdoor_day', 'od')):>12s}" for q, m in cols)
    print("\n== Table 3, divergence AREE (%) under different protocols")
    print(f"{'':44s}{hdr}")
    paper = [M.PAPER_TABLE3.get(m, {}).get(q, float("nan")) for q, m in cols]
    print(f"{'PAPER':44s}" + " ".join(f"{v:12.2f}" for v in paper))

    def row(name, vals):
        print(f"{name:44s}" + " ".join(f"{v:12.2f}" if np.isfinite(v) else f"{'-':>12s}" for v in vals))

    row("per event, all", [results[c]["all"] for c in cols])
    row("per event, median", [results[c]["median"] for c in cols])
    row("per event, error capped at 100 %", [results[c]["cap"] for c in cols])
    for v in MU_THR:
        row(f"per event, |mu_gt| >= {v} 1/s", [results[c][f"mu{v}"][0] for c in cols])
        row(f"   (% of events kept)", [results[c][f"mu{v}"][1] for c in cols])
    for v in VZ_THR:
        row(f"per event, frames |V_z| >= {v} m/s", [results[c][f"vz{v}"][0] for c in cols])
        row(f"   (% of events kept)", [results[c][f"vz{v}"][1] for c in cols])
    row(f"TTC map at depth frames (<= {args.map_age} s old)", [results[c]["map"] for c in cols])
    row("   same, mean of per-frame AREE", [results[c]["map_framemean"] for c in cols])
    print("\n== per event, all events, by GT camera-velocity source")
    for s in M.DivergenceGT.VEL_SOURCES:
        row(f"GT velocity: {s}", [results[c]["by_src"][s] for c in cols])
    print(f"\n(main rows use GT velocity: {results[cols[0]]['src']})")


if __name__ == "__main__":
    main()
