"""Table 6 and Fig. 8 of the supplementary material: VL global divergence REE vs the
resolution reduction multiplier r (event sampling by leaky firing, ettcm/sampling.py).

    python scripts/table6_vl.py --data ../datasets/VL --csv ../results_python_vl_table6.csv \\
        --plot ../figures/fig8_resolution.png

Settings as the official table_6_results.sh: -s {2, 3} -e 0.01 -m 1000 -l 1 -w r -g r,
models scaling, translation, six_dof, r in {2, 3, 4}; r = 1 is Table 2 (run too, for Fig. 8).
Each event's coordinates are divided by r, one integrator per reduced pixel and polarity
fires every r^2 events, and the estimator runs on the reduced grid with the camera
intrinsics divided by r (exactly as the official code).
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.calib import make_camera, rectify_events  # noqa: E402
from ettcm.estimator import run_ettcm  # noqa: E402
from ettcm.evaluate import read_vl_sequence, ree_at_timestamps  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402
from ettcm.sampling import leaky_sampling  # noqa: E402

SEQS = ["2D-1", "2D-2", "2D-3", "2D-4", "2D-5", "2D-6", "2D-7", "3D"]
MODELS = ["scaling", "translation", "six_dof"]

# Supplementary Table 6 (REE %), per sequence in SEQS order, then the average.
# r = 1 rows are Table 2.
PAPER = {
    ("scaling", 2, 1):     [7.19, 8.79, 9.13, 12.18, 4.98, 7.79, 4.92, 11.34, 8.29],
    ("translation", 2, 1): [5.03, 7.02, 6.17, 5.25, 4.61, 4.34, 6.18, 6.54, 5.64],
    ("six_dof", 2, 1):     [6.77, 13.73, 14.97, 7.50, 6.91, 12.15, 12.77, 6.10, 10.11],
    ("scaling", 3, 1):     [14.99, 15.19, 11.69, 21.69, 11.19, 15.44, 11.01, 19.43, 15.08],
    ("translation", 3, 1): [11.15, 7.01, 7.98, 13.80, 6.37, 11.17, 4.55, 12.46, 9.31],
    ("six_dof", 3, 1):     [6.51, 6.81, 7.25, 6.77, 6.06, 5.59, 6.18, 6.19, 6.42],
    ("scaling", 2, 2):     [10.66, 8.61, 6.81, 13.22, 6.94, 9.53, 6.94, 13.82, 9.57],
    ("translation", 2, 2): [7.10, 5.04, 4.61, 9.55, 3.61, 7.36, 2.82, 8.08, 6.02],
    ("six_dof", 2, 2):     [3.77, 9.93, 10.28, 7.66, 6.61, 6.10, 7.57, 5.10, 7.13],
    ("scaling", 3, 2):     [17.50, 14.30, 12.91, 19.13, 12.92, 14.77, 12.31, 20.53, 15.55],
    ("translation", 3, 2): [13.33, 10.22, 11.55, 16.33, 6.97, 12.96, 6.30, 14.77, 11.55],
    ("six_dof", 3, 2):     [8.22, 4.84, 4.22, 8.83, 6.38, 5.95, 3.45, 8.97, 6.36],
    ("scaling", 2, 3):     [9.74, 8.74, 5.43, 10.45, 8.44, 8.37, 7.12, 14.79, 9.14],
    ("translation", 2, 3): [8.62, 7.13, 6.00, 10.13, 7.06, 8.62, 4.06, 10.28, 7.74],
    ("six_dof", 2, 3):     [7.85, 12.29, 10.50, 9.76, 17.93, 9.01, 9.62, 5.77, 10.34],
    ("scaling", 3, 3):     [17.08, 14.40, 12.27, 17.74, 14.38, 15.22, 12.47, 21.06, 15.58],
    ("translation", 3, 3): [13.03, 12.23, 12.33, 17.12, 8.54, 14.48, 7.90, 16.61, 12.78],
    ("six_dof", 3, 3):     [7.80, 7.04, 6.24, 10.21, 11.13, 8.15, 5.06, 11.45, 8.39],
    ("scaling", 2, 4):     [11.12, 10.89, 8.08, 12.56, 9.85, 9.42, 8.13, 17.77, 10.98],
    ("translation", 2, 4): [9.46, 10.11, 8.48, 10.75, 11.63, 7.60, 7.36, 14.91, 10.04],
    ("six_dof", 2, 4):     [23.90, 18.54, 14.22, 23.00, 38.03, 16.50, 35.58, 10.26, 22.50],
    ("scaling", 3, 4):     [20.24, 17.50, 14.89, 21.50, 16.98, 17.16, 14.01, 24.42, 18.34],
    ("translation", 3, 4): [15.90, 13.95, 14.48, 18.01, 10.18, 14.71, 10.12, 21.20, 14.82],
    ("six_dof", 3, 4):     [12.36, 9.56, 8.34, 11.89, 13.53, 8.36, 8.74, 14.54, 10.92],
}


def run_one(job):
    data, seq, model, s, r = job
    ev, w, h, ts, div_gt = read_vl_sequence(os.path.join(data, seq))
    cam = make_camera(os.path.join(data, "calib.txt"), sampling=r)
    ev = leaky_sampling(ev, w, h, r)
    ev, xr, yr = rectify_events(ev, cam)
    tic = time.perf_counter()
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=s, w_thresh=0.01,
                    l_theta=1000.0, l_rho=1.0)
    us = 1e6 * (time.perf_counter() - tic) / max(len(ev), 1)
    ree, _, _ = ree_at_timestamps(ev["t"], out["div"], out["rho"], ts, div_gt)
    return (model, s, r, seq), (ree, us, len(ev))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = sorted({k[2] for k in res})
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 5.0), sharex=True)
    width = 0.8 / len(rs)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(rs)))
    for ax, s in zip(axs, (2, 3)):
        for i, r in enumerate(rs):
            ours = [res[(m, s, r)] for m in MODELS]
            paper = [PAPER[(m, s, r)][-1] for m in MODELS]
            xs = np.arange(len(MODELS)) + (i - (len(rs) - 1) / 2) * width
            ax.bar(xs, ours, width * 0.95, color=colors[i], label=f"r={r}")
            ax.scatter(xs, paper, marker="_", s=120, color="k", zorder=3,
                       label="paper" if i == 0 else None)
        ax.set_ylabel("REE (%)")
        ax.set_title(f"s={s}", fontsize=9)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(["Scaling", "Translation", "6-DOF"])
    axs[0].legend(fontsize=7, ncol=5, frameon=False, loc="upper left")
    fig.suptitle("Avg. global divergence REE on VL vs reduction multiplier r\n"
                 "(bars: ours, black ticks: paper Table 6)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--r", default="1,2,3,4", help="comma list of reduction multipliers")
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("-s", default="2,3", help="comma list of neighbourhood sizes")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--csv", default=None, help="write per-sequence results here")
    ap.add_argument("--plot", default=None, help="save Fig. 8 here")
    args = ap.parse_args()

    rs = [int(v) for v in args.r.split(",")]
    ss = [int(v) for v in args.s.split(",")]
    models = MODELS if args.model == "all" else [args.model]
    seqs = [s for s in SEQS if os.path.isfile(os.path.join(args.data, s, "events.es"))]
    jobs = [(args.data, q, m, s, r) for m in models for s in ss for r in rs for q in seqs]
    with Pool(args.jobs) as pool:
        per = dict(pool.map(run_one, jobs))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("model,s,r,seq,REE,us_per_event,events\n")
            for (m, s, r, q), (ree, us, n) in sorted(per.items()):
                f.write(f"{m},{s},{r},{q},{ree:.4f},{us:.3f},{n}\n")

    avg = {}
    print(f"{'model':12s} {'s':>2s} {'r':>2s} " + " ".join(f"{q:>6s}" for q in seqs)
          + f" {'avg':>6s} {'paper':>6s} {'diff':>6s} {'us/ev':>6s} {'events':>9s}")
    for m in models:
        for s in ss:
            for r in rs:
                rees = [per[(m, s, r, q)][0] for q in seqs]
                a = float(np.mean(rees))
                avg[(m, s, r)] = a
                p = PAPER[(m, s, r)][-1] if len(seqs) == len(SEQS) else float("nan")
                us = np.mean([per[(m, s, r, q)][1] for q in seqs])
                n = int(np.mean([per[(m, s, r, q)][2] for q in seqs]))
                print(f"{m:12s} {s:2d} {r:2d} " + " ".join(f"{v:6.2f}" for v in rees)
                      + f" {a:6.2f} {p:6.2f} {a - p:+6.2f} {us:6.2f} {n:9d}")
                print(f"{'  paper':18s}" + " ".join(f"{v:6.2f}" for v in PAPER[(m, s, r)][:-1]))
    if args.plot and len(models) == len(MODELS):
        plot(avg, args.plot)


if __name__ == "__main__":
    main()
