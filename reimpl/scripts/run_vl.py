"""Table 2 of the paper with our implementation, on the Ventral Landing (VL) dataset.

Unzip VL.zip (https://figshare.com/ndownloader/files/44987794) so that
    datasets/VL/calib.txt
    datasets/VL/<seq>/events.es, divergence.txt, timestamps.txt   (8 sequences)

    python scripts/run_vl.py --data ../datasets/VL --model translation -s 2
    python scripts/run_vl.py --data ../datasets/VL --all --csv ../results_python_vl.csv

Each line prints our REE next to the paper's Table 2 number for that sequence.
Models: scaling, translation, six_dof (the official re-parameterised 6-DOF, which
is what the paper's 6-DOF rows were computed with). Settings are those of the
official script table_2_results.sh: -e 0.01 -m 1000 -l 1, s in {2, 3}.
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm.calib import make_camera, rectify_events  # noqa: E402
from ettcm.estimator import run_ettcm  # noqa: E402
from ettcm.evaluate import read_vl_sequence, ree_at_timestamps  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402


SEQS = ["2D-1", "2D-2", "2D-3", "2D-4", "2D-5", "2D-6", "2D-7", "3D"]

# Paper, Table 2 (REE %), per sequence in SEQS order, then the average.
PAPER_REE = {
    ("scaling", 2):     [7.19, 8.79, 9.13, 12.18, 4.98, 7.79, 4.92, 11.34, 8.29],
    ("translation", 2): [5.03, 7.02, 6.17, 5.25, 4.61, 4.34, 6.18, 6.54, 5.64],
    ("six_dof", 2):     [6.77, 13.73, 14.97, 7.50, 6.91, 12.15, 12.77, 6.10, 10.11],
    ("scaling", 3):     [14.99, 15.19, 11.69, 21.69, 11.19, 15.44, 11.01, 19.43, 15.08],
    ("translation", 3): [11.15, 7.01, 7.98, 13.80, 6.37, 11.17, 4.55, 12.46, 9.31],
    ("six_dof", 3):     [6.51, 6.81, 7.25, 6.77, 6.06, 5.59, 6.18, 6.19, 6.42],
}
ALL_MODELS = ["scaling", "translation", "six_dof"]


def run_one(seq_dir, cam, model, s):
    ev, w, h, ts, div_gt = read_vl_sequence(seq_dir)
    ev, xr, yr = rectify_events(ev, cam)
    t0 = time.perf_counter()
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=s, w_thresh=0.01,
                    l_theta=1000.0, l_rho=1.0)
    us = 1e6 * (time.perf_counter() - t0) / len(ev)
    ree, _, _ = ree_at_timestamps(ev["t"], out["div"], out["rho"], ts, div_gt)
    return ree, us, len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="translation", choices=ALL_MODELS)
    ap.add_argument("-s", type=int, default=2, choices=(2, 3))
    ap.add_argument("--all", action="store_true", help="all 3 models x s in {2, 3}")
    ap.add_argument("--csv", default=None, help="append results to this CSV file")
    args = ap.parse_args()

    cam = make_camera(os.path.join(args.data, "calib.txt"))
    seqs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isfile(os.path.join(d, "events.es")))
    combos = ([(m, s) for m in ALL_MODELS for s in (2, 3)] if args.all
              else [(args.model, args.s)])
    csv = open(args.csv, "a") if args.csv else None
    for model, s in combos:
        print(f"Model: {model}  s={s}")
        print(f"  {'seq':6s} {'ours':>7s} {'paper':>7s}   {'us/event':>8s}")
        rees = []
        paper = PAPER_REE[(model, s)]
        for d in seqs:
            name = os.path.basename(d)
            ree, us, n = run_one(d, cam, model, s)
            rees.append(ree)
            ref = f"{paper[SEQS.index(name)]:7.2f}" if name in SEQS else "      -"
            print(f"  {name:6s} {ree:7.2f} {ref}   {us:8.2f}   ({n} events)", flush=True)
            if csv:
                csv.write(f"{model},{s},{name},{ree:.4f},{us:.3f},{n}\n")
                csv.flush()
        avg = sum(rees) / len(rees)
        ref = f"{paper[-1]:7.2f}" if len(rees) == len(SEQS) else "      -"
        print(f"  {'avg':6s} {avg:7.2f} {ref}")


if __name__ == "__main__":
    main()
