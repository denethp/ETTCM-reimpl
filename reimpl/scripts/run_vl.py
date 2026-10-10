"""M4: Table 2 with your implementation on the VL dataset.

Expects the official release layout (unzip VL.zip into ettcm/datasets/VL):
    datasets/VL/calib.txt
    datasets/VL/<seq>/events.es, divergence.txt, timestamps.txt

    python scripts/run_vl.py --data ../ettcm/datasets/VL --model translation -s 2
    python scripts/run_vl.py --data ../ettcm/datasets/VL --all

Compare with the official binary (same data, same hyper-parameters):
    ./build/src/global_divergence_translation datasets/VL/2D-1/events.es \
        datasets/VL/calib.txt datasets/VL/2D-1/divergence.txt \
        datasets/VL/2D-1/timestamps.txt -s 2 -e 0.01 -m 1000 -l 1
Paper Table 2 (avg REE %): s=2 Scaling 8.29 / Translation 5.64 / 6-DOF 10.11;
s=3 15.08 / 9.31 / 6.42. The partial release has 2D-1, 2D-3, 2D-5, 2D-7, 3D.
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


def run_one(seq_dir, cam, model, s, l_theta):
    ev, w, h, ts, div_gt = read_vl_sequence(seq_dir)
    ev, xr, yr = rectify_events(ev, cam)
    t0 = time.perf_counter()
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[model], s=s, l_theta=l_theta)
    us = 1e6 * (time.perf_counter() - t0) / len(ev)
    ree, _, _ = ree_at_timestamps(ev["t"], out["div"], out["rho"], ts, div_gt)
    return ree, us, len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="translation", choices=list(MODEL_IDS))
    ap.add_argument("-s", type=int, default=2)
    ap.add_argument("--l-theta", type=float, default=1000.0)
    ap.add_argument("--all", action="store_true", help="all models x s in {2,3}")
    args = ap.parse_args()

    cam = make_camera(os.path.join(args.data, "calib.txt"))
    seqs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isfile(os.path.join(d, "events.es")))
    combos = ([(m, s) for m in MODEL_IDS for s in (2, 3)] if args.all
              else [(args.model, args.s)])
    for model, s in combos:
        print(f"Model: {model}  s={s}")
        rees = []
        for d in seqs:
            ree, us, n = run_one(d, cam, model, s, args.l_theta)
            rees.append(ree)
            print(f"  {os.path.basename(d):6s} REE {ree:6.2f} %   {us:7.2f} us/event   ({n} events)")
        print(f"  avg    REE {sum(rees) / len(rees):6.2f} %")


if __name__ == "__main__":
    main()
