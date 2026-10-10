"""M1-M3: run your estimator on synthetic data with known ground truth.

    python scripts/run_synthetic.py --model scaling
    python scripts/run_synthetic.py --model translation --nu 0.3,0,1
    python scripts/run_synthetic.py --model six_dof --nu 0.3,0,1 --save-es syn.es

What "working" looks like (official C++ on this data, s = 2): the global
divergence converges within ~0.1-0.15 s and then tracks GT within a few percent
(REE over t > 0.2 s roughly 1-3 %).

--save-es writes the events as .es so you can run the official binary / the
cpp/harness on exactly the same data (milestone M4).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ettcm._jit import HAVE_NUMBA  # noqa: E402
from ettcm.calib import make_camera, rectify_events  # noqa: E402
from ettcm.estimator import run_ettcm  # noqa: E402
from ettcm.io_es import write_es  # noqa: E402
from ettcm.models import MODEL_IDS  # noqa: E402
from ettcm.synthetic import make_plane_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="translation", choices=list(MODEL_IDS))
    ap.add_argument("--nu", default="0,0,1", help="camera velocity nu_x,nu_y,nu_z (m/s)")
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--C", type=float, default=0.6, help="contrast threshold")
    ap.add_argument("-s", type=int, default=2)
    ap.add_argument("--l-theta", type=float, default=1000.0)
    ap.add_argument("--save-es", default=None)
    ap.add_argument("--plot", default="synthetic_divergence.png")
    args = ap.parse_args()

    nu = tuple(float(v) for v in args.nu.split(","))
    print(f"generating events (nu={nu}, duration={args.duration}s) ...")
    ev, gt = make_plane_sequence(nu=nu, duration=args.duration, C=args.C)
    print(f"  {len(ev)} events, {gt['width']}x{gt['height']}")
    if args.save_es:
        write_es(args.save_es, ev, gt["width"], gt["height"])
        print(f"  wrote {args.save_es} (fx=fy={gt['fx']}, cx={gt['cx']}, cy={gt['cy']})")

    cam = make_camera(fx=gt["fx"], fy=gt["fy"], cx=gt["cx"], cy=gt["cy"],
                      width=gt["width"], height=gt["height"])
    ev, xr, yr = rectify_events(ev, cam)
    t0 = time.perf_counter()
    out = run_ettcm(ev, xr, yr, cam, MODEL_IDS[args.model], s=args.s,
                    l_theta=args.l_theta)
    dt = time.perf_counter() - t0
    print(f"  {1e6 * dt / len(ev):.2f} us/event (numba={'on' if HAVE_NUMBA else 'OFF'};"
          " first numba run includes compilation)")

    t = ev["t"] * 1e-6
    print("\n   t (s)   est div   GT div   REE %")
    for tq in np.linspace(0.1 * args.duration, args.duration, 10):
        i = min(np.searchsorted(t, tq), len(t) - 1)
        g = float(gt["divergence"](t[i]))
        print(f"  {t[i]:6.3f}  {out['div'][i]:8.4f}  {g:7.4f}  {100 * abs(out['div'][i] - g) / g:6.2f}")
    sel = t > 0.2 * args.duration
    g = gt["divergence"](t[sel])
    print(f"\nmean REE over t > {0.2 * args.duration:.2f}s: "
          f"{100 * np.mean(np.abs(out['div'][sel] - g) / g):.2f} %")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(t, gt["divergence"](t), "k", label="GT")
        ax[0].plot(t, out["div"], label=f"ETTCM {args.model}")
        ax[0].set_xlabel("t (s)")
        ax[0].set_ylabel("divergence (1/s)")
        ax[0].legend()
        # TTC map: last per-pixel TTC over the final 20 ms
        last = t > t[-1] - 0.02
        ttcm = np.full((gt["height"], gt["width"]), np.nan)
        ttcm[ev["y"][last], ev["x"][last]] = out["ttc"][last]
        im = ax[1].imshow(ttcm, vmin=0, vmax=2 * float(gt["ttc"](t[-1])), cmap="jet")
        ax[1].set_title(f"TTCM (GT TTC = {float(gt['ttc'](t[-1])):.2f}s everywhere)")
        fig.colorbar(im, ax=ax[1])
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print(f"saved {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
