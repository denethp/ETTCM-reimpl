"""Evaluation helpers (Eq. 12 REE, VL protocol of the official global_divergence.hpp)."""
import numpy as np


def ree_at_timestamps(t_us, div_est, rho, ts_us, div_gt, official=True):
    """Divergence REE (%) as the official benchmark computes it.

    For each GT timestamp, take the global divergence of the first event (with
    rho > 0) whose timestamp is >= the GT timestamp. REE_k = |est - gt| / gt.

    official=True also reproduces a quirk of the official C++ evaluation
    (src/global_divergence.hpp): after the last GT value, reading past the end
    of the file leaves the last (timestamp, divergence) unchanged, so the last
    GT point is scored a second time at the next valid event. This is what the
    paper's Table 2 numbers include (verified: 3D, Translation, s=2 gives
    6.5456 % with the quirk vs the official 6.5453 %, and 6.674 % without).

    Returns (mean REE %, per-measurement REE %, estimates used).
    """
    valid = rho > 0
    tv, dv = t_us[valid], div_est[valid]
    idx = np.searchsorted(tv, ts_us, side="left")
    ok = idx < len(tv)
    idx = idx[ok]
    gt = np.asarray(div_gt, dtype=np.float64)[ok]
    if official and len(idx) and idx[-1] + 1 < len(tv):
        idx = np.append(idx, idx[-1] + 1)
        gt = np.append(gt, gt[-1])
    est = dv[idx]
    ree = 100.0 * np.abs(est - gt) / np.abs(gt)
    return float(ree.mean()), ree, est


def read_vl_sequence(folder):
    """A VL sequence folder of the official release: events.es, divergence.txt, timestamps.txt."""
    import os

    from .io_es import read_es

    ev, w, h = read_es(os.path.join(folder, "events.es"))
    div_gt = np.loadtxt(os.path.join(folder, "divergence.txt")).ravel()
    ts = np.loadtxt(os.path.join(folder, "timestamps.txt")).ravel().astype(np.uint64)
    n = min(len(div_gt), len(ts))
    return ev, w, h, ts[:n], div_gt[:n]
