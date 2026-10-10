"""Evaluation helpers (Eq. 12 REE, VL protocol of the official global_divergence.hpp)."""
import numpy as np


def ree_at_timestamps(t_us, div_est, rho, ts_us, div_gt):
    """Divergence REE (%) as the official benchmark computes it.

    For each GT timestamp, take the global divergence of the first event (with
    rho > 0) whose timestamp is >= the GT timestamp. REE_k = |est - gt| / gt.
    Returns (mean REE %, per-timestamp REE %, estimates used).
    """
    valid = rho > 0
    tv, dv = t_us[valid], div_est[valid]
    idx = np.searchsorted(tv, ts_us, side="left")
    ok = idx < len(tv)
    est = dv[idx[ok]]
    gt = np.asarray(div_gt)[ok]
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
