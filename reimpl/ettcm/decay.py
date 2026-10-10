"""Adaptive global decay (Nunes et al., CVPR 2023, ref [31]); mirrors ETTCM/global_decay.hpp.

State per stream: t_last (us), n (a decayed event count = the "global activity" a(t)).
For every event (timestamp t in microseconds):

    if t > t_last:
        decay = 1 / (1e-6 * (t - t_last) * n + 1)
        n     = n * decay
        t_last = t
    else:
        decay = 1
    n += 1

`n` is used as a(t) in Eq. (11):  w_j(t) = 1 / (1 + a(t) * (t - t_j))  (t in seconds)
`decay` is used to exponentially-ish average the per-event divergence into a
global divergence (see GlobalDivergence in estimator.py).
"""
from ._jit import njit


@njit
def decay_step(t_us, t_last_us, n):
    """One update. Returns (decay, t_last_us, n). All floats except t's (int)."""
    if t_us > t_last_us:
        decay = 1.0 / (1e-6 * float(t_us - t_last_us) * n + 1.0)
        n = n * decay
        t_last_us = t_us
    else:
        decay = 1.0
    n += 1.0
    return decay, t_last_us, n
