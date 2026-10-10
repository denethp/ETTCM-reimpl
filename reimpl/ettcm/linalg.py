"""Tiny dense solver for the (n x n, n <= 7) normal equations.

The official code uses Eigen's LLT (Cholesky). (Psi + L) is symmetric positive
definite because Psi = sum w J^T J >= 0 and L > 0 on the diagonal.
Written by hand so it works inside Numba without SciPy/LAPACK.
"""
import math

from ._jit import njit


@njit
def cholesky_solve(A, b, n, x, Lw):
    """Solve A[:n,:n] x = b[:n]. Lw is an (n, n) scratch array. Returns False if not SPD."""
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= Lw[i, k] * Lw[j, k]
            if i == j:
                if s <= 0.0:
                    return False
                Lw[i, i] = math.sqrt(s)
            else:
                Lw[i, j] = s / Lw[j, j]
    # forward: L y = b
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= Lw[i, k] * x[k]
        x[i] = s / Lw[i, i]
    # backward: L^T x = y
    for i in range(n - 1, -1, -1):
        s = x[i]
        for k in range(i + 1, n):
            s -= Lw[k, i] * x[k]
        x[i] = s / Lw[i, i]
    return True
