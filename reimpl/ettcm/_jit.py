"""Optional Numba JIT.

`njit` compiles with Numba when it is installed, otherwise it is a no-op so the
code still runs (slowly) in plain Python. Debug in plain Python first:
    ETTCM_NO_JIT=1 python scripts/run_synthetic.py
"""
import os

try:
    if os.environ.get("ETTCM_NO_JIT"):
        raise ImportError
    from numba import njit as _njit

    def njit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", False)
        return _njit(*args, **kwargs)

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f
