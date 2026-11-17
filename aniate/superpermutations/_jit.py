"""The engine is compiled with numba when it is installed (``pip install 'aniate[fast]'``) and runs as plain
Python otherwise. The plain form gives the same answers, between fifty and a hundred times more slowly."""

try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover - depends on the installation
    _njit = None

COMPILED = _njit is not None


def jit(fn):
    return _njit(cache=True)(fn) if COMPILED else fn


def need_compiled(what):
    if not COMPILED:
        raise ImportError(f"{what} needs numba: pip install 'aniate[fast]'")
