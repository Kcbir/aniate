"""``ant()``: confirm that aniate and its dependencies are installed and working.

    from aniate import ant
    ant()

Prints the Fibonacci drawing, the installed versions of Python and of each
dependency, and the result of a short self-test, then returns True when every
required component works.  ``python -m aniate`` runs the same check.
"""

from __future__ import annotations

import importlib.metadata as metadata
import re
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["ant"]

PYTHON = "3.10"
REQUIRED = [("numpy", "1.24"), ("scipy", "1.12")]
OPTIONAL = [("matplotlib", "3.8", "plots in aniate.vis"),
            ("pytest", "7", "running the tests"),
            ("hypothesis", "6", "property-based tests")]


def _parse(version):
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


def _installed(name):
    """Installed version of a distribution, or None."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _status(version, minimum, absent):
    if version is None:
        return absent
    return "ok" if _parse(version) >= _parse(minimum) else "too old"


def _self_test(plots):
    """Build, solve, simulate and check a five-state corridor.  Returns (passed, summary line)."""
    import aniate as an
    from aniate import log

    start = time.perf_counter()
    steps, passed = [], True
    with log.capture("warning"):
        m = an.from_functions(
            states=range(5), actions=["left", "right"],
            transition=lambda s, a: {max(s - 1, 0): 1.0} if a == "left" else {min(s + 1, 4): 0.8, s: 0.2},
            reward=lambda s, a, s2: 10.0 if s2 == 4 else -1.0,
            gamma=0.9, terminal=[4], initial=0)
        steps.append("model valid")

        sol = m.solve()
        exact = float(m.initial @ an.evaluate(m, sol))
        solved = abs(sol.start_value - exact) < 1e-9
        steps.append(f"solved, value {exact:.6g}" if solved else "solver disagrees with exact evaluation")

        runs = an.simulate(m, sol, episodes=2000, seed=0)
        agree = abs(runs.mean_return - exact) < 5 * runs.stderr
        steps.append(f"{len(runs):,} episodes simulated" if agree else "simulation disagrees with the exact value")

        report = an.check(m, policy=sol, episodes=300, seed=0)
        steps.append("check passed" if report.ok else "check failed")
        passed = solved and agree and report.ok

        if plots:
            import matplotlib.pyplot as plt

            from aniate import vis

            with tempfile.TemporaryDirectory() as tmp:
                pdf = vis.save(vis.overview(runs), Path(tmp) / "overview.pdf").read_bytes().startswith(b"%PDF")
            plt.close("all")
            steps.append("PDF plot written" if pdf else "PDF plot failed")
            passed = passed and pdf

    steps.append(f"{(time.perf_counter() - start) * 1000:.0f} ms")
    return passed, " | ".join(steps)


def ant(stream=None):
    """Print the installation report and self-test; return True when aniate is working."""
    from aniate import __version__, art

    out = stream if stream is not None else sys.stdout

    def write(line=""):
        out.write(line + "\n")

    if not art.shown:                       # importing aniate may already have drawn it
        out.write(art.ART.lstrip("\n"))
    write()
    write(f"aniate {__version__}   Approximate Numerics in Applied Transition Environments")
    write()

    python = ".".join(str(x) for x in sys.version_info[:3])
    rows = [("python", python, _status(python, PYTHON, "missing"), f"required, {PYTHON} or later")]
    for name, minimum in REQUIRED:
        rows.append((name, _installed(name), _status(_installed(name), minimum, "missing"),
                     f"required, {minimum} or later"))
    for name, minimum, purpose in OPTIONAL:
        rows.append((name, _installed(name), _status(_installed(name), minimum, "not installed"),
                     f"optional, for {purpose}"))
    for name, version, status, note in rows:
        write(f"{name:<12}{version or '-':<12}{status:<15}{note}")
    write()

    required_ok = all(status == "ok" for _, _, status, note in rows if note.startswith("required"))
    plots = dict((name, status) for name, _, status, _ in rows)["matplotlib"] == "ok"
    try:
        passed, summary = _self_test(plots)
    except Exception as exc:                # report any failure instead of raising it
        passed, summary = False, f"raised {type(exc).__name__}: {exc}"
    write(f"{'self-test':<12}{summary}")
    if not plots:
        write(f"{'':<12}plots unavailable; install them with: pip install 'aniate[vis]'")
    write()

    working = required_ok and passed
    write("aniate is working." if working else "aniate is not working correctly; see the lines above.")
    out.flush()
    return working
