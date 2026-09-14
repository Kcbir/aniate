"""Plain, rule-based logs: every operation says what it did, in numbers.

    aniate | model    | 16 states | 4 actions | gamma 0.95 | horizon inf | 60 transitions | valid
    aniate | simulate | 10,000 episodes | 71,406 steps | 0.41 s | mean return 0.4731 +/- 0.0021

Levels: ``"quiet"``, ``"warning"``, ``"info"`` (default), ``"debug"`` (one
line per environment step).  Set with ``aniate.verbosity(...)`` or the
environment variable ``ANIATE_LOG``.  Output goes to stderr, ASCII only.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys

__all__ = ["verbosity", "capture"]

LEVELS = {"quiet": logging.CRITICAL + 10, "warning": logging.WARNING,
          "info": logging.INFO, "debug": logging.DEBUG}

logger = logging.getLogger("aniate")
logger.propagate = False


class _Stderr(logging.Handler):
    """Writes to whatever sys.stderr is *now* (notebooks and test runners swap it)."""

    def emit(self, record):
        try:
            sys.stderr.write(record.getMessage() + "\n")
        except Exception:  # pragma: no cover
            self.handleError(record)


_stream = _Stderr()
logger.addHandler(_stream)


def verbosity(level=None):
    """Get or set how much aniate logs: "quiet", "warning", "info" or "debug"."""
    if level is None:
        return next((k for k, v in LEVELS.items() if v == logger.level), "info")
    if level not in LEVELS:
        raise ValueError(f"verbosity must be one of {list(LEVELS)}; got {level!r}")
    logger.setLevel(LEVELS[level])
    return level


verbosity(os.environ.get("ANIATE_LOG", "info") if os.environ.get("ANIATE_LOG") in LEVELS else "info")


def enabled(level):
    return logger.isEnabledFor(LEVELS[level])


def _line(topic, parts):
    return f"aniate | {topic:<8} | " + " | ".join(str(p) for p in parts if p not in (None, ""))


def debug(topic, *parts):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(_line(topic, parts))


def info(topic, *parts):
    if logger.isEnabledFor(logging.INFO):
        logger.info(_line(topic, parts))


def warning(topic, *parts):
    if logger.isEnabledFor(logging.WARNING):
        logger.warning(_line(topic, parts))


@contextlib.contextmanager
def capture(level="debug"):
    """Collect log lines in a list instead of printing them.

    >>> with aniate.log.capture() as lines:
    ...     model.solve()
    """
    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler, old = _Collect(), logger.level
    logger.removeHandler(_stream)
    logger.addHandler(handler)
    logger.setLevel(LEVELS[level])
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
        logger.addHandler(_stream)
        logger.setLevel(old)
