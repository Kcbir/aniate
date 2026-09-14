"""Fibonacci squares 1, 1, 2, 3, 5, 8 tiling a 13 x 8 rectangle, with the golden spiral through them.

Printed to stderr once per process when aniate is imported, and by
``python -m aniate``.  Set ``ANIATE_BANNER=0`` to switch the import banner off.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ART", "banner"]

ART = """
+-------------------------------+-------------------+
|                  ......       |     .....         |
|              ....             |          ...      |
|           ....                |            ...    |
|         ...                   |              ...  |
|       ...                     |                .. |
|     ...                       |                 ..|
|    ..                         |                  .|
|   ..                          |                  .|
|  ..                           |                   |
| ..                            +---+---+-----------+
|..                             |.  |  .|           |
|.                              +---+---+          .|
|.                              |.      |         ..|
|                               |..     |       ... |
|                               | ...   |    ....   |
+-------------------------------+-------+-----------+
1 1 2 3 5 8 13        F(n+1) / F(n) -> 1.618...
"""


shown = False       # set once the import banner has been printed in this process


def banner(stream=None):
    """Write the art to ``stream`` (default stderr)."""
    global shown
    stream = stream if stream is not None else sys.stderr
    stream.write(ART.lstrip("\n"))
    stream.flush()
    shown = True


def _on_import():
    if os.environ.get("ANIATE_BANNER", "1").strip().lower() not in ("0", "false", "no", "off"):
        try:
            banner()
        except (OSError, ValueError, AttributeError):   # closed or missing stderr
            pass
