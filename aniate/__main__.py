"""``python -m aniate`` (or ``aniate``): the installation report and self-test, as ``ant()``."""

from __future__ import annotations


def main():
    from aniate import ant

    return 0 if ant() else 1


if __name__ == "__main__":
    raise SystemExit(main())
