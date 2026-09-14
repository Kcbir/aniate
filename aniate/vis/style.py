"""The one look: black Computer Modern text on white, thin lines, nothing else.

Computer Modern (``cmr10``) is the LaTeX default typeface and ships with
matplotlib, so figures match the body text of a paper with no extra
installs.  Fonts are set on every artist directly, not through global
``rcParams``, so importing aniate never changes anyone else's figures.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["save"]

FONT = "cmr10"
SIZE = 9
INK = "black"
GRAY = "0.45"
LIGHT = "0.88"
ACCENT = "#1f4e8c"
PALETTE = ["#1f4e8c", "#b35806", "#1b7837", "#762a83", "#8c510a", "#01665e", "#c51b7d", "#4d4d4d"]

# Applied only while saving, so anything formatted at draw time matches too.
RC = {"font.family": FONT, "mathtext.fontset": "cm", "axes.unicode_minus": False,
      "pdf.fonttype": 42, "font.weight": "normal", "text.color": INK}


def text(size=SIZE):
    """Keyword arguments for plain black Computer Modern text."""
    return {"family": FONT, "size": size, "color": INK, "weight": "normal", "style": "normal",
            "math_fontfamily": "cm"}


_ESCAPE = {"\\": r"\backslash", "{": r"\{", "}": r"\}", "$": r"\$", "#": r"\#", "%": r"\%",
           "&": r"\&", "^": r"\wedge", "~": r"\sim", " ": r"\ ", "_": r"\_"}


def plain(label):
    """A user label, drawn in Computer Modern even though cmr10 has no underscore.

    Lines containing ``_`` go through mathtext as ``\\mathrm{...}`` with the
    underscore escaped; everything else is left exactly as written.
    """
    lines = []
    for line in str(label).split("\n"):
        if "_" in line:
            line = r"$\mathrm{" + "".join(_ESCAPE.get(ch, ch) for ch in line) + "}$"
        lines.append(line)
    return "\n".join(lines)


def number(x, pos=None):
    """Tick label with an ASCII minus (cmr10 has no Unicode minus sign)."""
    s = f"{x:.6g}"
    return "0" if s == "-0" else s


def axes(ax=None, figsize=(4.2, 3.0)):
    """A plain Axes: the one given, or a new one."""
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        from aniate.vis import _plt

        _, ax = _plt().subplots(figsize=figsize)
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.6)
    ticks(ax)
    ax.xaxis.set_major_formatter(FuncFormatter(number))
    ax.yaxis.set_major_formatter(FuncFormatter(number))
    return ax


def ticks(ax):
    ax.tick_params(colors=INK, labelcolor=INK, labelsize=SIZE - 1, length=3, width=0.6,
                   labelfontfamily=FONT)


def labels(ax, x=None, y=None):
    if x is not None:
        ax.set_xlabel(x, **text())
    if y is not None:
        ax.set_ylabel(y, **text())


def legend(ax, items, loc="best"):
    """``items``: (label, color, linestyle) triples.  Frameless, black text."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=col, ls=ls, lw=1.2, label=lab) for lab, col, ls in items]
    ax.legend(handles=handles, loc=loc, frameon=False, handlelength=2.0,
              prop={"family": FONT, "size": SIZE - 1}, labelcolor=INK)


def save(figure, path):
    """Save a figure (or the figure of an Axes) as a PDF with embedded fonts.

    Returns the path written.  Only PDF is supported: vector output that
    drops straight into LaTeX.
    """
    from matplotlib.figure import Figure

    from aniate import log as _log
    from aniate.vis import _plt

    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".pdf")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"aniate saves figures as PDF; got {path.name!r}")
    import logging

    fig = figure if isinstance(figure, Figure) else figure.figure
    # Embedding cmr10 makes fontTools complain about the font file's 1990s
    # timestamps; the PDF is unaffected, so keep that noise out of the logs.
    font_log = logging.getLogger("fontTools")
    level = font_log.level
    font_log.setLevel(logging.ERROR)
    try:
        with _plt().rc_context(RC):
            fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")
    finally:
        font_log.setLevel(level)
    _log.info("vis", f"saved {path}")
    return path
