"""Shared figure style: TrueType fonts embedded as Type 42, transparent background.

The paper figures use Arial. Arial is not redistributed here; if it is installed (or its
.ttf files are in the folder named by the environment variable LCV_FONT_DIR) it is used,
otherwise the metric-compatible Liberation Sans or Arimo, and finally DejaVu Sans.
"""
import os

import matplotlib
from matplotlib import font_manager

PREFERRED = ["Arial", "Liberation Sans", "Arimo", "DejaVu Sans"]


def _register_font_dir():
    font_dir = os.environ.get("LCV_FONT_DIR")
    if font_dir and os.path.isdir(font_dir):
        for f in os.listdir(font_dir):
            if f.lower().endswith((".ttf", ".otf")):
                font_manager.fontManager.addfont(os.path.join(font_dir, f))


def family():
    """First preferred family that matplotlib can find."""
    _register_font_dir()
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in PREFERRED:
        if name in available:
            return name
    return "DejaVu Sans"


def apply(base=8, transparent=True):
    fam = family()
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": fam, "font.sans-serif": [fam],
        "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold", "mathtext.default": "it",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base, "legend.fontsize": base - 1.2,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    })
    if transparent:
        matplotlib.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                                    "savefig.facecolor": "none", "savefig.transparent": True})
    return fam
