"""Animated blackout map: estimated outage probability of every Surabaya feeder as the
number of simulated days grows.

Reads results/relerr.json (scripts/run_relerr.py). Every feeder carries a charging station
and the reserve that meets the one-day-in-ten-years target exactly, so the true daily outage
probability is 2.7e-4 everywhere and the "true" panel is one uniform colour. The other panels
colour each sub-district by its feeder's estimate: the Gaussian load model of the earlier
work, plain Monte Carlo, and the learned control variate (replication 0 of each feeder).
Any colour different from the true panel is estimation error.

Writes figures/relerr_map_light.gif and figures/relerr_map_dark.gif, plus a PNG of the
last frame of each. Needs data/boundaries/ (scripts/fetch_boundaries.py).
"""
import difflib
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Polygon as MplPolygon

import lcv
from lcv import figstyle

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "results" / "relerr.json"))
CASE = lcv.load_surabaya(ROOT / "data")
GJ_PATH = ROOT / "data" / "boundaries" / "surabaya_kecamatan.geojson"
if not GJ_PATH.exists():
    sys.exit("data/boundaries/surabaya_kecamatan.geojson is missing: run python scripts/fetch_boundaries.py")

GRID = np.array(D["grid"])
TARGET = D["target"]
FEED = D["feeders"]
names = list(FEED)
REP = 0                                                   # replication drawn on the map
HOLD = 8                                                  # extra frames on the last sample size

# sub-district polygons and serving feeder (nearest by motorcycle travel time), as in make_map.py
GJ = json.load(open(GJ_PATH))
def _key(s): return "".join(ch for ch in s.lower() if ch.isalpha())
gadm = [_key(f["properties"]["NAME_3"]) for f in GJ["features"]]
kec = CASE["subdistricts"]
poly_of = {k: GJ["features"][gadm.index(difflib.get_close_matches(_key(k), gadm, n=1, cutoff=0.6)[0])]["geometry"]
           for k in kec}
serve = {k: min(names, key=lambda j: CASE["minutes"][j][k]) for k in kec}
def rings(geom):
    parts = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for p in parts:
        yield p[0]

BM = ROOT / "data" / "basemap"
bm_meta = json.load(open(BM / "surabaya_canvas.json"))
bm_img = plt.imread(BM / "surabaya_canvas.jpg")[::3, ::3]  # downsampled: smaller GIF, same look
bm_extent = (bm_meta["west"], bm_meta["east"], bm_meta["south"], bm_meta["north"])
WEST, EAST, SOUTH, NORTH = 112.56, 112.845, -7.374, -7.185
ASPECT = 1 / math.cos(math.radians(-7.28))

cmap = LinearSegmentedColormap.from_list("wk", ["#ffffff", "#808080", "#000000"])   # paper colours
norm = LogNorm(vmin=TARGET / 10, vmax=TARGET * 10)

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", sub="#57606a"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", sub="#8b949e"),
}


def theta_hat(method, gi):
    if method == "true":
        return {j: FEED[j]["theta_true"] for j in names}
    if method == "gauss":
        return {j: FEED[j]["theta_gauss"] for j in names}
    return {j: FEED[j][method][REP][gi] for j in names}


def spread(method, gi):
    """Relative error that 95% of all estimates (every feeder, every replication) stay within."""
    rel = np.concatenate([np.array(FEED[j][method])[:, gi] / FEED[j]["theta_true"] - 1 for j in names])
    return 100 * np.quantile(np.abs(rel), 0.95)


def colour(v):
    return cmap(norm(min(max(v, norm.vmin), norm.vmax)))


def render(theme_name):
    th = THEMES[theme_name]
    figstyle.apply(9, transparent=False)
    plt.rcParams.update({"figure.facecolor": th["bg"], "savefig.facecolor": th["bg"], "text.color": th["fg"],
                         "xtick.color": th["fg"], "axes.labelcolor": th["fg"]})
    panels = [("true", "True value"), ("gauss", "Gaussian load model (earlier work)"),
              ("mc", "Monte Carlo"), ("lcv", "Learned control variate")]
    pw = 2.5; ph = pw * (NORTH - SOUTH) * ASPECT / (EAST - WEST)
    fig = plt.figure(figsize=(4 * pw + 0.5, ph + 2.45), dpi=100)
    B = 1.15                                              # bottom of the map panels, inches
    W, H = fig.get_size_inches()
    axes, colls, subs = [], [], []
    for i, (key, title) in enumerate(panels):
        ax = fig.add_axes([(0.15 + i * (pw + 0.1)) / W, B / H, pw / W, ph / H])
        ax.imshow(bm_img, extent=bm_extent, zorder=0, interpolation="bilinear", aspect="auto")
        patches, owner = [], []
        for k in kec:
            for ring in rings(poly_of[k]):
                patches.append(MplPolygon(ring, closed=True)); owner.append(k)
        pc = PatchCollection(patches, edgecolor="#ffffff", linewidth=0.35, alpha=0.92, zorder=2)
        ax.add_collection(pc)
        ax.set_xlim(WEST, EAST); ax.set_ylim(SOUTH, NORTH); ax.set_aspect(ASPECT)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for side in ax.spines.values():
            side.set_edgecolor(th["sub"]); side.set_linewidth(0.5)
        ax.text(0.99, 0.01, bm_meta["attribution"], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.5, color="#333333", zorder=6)
        fig.text((0.15 + i * (pw + 0.1)) / W, (B + ph + 0.27) / H, title, fontsize=9.5, weight="bold", color=th["fg"])
        subs.append(fig.text((0.15 + i * (pw + 0.1)) / W, (B + ph + 0.08) / H, "", fontsize=7.5, color=th["sub"]))
        axes.append(ax); colls.append((pc, owner, key))
    fig.text(0.15 / W, 1 - 0.32 / H, "Estimated daily outage probability of each Surabaya feeder", fontsize=13,
             weight="bold", color=th["fg"])
    fig.text(0.15 / W, 1 - 0.55 / H, "Each feeder has a charging station and the reserve that meets the "
             "one-day-in-ten-years target exactly, so the true value is the same everywhere. "
             "Any colour that differs from the first panel is estimation error.", fontsize=7.5, color=th["sub"])
    nlabel = fig.text(0.15 / W, 0.45 / H, "", fontsize=12, weight="bold", color=th["fg"])
    cax = fig.add_axes([(0.15 + 2 * (pw + 0.1)) / W, 0.72 / H, (2 * pw + 0.1) / W, 0.12 / H])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both")
    ticks = [TARGET / 10, TARGET / 3, TARGET, 3 * TARGET, 10 * TARGET]
    cb.set_ticks(ticks); cb.ax.minorticks_off()
    cb.set_ticklabels(["2.7e-5", "9.1e-5", "2.7e-4\n(true)", "8.2e-4", "2.7e-3"])
    cb.ax.tick_params(labelsize=7, colors=th["fg"])
    cb.outline.set_edgecolor(th["sub"])
    cb.set_label("daily outage probability (lighter: underestimated, darker: overestimated)", fontsize=7.5, color=th["fg"], labelpad=2)

    def draw(frame):
        gi = min(frame, len(GRID) - 1)
        for (pc, owner, key), sub in zip(colls, subs):
            est = theta_hat(key, gi)
            pc.set_facecolor([colour(est[serve[k]]) for k in owner])
            if key == "true":
                sub.set_text("2.7e-4 per day on every feeder")
            elif key == "gauss":
                r = [FEED[j]["theta_gauss"] / FEED[j]["theta_true"] for j in names]
                sub.set_text(f"{100 * (1 - max(r)):.0f}% to {100 * (1 - min(r)):.0f}% too low, for any number of days")
            else:
                sub.set_text(f"95% of estimates within ±{spread(key, gi):.0f}% of the true value")
        nlabel.set_text(f"after {GRID[gi]:,} simulated days per feeder")
        return []

    frames = len(GRID) + HOLD
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out = ROOT / "figures" / f"relerr_map_{theme_name}.gif"
    out.parent.mkdir(exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=3))
    draw(frames - 1)
    fig.savefig(ROOT / "figures" / f"relerr_map_{theme_name}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    for name in THEMES:
        render(name)
