"""Blackout maps of Surabaya (Fig. 2) and the city-wide table (Table II) from results/map.json.

Inputs: data/feeders.csv, data/demand.csv, data/travel_time.csv, data/basemap/, and the
sub-district boundaries in data/boundaries/surabaya_kecamatan.geojson, which
scripts/fetch_boundaries.py downloads (GADM 4.1 is not redistributed here). Without the
boundaries the map falls back to one disc per sub-district.
"""
import json, os, math, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import lcv

M = json.load(open(os.path.join(ROOT, "results", "map.json")))
CASE = lcv.load_surabaya(ROOT / "data")
FIG = os.path.join(ROOT, "figures"); TAB = os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

from lcv import figstyle
figstyle.apply(8)
plt.rcParams.update({"axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})

target_days = 365 * M["target"]
names = list(M["feeders"].keys())
st_lat, st_lon = CASE["feeder_lat"], CASE["feeder_lon"]
kec = CASE["subdistricts"]; d_lat, d_lon = CASE["sub_lat"], CASE["sub_lon"]
demand, pop = CASE["demand"], CASE["population"]
# serving feeder: nearest candidate station by motorcycle travel time
serve = {k: min(names, key=lambda j: CASE["minutes"][j][k]) for k in kec}

def days_scenario(which, rep=0):
    out = {}
    for j in names:
        if which == "gauss":
            out[j] = 365 * M["feeders"][j]["theta_at_gauss"]
        else:
            out[j] = 365 * M["decisions"][j][which]["theta"][rep]
    return out

scen = [("gauss", "(a)"), ("mc_plug", "(b)"), ("lcv_ucl", "(c)")]   # descriptions are in the caption
cmap = LinearSegmentedColormap.from_list("wk", ["#ffffff", "#808080", "#000000"])   # white, grey, black
norm = LogNorm(vmin=target_days / 3, vmax=target_days * 3)   # 0.033 ... 0.30 days per year, target in the middle

# --- basemap of the earlier work (Carto Positron render cached in ai-vnv/rare-event-ev-charging) ---
BM = os.path.join(ROOT, "data", "basemap")
bm_meta = json.load(open(os.path.join(BM, "surabaya_canvas.json")))
bm_img = plt.imread(os.path.join(BM, "surabaya_canvas.jpg"))
bm_extent = (bm_meta["west"], bm_meta["east"], bm_meta["south"], bm_meta["north"])
WEST, EAST, SOUTH, NORTH = 112.56, 112.845, -7.374, -7.185      # city window inside the render
ASPECT = 1 / math.cos(math.radians(-7.28))
panel_w = 2.3; panel_h = panel_w * (NORTH - SOUTH) * ASPECT / (EAST - WEST)
LABEL_OFFSET = {"Embong Wungu": (-6, 5, "center"), "Ngagel": (6, -5, "center"), "Indrapura": (3, -6, "center"),
                "Rayon Perak": (-3, 6, "center")}   # nudges for the two pairs of feeders that sit close together

# --- sub-district polygons (GADM 4.1 level 3, fetched by scripts/fetch_boundaries.py) ---
import difflib
from matplotlib.patches import Polygon as MplPolygon
GJ_PATH = ROOT / "data" / "boundaries" / "surabaya_kecamatan.geojson"
poly_of = None
if GJ_PATH.exists():
    GJ = json.load(open(GJ_PATH))
    def _key(s): return "".join(ch for ch in s.lower() if ch.isalpha())
    gadm_names = [f["properties"]["NAME_3"] for f in GJ["features"]]
    poly_of = {}
    for k in kec:
        m = difflib.get_close_matches(_key(k), [_key(g) for g in gadm_names], n=1, cutoff=0.6)
        assert m, k
        poly_of[k] = GJ["features"][[_key(g) for g in gadm_names].index(m[0])]["geometry"]
    assert len(set(id(v) for v in poly_of.values())) == len(kec)
else:
    print(f"note: {GJ_PATH.relative_to(ROOT)} not found; run scripts/fetch_boundaries.py for the choropleth. "
          "Drawing one disc per sub-district instead.", file=sys.stderr)
def rings(geom):
    if geom["type"] == "Polygon":
        yield geom["coordinates"][0]
    else:
        for part in geom["coordinates"]:
            yield part[0]

fig, axes = plt.subplots(1, 3, figsize=(7.1, panel_h + 0.75))
for ax, (key, title) in zip(axes, scen):
    days = days_scenario(key)
    ax.imshow(bm_img, extent=bm_extent, zorder=0, interpolation="bilinear", aspect="auto")
    # colour every sub-district by the blackout rate of its serving feeder
    for k in kec:
        v = min(max(days[serve[k]], norm.vmin), norm.vmax)
        if poly_of is not None:
            for ring in rings(poly_of[k]):
                ax.add_patch(MplPolygon(ring, closed=True, facecolor=cmap(norm(v)), edgecolor="#ffffff", lw=0.35, alpha=0.92, zorder=2))
        else:
            ax.scatter(d_lon[k], d_lat[k], s=60, color=cmap(norm(v)), edgecolor="#555555", lw=0.4, zorder=2)
    for j in names:   # value label of each feeder, placed at the feeder position (locations are fixed, not optimised)
        dx, dy, ha = LABEL_OFFSET.get(j, (0, 0, "center"))
        u = norm(min(max(days[j], norm.vmin), norm.vmax))
        dark = u > 0.55   # fill darker than mid grey: white text, else black; thin halo in the fill colour hides boundary lines behind the text
        ax.annotate(f"{days[j]:.2f}", (st_lon[j], st_lat[j]), xytext=(dx, dy), textcoords="offset points", fontsize=6,
                    ha=ha, va="center", color="#ffffff" if dark else "#000000", zorder=5,
                    path_effects=[pe.withStroke(linewidth=1.6, foreground=cmap(u))])
    ax.set_xlim(WEST, EAST); ax.set_ylim(SOUTH, NORTH)
    ax.set_aspect(ASPECT)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for side in ax.spines.values():
        side.set_edgecolor("#999999"); side.set_linewidth(0.5)
    ax.set_title(title, loc="left", fontsize=8, pad=3)
    ax.text(0.995, 0.01, bm_meta["attribution"], transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2, color="#333333", zorder=6)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.05, aspect=45)
cbar.set_label("expected blackout days per year (target 0.10)", fontsize=6.5)
cbar.set_ticks([target_days / 3, target_days / 2, target_days, 2 * target_days, 3 * target_days])
cbar.set_ticklabels(["0.033", "0.05", "0.10", "0.20", "0.30"])
cbar.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); cbar.ax.minorticks_off()
cbar.ax.tick_params(labelsize=6.5)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.2, wspace=0.04)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_map.{ext}"), dpi=300, bbox_inches="tight", transparent=True)

# city-wide statistics over all replications
tot_demand = sum(demand.values()); tot_pop = sum(pop.values())
stats_rows = {}
for key, lab in [("gauss", "Gaussian model"), ("mc_plug", "MC, point est."), ("mc_ucl", "MC, 95\\% limit"),
                 ("lcv_plug", "CV, point est."), ("lcv_ucl", "CV, 95\\% limit")]:
    if key == "gauss":
        per = {j: np.array([365 * M["feeders"][j]["theta_at_gauss"]]) for j in names}
    else:
        per = {j: 365 * np.array(M["decisions"][j][key]["theta"]) for j in names}
    city_days = sum(per[j].mean() for j in names)
    frac_above = np.mean(np.concatenate([per[j] > target_days for j in names]))
    frac_2x = np.mean(np.concatenate([per[j] > 2 * target_days for j in names]))
    worst = np.mean([per[j].max() if key == "gauss" else np.quantile(per[j], 0.9) for j in names])
    exposure = sum(demand[k] * per[serve[k]].mean() for k in kec) / tot_demand   # demand-weighted days/yr
    pop_above = np.mean([sum(pop[k] for k in kec if per[serve[k]][r] > target_days) for r in range(len(per[names[0]]))]) / tot_pop
    stats_rows[key] = dict(label=lab, city_days=city_days, frac_above=frac_above, frac_2x=frac_2x, q90=worst, exposure=exposure, pop_above=pop_above)
    print(f"{lab:30s} city feeder-blackout days/yr {city_days:.2f}; feeders above target {frac_above:.2f}; above 2x {frac_2x:.2f}; "
          f"90th pct per feeder {worst:.3f}; demand-weighted days/yr {exposure:.3f}; population served by a feeder above target {pop_above:.2f}")
rows = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccc@{}}", r"\toprule",
        r"Procedure & city days/yr & $>$target & $>2\times$ & 90th pct. & pop.\\", r"\midrule"]
for key in ["gauss", "mc_plug", "lcv_plug", "mc_ucl", "lcv_ucl"]:
    s = stats_rows[key]
    rows.append(f"{s['label']} & {s['city_days']:.2f} & {s['frac_above']:.2f} & {s['frac_2x']:.2f} & {s['q90']:.3f} & {s['pop_above']:.2f}\\\\")
rows += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(TAB, "tab_map.tex"), "w").write("\n".join(rows) + "\n")
json.dump(stats_rows, open(os.path.join(ROOT, "results", "map_stats.json"), "w"), indent=1)
print("first-campaign values (b):", {j: round(days_scenario('mc_plug')[j], 3) for j in names})
print("first-campaign values (c):", {j: round(days_scenario('lcv_ucl')[j], 3) for j in names})
