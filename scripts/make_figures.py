"""Figures for the ICSGSC 2026 paper from results/results.json.
Colour-blind-safe (Okabe-Ito) palette, distinct line styles and markers so the
figures survive black-and-white printing, transparent backgrounds, TrueType fonts."""
import json, os, math
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator

RES = json.load(open(os.path.join(ROOT, "results", "results.json")))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

from lcv import figstyle
figstyle.apply(8)

# Okabe-Ito; fixed assignment per estimator, with a line style and marker each
STY = {
    "mc":      dict(color="#000000", ls="-",  marker="o", label="Monte Carlo"),
    "gauss":   dict(color="#E69F00", ls="--", marker="v", label="Gaussian proxy CV (prior work)"),
    "stump":   dict(color="#56B4E9", ls="-.", marker="s", label="Stump CV (weak)"),
    "linear":  dict(color="#009E73", ls=":",  marker="D", label="Linear CV (weak, misspecified)"),
    "gbm":     dict(color="#D55E00", ls="-",  marker="^", label="Boosted-tree CV (strong)"),
    "oracle":  dict(color="#CC79A7", ls=(0, (4, 1, 1, 1)), marker="x", label="Bayes posterior CV (ceiling)"),
    "gbm_nom": dict(color="#0072B2", ls=(0, (6, 2)), marker="P", label="Boosted-tree CV, nominal training"),
}
THETAS = [float(t) for t in RES["settings"]["thetas"]]
TH_STR = [str(t) for t in RES["settings"]["thetas"]]


def line(ax, x, y, key, **kw):
    s = dict(STY[key]); s.update(kw)
    return ax.plot(x, y, **s)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: theory check (stump threshold sweep at theta = 1e-3)
# ---------------------------------------------------------------------------
E1 = RES["E1"]
fig, ax = plt.subplots(figsize=(3.45, 2.2))
t = np.array(E1["t"])
ax.plot(t, E1["theory"], color=STY["gbm"]["color"], lw=1.8, ls="-", label=r"Theorem 1: $1-\rho^2$")
ax.plot(t, E1["rare_limit"], color=STY["oracle"]["color"], lw=1.4, ls=(0, (1, 1)),
        label=r"Corollary 1: $1-\mathrm{recall}\times\mathrm{precision}$")
ax.plot(t, E1["empirical"], ls="none", marker="o", mfc="none", color="#000000",
        label=f"empirical ({E1['reps']} replications, $n$={E1['n']:,})")
ax.set_xlabel(r"stump threshold $t$ in $g(x)=1\{x_{\mathrm{temp}} \geq t\}$")
ax.set_ylabel(r"variance ratio $\mathrm{Var}(\hat\theta^{\mathrm{CV}})/\mathrm{Var}(\hat\theta^{\mathrm{MC}})$")
ax.set_ylim(0.6, 1.05)
ax.legend(loc="lower left")
save(fig, "fig_theory_check")

# ---------------------------------------------------------------------------
# Fig 2: main ablation (three panels, two-column width)
# ---------------------------------------------------------------------------
E0, E2, E6 = RES["E0"], RES["E2"], RES["E6"]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.05))
# (a) rho^2 vs theta
ax = axes[0]
for k in ["gauss", "stump", "linear", "gbm", "gbm_nom", "oracle"]:
    y = [E0[s][k] for s in TH_STR]
    line(ax, THETAS, y, k)
for r, ls in (("0.98", (0, (1, 1))), ("0.9", (0, (3, 2)))):
    y = [E6[r][s]["rho2"] for s in TH_STR]
    ax.plot(THETAS, y, color="#7f7f7f", ls=ls, lw=1.1, marker="none", label=f"Gaussian proxy with r = {r}, exact (Prop. 1)")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("outage probability θ (rarer →)")
ax.set_ylabel("variance reduction ρ²")
ax.xaxis.set_major_locator(FixedLocator(THETAS))
ax.set_ylim(-0.02, 0.72)
ax.set_title("(a)", loc="left")
# (b) running estimate as the campaign grows (theta = 1e-4): median and 95% range over 400 independent campaigns
RUN = json.load(open(os.path.join(ROOT, "results", "running_multi.json")))
ax = axes[1]
nn = np.array(RUN["grid"])
for k, col, lab, ls in (("mc", STY["mc"]["color"], "Monte Carlo", (0, (4, 2))), ("cv", STY["gbm"]["color"], "Boosted-tree CV, same days", "-")):
    est = 365 * np.array(RUN[k])
    lo, med, hi = np.quantile(est, [0.025, 0.5, 0.975], axis=0)
    ax.fill_between(nn, lo, hi, color=col, alpha=0.22 if k == "cv" else 0.14, lw=0,
                    hatch=None if k == "cv" else "////", edgecolor=col, zorder=1)
    ax.plot(nn, med, color=col, lw=1.5, ls=ls, label=lab, zorder=3)
ax.axhline(365 * RUN["theta"], color="#000000", lw=0.8, ls=(0, (1, 2)), label="true value")
ax.set_xscale("log")
ax.set_xlabel("number of simulated days")
ax.set_ylabel("estimated blackout days per year")
ax.set_ylim(0, 0.10)
ax.set_title("(b)", loc="left")
# (c) runs needed for 10% relative error
ax = axes[2]
from scipy import stats
z = stats.norm.ppf(0.975)
tt = np.logspace(-6, -3, 60)
for k in ["mc", "gauss", "stump", "linear", "gbm", "oracle"]:
    rho = np.interp(np.log10(tt), np.log10(THETAS)[::-1], np.array([E0[s][k] if k != "mc" else 0.0 for s in TH_STR])[::-1])
    n_req = z ** 2 * (1 - tt) / (tt * 0.1 ** 2) * (1 - rho)
    s = dict(STY[k]); s.pop("marker")
    ax.plot(tt, n_req, **s)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("outage probability θ (rarer →)")
ax.set_ylabel("simulator runs for 10% rel. error")
ax.xaxis.set_major_locator(FixedLocator(THETAS))
ax.set_title("(c)", loc="left")
from matplotlib.lines import Line2D
h, l = axes[0].get_legend_handles_labels()
h = [Line2D([], [], color=STY["mc"]["color"], ls="-", marker="o", mfc="none")] + h; l = ["Monte Carlo"] + l
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.2), handlelength=2.6)
fig.tight_layout(w_pad=1.2)
save(fig, "fig_ablation")

# ---------------------------------------------------------------------------
# Fig 3: few-events regime and unlabelled pool
# ---------------------------------------------------------------------------
E3, E4 = RES["E3"], RES["E4"]
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
ax = axes[0]
ev = np.array(E3["events"])
for k in ["stump", "linear", "gbm", "oracle"]:
    vr = [row[k]["var_ratio"] for row in E3["rows"]]
    s = dict(STY[k]); s.pop("label")
    ax.axhline(1 - E3["rho2"][k], color=s["color"], ls=s["ls"], lw=0.9, alpha=0.8)
    ax.plot(ev, vr, ls="none", marker=s["marker"], color=s["color"], mfc="none", label=STY[k]["label"])
ax.axhline(1.0, color="#000000", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"expected number of observed outages $n\theta$ ($\theta=10^{-4}$)")
ax.set_ylabel("empirical variance ratio CV / MC")
ax.set_ylim(0.27, 3.2)
ax.set_yticks([0.3, 0.5, 1, 2, 3]); ax.set_yticklabels(["0.3", "0.5", "1", "2", "3"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title(r"(a) few-events regime (lines: $1-\rho^2$)", loc="left")
ax.legend(loc="upper right", ncol=1, bbox_to_anchor=(1.0, 1.0))
ax = axes[1]
q = np.array(E4["ratios"])
ax.plot(q, E4["theory"], color=STY["gbm"]["color"], lw=1.8, label=r"Theorem 3(b): $1-\rho^2\,m/(n+m)$")
ax.plot(q, E4["empirical"], ls="none", marker="o", mfc="none", color="#000000",
        label=f"empirical ({E4['reps']} replications, $n$={E4['n']:,})")
ax.axhline(1 - E4["rho2"], color=STY["oracle"]["color"], ls=(0, (1, 1)), lw=1.2, label=r"known control mean: $1-\rho^2$")
ax.axhline(1.0, color="#000000", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"pool size / labelled sample, $m/n$ ($\theta=10^{-3}$)")
ax.set_ylabel("variance ratio CV / MC")
ax.set_ylim(0.3, 1.05)
ax.set_title("(b) control mean from an unlabelled pool", loc="left")
ax.legend(loc="upper right")
fig.tight_layout(w_pad=1.5)
save(fig, "fig_regimes")
print("figures written to", FIG)
