"""Figure and table for the Surabaya case from results/surabaya.json."""
import json, os
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

R = json.load(open(os.path.join(ROOT, "results", "surabaya.json")))
FIG = os.path.join(ROOT, "figures"); TAB = os.path.join(ROOT, "tables")

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False})
C_MC, C_LCV, C_GAUSS, C_TRUE = "#000000", "#D55E00", "#E69F00", "#0072B2"

h = np.array(R["h_grid"]); th = np.array(R["theta_exact"]); tg = np.array(R["theta_gauss"])
SIG, MU = R["sigma"], R["mu"]
target = R["required"]["2.7e-4"]
ex = R["example_campaign_n100k"]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
ax = axes[0]
ax.plot(h, th, color=C_TRUE, lw=1.8, label="feeder simulator (exact)")
ax.plot(h, tg, color=C_GAUSS, lw=1.4, ls="--", label="Gaussian load model of prior work")
sel = np.arange(0, len(h), 4)
for k, col, mk, lab, off in (("mc", C_MC, "o", "Monte Carlo: (a) estimate with 95% CI from $n=10^5$ days; (b) mean $\\pm$ sd of $h^*$", -0.03), ("lcv", C_LCV, "^", "Boosted-tree CV: (a) same days, 95% CI; (b) mean $\\pm$ sd of $h^*$", 0.03)):
    est = np.array(ex[k][0]); se = np.array(ex[k][1])
    m = (est > 0) & (h >= 3.0)
    idx = np.intersect1d(np.flatnonzero(m), sel)
    lo = np.maximum(est[idx] - 1.96 * se[idx], 1e-6)
    ax.errorbar(h[idx] + off, est[idx], yerr=[est[idx] - lo, 1.96 * se[idx]], fmt=mk, color=col, mfc="none", ms=4, lw=0.8, capsize=1.5, label=lab)
ax.axhline(target["target"], color="#555555", ls=(0, (1, 1)), lw=1.0)
ax.annotate("target: one day in ten years", (2.05, target["target"]), xytext=(0, 3), textcoords="offset points", fontsize=6.5, color="#555555")
ax.axvline(target["h_true"], color=C_TRUE, lw=0.7, ls=":"); ax.axvline(target["h_gauss"], color=C_GAUSS, lw=0.7, ls=":")
ax.set_yscale("log"); ax.set_ylim(3e-6, 5e-2); ax.set_xlim(2.0, 6.0)
ax.set_xlabel(r"headroom $h$ above the base load, in $\sigma$ units ($\sigma$ = 3,622)")
ax.set_ylabel(r"daily outage probability $\theta(h)$")
ax.set_title("(a) outage probability against headroom", loc="left", pad=26)
secax = ax.secondary_xaxis("top", functions=(lambda x: (x - 2.0) * SIG / 1000, lambda u: 2.0 + u * 1000 / SIG))
secax.set_xlabel("feeder upgrade beyond the current rating (thousand units)", fontsize=7)
secax.tick_params(labelsize=7)

ax = axes[1]
budgets = R["budgets"]; x = np.arange(len(budgets))
for k, col, mk, lab, off in (("mc", C_MC, "o", "Monte Carlo", -0.12), ("lcv", C_LCV, "^", "Boosted-tree CV", 0.12)):
    means = [R["decisions"][f"{n}|{k}|ucl|2.7e-4"]["mean_h"] for n in budgets]
    sds = [R["decisions"][f"{n}|{k}|ucl|2.7e-4"]["sd_h"] for n in budgets]
    ax.errorbar(x + off, means, yerr=sds, fmt=mk, color=col, mfc="none", capsize=3, lw=1.0, label="_nolegend_")
ax.axhline(target["h_true"], color=C_TRUE, lw=1.0, ls=":", label="required headroom, feeder simulator (exact)")
ax.axhline(target["h_gauss"], color=C_GAUSS, lw=1.0, ls="--", label="required headroom, Gaussian model of prior work")
ax.set_xticks(x); ax.set_xticklabels([f"$n=${n:,}" for n in budgets])
ax.set_xlabel("simulation campaign (days)")
ax.set_ylabel(r"chosen headroom $h^*$ ($\sigma$ units)")
ax.set_ylim(3.2, 5.4)
ax.set_title("(b) headroom chosen with a 95% upper confidence limit", loc="left")
h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=2.4)
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig_surabaya.{ext}"), dpi=300, bbox_inches="tight", transparent=True)

# table: decision quality for the 1-in-10-years target, UCL rule and plug-in rule
rows = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
        r"$n$ & Estimator & $h^*$ & over-design & unsafe & days/yr\\", r"\midrule"]
for n in budgets:
    for k, name in (("mc", "MC"), ("lcv", "Boosted CV")):
        d = R["decisions"][f"{n}|{k}|ucl|2.7e-4"]
        rows.append(f"{n:,} & {name} & ${d['mean_h']:.2f}\\pm{d['sd_h']:.2f}$ & {d['mean_overdesign_units']:,.0f} & {d['p_unsafe']:.2f} & {d['mean_outage_days_per_year']:.3f}\\\\")
    if n != budgets[-1]:
        rows.append(r"\addlinespace[1pt]")
rows.append(r"\midrule")
rows.append(f"-- & Gaussian model & ${target['h_gauss']:.2f}$ & {(target['h_gauss']-target['h_true'])*SIG:,.0f} & 1.00 & {365*target['theta_true_at_h_gauss']:.3f}\\\\")
rows.append(f"-- & Exact requirement & ${target['h_true']:.2f}$ & 0 & 0.00 & {365*target['target']:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "tab_surabaya.tex"), "w").write("\n".join(rows) + "\n")

for tname, t in R["required"].items():
    print(tname, f"h_true={t['h_true']:.3f} h_gauss={t['h_gauss']:.3f} upgrade_true={t['upgrade_true_units']:,.0f} upgrade_gauss={t['upgrade_gauss_units']:,.0f} theta_at_gauss={t['theta_true_at_h_gauss']:.2e} days/yr={365*t['theta_true_at_h_gauss']:.2f}")
for key, d in R["decisions"].items():
    if d["target"] == "2.7e-4":
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k not in ("estimator", "rule", "target", "n")})
print("rho2 gbm range", min(R["rho2"]["gbm"]), max(R["rho2"]["gbm"]))
