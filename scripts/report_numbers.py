"""Print every number quoted in the paper from the stored results and write
results/paper_numbers.md. Run after the experiments (or on the shipped results)."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
R = json.load(open(RES / "results.json"))
RUN = json.load(open(RES / "running_multi.json"))
SUR = json.load(open(RES / "surabaya.json"))
MAP = json.load(open(RES / "map.json"))
MST = json.load(open(RES / "map_stats.json"))
TH = ["0.001", "0.0001", "1e-05", "1e-06"]
lines = []


def out(section, text):
    lines.append((section, text))


def rng_(vals, fmt="{:.2f}"):
    return f"{fmt.format(min(vals))} to {fmt.format(max(vals))}"


# --- Section V-B: variance reduction (exact rho^2, Fig. 1a) ---------------------------------
E0 = R["E0"]
out("V-B", f"Gaussian proxy rho^2 at 1e-3: {E0['0.001']['gauss']:.4f} (< 0.2%); at 1e-4..1e-6: "
           + ", ".join(f"{E0[t]['gauss']:.1e}" for t in TH[1:]))
E6 = R["E6"]
out("V-B", f"hypothetical proxy r=0.98: rho^2 {E6['0.98']['0.001']['rho2']:.2f} -> {E6['0.98']['1e-06']['rho2']:.2f}; "
           f"r=0.9: {E6['0.9']['0.001']['rho2']:.3f} -> {E6['0.9']['1e-06']['rho2']:.3f}")
for k, lab in (("stump", "stump"), ("linear", "linear"), ("gbm", "boosted trees"), ("oracle", "ceiling (Bayes posterior)")):
    v = [E0[t][k] for t in TH]
    out("V-B", f"{lab}: rho^2 {rng_(v)}, runs saved 1/(1-rho^2) {rng_([1 / (1 - x) for x in v])}")
gb = [E0[t]["gbm"] for t in TH]; ce = [E0[t]["ceiling"] for t in TH]
out("V-B", f"boosted trees within {100 * max(1 - g / c for g, c in zip(gb, ce)):.1f}% of the ceiling")
for t in TH:
    tau, fpr, prec, gam = E0[t]["stump_rates"]
    out("V-B", f"stump at theta={t}: recall {tau:.3f} x precision {prec:.3f} = {tau * prec:.3f} vs exact rho^2 {E0[t]['stump']:.3f}")
for t in TH:
    tau, fpr, prec, gam = E0[t]["gauss_rates"]
    out("V-B", f"Gaussian proxy at theta={t}: false-positive rate {fpr:.3f}, precision {prec:.4f}")
out("V-B", "nominal-trained boosted trees rho^2: " + ", ".join(f"{E0[t]['gbm_nom']:.2f}" for t in TH)
           + f"; nominal linear at 1e-3: {E0['0.001']['linear_nom']:.2f}")

# --- Table I ------------------------------------------------------------------------------
E2 = R["E2"]
for k in ("mc", "gbm", "linear", "stump"):
    out("Table I", f"{k}: relative RMSE {rng_([E2[t][k]['rel_rmse'] for t in TH])}, coverage "
                   f"{rng_([E2[t][k]['coverage'] for t in TH])}")
out("Table I", "replications per level: " + ", ".join(f"{t}: {R['settings']['reps'][t]}" for t in TH))
z = stats.norm.ppf(0.975)
n_mc = z ** 2 * (1 - 1e-6) / (1e-6 * 0.01)
out("V-B", f"runs for 10% relative error at 1e-6: MC {n_mc:.2e}, boosted-tree CV {n_mc * (1 - E0['1e-06']['gbm']):.2e}; "
           f"training {R['settings']['n_train'] + R['settings']['n_val']:,} runs")

# --- Fig. 1b: 400 replications at theta = 1e-4 ----------------------------------------------
g = np.array(RUN["grid"]); mc = 365 * np.array(RUN["mc"]); cv = 365 * np.array(RUN["cv"])
half = lambda a, i: (np.quantile(a[:, i], 0.975) - np.quantile(a[:, i], 0.025)) / 2
for n0 in (1e5, 1e6):
    i = int(np.argmin(np.abs(g - n0))); w = half(mc, i)
    j = next(j for j in range(len(g)) if half(cv, j) <= w)
    out("Fig. 1b", f"MC 95% range at {g[i]:,} days: +-{w:.3f} days/yr; CV reaches it at {g[j]:,} days")
out("Fig. 1b", f"true value {365 * RUN['theta']:.4f} days/yr; mean MC/CV at the last n: {mc[:, -1].mean():.4f} / {cv[:, -1].mean():.4f}")

# --- pool and ceiling ablation --------------------------------------------------------------
E4 = R["E4"]
i10 = E4["ratios"].index(10)
out("V-B", f"pool m=10n: variance ratio theory {E4['theory'][i10]:.3f}, empirical {E4['empirical'][i10]:.3f}, "
           f"known mean {E4['known_mean_empirical']:.3f}, rho^2 {E4['rho2']:.3f}")
E5 = R["E5"]
out("VI", "ceiling vs unobserved noise a0: " + ", ".join(f"a0={a}: {E5[a]['ceiling']:.2f}" for a in E5))

# --- Section V-C: Surabaya map and Embong Wungu ---------------------------------------------
fe = MAP["feeders"]
out("V-C", f"required headroom {rng_([f['h_true'] for f in fe.values()])} vs Gaussian {list(fe.values())[0]['h_gauss']:.2f}")
out("V-C", f"blackout days/yr at the Gaussian design: {rng_([365 * f['theta_at_gauss'] for f in fe.values()])}")
for k, s in MST.items():
    out("Table II", f"{s['label']}: city {s['city_days']:.2f} days/yr, >target {s['frac_above']:.2f}, >2x {s['frac_2x']:.2f}, "
                    f"90th pct {s['q90']:.3f}, population {s['pop_above']:.2f}")
rq = SUR["required"]["2.7e-4"]
out("V-C", f"Embong Wungu: exact upgrade {rq['upgrade_true_units']:,.0f} units, Gaussian {rq['upgrade_gauss_units']:,.0f}, "
           f"true theta at Gaussian design {rq['theta_true_at_h_gauss']:.2e} = {365 * rq['theta_true_at_h_gauss']:.2f} days/yr")
for n in SUR["budgets"]:
    a = SUR["decisions"][f"{n}|mc|ucl|2.7e-4"]; b = SUR["decisions"][f"{n}|lcv|ucl|2.7e-4"]
    out("V-C", f"{n:,} days, 95% limit: over-design MC {a['mean_overdesign_units']:,.0f} vs CV {b['mean_overdesign_units']:,.0f} units; "
               f"safe in {100 * (1 - a['p_unsafe']):.0f}% / {100 * (1 - b['p_unsafe']):.0f}% of replications")

# --- animated map (results/relerr.json) ----------------------------------------------------
p = RES / "relerr.json"
if p.exists():
    E = json.load(open(p)); G = np.array(E["grid"]); F = E["feeders"]
    for k in ("mc", "lcv"):
        rel = np.concatenate([np.array(F[j][k]) / F[j]["theta_true"] - 1 for j in F])
        q = np.quantile(np.abs(rel), 0.95, axis=0)
        out("GIF", f"{k}: 95% of estimates within {100 * np.interp(5, np.log10(G), q):.0f}% at 1e5 days, "
                   f"{100 * q[-1]:.0f}% at {G[-1]:,} days (all feeders, {E['reps']} replications each)")
    r = [F[j]["theta_gauss"] / F[j]["theta_true"] - 1 for j in F]
    out("GIF", f"Gaussian load model: {100 * min(r):.0f}% to {100 * max(r):.0f}% at every n")

md = ["# Numbers quoted in the paper", "", "Generated by `scripts/report_numbers.py` from `results/`.", ""]
sec = None
for s, t in lines:
    if s != sec:
        md += ["", f"## {s}", ""]; sec = s
    md.append(f"- {t}")
open(RES / "paper_numbers.md", "w").write("\n".join(md) + "\n")
print("\n".join(f"[{s}] {t}" for s, t in lines))
