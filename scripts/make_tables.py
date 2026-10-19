"""LaTeX tables for the paper from results/results.json."""
import json, os
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

R = json.load(open(os.path.join(ROOT, "results", "results.json")))
TAB = os.path.join(ROOT, "tables")
os.makedirs(TAB, exist_ok=True)
TH = R["settings"]["thetas"]
TS = [str(t) for t in TH]
E0, E2, E3, E5 = R["E0"], R["E2"], R["E3"], R["E5"]


def sci(t):
    e = int(round(np.log10(float(t))))
    return f"$10^{{{e}}}$"


# Table: main ablation
rows = []
rows.append(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llcccccc@{}}")
rows.append(r"\toprule")
rows.append(r"$\theta$ & Estimator & $\rho^2$ & $\frac{1}{1-\rho^2}$ & RMSE & theory & bias & cov.\\")
rows.append(r"\midrule")
names = {"mc": "Monte Carlo", "gauss": "Gaussian proxy", "stump": "Stump (weak)", "linear": "Linear (weak)",
         "gbm": "Boosted trees", "oracle": "Bayes posterior"}
for s in TS:
    first = True
    n = E2[s]["n"]
    for k in ["mc", "gauss", "stump", "linear", "gbm", "oracle"]:
        r2 = 0.0 if k == "mc" else E0[s][k]
        e = E2[s][k]
        lab = (sci(s) if first else "")
        rows.append(f"{lab} & {names[k]} & {r2:.3f} & {1/(1-r2):.2f} & {e['rel_rmse']:.3f} & "
                    f"{E2[s]['theory_rel_se'][k]:.3f} & {e['rel_bias']:+.3f} & {e['coverage']:.2f}\\\\")
        first = False
    if s != TS[-1]:
        rows.append(r"\midrule")
rows.append(r"\bottomrule")
rows.append(r"\end{tabular*}")
open(os.path.join(TAB, "tab_ablation.tex"), "w").write("\n".join(rows) + "\n")

# Table: binary rates of the stump and hard-thresholded learners (Theorem 1 / Corollary 1)
rows = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccccccc@{}}", r"\toprule",
        r"$\theta$ & $t$ & $\gamma$ & $\tau$ & $\phi$ & $\pi$ & $\tau\pi$ & $\rho^2$ & $1-\mathrm{BER}$\\", r"\midrule"]
for s in TS:
    tau, fpr, prec, gam = E0[s]["stump_rates"]
    th = float(s)
    r2 = (tau - fpr) ** 2 * th * (1 - th) / (gam * (1 - gam))
    ber = 0.5 * ((1 - tau) + fpr)
    rows.append(f"{sci(s)} & {E0[s]['stump_t']:.2f} & {gam:.1e} & {tau:.3f} & {fpr:.1e} & {prec:.3f} & {tau*prec:.3f} & {r2:.3f} & {1-ber:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(TAB, "tab_stump.tex"), "w").write("\n".join(rows) + "\n")

# Table: ceiling ablation (a0)
rows = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccccccc@{}}", r"\toprule",
        r"$a_0$ & $\kappa$ & $\rho^2_{\max}$ & Gaussian proxy & Stump & Linear & Boosted trees & Bayes posterior\\", r"\midrule"]
for a0 in ["0.15", "0.3", "0.6"]:
    e = E5[a0]
    rows.append(f"{float(a0):.2f} & {e['kappa']:.2f} & {e['ceiling']:.3f} & {e['gauss']:.3f} & {e['stump']:.3f} & {e['linear']:.3f} & {e['gbm']:.3f} & {e['oracle']:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(TAB, "tab_ceiling.tex"), "w").write("\n".join(rows) + "\n")

# Table: training design effect
rows = [r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccc@{}}", r"\toprule",
        r"$\theta$ & Linear, nom. & Linear, tail & Boosted, nom. & Boosted, tail\\", r"\midrule"]
for s in TS:
    rows.append(f"{sci(s)} & {E0[s]['linear_nom']:.3f} & {E0[s]['linear']:.3f} & {E0[s]['gbm_nom']:.3f} & {E0[s]['gbm']:.3f}\\\\")
rows += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(TAB, "tab_design.tex"), "w").write("\n".join(rows) + "\n")

# a few numbers for the text
txt = {}
txt["gap_e1"] = R["E1"]["largest_gap"]
txt["max_seeddiff"] = max(E0[s].get(k + "_seeddiff", 0) for s in TS for k in ["stump", "linear", "gbm", "oracle", "ceiling"])
txt["is_theta_relerr_max"] = max(abs(E0[s]["theta_is"] / float(s) - 1) for s in TS)
txt["E3_rows"] = [(r["events"], {k: (r[k]["var_ratio"], r[k]["rel_bias"], r[k]["coverage"]) for k in ["mc", "stump", "linear", "gbm", "oracle"]}) for r in E3["rows"]]
txt["E4"] = list(zip(R["E4"]["ratios"], R["E4"]["theory"], R["E4"]["empirical"]))
txt["E4_known"] = R["E4"]["known_mean_empirical"]
txt["training"] = R["training"]
txt["stump_events"] = {s: E0[s]["stump_events_train"] for s in TS}
txt["events_mean"] = {s: E2[s]["mean_events"] for s in TS}
txt["gauss_t"] = {s: E0[s]["gauss_t"] for s in TS}
txt["gauss_rates"] = {s: E0[s]["gauss_rates"] for s in TS}
print(json.dumps(txt, indent=1, default=str))
