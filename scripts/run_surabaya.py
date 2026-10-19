"""Surabaya case: headroom decision for the Embong Wungu feeder with a fixed EV charging
station.  Writes results/surabaya.json.

Feeder model (masked units of the utility): daily peak load Z = mu + sigma * S~, where S~ is the
standardised nonlinear severity of lcv.Simulator, scaled so that P(S~ >= 2) = 0.023, the
utility-reported outage probability at the current rating z = mu + 2 sigma.  The planner chooses
the headroom h (feeder upgrade or charging cap) so that theta(h) = P(S~ >= h) <= theta*.
"""
import json, math, os, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from scipy import stats, optimize
from scipy.special import ndtr
import lcv

OUT = os.path.join(ROOT, "results")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

MU, SIG, Z = 28979.0, 3622.0, 36224.0          # Embong Wungu, masked units (Akhra et al.)
THETA_NOW = 0.023                                # 1 - 0.977 reported reliability
H0 = (Z - MU) / SIG                              # = 2.0
TARGETS = {"1e-3": 1e-3, "2.7e-4": 1.0 / 3652.5, "1e-4": 1e-4}
H_GRID = np.round(np.arange(2.0, 6.01, 0.05), 3)
BUDGETS = [20_000, 100_000, 500_000]
R = 200
N_TRAIN, N_VAL = 10_000, 2_000

sim = lcv.Simulator()
rng = np.random.default_rng(777)

# --- standardisation of S: m = E[S]; s such that P(S >= m + 2 s) = THETA_NOW ----------------
Xp, Sp, _ = sim.sample(4_000_000, rng)
m_S = float(Sp.mean())
q = float(np.quantile(Sp, 1 - THETA_NOW))
s_S = (q - m_S) / H0
# refine s_S with the exact quadrature so that theta_exact(m + 2 s) = THETA_NOW
s_S = optimize.brentq(lambda s: sim.theta_exact(m_S + H0 * s) - THETA_NOW, 0.5 * s_S, 2 * s_S, xtol=1e-10)
kappa_of_h = lambda h: m_S + h * s_S
log(f"m_S={m_S:.4f} s_S={s_S:.4f}; check theta(h0)={sim.theta_exact(kappa_of_h(H0)):.4f}")

# --- exact curves ---------------------------------------------------------------------------
theta_exact = np.array([sim.theta_exact(kappa_of_h(h)) for h in H_GRID])
theta_gauss = ndtr(-H_GRID)
required = {}
for name, t in TARGETS.items():
    h_true = optimize.brentq(lambda h: sim.theta_exact(kappa_of_h(h)) - t, 2.0, 12.0, xtol=1e-8)
    h_gauss = float(stats.norm.isf(t))
    required[name] = dict(target=t, h_true=h_true, h_gauss=h_gauss,
                          theta_true_at_h_gauss=sim.theta_exact(kappa_of_h(h_gauss)),
                          upgrade_true_units=(h_true - H0) * SIG, upgrade_gauss_units=(h_gauss - H0) * SIG)
    log(f"target {name}: h_true={h_true:.3f} ({(h_true-H0)*SIG:,.0f} units), h_gauss={h_gauss:.3f}, "
        f"theta at h_gauss={required[name]['theta_true_at_h_gauss']:.2e} "
        f"({365*required[name]['theta_true_at_h_gauss']:.2f} days/yr vs target {365*t:.3f})")

# --- learners (tail-focused design), as in the main study ------------------------------------
def tail_features(n, rng):
    X = np.empty((n, 3))
    X[:, 0] = 2.0 + 1.5 * rng.standard_normal(n)
    X[:, 1] = 1.0 + 1.5 * rng.standard_normal(n)
    X[:, 2] = rng.random(n) < 0.15
    return X
Xtr = tail_features(N_TRAIN, rng); Str = sim.severity(Xtr, rng)
Xv = tail_features(N_VAL, rng); Sv = sim.severity(Xv, rng)
gbm = lcv.GBMScore(seed=0).fit(Xtr, Str, kappa_of_h(4.0), Xv, Sv)
lin = lcv.LinearScore().fit(Xtr, Str, kappa_of_h(4.0))
def score(model, X, h):
    return ndtr(-(kappa_of_h(h) - model.shat(X)) / model.sigma)

# --- control means and exact rho^2 per h (importance sampling, evaluation only) --------------
sim_mid = lcv.Simulator(kappa=kappa_of_h(4.0))
ev = lcv.ISEvaluator(sim_mid, n=4_000_000, seed=99)
shat_ev = {"gbm": gbm.shat(ev.X), "linear": lin.shat(ev.X)}
gamma = {k: np.empty(len(H_GRID)) for k in shat_ev}; gamma_relse = {k: np.empty(len(H_GRID)) for k in shat_ev}
rho2 = {k: np.empty(len(H_GRID)) for k in shat_ev}
for i, h in enumerate(H_GRID):
    th = theta_exact[i]
    eta_h = sim.eta(ev.X, kappa=kappa_of_h(h))
    for k, sh in shat_ev.items():
        model = gbm if k == "gbm" else lin
        C = ndtr(-(kappa_of_h(h) - sh) / model.sigma)
        w = ev.w
        EC = float(np.mean(w * C)); EC2 = float(np.mean(w * C * C)); ECe = float(np.mean(w * C * eta_h))
        gamma[k][i] = EC; gamma_relse[k][i] = float(np.std(w * C) / math.sqrt(ev.n)) / EC
        varC = EC2 - EC ** 2; cov = ECe - th * EC
        rho2[k][i] = cov ** 2 / (th * (1 - th) * varC) if varC > 0 else 0.0
log(f"gamma rel. SE max: gbm {gamma_relse['gbm'].max():.3%}, linear {gamma_relse['linear'].max():.3%}")
log("rho2 gbm at h=2,3,4,5,6: " + ", ".join(f"{rho2['gbm'][np.argmin(abs(H_GRID-h))]:.3f}" for h in [2, 3, 4, 5, 6]))

# --- decision study ------------------------------------------------------------------------
def campaign(n, rng):
    """One simulation campaign of n days: MC and LCV curves over H_GRID with plug-in SEs."""
    X, S, _ = sim.sample(n, rng)
    sh = gbm.shat(X)
    out = {"mc": (np.empty(len(H_GRID)), np.empty(len(H_GRID))), "lcv": (np.empty(len(H_GRID)), np.empty(len(H_GRID)))}
    for i, h in enumerate(H_GRID):
        Y = (S >= kappa_of_h(h)).astype(float)
        out["mc"][0][i], out["mc"][1][i] = lcv.mc_estimate(Y)
        C = ndtr(-(kappa_of_h(h) - sh) / gbm.sigma)
        e, se, _, _ = lcv.cv_estimate(Y, C, gamma=gamma["gbm"][i])
        out["lcv"][0][i], out["lcv"][1][i] = e, se
    return out

def decide(est, se, target, rule):
    thr = est if rule == "plugin" else est + stats.norm.ppf(0.95) * se
    ok = np.flatnonzero(thr <= target)
    return H_GRID[ok[0]] if len(ok) else float(H_GRID[-1])

z95 = stats.norm.ppf(0.95)
decisions = {}
example = None
rng_d = np.random.default_rng(2027)
for n in BUDGETS:
    log(f"budget n={n}")
    hs = {(k, rule, tname): [] for k in ["mc", "lcv"] for rule in ["plugin", "ucl"] for tname in TARGETS}
    for rep in range(R):
        c = campaign(n, rng_d)
        if n == 100_000 and rep == 0:
            example = {k: (c[k][0].tolist(), c[k][1].tolist()) for k in c}
        for k in ["mc", "lcv"]:
            for rule in ["plugin", "ucl"]:
                for tname, t in TARGETS.items():
                    hs[(k, rule, tname)].append(decide(c[k][0], c[k][1], t, rule))
    for (k, rule, tname), lst in hs.items():
        hstar = np.array(lst); t = TARGETS[tname]; h_true = required[tname]["h_true"]
        th_true = np.array([sim.theta_exact(kappa_of_h(h)) for h in hstar])
        decisions[f"{n}|{k}|{rule}|{tname}"] = dict(
            n=n, estimator=k, rule=rule, target=tname,
            mean_h=float(hstar.mean()), sd_h=float(hstar.std(ddof=1)),
            mean_overdesign_sigma=float(np.mean(hstar - h_true)),
            mean_overdesign_units=float(np.mean(hstar - h_true) * SIG),
            rmse_h=float(np.sqrt(np.mean((hstar - h_true) ** 2))),
            p_unsafe=float(np.mean(th_true > t)),
            mean_outage_days_per_year=float(365 * th_true.mean()),
            p_no_feasible=float(np.mean(hstar >= H_GRID[-1])))
        if tname == "2.7e-4":
            d = decisions[f"{n}|{k}|{rule}|{tname}"]
            log(f"  {k:3s} {rule:6s} target 2.7e-4: h*={d['mean_h']:.3f}+-{d['sd_h']:.3f} overdesign={d['mean_overdesign_units']:+,.0f} units "
                f"unsafe={d['p_unsafe']:.2f} days/yr={d['mean_outage_days_per_year']:.3f}")

res = dict(mu=MU, sigma=SIG, z=Z, theta_now=THETA_NOW, h0=H0, m_S=m_S, s_S=s_S,
           h_grid=H_GRID.tolist(), theta_exact=theta_exact.tolist(), theta_gauss=theta_gauss.tolist(),
           rho2={k: v.tolist() for k, v in rho2.items()}, gamma_relse_max={k: float(v.max()) for k, v in gamma_relse.items()},
           required=required, budgets=BUDGETS, reps=R, decisions=decisions, example_campaign_n100k=example)
json.dump(res, open(os.path.join(OUT, "surabaya.json"), "w"), indent=1)
log("done")
