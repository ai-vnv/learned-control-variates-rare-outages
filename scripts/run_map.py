"""City-wide blackout map experiment for the 11 Surabaya feeders.  Writes results/map.json.

For each feeder j (masked mean mu_j, sd sigma_j, rating z_j, reported reliability p_j), the daily
peak load is mu_j + sigma_j * S~_j with S~_j the standardised nonlinear severity scaled so that
P(S~_j >= h0_j) = 1 - p_j at the current headroom h0_j = (z_j - mu_j)/sigma_j.  A charging station is
added on every feeder and the planner sizes the reserve for the one-day-in-ten-years target with
(a) the Gaussian load model of the earlier work, (b) plain MC with n days and the point estimate,
(c) the learned control variate with the same n days and a 95% upper confidence limit.
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

# Table I of Akhra, Vanany, Arief (masked units): mean, sd, rating, reported reliability
STATIONS, P_J = lcv.load_feeders(ROOT / "data" / "feeders.csv")
TARGET = 1.0 / 3652.5
H_GRID = np.round(np.arange(1.9, 6.01, 0.05), 3)
N_CAMPAIGN = 10_000
R = 200
Z95 = stats.norm.ppf(0.95)

sim = lcv.Simulator()
rng = np.random.default_rng(4242)
Sp = sim.severity(sim.draw_features(2_000_000, rng), rng)
m_S = float(Sp.mean())

def tail_features(n, rng):
    X = np.empty((n, 3)); X[:, 0] = 2.0 + 1.5 * rng.standard_normal(n); X[:, 1] = 1.0 + 1.5 * rng.standard_normal(n)
    X[:, 2] = rng.random(n) < 0.15; return X
Xtr = tail_features(10_000, rng); Str = sim.severity(Xtr, rng)
Xv = tail_features(2_000, rng); Sv = sim.severity(Xv, rng)
gbm = lcv.GBMScore(seed=0).fit(Xtr, Str, m_S + 4.0, Xv, Sv)
ev = lcv.ISEvaluator(lcv.Simulator(kappa=m_S + 4.0), n=2_000_000, seed=17)
sh_ev = gbm.shat(ev.X)

feeders = {}
for name, (mu, sig, z) in STATIONS.items():
    h0 = (z - mu) / sig
    th_now = 1 - P_J[name]
    s = optimize.brentq(lambda s: sim.theta_exact(m_S + h0 * s) - th_now, 0.3, 3.0, xtol=1e-10)
    kap = lambda h, s=s: m_S + h * s
    th_exact = np.array([sim.theta_exact(kap(h)) for h in H_GRID])
    h_true = optimize.brentq(lambda h: sim.theta_exact(kap(h)) - TARGET, 2.0, 12.0, xtol=1e-8)
    h_gauss = float(stats.norm.isf(TARGET))
    gamma = np.array([float(np.mean(ev.w * ndtr(-(kap(h) - sh_ev) / gbm.sigma))) for h in H_GRID])
    feeders[name] = dict(mu=mu, sigma=sig, z=z, h0=h0, theta_now=th_now, s=s, h_true=h_true, h_gauss=h_gauss,
                         theta_at_gauss=sim.theta_exact(kap(h_gauss)), theta_exact=th_exact, gamma=gamma)
    log(f"{name:20s} h0={h0:.2f} s={s:.3f} h_true={h_true:.3f} days/yr at Gaussian design={365*feeders[name]['theta_at_gauss']:.2f}")

def theta_of(name, h):
    return float(np.interp(h, H_GRID, feeders[name]["theta_exact"]))

results = {}
for name, f in feeders.items():
    rng_f = np.random.default_rng(sum(map(ord, name)))
    kap = lambda h, s=f["s"]: m_S + h * s
    rec = {k: [] for k in ["mc_plug", "mc_ucl", "lcv_plug", "lcv_ucl"]}
    for rep in range(R):
        X, S, _ = sim.sample(N_CAMPAIGN, rng_f); sh = gbm.shat(X)
        est = {"mc": np.empty(len(H_GRID)), "lcv": np.empty(len(H_GRID))}; se = {k: np.empty(len(H_GRID)) for k in est}
        for i, h in enumerate(H_GRID):
            Y = (S >= kap(h)).astype(float)
            est["mc"][i], se["mc"][i] = lcv.mc_estimate(Y)
            C = ndtr(-(kap(h) - sh) / gbm.sigma)
            e, s_, _, _ = lcv.cv_estimate(Y, C, gamma=f["gamma"][i]); est["lcv"][i], se["lcv"][i] = e, s_
        for k in ["mc", "lcv"]:
            for rule in ["plug", "ucl"]:
                thr = est[k] if rule == "plug" else est[k] + Z95 * se[k]
                ok = np.flatnonzero(thr <= TARGET)
                hs = float(H_GRID[ok[0]]) if len(ok) else float(H_GRID[-1])
                rec[f"{k}_{rule}"].append((hs, theta_of(name, hs)))
    results[name] = {k: dict(h=[a for a, b in v], theta=[b for a, b in v]) for k, v in rec.items()}
    d = results[name]
    log(f"{name:20s} mc_plug mean days/yr {365*np.mean(d['mc_plug']['theta']):.3f}  lcv_ucl {365*np.mean(d['lcv_ucl']['theta']):.3f}  "
        f"first campaign: mc_plug {365*d['mc_plug']['theta'][0]:.3f}, lcv_ucl {365*d['lcv_ucl']['theta'][0]:.3f}")

out = dict(target=TARGET, n_campaign=N_CAMPAIGN, reps=R, h_grid=H_GRID.tolist(), m_S=m_S,
           feeders={k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in feeders.items()},
           decisions=results)
json.dump(out, open(os.path.join(OUT, "map.json"), "w"))
log("done")
