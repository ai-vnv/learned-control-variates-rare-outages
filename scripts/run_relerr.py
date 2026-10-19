"""Estimated outage probability of every Surabaya feeder as the simulation sample grows.

Each feeder gets a charging station and the reserve that meets the one-day-in-ten-years
target exactly, so the true daily outage probability is 2.7e-4 on every feeder. The
estimate of that probability is then computed with
  * the Gaussian load model of the earlier work (deterministic, no simulation),
  * plain Monte Carlo (MC) on n simulated days,
  * the learned control variate (boosted trees, tail-focused training) on the same days,
for R independent replications per feeder and n from 2,000 to 1,000,000 days. The feeder
model, the learner and the evaluator are set up exactly as in run_map.py (same seeds).
Writes results/relerr.json, which make_gif.py turns into the animated map.
Runtime: about 4 minutes on 2 cores.
"""
import json, time
import numpy as np
from pathlib import Path
from scipy import optimize
from scipy.special import ndtr
import lcv

ROOT = Path(__file__).resolve().parents[1]
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

TARGET = 1.0 / 3652.5            # one blackout day in ten years
R = 40                           # replications per feeder (replication 0 is drawn on the map)
N_MAX = 1_000_000
GRID = np.unique(np.round(np.logspace(np.log10(2_000), np.log10(N_MAX), 31)).astype(int))
CHUNK = 250_000

STATIONS, P_J = lcv.load_feeders(ROOT / "data" / "feeders.csv")

# --- identical to run_map.py ---------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------

feeders = {}
for name, (mu, sig, z) in STATIONS.items():
    h0 = (z - mu) / sig
    s = optimize.brentq(lambda s: sim.theta_exact(m_S + h0 * s) - (1 - P_J[name]), 0.3, 3.0, xtol=1e-10)
    h_true = optimize.brentq(lambda h: sim.theta_exact(m_S + h * s) - TARGET, 2.0, 12.0, xtol=1e-8)
    kappa = m_S + h_true * s
    feeders[name] = dict(h_true=h_true, kappa=kappa, theta_true=sim.theta_exact(kappa),
                         theta_gauss=float(lcv.Phibar(h_true)),
                         gamma=float(np.mean(ev.w * ndtr(-(kappa - sh_ev) / gbm.sigma))))
    log(f"{name:20s} h_true={h_true:.3f} true {feeders[name]['theta_true']:.2e}  Gaussian model {feeders[name]['theta_gauss']:.2e}")


def cv_from_sums(n, sy, sc, scc, syc, gamma):
    my, mc = sy / n, sc / n
    varC = scc / n - mc * mc
    cov = syc / n - my * mc
    if varC <= 0:
        return my
    return my - (cov / varC) * (mc - gamma)


for fi, (name, f) in enumerate(feeders.items()):
    kappa, gamma = f["kappa"], f["gamma"]
    est = {k: np.empty((R, len(GRID))) for k in ("mc", "lcv")}
    for rep in range(R):
        rr = np.random.default_rng(10_000 * (fi + 1) + rep)
        acc = np.zeros(4); done, gi = 0, 0
        while done < N_MAX:
            b = min(CHUNK, N_MAX - done)
            X, S, _ = sim.sample(b, rr)
            Y = (S >= kappa).astype(float)
            C = ndtr(-(kappa - gbm.shat(X)) / gbm.sigma)
            csum = np.cumsum(np.column_stack([Y, C, C * C, Y * C]), axis=0)
            while gi < len(GRID) and GRID[gi] <= done + b:
                n = GRID[gi]; t = acc + csum[n - done - 1]
                est["mc"][rep, gi] = t[0] / n
                est["lcv"][rep, gi] = cv_from_sums(n, t[0], t[1], t[2], t[3], gamma)
                gi += 1
            acc += csum[-1]; done += b
    f["mc"] = est["mc"].tolist(); f["lcv"] = est["lcv"].tolist()
    rel = {k: np.abs(est[k][:, -1] / f["theta_true"] - 1) for k in est}
    log(f"{name:20s} done; median |relative error| at {N_MAX:,} days: MC {np.median(rel['mc']):.3f}, "
        f"learned CV {np.median(rel['lcv']):.3f}")

json.dump(dict(target=TARGET, reps=R, grid=GRID.tolist(), feeders=feeders), open(ROOT / "results" / "relerr.json", "w"))
log("done")
