"""Running estimate over many independent campaigns (theta = 1e-4), for the spread of the estimate
against campaign length.  Writes results/running_multi.json: for each grid n, the MC and CV estimates
of every campaign and the plug-in 95% half-widths."""
import json, math, os, sys, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import lcv

THETA = 1e-4
N_MAX = 2_000_000
R = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rng = np.random.default_rng(20261019)
sim = lcv.Simulator().calibrate(THETA)

def tail_features(n, rng):
    X = np.empty((n, 3)); X[:, 0] = 2.0 + 1.5 * rng.standard_normal(n); X[:, 1] = 1.0 + 1.5 * rng.standard_normal(n)
    X[:, 2] = rng.random(n) < 0.15; return X
Xtr = tail_features(10_000, rng); Str = sim.severity(Xtr, rng)
Xv = tail_features(2_000, rng); Sv = sim.severity(Xv, rng)
gbm = lcv.GBMScore(seed=0).fit(Xtr, Str, sim.kappa, Xv, Sv)
ev = lcv.ISEvaluator(sim, n=4_000_000, seed=5)
gamma = float(np.mean(ev.w * gbm(ev.X)))
rho2 = ev.rho2(gbm(ev.X), EC=gamma)[0]

grid = np.unique(np.round(np.logspace(np.log10(2e4), np.log10(N_MAX), 40)).astype(int))
CH = 200_000
out = dict(theta=THETA, rho2=rho2, gamma=gamma, grid=grid.tolist(), mc=[], mc_se=[], cv=[], cv_se=[], events=[])
t0 = time.time()
for rep in range(R):
    rr = np.random.default_rng(1000 + rep)
    sy = sc = scc = syc = 0.0; done = 0; gi = 0
    rows = {k: [] for k in ("mc", "mc_se", "cv", "cv_se", "events")}
    while done < N_MAX:
        b = min(CH, N_MAX - done)
        X, S, Y = sim.sample(b, rr); C = gbm(X)
        cy = np.cumsum(Y); cc = np.cumsum(C); ccc = np.cumsum(C * C); cyc = np.cumsum(Y * C)
        while gi < len(grid) and grid[gi] <= done + b:
            k = grid[gi] - done; n = grid[gi]
            Sy = sy + cy[k - 1]; Sc = sc + cc[k - 1]; Scc = scc + ccc[k - 1]; Syc = syc + cyc[k - 1]
            my = Sy / n; mc = Sc / n; varY = my * (1 - my); varC = Scc / n - mc * mc; cov = Syc / n - my * mc
            beta = cov / varC if varC > 0 else 0.0
            resid = max(varY - 2 * beta * cov + beta * beta * varC, 0.0)
            rows["mc"].append(my); rows["mc_se"].append(math.sqrt(varY / n))
            rows["cv"].append(my - beta * (mc - gamma)); rows["cv_se"].append(math.sqrt(resid / n)); rows["events"].append(int(Sy))
            gi += 1
        sy += cy[-1]; sc += cc[-1]; scc += ccc[-1]; syc += cyc[-1]; done += b
    for k in rows: out[k].append(rows[k])
    if rep % 5 == 0: print(f"rep {rep} {time.time()-t0:.0f}s final mc {365*rows['mc'][-1]:.4f} cv {365*rows['cv'][-1]:.4f}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "running_multi.json"), "w"))
print("done", time.time() - t0)
