"""Run every experiment of the paper and write results/results.json.

E0  exact rho^2 of every control at theta in {1e-3,...,1e-6} (IS evaluation, two seeds)
E1  theory check: stump threshold sweep at theta=1e-3 (theory vs rare limit vs empirical)
E2  replication study at n = 20/theta: relative RMSE, variance ratio, CI coverage
E3  few-events regime at theta=1e-4: n*theta in {2,...,100}
E4  unlabelled pool at theta=1e-3: m/n in {0.5,...,50}
E5  feature-informativeness ablation: a0 in {0.15, 0.30, 0.60} at theta=1e-4
E6  Gaussian proxy with latent correlation r (Model A, exact, bivariate normal)
"""
import json, math, os, sys, time
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from scipy import stats
from scipy.special import ndtr
import lcv

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

THETAS = [1e-3, 1e-4, 1e-5, 1e-6]
N_TRAIN = 10_000
N_VAL = 2_000
EVENTS_PER_REP = 20
REPS = {1e-3: 400, 1e-4: 300, 1e-5: 120, 1e-6: 60}
N_IS = 4_000_000
TRAIN_DESIGN = dict(mu1=2.0, s1=1.5, mu2=1.0, s2=1.5)

results = dict(settings=dict(thetas=THETAS, n_train=N_TRAIN, n_val=N_VAL, events_per_rep=EVENTS_PER_REP,
                             reps={str(k): v for k, v in REPS.items()}, n_is=N_IS, train_design=TRAIN_DESIGN,
                             simulator=dict(a1=0.45, a2=0.20, a3=0.55, a4=0.40, a0=0.30, p_storm=0.15)))


def train_features(n, rng, design):
    if design == "nominal":
        X = lcv.Simulator().draw_features(n, rng)
        return X, np.ones(n)
    d = TRAIN_DESIGN
    X = np.empty((n, 3))
    X[:, 0] = d["mu1"] + d["s1"] * rng.standard_normal(n)
    X[:, 1] = d["mu2"] + d["s2"] * rng.standard_normal(n)
    X[:, 2] = rng.random(n) < 0.15
    logw = (-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2)
            + 0.5 * ((X[:, 0] - d["mu1"]) ** 2 / d["s1"] ** 2 + (X[:, 1] - d["mu2"]) ** 2 / d["s2"] ** 2)
            + math.log(d["s1"] * d["s2"]))
    return X, np.exp(logw)


class StumpRule:
    """Weak learner: temperature stump 1{x1 >= t}; t maximises the importance-weighted
    estimate of recall x precision on the training runs (the rare-event limit of rho^2)."""
    name = "stump"

    def fit(self, X, S, w, kappa, grid=np.linspace(1.0, 7.5, 131)):
        Y = (S >= kappa).astype(float)
        best = (-1.0, None)
        for t in grid:
            a = X[:, 0] >= t
            den = np.sum(w * a)
            if den <= 0:
                continue
            tau = np.sum(w * Y * a) / np.sum(w * Y)
            prec = np.sum(w * Y * a) / den
            if tau * prec > best[0]:
                best = (tau * prec, t)
        self.t = float(best[1]); self.crit = float(best[0])
        self.gamma = float(ndtr(-self.t))
        self.n_events_train = int(Y.sum())
        return self

    def __call__(self, X):
        return (X[:, 0] >= self.t).astype(float)


def fit_controls(sim, Xtr, Str, wtr, Xv, Sv, gbm_seed=0):
    k = sim.kappa
    ctr = {}
    ctr["stump"] = StumpRule().fit(Xtr, Str, wtr, k)
    ctr["linear"] = lcv.LinearScore().fit(Xtr, Str, k)
    ctr["gbm"] = lcv.GBMScore(seed=gbm_seed).fit(Xtr, Str, k, Xv, Sv)
    ctr["oracle"] = lcv.Oracle(sim)
    return ctr


def best_gauss_proxy(ev, grid=np.linspace(0.5, 6.0, 111)):
    best = (-1.0, None)
    for t in grid:
        g = lcv.GaussianProxy(t)
        r2 = ev.rho2(g(ev.X), EC=g.gamma)[0]
        if r2 > best[0]:
            best = (r2, t)
    return lcv.GaussianProxy(best[1])


def known_gamma(ctrl):
    return getattr(ctrl, "gamma", None)


# ---------------------------------------------------------------------------
rng = np.random.default_rng(20261019)
sim0 = lcv.Simulator()
Xtr, wtr = train_features(N_TRAIN, rng, "tail")
Str = sim0.severity(Xtr, rng)
Xv, wv = train_features(N_VAL, rng, "tail")
Sv = sim0.severity(Xv, rng)
Xtr_nom, _ = train_features(N_TRAIN, rng, "nominal")
Str_nom = sim0.severity(Xtr_nom, rng)
Xv_nom, _ = train_features(N_VAL, rng, "nominal")
Sv_nom = sim0.severity(Xv_nom, rng)
results["training"] = dict(max_x1_tail=float(Xtr[:, 0].max()), max_x1_nominal=float(Xtr_nom[:, 0].max()),
                           frac_x1_above_4_tail=float(np.mean(Xtr[:, 0] > 4)),
                           frac_x1_above_4_nominal=float(np.mean(Xtr_nom[:, 0] > 4)))

# ---------------------------------------------------------------------------
# E0: exact rho^2 per control and theta (two IS seeds)
# ---------------------------------------------------------------------------
log("E0: exact rho^2")
E0 = {}
CONTROLS = {}
for th in THETAS:
    sim = lcv.Simulator().calibrate(th)
    ctr = fit_controls(sim, Xtr, Str, wtr, Xv, Sv)
    ctr_nom = dict(linear_nom=lcv.LinearScore().fit(Xtr_nom, Str_nom, sim.kappa),
                   gbm_nom=lcv.GBMScore(seed=0).fit(Xtr_nom, Str_nom, sim.kappa, Xv_nom, Sv_nom))
    CONTROLS[th] = (sim, ctr)
    row = dict(kappa=sim.kappa, theta_exact=sim.theta_exact(), stump_t=ctr["stump"].t,
               stump_gamma=ctr["stump"].gamma, stump_events_train=ctr["stump"].n_events_train,
               linear_sigma=ctr["linear"].sigma, gbm_sigma=ctr["gbm"].sigma)
    vals = {}
    for seed in (11, 12):
        ev = lcv.ISEvaluator(sim, n=N_IS, seed=seed)
        X = ev.X
        v = dict(theta_is=ev.theta, theta_is_se=ev.theta_se, ess=ev.ess, ceiling=ev.ceiling())
        gp = best_gauss_proxy(ev)
        v["gauss_t"] = gp.t
        v["gauss"] = ev.rho2(gp(X), EC=gp.gamma)[0]
        tau, fpr, prec, gam = ev.binary_rates(gp(X), EC=gp.gamma)
        v["gauss_rates"] = (tau, fpr, prec, gam)
        for name, c in list(ctr.items()) + list(ctr_nom.items()):
            C = c(X)
            v[name] = ev.rho2(C, EC=known_gamma(c))[0]
            mo = ev.moments(C)
            v[name + "_gamma"] = known_gamma(c) if known_gamma(c) is not None else mo["EC"]
            v[name + "_gamma_relse"] = (mo["EC_se"] / mo["EC"]) if mo["EC"] > 0 else None
            if hasattr(c, "hard"):
                Ch = c.hard(X)
                v[name + "_hard"] = ev.rho2(Ch)[0]
                v[name + "_hard_rates"] = ev.binary_rates(Ch)
        tau, fpr, prec, gam = ev.binary_rates(ctr["stump"](X), EC=ctr["stump"].gamma)
        v["stump_rates"] = (tau, fpr, prec, gam)
        vals[seed] = v
    # average the two seeds, keep the spread
    merged = {}
    for k in vals[11]:
        a, b = vals[11][k], vals[12][k]
        if isinstance(a, tuple):
            merged[k] = [0.5 * (x + y) for x, y in zip(a, b)]
        elif a is None or b is None:
            merged[k] = a if a is not None else b
        else:
            merged[k] = 0.5 * (a + b)
            merged[k + "_seeddiff"] = abs(a - b)
    row.update(merged)
    E0[str(th)] = row
    log(f"  theta={th:.0e}: ceiling={row['ceiling']:.3f} gauss={row['gauss']:.4f} stump={row['stump']:.3f} "
        f"linear={row['linear']:.3f} gbm={row['gbm']:.3f} oracle={row['oracle']:.3f} "
        f"gbm_nom={row['gbm_nom']:.3f} linear_nom={row['linear_nom']:.3f} (IS theta rel err {abs(row['theta_is']/th-1):.2%})")
results["E0"] = E0

# ---------------------------------------------------------------------------
# E6: Gaussian proxy with latent correlation r (Model A), exact
# ---------------------------------------------------------------------------
log("E6: Model A Gaussian proxy, exact")
def gauss_proxy_rho2_exact(kappa, r, t):
    # Y = 1{S>=kappa}, C = 1{X>=t}, (S,X) standard bivariate normal with correlation r
    th = ndtr(-kappa); gam = ndtr(-t)
    joint = stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-kappa, -t])
    cov = joint - th * gam
    return cov ** 2 / (th * (1 - th) * gam * (1 - gam))
E6 = {}
for r in (0.98, 0.9):
    E6[str(r)] = {}
    for th in THETAS:
        kappa = stats.norm.ppf(1 - th)
        grid = np.linspace(kappa - 1.5, kappa + 0.6, 43)
        vals = [gauss_proxy_rho2_exact(kappa, r, t) for t in grid]
        i = int(np.argmax(vals))
        E6[str(r)][str(th)] = dict(rho2=float(vals[i]), t=float(grid[i]))
    log(f"  r={r}: " + ", ".join(f"{th:.0e}:{E6[str(r)][str(th)]['rho2']:.3f}" for th in THETAS))
results["E6"] = E6

# ---------------------------------------------------------------------------
# E1: theory check at theta = 1e-3 (stump threshold sweep)
# ---------------------------------------------------------------------------
log("E1: theory check")
sim, _ = CONTROLS[1e-3]
ev = lcv.ISEvaluator(sim, n=N_IS, seed=21)
ts = np.linspace(1.6, 4.6, 16)
n1, R1 = 20_000, 400
E1 = dict(theta=1e-3, n=n1, reps=R1, t=ts.tolist(), theory=[], rare_limit=[], empirical=[], tau=[], fpr=[], prec=[])
for t in ts:
    C = (ev.X[:, 0] >= t).astype(float)
    r2 = ev.rho2(C, EC=float(ndtr(-t)))[0]
    tau, fpr, prec, gam = ev.binary_rates(C, EC=float(ndtr(-t)))
    E1["theory"].append(1 - r2); E1["rare_limit"].append(1 - tau * prec)
    E1["tau"].append(tau); E1["fpr"].append(fpr); E1["prec"].append(prec)
rng1 = np.random.default_rng(31)
mc = np.empty(R1); cv = np.empty((len(ts), R1))
for rep in range(R1):
    X, S, Y = sim.sample(n1, rng1)
    mc[rep] = Y.mean()
    for i, t in enumerate(ts):
        cv[i, rep] = lcv.cv_estimate(Y, (X[:, 0] >= t).astype(float), gamma=float(ndtr(-t)))[0]
E1["empirical"] = (cv.var(axis=1) / mc.var()).tolist()
E1["largest_gap"] = float(np.max(np.abs(np.array(E1["empirical"]) - np.array(E1["theory"]))))
log(f"  largest gap theory vs empirical: {E1['largest_gap']:.3f}")
results["E1"] = E1

# ---------------------------------------------------------------------------
# E2: replication study, n = 20/theta
# ---------------------------------------------------------------------------
log("E2: replication study")
E2 = {}
EST_NAMES = ["mc", "gauss", "stump", "linear", "gbm", "oracle"]
for th in THETAS:
    sim, ctr = CONTROLS[th]
    gp = lcv.GaussianProxy(E0[str(th)]["gauss_t"])
    gam = {"gauss": gp.gamma, "stump": ctr["stump"].gamma,
           "linear": E0[str(th)]["linear_gamma"], "gbm": E0[str(th)]["gbm_gamma"], "oracle": E0[str(th)]["oracle_gamma"]}
    ctrls = {"gauss": gp, "stump": ctr["stump"], "linear": ctr["linear"], "gbm": ctr["gbm"], "oracle": ctr["oracle"]}
    n = int(round(EVENTS_PER_REP / th)); R = REPS[th]
    rng2 = np.random.default_rng(int(41 + 1e3 * math.log10(1 / th)))
    est = {k: np.empty(R) for k in EST_NAMES}; se = {k: np.empty(R) for k in EST_NAMES}
    nev = np.empty(R)
    t0 = time.time()
    CH = 1_000_000
    for rep in range(R):
        acc = {k: lcv.CVStats() for k in ctrls}
        done = 0
        while done < n:
            b = min(CH, n - done)
            X, S, Y = sim.sample(b, rng2)
            for k, c in ctrls.items():
                acc[k].add(Y, c(X))
            done += b
        nev[rep] = acc["stump"].sy
        est["mc"][rep], se["mc"][rep] = acc["stump"].mc()
        for k in ctrls:
            est[k][rep], se[k][rep] = acc[k].estimate(gam[k])
        if rep == 0:
            log(f"  theta={th:.0e} n={n} first rep {time.time()-t0:.1f}s")
    z = stats.norm.ppf(0.975)
    row = dict(n=n, reps=R, mean_events=float(nev.mean()))
    for k in EST_NAMES:
        err = est[k] - th
        row[k] = dict(rel_rmse=float(np.sqrt(np.mean(err ** 2)) / th),
                      rel_bias=float(np.mean(err) / th),
                      rel_sd=float(np.std(est[k], ddof=1) / th),
                      var_ratio=float(np.var(est[k], ddof=1) / np.var(est["mc"], ddof=1)),
                      coverage=float(np.mean(np.abs(err) <= z * se[k])),
                      mean_rel_se=float(np.mean(se[k]) / th))
    row["theory_rel_se"] = {"mc": math.sqrt((1 - th) / (n * th))}
    for k in ["gauss", "stump", "linear", "gbm", "oracle"]:
        row["theory_rel_se"][k] = math.sqrt((1 - E0[str(th)][k]) * (1 - th) / (n * th))
    E2[str(th)] = row
    log(f"  theta={th:.0e}: " + ", ".join(f"{k}: rmse={row[k]['rel_rmse']:.3f} vr={row[k]['var_ratio']:.2f} cov={row[k]['coverage']:.2f}" for k in EST_NAMES))
results["E2"] = E2
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)

# ---------------------------------------------------------------------------
# E3: few-events regime at theta = 1e-4
# ---------------------------------------------------------------------------
log("E3: few-events regime")
th = 1e-4
sim, ctr = CONTROLS[th]
gam = {"stump": ctr["stump"].gamma, "linear": E0[str(th)]["linear_gamma"], "gbm": E0[str(th)]["gbm_gamma"], "oracle": E0[str(th)]["oracle_gamma"]}
ctrls = {"stump": ctr["stump"], "linear": ctr["linear"], "gbm": ctr["gbm"], "oracle": ctr["oracle"]}
EV = [2, 5, 10, 20, 50, 100]
R3 = 400
E3 = dict(theta=th, events=EV, reps=R3, rho2={k: E0[str(th)][k] for k in ctrls}, rows=[])
rng3 = np.random.default_rng(51)
z = stats.norm.ppf(0.975)
for ne in EV:
    n = int(round(ne / th))
    est = {k: np.empty(R3) for k in ["mc"] + list(ctrls)}; se = {k: np.empty(R3) for k in est}
    CH = 1_000_000
    for rep in range(R3):
        acc = {k: lcv.CVStats() for k in ctrls}
        done = 0
        while done < n:
            b = min(CH, n - done)
            X, S, Y = sim.sample(b, rng3)
            for k, c in ctrls.items():
                acc[k].add(Y, c(X))
            done += b
        est["mc"][rep], se["mc"][rep] = acc["stump"].mc()
        for k in ctrls:
            est[k][rep], se[k][rep] = acc[k].estimate(gam[k])
    row = dict(n=n, events=ne)
    for k in est:
        err = est[k] - th
        row[k] = dict(var_ratio=float(np.var(est[k], ddof=1) / np.var(est["mc"], ddof=1)),
                      rel_rmse=float(np.sqrt(np.mean(err ** 2)) / th), rel_bias=float(np.mean(err) / th),
                      coverage=float(np.mean(np.abs(err) <= z * se[k])))
    E3["rows"].append(row)
    log(f"  events={ne}: " + ", ".join(f"{k}: vr={row[k]['var_ratio']:.2f} bias={row[k]['rel_bias']:+.3f} cov={row[k]['coverage']:.2f}" for k in est))
results["E3"] = E3

# ---------------------------------------------------------------------------
# E4: unlabelled pool at theta = 1e-3
# ---------------------------------------------------------------------------
log("E4: pool")
th = 1e-3
sim, ctr = CONTROLS[th]
g = ctr["gbm"]; rho2_g = E0[str(th)]["gbm"]
n4, R4 = 20_000, 300
ratios = [0.5, 1, 2, 5, 10, 20, 50]
rng4 = np.random.default_rng(61)
mc = np.empty(R4); cvs = {q: np.empty(R4) for q in ratios}; known = np.empty(R4)
for rep in range(R4):
    X, S, Y = sim.sample(n4, rng4)
    C = g(X); mc[rep] = Y.mean()
    known[rep] = lcv.cv_estimate(Y, C, gamma=E0[str(th)]["gbm_gamma"])[0]
    for q in ratios:
        Cp = g(sim.draw_features(int(q * n4), rng4))
        cvs[q][rep] = lcv.cv_estimate(Y, C, C_pool=Cp)[0]
E4 = dict(theta=th, n=n4, reps=R4, ratios=ratios, rho2=rho2_g,
          theory=[1 - rho2_g * (q * n4) / (n4 + q * n4) for q in ratios],
          empirical=[float(np.var(cvs[q], ddof=1) / np.var(mc, ddof=1)) for q in ratios],
          known_mean_empirical=float(np.var(known, ddof=1) / np.var(mc, ddof=1)))
log("  empirical: " + ", ".join(f"{q}:{v:.3f}" for q, v in zip(ratios, E4["empirical"])) + f" known={E4['known_mean_empirical']:.3f}")
results["E4"] = E4

# ---------------------------------------------------------------------------
# E5: feature informativeness (a0) at theta = 1e-4
# ---------------------------------------------------------------------------
log("E5: ceiling ablation")
E5 = {}
for a0 in (0.15, 0.30, 0.60):
    simA = lcv.Simulator(a0=a0).calibrate(1e-4)
    rngA = np.random.default_rng(71)
    XtA, wA = train_features(N_TRAIN, rngA, "tail"); StA = simA.severity(XtA, rngA)
    XvA, _ = train_features(N_VAL, rngA, "tail"); SvA = simA.severity(XvA, rngA)
    ctrA = fit_controls(simA, XtA, StA, wA, XvA, SvA)
    evA = lcv.ISEvaluator(simA, n=2_000_000, seed=81)
    gpA = best_gauss_proxy(evA)
    row = dict(kappa=simA.kappa, ceiling=evA.ceiling(), gauss=evA.rho2(gpA(evA.X), EC=gpA.gamma)[0])
    for k, c in ctrA.items():
        row[k] = evA.rho2(c(evA.X), EC=known_gamma(c))[0]
    E5[str(a0)] = row
    log(f"  a0={a0}: " + ", ".join(f"{k}={v:.3f}" for k, v in row.items()))
results["E5"] = E5

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
log("done")
