"""
lcv.py -- Learned control variates for rare grid-outage probabilities at EV charging
stations.  Companion code for the ICSGSC 2026 paper.

Everything needed for the experiments:
  * a feature-driven outage simulator (cheap features X, expensive severity S, label Y),
  * exact outage probability theta(kappa) and exact Bayes posterior eta(x) by quadrature,
  * an importance-sampling evaluator for rho^2 of any control variate (uses the exact
    eta, which the estimators themselves never see),
  * the control variates compared in the paper: Gaussian proxy (previous work),
    temperature stump (weak), linear model (weak, misspecified), gradient-boosted
    trees (strong), and the Bayes posterior (ceiling),
  * control-variate estimators with estimated coefficient and (optionally) a pool
    estimate of the control mean, with plug-in standard errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats, optimize, integrate
from scipy.special import ndtr

SQRT2PI = math.sqrt(2.0 * math.pi)


def phi(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def Phibar(x):
    return ndtr(-np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
@dataclass
class Simulator:
    """Severity  S = a1 x1 + a2 max(x1,0)^2 + a3 x2 + a4 x3 max(e2,0) + a0 e1,
    outage Y = 1{S >= kappa}.  x1: temperature anomaly N(0,1); x2: day-ahead load
    forecast anomaly N(0,1); x3: storm flag Bernoulli(p_storm); e1,e2: internal noise
    of the expensive simulator (not observable through the features)."""
    a1: float = 0.45
    a2: float = 0.20
    a3: float = 0.55
    a4: float = 0.40
    a0: float = 0.30
    p_storm: float = 0.15
    kappa: float = 2.0

    # --- sampling -------------------------------------------------------------
    def draw_features(self, n, rng):
        X = np.empty((n, 3))
        X[:, 0] = rng.standard_normal(n)
        X[:, 1] = rng.standard_normal(n)
        X[:, 2] = rng.random(n) < self.p_storm
        return X

    def mean_given_x(self, X):
        x1 = X[:, 0]
        return self.a1 * x1 + self.a2 * np.maximum(x1, 0.0) ** 2 + self.a3 * X[:, 1]

    def severity(self, X, rng):
        n = X.shape[0]
        e1 = rng.standard_normal(n)
        e2 = rng.standard_normal(n)
        return self.mean_given_x(X) + self.a4 * X[:, 2] * np.maximum(e2, 0.0) + self.a0 * e1

    def sample(self, n, rng):
        X = self.draw_features(n, rng)
        S = self.severity(X, rng)
        return X, S, (S >= self.kappa).astype(float)

    # --- exact quantities -------------------------------------------------------
    # Gauss-Legendre nodes on [0, U] for the half-normal integral over max(e2,0)
    _GL_N = 64
    _GL_U = 9.0
    _gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
    _u_nodes = 0.5 * _GL_U * (_gl_x + 1.0)
    _u_weights = 0.5 * _GL_U * _gl_w * phi(_u_nodes)   # weights include the N(0,1) density

    def eta(self, X, kappa=None):
        """Exact Bayes posterior P(Y=1 | X=x)."""
        kappa = self.kappa if kappa is None else kappa
        mu = self.mean_given_x(X)
        out = Phibar((kappa - mu) / self.a0)
        storm = np.flatnonzero(X[:, 2] > 0.5)
        B = 250_000                                   # block size keeps memory bounded
        for i in range(0, len(storm), B):
            idx = storm[i:i + B]
            ms = mu[idx]
            z = (kappa - ms[:, None] - self.a4 * self._u_nodes[None, :]) / self.a0
            integ = Phibar(z) @ self._u_weights          # int_0^inf Phibar(.) phi(u) du
            out[idx] = 0.5 * Phibar((kappa - ms) / self.a0) + integ
        return out

    def theta_exact(self, kappa=None):
        """Exact outage probability by quadrature over x1 (and the storm integral)."""
        kappa = self.kappa if kappa is None else kappa
        s = math.sqrt(self.a3 ** 2 + self.a0 ** 2)   # a3 x2 + a0 e1 is N(0, s^2)
        a1, a2, a4, p = self.a1, self.a2, self.a4, self.p_storm

        def inner(x1):
            m = a1 * x1 + a2 * max(x1, 0.0) ** 2
            no_storm = Phibar((kappa - m) / s)
            integ = np.dot(Phibar((kappa - m - a4 * self._u_nodes) / s), self._u_weights)
            storm = 0.5 * Phibar((kappa - m) / s) + integ
            return ((1 - p) * no_storm + p * storm) * phi(x1)

        val, err = integrate.quad(inner, -12.0, 12.0, limit=400, epsabs=0.0, epsrel=1e-10,
                                  points=[0.0])
        return float(val)

    def calibrate(self, theta_target):
        f = lambda k: self.theta_exact(k) - theta_target
        self.kappa = optimize.brentq(f, -2.0, 15.0, xtol=1e-12, rtol=1e-12)
        return self

    def latent_corr_forecast(self, n=2_000_000, seed=0):
        """Correlation between the severity S and the Gaussian forecast anomaly x2."""
        rng = np.random.default_rng(seed)
        X, S, _ = self.sample(n, rng)
        return float(np.corrcoef(S, X[:, 1])[0, 1])


# ---------------------------------------------------------------------------
# Importance-sampling evaluator (uses the exact posterior; evaluation only)
# ---------------------------------------------------------------------------
class ISEvaluator:
    """Draw features from a defensive mixture q = w0 p + (1-w0) p_shift and weight by
    p/q; returns weighted moments of (eta, C) for any control C = c(X)."""

    def __init__(self, sim: Simulator, n=4_000_000, seed=123, w0=0.3, pilot=True):
        self.sim = sim
        self.n = n
        self.w0 = w0
        rng = np.random.default_rng(seed)
        # pilot: locate the conditional mean of (x1, x2) given an outage
        mu = np.array([min(sim.kappa, 3.0), min(sim.kappa, 3.0)]) * 0.6
        if pilot:
            for _ in range(3):
                Xp, wp = self._draw(400_000, rng, mu)
                et = sim.eta(Xp)
                z = wp * et
                mu = np.array([np.sum(z * Xp[:, 0]), np.sum(z * Xp[:, 1])]) / np.sum(z)
        self.mu = mu
        self.X, self.w = self._draw(n, rng, mu)
        self.eta = sim.eta(self.X)
        self.theta = float(np.mean(self.w * self.eta))
        self.theta_se = float(np.std(self.w * self.eta) / math.sqrt(n))
        self.ess = float(np.sum(self.w) ** 2 / np.sum(self.w ** 2))

    def _draw(self, n, rng, mu):
        sim, w0 = self.sim, self.w0
        n0 = int(round(w0 * n))
        X = np.empty((n, 3))
        X[:n0, 0] = rng.standard_normal(n0)
        X[:n0, 1] = rng.standard_normal(n0)
        X[n0:, 0] = mu[0] + rng.standard_normal(n - n0)
        X[n0:, 1] = mu[1] + rng.standard_normal(n - n0)
        X[:, 2] = rng.random(n) < sim.p_storm         # storm flag kept at its nominal law
        logp = -0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2)
        logq_shift = -0.5 * ((X[:, 0] - mu[0]) ** 2 + (X[:, 1] - mu[1]) ** 2)
        # p/q = exp(logp) / (w0 exp(logp) + (1-w0) exp(logq_shift))
        m = np.maximum(logp, logq_shift)
        w = np.exp(logp - m) / (w0 * np.exp(logp - m) + (1 - w0) * np.exp(logq_shift - m))
        return X, w

    def moments(self, C):
        """Weighted E[C], E[C^2], E[C eta] with standard errors."""
        w, eta, n = self.w, self.eta, self.n
        def m(z):
            return float(np.mean(z)), float(np.std(z) / math.sqrt(n))
        EC, EC_se = m(w * C)
        EC2, _ = m(w * C * C)
        ECe, ECe_se = m(w * C * eta)
        return dict(EC=EC, EC_se=EC_se, EC2=EC2, ECeta=ECe, ECeta_se=ECe_se)

    def rho2(self, C, EC=None, theta=None):
        """rho^2(Y, C) for a control C evaluated on self.X.  If the control mean is known
        exactly (binary stump on a known feature), pass EC."""
        th = self.theta if theta is None else theta
        mo = self.moments(C)
        EC_ = mo["EC"] if EC is None else EC
        varC = mo["EC2"] - EC_ ** 2
        cov = mo["ECeta"] - th * EC_
        if varC <= 0:
            return 0.0, cov, varC, mo
        return cov ** 2 / (th * (1 - th) * varC), cov, varC, mo

    def ceiling(self):
        """rho2_max = Var(eta)/(theta(1-theta))."""
        th = self.theta
        Eeta2 = float(np.mean(self.w * self.eta ** 2))
        return (Eeta2 - th ** 2) / (th * (1 - th))

    def binary_rates(self, C, EC=None):
        """(recall tau, false-positive rate phi, precision pi) of a binary control."""
        th = self.theta
        mo = self.moments(C)
        gamma = mo["EC"] if EC is None else EC
        joint = mo["ECeta"]                 # P(Y=1, C=1)
        tau = joint / th
        fpr = (gamma - joint) / (1 - th)
        prec = joint / gamma if gamma > 0 else float("nan")
        return tau, fpr, prec, gamma


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
class GaussianProxy:
    """Control of the previous work: C = 1{x2 >= t}, known mean Phibar(t)."""
    name = "gauss"

    def __init__(self, t):
        self.t = float(t)
        self.gamma = float(Phibar(t))

    def __call__(self, X):
        return (X[:, 1] >= self.t).astype(float)


class TempStump:
    """Weak learner: univariate linear fit of S on x1, thresholded at kappa.
    C = 1{a x1 + b >= kappa} = 1{x1 >= t}, known mean Phibar(t)."""
    name = "stump"

    def fit(self, X, S, kappa):
        a, b = np.polyfit(X[:, 0], S, 1)
        self.a, self.b = float(a), float(b)
        self.t = (kappa - self.b) / self.a
        self.gamma = float(Phibar(self.t))
        return self

    def __call__(self, X):
        return (X[:, 0] >= self.t).astype(float)


class LinearScore:
    """Weak (misspecified) learner: OLS of S on (x1, x2, x3); score Phibar((kappa - s)/sigma)."""
    name = "linear"

    def fit(self, X, S, kappa):
        A = np.column_stack([X, np.ones(len(S))])
        coef, *_ = np.linalg.lstsq(A, S, rcond=None)
        self.coef = coef
        resid = S - A @ coef
        self.sigma = float(np.std(resid))
        self.kappa = kappa
        return self

    def shat(self, X):
        return X @ self.coef[:3] + self.coef[3]

    def __call__(self, X):
        return Phibar((self.kappa - self.shat(X)) / self.sigma)

    def hard(self, X):
        return (self.shat(X) >= self.kappa).astype(float)


class GBMScore:
    """Strong learner: gradient-boosted trees regressing S on X; same score transform.
    Prediction is done through an exact lookup table over the model's feature bins,
    which reproduces HistGradientBoostingRegressor.predict and is much faster."""
    name = "gbm"

    def __init__(self, max_iter=300, learning_rate=0.1, max_leaf_nodes=31, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                                   max_leaf_nodes=max_leaf_nodes, random_state=seed,
                                                   early_stopping=False)

    def fit(self, X, S, kappa, X_val=None, S_val=None):
        self.model.fit(X, S)
        self.kappa = kappa
        # residual sd on held-out data if given (otherwise in-sample)
        Xv, Sv = (X_val, S_val) if X_val is not None else (X, S)
        self.sigma = float(np.std(Sv - self.model.predict(Xv)))
        self._build_lookup()
        return self

    def _build_lookup(self):
        thr = self.model._bin_mapper.bin_thresholds_
        self.thr = [np.asarray(t) for t in thr]
        n1, n2, n3 = len(self.thr[0]) + 1, len(self.thr[1]) + 1, 2
        # representative point for each bin: midpoints (any point inside the bin works)
        def reps(t):
            if len(t) == 0:
                return np.array([0.0])
            edges = np.concatenate([[t[0] - 1.0], t, [t[-1] + 1.0]])
            return 0.5 * (edges[:-1] + edges[1:])
        r1, r2 = reps(self.thr[0]), reps(self.thr[1])
        g1, g2, g3 = np.meshgrid(r1, r2, np.array([0.0, 1.0]), indexing="ij")
        G = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel()])
        self.table = self.model.predict(G).reshape(n1, n2, n3)

    def shat(self, X):
        i1 = np.searchsorted(self.thr[0], X[:, 0], side="right")
        i2 = np.searchsorted(self.thr[1], X[:, 1], side="right")
        i3 = (X[:, 2] > 0.5).astype(int)
        return self.table[i1, i2, i3]

    def check_lookup(self, X):
        return float(np.max(np.abs(self.shat(X) - self.model.predict(X))))

    def __call__(self, X):
        return Phibar((self.kappa - self.shat(X)) / self.sigma)

    def hard(self, X):
        return (self.shat(X) >= self.kappa).astype(float)


class Oracle:
    """Bayes posterior eta(X): the best possible control from the features (ceiling)."""
    name = "oracle"

    def __init__(self, sim):
        self.sim = sim

    def __call__(self, X):
        return self.sim.eta(X)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def mc_estimate(Y):
    n = len(Y)
    m = Y.mean()
    se = math.sqrt(max(m * (1 - m), 0.0) / n)
    return m, se


def cv_estimate(Y, C, gamma=None, C_pool=None):
    """Control-variate estimate with regression coefficient estimated from the n pairs.
    gamma: known control mean; else C_pool (unlabelled draws) gives the pooled mean.
    Returns (estimate, plug-in standard error, beta_hat, sample rho2)."""
    n = len(Y)
    Cc = C - C.mean()
    varC = float(np.dot(Cc, Cc) / (n - 1))
    if varC <= 0:
        return mc_estimate(Y) + (0.0, 0.0)
    cov = float(np.dot(Cc, Y - Y.mean()) / (n - 1))
    beta = cov / varC
    varY = float(Y.var(ddof=1))
    rho2 = cov ** 2 / (varY * varC) if varY > 0 else 0.0
    if gamma is None:
        m = len(C_pool)
        gamma_hat = (C.sum() + C_pool.sum()) / (n + m)
        est = Y.mean() - beta * (C.mean() - gamma_hat)
        # plug-in version of the exact variance in Theorem 3(b):
        # Var = [varY - 2 b f cov + b^2 f^2 varC]/n + b^2 f^2 varC/m,  f = m/(n+m)
        f = m / (n + m)
        se2 = (varY - 2 * beta * f * cov + beta ** 2 * f ** 2 * varC) / n + beta ** 2 * f ** 2 * varC / m
        se2 = max(se2, 0.0)
    else:
        est = Y.mean() - beta * (C.mean() - gamma)
        resid_var = float(np.var(Y - beta * C, ddof=1))
        se2 = resid_var / n
    return float(est), math.sqrt(se2), float(beta), float(rho2)


def n_required_mc(theta, rel_err, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    return z ** 2 * (1 - theta) / (theta * rel_err ** 2)


def n_required_cv(theta, rel_err, rho2, alpha=0.05):
    return n_required_mc(theta, rel_err, alpha) * (1 - rho2)


class CVStats:
    """Accumulates sufficient statistics of (Y, C) over chunks and returns the same
    estimate and plug-in standard error as cv_estimate with a known control mean."""

    def __init__(self):
        self.n = 0; self.sy = 0.0; self.syy = 0.0; self.sc = 0.0; self.scc = 0.0; self.syc = 0.0

    def add(self, Y, C):
        self.n += len(Y); self.sy += Y.sum(); self.syy += np.dot(Y, Y)
        self.sc += C.sum(); self.scc += np.dot(C, C); self.syc += np.dot(Y, C)

    def estimate(self, gamma):
        n = self.n
        my, mc = self.sy / n, self.sc / n
        varY = (self.syy - n * my * my) / (n - 1)
        varC = (self.scc - n * mc * mc) / (n - 1)
        cov = (self.syc - n * my * mc) / (n - 1)
        if varC <= 0:
            return my, math.sqrt(max(varY, 0.0) / n)
        beta = cov / varC
        est = my - beta * (mc - gamma)
        resid_var = max(varY - 2 * beta * cov + beta * beta * varC, 0.0)
        return float(est), math.sqrt(resid_var / n)

    def mc(self):
        n = self.n; my = self.sy / n
        return my, math.sqrt(max(my * (1 - my), 0.0) / n)
