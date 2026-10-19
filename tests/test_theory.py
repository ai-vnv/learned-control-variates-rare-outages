"""Fast checks of the analytical results and of the estimators (about 30 s)."""
import math

import numpy as np
import pytest
from scipy.special import ndtr

import lcv


def joint_binary(theta, tau, fpr):
    """Exact moments of (Y, C) for a binary control with recall tau and false-positive rate fpr."""
    gamma = theta * tau + (1 - theta) * fpr
    cov = theta * tau - theta * gamma
    rho2 = cov ** 2 / (theta * (1 - theta) * gamma * (1 - gamma))
    return gamma, rho2


@pytest.mark.parametrize("theta,tau,fpr", [(1e-2, 0.6, 0.01), (1e-3, 0.3, 0.2), (1e-4, 0.9, 1e-4), (0.2, 0.1, 0.5)])
def test_theorem1_formula(theta, tau, fpr):
    gamma, rho2 = joint_binary(theta, tau, fpr)
    formula = (tau - fpr) ** 2 * theta * (1 - theta) / (gamma * (1 - gamma))
    assert rho2 == pytest.approx(formula, rel=1e-12)
    # Corollary 1, finite-theta form
    prec = theta * tau / gamma
    assert rho2 == pytest.approx(tau * prec * (1 - fpr / tau) ** 2 * (1 - theta) / (1 - gamma), rel=1e-12)
    # rho is the phi coefficient (Matthews correlation) of the confusion matrix
    tp, fn, fp, tn = theta * tau, theta * (1 - tau), (1 - theta) * fpr, (1 - theta) * (1 - fpr)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert rho2 == pytest.approx(mcc ** 2, rel=1e-10)


def test_corollary1_rare_limit():
    tau, prec = 0.5, 0.4
    for theta in (1e-4, 1e-6, 1e-8):
        gamma = theta * tau / prec
        fpr = (gamma - theta * tau) / (1 - theta)
        _, rho2 = joint_binary(theta, tau, fpr)
        assert 1 - rho2 == pytest.approx(1 - tau * prec, abs=10 * theta)
    # fixed false-positive rate: rho2 = O(theta)
    r = [joint_binary(t, 0.5, 0.05)[1] / t for t in (1e-4, 1e-6)]
    assert r[1] == pytest.approx(r[0], rel=0.01)


def test_theta_exact_matches_monte_carlo():
    sim = lcv.Simulator().calibrate(1e-2)
    assert sim.theta_exact() == pytest.approx(1e-2, rel=1e-6)
    rng = np.random.default_rng(1)
    _, _, Y = sim.sample(2_000_000, rng)
    assert abs(Y.mean() - 1e-2) < 4 * math.sqrt(1e-2 * 0.99 / 2e6)


def test_control_variate_unbiased_and_variance_reduced():
    sim = lcv.Simulator().calibrate(1e-2)
    stump_t = 1.6
    gamma = float(lcv.Phibar(stump_t))
    rng = np.random.default_rng(2)
    n, reps = 5_000, 400
    mc, cv = np.empty(reps), np.empty(reps)
    for r in range(reps):
        X, _, Y = sim.sample(n, rng)
        C = (X[:, 0] >= stump_t).astype(float)
        mc[r] = Y.mean()
        cv[r] = lcv.cv_estimate(Y, C, gamma=gamma)[0]
    theta = sim.theta_exact()
    assert abs(cv.mean() - theta) < 4 * cv.std() / math.sqrt(reps)
    assert cv.var() < mc.var()


def test_pool_variance_theorem3b():
    """Var with a pool of m unlabelled draws ~ Var(Y)/n (1 - rho2 m/(n+m)), fixed beta*."""
    sim = lcv.Simulator().calibrate(5e-2)
    t = 1.2
    rng = np.random.default_rng(3)
    # beta*, rho2 from a large pilot
    X, _, Y = sim.sample(2_000_000, rng)
    C = (X[:, 0] >= t).astype(float)
    cov = np.cov(Y, C)[0, 1]; beta = cov / C.var(); rho2 = cov ** 2 / (Y.var() * C.var())
    n, m, reps = 2_000, 8_000, 1_500
    est = np.empty(reps)
    for r in range(reps):
        X, _, Y = sim.sample(n, rng)
        C = (X[:, 0] >= t).astype(float)
        Cp = (sim.draw_features(m, rng)[:, 0] >= t).astype(float)
        g_hat = (C.sum() + Cp.sum()) / (n + m)
        est[r] = Y.mean() - beta * (C.mean() - g_hat)
    th = sim.theta_exact()
    predicted = th * (1 - th) / n * (1 - rho2 * m / (n + m))
    assert est.var() == pytest.approx(predicted, rel=0.12)


def test_gbm_lookup_matches_sklearn():
    sim = lcv.Simulator()
    rng = np.random.default_rng(4)
    X = sim.draw_features(3_000, rng); S = sim.severity(X, rng)
    g = lcv.GBMScore(max_iter=50, seed=0).fit(X, S, 3.0)
    assert g.check_lookup(sim.draw_features(5_000, rng)) < 1e-9


def test_gaussian_proxy_vanishes():
    """Proposition 1: the best threshold control on a correlated Gaussian loses its effect."""
    r = 0.9
    best = []
    for kappa in (2.0, 3.0, 4.0):
        theta = float(lcv.Phibar(kappa))
        rhos = []
        for t in np.linspace(-1, kappa + 1, 200):
            from scipy.stats import multivariate_normal
            joint = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-kappa, -t])   # P(S>=k, P>=t)
            gamma = float(lcv.Phibar(t))
            cov = joint - theta * gamma
            rhos.append(cov ** 2 / (theta * (1 - theta) * gamma * (1 - gamma)))
        best.append(max(rhos))
    assert best[0] > best[1] > best[2]
