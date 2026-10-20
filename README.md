# Learned control variates for rare grid-outage estimation


A public EV charging station adds load to a distribution feeder, and on rare days the feeder
overloads. Planners need the probability of such a day, often 1e-3 to 1e-6, but plain Monte
Carlo needs about 400/θ runs of an expensive grid simulator to reach a 10% relative error. The
study builds a control variate from a classifier of the outage trained on cheap features
(temperature, load forecast, storm flag) and shows that its variance reduction is set by the
classifier's recall and precision on the rare days. Every number, figure and table of the paper
can be regenerated from this repository.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/relerr_map_dark.gif">
    <img src="figures/relerr_map_light.gif" width="100%"
         alt="Animated map of Surabaya: estimated daily outage probability of each feeder from the Gaussian load model, Monte Carlo and the learned control variate, as the number of simulated days grows">
  </picture>
</p>

*Each feeder has a charging station and the reserve that meets the one-day-in-ten-years target
exactly, so the true outage probability (2.7e-4 per day) is the same everywhere. The Gaussian
load model of the earlier work underestimates it by 88% to 98% on every feeder, whatever the
number of simulated days. After 10^5 simulated days per feeder, 95% of the Monte Carlo estimates
are within ±37% of the truth and 95% of the learned-control-variate estimates within ±23%; after
10^6 days, ±13% and ±7%.*

## Quick start

```bash
git clone https://github.com/ai-vnv/learned-control-variates-rare-outages
cd learned-control-variates-rare-outages
python -m pip install -e ".[test]"     # numpy, scipy, scikit-learn, matplotlib, pillow, pytest
make test                              # checks of Theorems 1 and 3, Corollary 1, Proposition 1 (seconds)
make figures                           # rebuild figures/ and tables/ from the stored results/ (seconds)
make numbers                           # print every number quoted in the paper
```

`make all` reruns every simulation from scratch (about one hour on two cores) and then
rebuilds the figures, the animation and the numbers. All runs use fixed seeds; with the
versions in `requirements.txt`.



## Examples

```python
import numpy as np, lcv
sim = lcv.Simulator().calibrate(1e-3)                  # outage probability 1e-3 per simulated day
X, S, Y = sim.sample(500_000, np.random.default_rng(0))   # features, severity, outage indicator
t = 3.1                                                # stump: alarm on a hot day, x1 >= t
C = (X[:, 0] >= t).astype(float)
print(lcv.mc_estimate(Y))                              # (estimate, standard error)
print(lcv.cv_estimate(Y, C, gamma=float(lcv.Phibar(t))))  # (estimate, standard error, beta, rho^2)

ev = lcv.ISEvaluator(sim, n=2_000_000, seed=1)         # exact evaluation of the control
Ce = (ev.X[:, 0] >= t).astype(float)
recall, fpr, precision, gamma = ev.binary_rates(Ce, EC=float(lcv.Phibar(t)))
print(ev.rho2(Ce, EC=gamma)[0], recall * precision)    # 0.273 and 0.274: Corollary 1
```

For a binary learner with recall τ and false-positive rate φ, the variance ratio to Monte
Carlo is 1 − ρ² with ρ² = (τ − φ)² θ(1 − θ) / (γ(1 − γ)), i.e., one minus the squared Matthews
correlation of the learner, which tends to 1 − recall × precision as θ → 0 (Theorem 1,
Corollary 1).



## Licence

Code: MIT (see `LICENSE`). 
