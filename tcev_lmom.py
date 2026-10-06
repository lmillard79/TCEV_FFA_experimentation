"""L-moment fitting for the TCEV distribution.

Totaro et al. (2024) estimate TCEV by matching sample L-moments, not by maximum
likelihood. Four parameters (l1, t1, l2, t2) are matched to four sample
L-moment statistics: l1 (mean), l2 (L-scale), t3 (L-skew), t4 (L-kurtosis).

Population L-moments are computed numerically from the CDF, so no closed-form
TCEV L-moment expressions are needed:
    beta_r = E[X F(X)^r] = integral x F(x)^r f(x) dx,    r = 0..3
    L1 = b0,  L2 = 2b1 - b0,  L3 = 6b2 - 6b1 + b0,  L4 = 20b3 - 30b2 + 12b1 - b0
"""
import numpy as np
from scipy.optimize import least_squares

import tcev

GRID_N = 8000


def sample_lmoments(x):
    """Unbiased sample L-moments (Hosking & Wallis 1995). Returns l1, l2, t3, t4."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3 / l2, l4 / l2


def _grid(l1, t1, l2, t2):
    """x range that captures essentially all probability mass (F from ~e^-40 to 1-1e-13)."""
    lo = t1 * (np.log(l1) - np.log(40.0))
    hi = max(t1 * np.log(l1 * 1e13), t2 * np.log(max(l2, 1e-12) * 1e13))
    return np.linspace(lo, hi, GRID_N)


def population_lmoments(par):
    """(L1, L2, tau3, tau4) of the TCEV with parameters (l1, t1, l2, t2)."""
    x = _grid(*par)
    with np.errstate(all="ignore"):
        F = tcev.cdf(x, *par)
        f = tcev.pdf(x, *par)
    xf = np.nan_to_num(x * f)
    trap = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renamed trapz
    b = [trap(xf * F ** r, x) for r in range(4)]
    L1 = b[0]
    L2 = 2 * b[1] - b[0]
    L3 = 6 * b[2] - 6 * b[1] + b[0]
    L4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return L1, L2, L3 / L2, L4 / L2


def _unpack(p):
    """log-parameters -> (l1, t1, l2, t2), with t2 = t1 * (1 + exp(s)) so t2 > t1."""
    l1, t1, l2, s = p
    t1 = np.exp(t1)
    return np.exp(l1), t1, np.exp(l2), t1 * (1 + np.exp(s))


def _resid(p, target):
    try:
        pop = population_lmoments(_unpack(p))
    except Exception:
        return np.full(4, 1e3)
    r = np.array([(pop[0] - target[0]) / target[0], (pop[1] - target[1]) / target[1],
                  pop[2] - target[2], pop[3] - target[3]])
    return np.where(np.isfinite(r), r, 1e3)


def fit_lmom(x, rng, starts=40):
    """Match (l1, l2, t3, t4). Returns (params, cost, sample_lmoments, starts_table).

    cost is the sum of squared residuals; ~0 means the sample L-moments lie inside
    the TCEV domain (an exact solution exists), larger means they do not.
    """
    target = sample_lmoments(x)
    m = target[0]
    rows, best = [], None
    for k in range(starts):
        p0 = np.log([rng.uniform(1, 5), rng.uniform(.05, .6) * m, rng.uniform(.1, 2)])
        p0 = np.append(p0, np.log(rng.uniform(.3, 2.5)))
        # p0 = [log l1, log t1, log l2, s]  (t1 entered above already in log via np.log)
        try:
            r = least_squares(_resid, p0, args=(target,), method="lm", xtol=1e-12, ftol=1e-12,
                              max_nfev=2000)
        except Exception:
            continue
        par = _unpack(r.x)
        rows.append({"start": k, "cost": 2 * r.cost, **dict(zip(tcev.PARAM_NAMES, par))})
        if best is None or r.cost < best[0]:
            best = (r.cost, par)
    return np.array(best[1]), 2 * best[0], target, rows
