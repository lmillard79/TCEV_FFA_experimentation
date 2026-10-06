"""Two-Component Extreme Value (TCEV) distribution: CDF, PDF, MLE fit, quantiles.

F(x) = exp(-l1*exp(-x/t1) - l2*exp(-x/t2)),  t2 > t1
Component 1 = frequent "ordinary" annual events (l1 large, t1 small).
Component 2 = rare, large events (l2 small, t2 large).
F is the product of two Gumbel-type CDFs, one per component.

Also holds the GEV / Gumbel baselines so every distribution's quantiles come
from one place.
"""
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize

PARAM_NAMES = ("l1", "t1", "l2", "t2")


def cdf(x, l1, t1, l2, t2):
    return np.exp(-l1 * np.exp(-x / t1) - l2 * np.exp(-x / t2))


def cdf_components(x, l1, t1, l2, t2):
    """CDF of each component on its own; their product is the TCEV CDF."""
    return np.exp(-l1 * np.exp(-x / t1)), np.exp(-l2 * np.exp(-x / t2))


def pdf(x, l1, t1, l2, t2):
    return cdf(x, l1, t1, l2, t2) * (l1 / t1 * np.exp(-x / t1) + l2 / t2 * np.exp(-x / t2))


def nll(logp, x):
    l1, t1, l2, t2 = np.exp(logp)  # log scale keeps params positive
    if t2 <= t1:
        return 1e12  # component 2 = larger events
    with np.errstate(all="ignore"):
        f = pdf(x, l1, t1, l2, t2)
    return 1e12 if (not np.all(np.isfinite(f)) or np.any(f <= 0)) else -np.sum(np.log(f))


def fit(x, rng, starts=30):
    """Multi-start Nelder-Mead MLE. Returns (best_params, best_nll, starts_table).

    starts_table has one row per start so convergence can be inspected.
    """
    x = np.asarray(x, dtype=float)
    m = x.mean()
    rows = []
    best = None
    for k in range(starts):
        p0 = np.log([rng.uniform(1, 5), rng.uniform(.1, .6) * m,
                     rng.uniform(.05, 1), rng.uniform(.8, 3) * m])
        r = minimize(nll, p0, args=(x,), method="Nelder-Mead",
                     options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
        p = np.exp(r.x)
        rows.append({"start": k, "nll": r.fun, "success": r.success, "nit": r.nit,
                     **dict(zip(PARAM_NAMES, p))})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), best.fun, rows


def quantile(aep, par):
    return brentq(lambda x: cdf(x, *par) - (1 - aep), 1e-9, 1e9)


def boundary_flags(par):
    """Signs that the 4-parameter fit has collapsed towards a simpler model."""
    l1, t1, l2, t2 = par
    return {
        "t2_close_to_t1": bool(t2 / t1 < 1.01),   # components indistinguishable
        "l2_near_zero": bool(l2 < 1e-3),          # second component vanished
        "l1_near_zero": bool(l1 < 1e-3),
    }


# ---- baselines -------------------------------------------------------------

def fit_gev(x):
    """MLE GEV. scipy shape c = -xi. Returns (params, nll, quantile_fn).

    scipy's default start can land in a bad local optimum (seen on Gargett:
    NLL worse than the nested Gumbel), so start from the Gumbel fit at several
    shapes and keep the best.
    """
    g_loc, g_scale = stats.gumbel_r.fit(x)
    best = None
    for c0 in (-0.3, -0.15, -0.05, 0.05, 0.15, 0.3):
        try:
            c, loc, scale = stats.genextreme.fit(x, c0, loc=g_loc, scale=g_scale)
        except Exception:
            continue
        n = -np.sum(stats.genextreme.logpdf(x, c, loc, scale))
        if np.isfinite(n) and (best is None or n < best[1]):
            best = ((c, loc, scale), n)
    (c, loc, scale), n = best
    return (c, loc, scale), n, lambda aep: stats.genextreme.isf(aep, c, loc, scale)


def fit_gumbel(x):
    loc, scale = stats.gumbel_r.fit(x)
    n = -np.sum(stats.gumbel_r.logpdf(x, loc, scale))
    return (loc, scale), n, lambda aep: stats.gumbel_r.isf(aep, loc, scale)


def aic(nll_min, k):
    return 2 * k + 2 * nll_min
