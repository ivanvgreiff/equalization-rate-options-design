"""Synthetic data generators for testing and sanity checks.

Two-regime Markov model matching the worked example from work-until-now.md.
"""

from __future__ import annotations

import numpy as np


def two_regime_markov(
    n_intervals: int = 90,
    n_paths: int = 10_000,
    good_mean_apr: float = 0.12,
    good_vol_apr: float = 0.10,
    bad_mean_apr: float = -0.08,
    bad_vol_apr: float = 0.20,
    p_good_to_bad: float = 0.03,
    p_bad_to_good: float = 0.15,
    dt_years: float = 8 / (24 * 365),
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic funding CF paths under a two-regime Markov model.

    Returns array of shape (n_paths, n_intervals) in per-interval CF units.

    Parameters
    ----------
    good_mean_apr, good_vol_apr : annualized APR mean/vol in good regime
    bad_mean_apr, bad_vol_apr   : annualized APR mean/vol in bad regime
    p_good_to_bad               : per-interval transition probability
    p_bad_to_good               : per-interval transition probability
    """
    rng = np.random.default_rng(seed)

    # Stationary probability of starting in bad state
    p_bad_stationary = p_good_to_bad / (p_good_to_bad + p_bad_to_good)

    paths = np.empty((n_paths, n_intervals))
    for p in range(n_paths):
        state = 1 if rng.random() < p_bad_stationary else 0  # 0=good, 1=bad
        for i in range(n_intervals):
            if state == 0:
                apr = rng.normal(good_mean_apr, good_vol_apr)
                if rng.random() < p_good_to_bad:
                    state = 1
            else:
                apr = rng.normal(bad_mean_apr, bad_vol_apr)
                if rng.random() < p_bad_to_good:
                    state = 0
            paths[p, i] = apr * dt_years

    return paths
