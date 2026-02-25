"""Block bootstrap for funding-rate time series.

Provides circular block bootstrap resampling and premium confidence intervals.
Block bootstrap preserves temporal dependence (streaks, regimes) that i.i.d.
resampling would destroy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ddx.backtest.rolling import rolling_payoffs, rolling_windows
from ddx.pricing.premium import full_premium


def circular_block_bootstrap(
    series: np.ndarray,
    block_size: int,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate bootstrap resamples using circular block bootstrap.

    The series is treated as circular (wraps from end to start).  For each
    sample, ``ceil(n / block_size)`` random start positions are drawn
    uniformly, blocks of ``block_size`` consecutive observations are
    extracted (wrapping around), concatenated, and trimmed to length ``n``.

    Parameters
    ----------
    series : 1-D array of length n.
    block_size : Number of consecutive observations per block.
    n_samples : Number of bootstrap resamples to generate.
    rng : NumPy random Generator (for reproducibility).

    Returns
    -------
    2-D array of shape ``(n_samples, n)``.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(series)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) > series length ({n})")

    n_blocks = int(np.ceil(n / block_size))
    total_len = n_blocks * block_size

    starts = rng.integers(0, n, size=(n_samples, n_blocks))

    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_samples, total_len)[:, :n]

    return series[idx]


def _vectorized_floor_payoffs(windows: np.ndarray, deductible: float) -> np.ndarray:
    """Vectorized vanilla floor payoffs across all windows at once."""
    return np.sum(np.maximum(0.0, -windows - deductible), axis=1)


def _vectorized_asl_payoffs(windows: np.ndarray, deductible_D: float) -> np.ndarray:
    """Vectorized ASL payoffs across all windows at once."""
    lambdas = np.sum(np.maximum(0.0, -windows), axis=1)
    return np.maximum(0.0, lambdas - deductible_D)


def _vectorized_daf_payoffs(
    windows: np.ndarray, threshold_b: float, streak_m: int, deductible: float
) -> np.ndarray:
    """Vectorized DAF payoffs — processes all windows in parallel per timestep."""
    n_windows, w_size = windows.shape
    bad = (windows < -threshold_b).astype(np.int32)
    runs = np.zeros(n_windows, dtype=np.int32)
    total = np.zeros(n_windows, dtype=np.float64)
    for j in range(w_size):
        runs = (runs + 1) * bad[:, j]
        active = runs >= streak_m
        total += active * np.maximum(0.0, -windows[:, j] - deductible)
    return total


def bootstrap_premiums(
    funding_cf: np.ndarray,
    window_size: int,
    payoff_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 1000,
    block_size: int = 90,
    ci_level: float = 0.90,
    rng: np.random.Generator | None = None,
    lam: float = 0.35,
    cost_of_capital: float = 0.12,
    horizon_years: float = 30 / 365,
    alpha: float = 0.01,
    _vectorized_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Block-bootstrap confidence intervals for premium decomposition.

    For each bootstrap sample:
    1. Resample the funding series via circular block bootstrap.
    2. Compute rolling-window payoffs on the resampled series.
    3. Compute full premium decomposition.

    Parameters
    ----------
    funding_cf : Original funding CF series (1-D).
    window_size : Rolling window size in intervals.
    payoff_fn : Function(window_cf) -> scalar payoff.
    n_bootstrap : Number of bootstrap resamples.
    block_size : Block size for circular block bootstrap.
    ci_level : Confidence level (e.g. 0.90 for 90% CI).
    rng : NumPy random Generator.
    lam, cost_of_capital, horizon_years, alpha : Premium parameters.
    _vectorized_fn : Optional vectorized payoff function(windows_2d) -> 1d payoffs.
        When provided, bypasses the per-window Python loop for a ~10x speedup.

    Returns
    -------
    Dict with keys:
        ``samples_pure``, ``samples_total``, ``samples_risk_load``,
        ``samples_capital_charge`` : 1-D arrays of length n_bootstrap.
        ``ci_lower``, ``ci_upper`` : Dicts with per-component CI bounds.
        ``mean`` : Dict with per-component means across bootstrap samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    resamples = circular_block_bootstrap(
        funding_cf, block_size, n_bootstrap, rng
    )

    tail = (1 - ci_level) / 2
    q_lo, q_hi = tail, 1 - tail

    pure_arr = np.empty(n_bootstrap)
    rl_arr = np.empty(n_bootstrap)
    cc_arr = np.empty(n_bootstrap)
    total_arr = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        if _vectorized_fn is not None:
            wins = rolling_windows(resamples[i], window_size)
            payoffs = _vectorized_fn(wins)
        else:
            payoffs = rolling_payoffs(resamples[i], window_size, payoff_fn)
        fp = full_premium(payoffs, lam, cost_of_capital, horizon_years, alpha)
        pure_arr[i] = fp["pure"]
        rl_arr[i] = fp["risk_load"]
        cc_arr[i] = fp["capital_charge"]
        total_arr[i] = fp["total"]

    components = {
        "pure": pure_arr,
        "risk_load": rl_arr,
        "capital_charge": cc_arr,
        "total": total_arr,
    }

    ci_lower = {k: float(np.quantile(v, q_lo)) for k, v in components.items()}
    ci_upper = {k: float(np.quantile(v, q_hi)) for k, v in components.items()}
    mean = {k: float(np.mean(v)) for k, v in components.items()}

    return {
        "samples_pure": pure_arr,
        "samples_total": total_arr,
        "samples_risk_load": rl_arr,
        "samples_capital_charge": cc_arr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean": mean,
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }
