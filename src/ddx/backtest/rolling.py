"""Rolling-window engine for computing payoff distributions over history."""

from __future__ import annotations

from typing import Callable

import numpy as np


def rolling_windows(funding_cf: np.ndarray, window_size: int) -> np.ndarray:
    """Generate rolling windows from a 1-D array.

    Returns a 2-D array of shape (n_windows, window_size).
    """
    n = len(funding_cf)
    if n < window_size:
        raise ValueError(f"Series length {n} < window size {window_size}")
    n_windows = n - window_size + 1
    idx = np.arange(window_size)[None, :] + np.arange(n_windows)[:, None]
    return funding_cf[idx]


def rolling_payoffs(
    funding_cf: np.ndarray,
    window_size: int,
    payoff_fn: Callable[[np.ndarray], float],
) -> np.ndarray:
    """Apply a payoff function to each rolling window.

    Parameters
    ----------
    funding_cf  : full history of per-interval CFs
    window_size : number of intervals per window
    payoff_fn   : function(window_cf) -> scalar payoff

    Returns
    -------
    Array of payoff values, one per window.
    """
    windows = rolling_windows(funding_cf, window_size)
    return np.array([payoff_fn(w) for w in windows])
