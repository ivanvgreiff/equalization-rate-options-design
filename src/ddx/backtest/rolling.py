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


def rolling_windows_regular(
    funding_cf: np.ndarray,
    is_regular: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rolling windows, skipping any that contain an irregular interval.

    Parameters
    ----------
    funding_cf  : 1-D CF array
    is_regular  : boolean array, same length as funding_cf
    window_size : number of intervals per window

    Returns
    -------
    (windows, start_indices) where windows has shape (n_valid, window_size)
    and start_indices records which positions in the original series each
    valid window starts at.
    """
    n = len(funding_cf)
    if n < window_size:
        raise ValueError(f"Series length {n} < window size {window_size}")

    is_reg = np.asarray(is_regular, dtype=bool)
    n_candidates = n - window_size + 1

    # For each candidate window, check that all intervals are regular.
    # Use a cumulative sum of *irregular* flags: a window is valid iff the
    # count of irregulars inside it is zero.
    irreg_cumsum = np.concatenate(([[0]], [np.cumsum(~is_reg)]), axis=None)
    # irreg_cumsum has length n+1; irregulars in window [i, i+w) =
    #   irreg_cumsum[i+w] - irreg_cumsum[i]
    window_irreg_count = (
        irreg_cumsum[window_size: window_size + n_candidates]
        - irreg_cumsum[:n_candidates]
    )
    valid_mask = window_irreg_count == 0
    valid_starts = np.where(valid_mask)[0]

    if len(valid_starts) == 0:
        return np.empty((0, window_size), dtype=funding_cf.dtype), valid_starts

    idx = np.arange(window_size)[None, :] + valid_starts[:, None]
    return funding_cf[idx], valid_starts


def rolling_payoffs(
    funding_cf: np.ndarray,
    window_size: int,
    payoff_fn: Callable[[np.ndarray], float],
    is_regular: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a payoff function to each rolling window.

    Parameters
    ----------
    funding_cf  : full history of per-interval CFs
    window_size : number of intervals per window
    payoff_fn   : function(window_cf) -> scalar payoff
    is_regular  : optional boolean array; if provided, windows containing
                  irregular intervals are skipped

    Returns
    -------
    Array of payoff values, one per valid window.
    """
    if is_regular is not None:
        windows, _ = rolling_windows_regular(funding_cf, is_regular, window_size)
    else:
        windows = rolling_windows(funding_cf, window_size)
    return np.array([payoff_fn(w) for w in windows])
