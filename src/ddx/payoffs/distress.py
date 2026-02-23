"""Distress-Activated Floor & Soft-Duration Cover — Products 2 and 4.

Product 2 (hard activation):
    A_i = 1 if L_i >= m  (L_i = consecutive-bad run length)
    Payoff = min(L, sum_i A_i * max(0, -f_i - d))

Product 4 (soft ramp):
    w(L_i) ramps from 0->1 over [m, m+s]
    Payoff = min(L, sum_i w(L_i) * max(0, -f_i - d))
"""

from __future__ import annotations

import numpy as np


def _run_lengths(funding_cf: np.ndarray, threshold_b: float) -> np.ndarray:
    """Compute consecutive-bad run lengths.

    Bad state: f_i < -threshold_b  (i.e. loss exceeds threshold).
    threshold_b >= 0 in CF units.
    """
    bad = (funding_cf < -threshold_b).astype(np.int32)
    runs = np.zeros(len(bad), dtype=np.int32)
    runs[0] = bad[0]
    for i in range(1, len(bad)):
        runs[i] = (runs[i - 1] + 1) * bad[i]
    return runs


def distress_activated_floor(
    funding_cf: np.ndarray,
    threshold_b: float = 0.0,
    streak_m: int = 3,
    deductible: float = 0.0,
    cap: float | None = None,
) -> float:
    """Product 2: persistence-gated floor.

    Parameters
    ----------
    funding_cf   : per-interval CFs (buyer perspective)
    threshold_b  : bad-state threshold (CF units, >= 0)
    streak_m     : consecutive bad intervals to activate
    deductible   : per-interval deductible after activation
    cap          : aggregate payout cap
    """
    runs = _run_lengths(funding_cf, threshold_b)
    active = (runs >= streak_m).astype(np.float64)
    interval_payoffs = active * np.maximum(0.0, -funding_cf - deductible)
    total = float(np.sum(interval_payoffs))
    if cap is not None:
        total = min(total, cap)
    return total


def soft_duration_cover(
    funding_cf: np.ndarray,
    threshold_b: float = 0.0,
    streak_m: int = 3,
    ramp_s: int = 3,
    deductible: float = 0.0,
    cap: float | None = None,
) -> float:
    """Product 4: soft-ramp activation to reduce cliff effects.

    w(L) = 0            if L < m
    w(L) = (L-m)/s      if m <= L < m+s
    w(L) = 1            if L >= m+s
    """
    runs = _run_lengths(funding_cf, threshold_b)
    weights = np.clip((runs - streak_m) / ramp_s, 0.0, 1.0)
    interval_payoffs = weights * np.maximum(0.0, -funding_cf - deductible)
    total = float(np.sum(interval_payoffs))
    if cap is not None:
        total = min(total, cap)
    return total
