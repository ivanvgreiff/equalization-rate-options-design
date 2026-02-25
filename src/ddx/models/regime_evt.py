"""2-state regime-switching model with EVT tail augmentation.

Fits a Markov regime model to funding-rate data using observable state
assignment (not hidden), with empirical body resampling and GPD tail
extrapolation in the stress regime.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def fit_regime_model(
    funding_cf: np.ndarray,
    threshold_b: float = 0.0001,
) -> dict:
    """Fit a 2-state Markov regime model from a funding CF series.

    State 0 (normal): f_t >= -b
    State 1 (stress): f_t < -b

    Returns dict with transition matrix, state samples, and statistics.
    """
    states = (funding_cf < -threshold_b).astype(np.int32)
    n = len(states)

    counts = np.zeros((2, 2), dtype=np.int64)
    for t in range(n - 1):
        counts[states[t], states[t + 1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    transition_matrix = counts / row_sums

    p01 = transition_matrix[0, 1]
    p10 = transition_matrix[1, 0]
    denom = p01 + p10
    stationary = np.array([p10 / denom, p01 / denom]) if denom > 0 else np.array([0.5, 0.5])

    normal_samples = funding_cf[states == 0]
    stress_samples = funding_cf[states == 1]

    return {
        "threshold_b": threshold_b,
        "transition_matrix": transition_matrix,
        "stationary_dist": stationary,
        "normal_samples": normal_samples,
        "stress_samples": stress_samples,
        "n_normal": len(normal_samples),
        "n_stress": len(stress_samples),
        "n_total": n,
        "states": states,
        "expected_run_normal": 1.0 / p01 if p01 > 0 else np.inf,
        "expected_run_stress": 1.0 / p10 if p10 > 0 else np.inf,
    }


def fit_evt_tail(
    losses: np.ndarray,
    quantile_threshold: float = 0.95,
) -> dict:
    """Fit a GPD to the tail of a loss array (Peaks Over Threshold).

    Parameters
    ----------
    losses : positive loss values (e.g., max(0, -f_t) for stress intervals).
    quantile_threshold : quantile of losses to use as the POT threshold.

    Returns dict with GPD parameters (shape xi, scale sigma), threshold u,
    and exceedance count.
    """
    u = float(np.quantile(losses, quantile_threshold))
    exceedances = losses[losses > u] - u

    if len(exceedances) < 5:
        return {
            "threshold_u": u,
            "n_exceedances": len(exceedances),
            "shape_xi": 0.0,
            "scale_sigma": float(np.std(exceedances)) if len(exceedances) > 0 else 1e-6,
            "fit_success": False,
        }

    shape, _, scale = genpareto.fit(exceedances, floc=0)

    return {
        "threshold_u": u,
        "n_exceedances": len(exceedances),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "fit_success": True,
    }


def simulate_regime_evt(
    fitted_model: dict,
    evt_params: dict | None = None,
    n_intervals: int = 10_000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic funding series from the fitted regime-EVT model.

    Normal state: empirical resampling from normal_samples.
    Stress state: empirical body + GPD tail (if evt_params provided).

    Returns array of shape (n_paths, n_intervals).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    P = fitted_model["transition_matrix"]
    pi = fitted_model["stationary_dist"]
    normal_pool = fitted_model["normal_samples"]
    stress_pool = fitted_model["stress_samples"]

    if evt_params is not None and evt_params.get("fit_success", False):
        u = evt_params["threshold_u"]
        xi = evt_params["shape_xi"]
        sigma = evt_params["scale_sigma"]
        stress_losses = np.maximum(0.0, -stress_pool)
        body_mask = stress_losses <= u
        stress_body = stress_pool[body_mask]
        p_tail = 1.0 - np.mean(body_mask)
        use_evt = len(stress_body) > 0 and p_tail > 0
    else:
        use_evt = False
        stress_body = stress_pool

    paths = np.empty((n_paths, n_intervals))
    for p in range(n_paths):
        state = 1 if rng.random() < pi[1] else 0
        for t in range(n_intervals):
            if state == 0:
                paths[p, t] = rng.choice(normal_pool)
            else:
                if use_evt and rng.random() < p_tail:
                    exc = genpareto.rvs(xi, scale=sigma, random_state=rng)
                    paths[p, t] = -(u + exc)
                else:
                    paths[p, t] = rng.choice(stress_body)

            r = rng.random()
            state = 1 if r < P[state, 1] else 0

    return paths
