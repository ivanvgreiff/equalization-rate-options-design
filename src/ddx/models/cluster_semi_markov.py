"""Episode-based semi-Markov simulator for funding-rate series.

Decomposes the historical funding series into alternating calm segments and
stress clusters, then generates synthetic paths by resampling whole segments.
This preserves within-episode severity correlation, non-geometric durations,
and macro-level crisis clustering that the i.i.d.-emission Markov model misses.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


def extract_episodes_and_clusters(
    funding_cf: np.ndarray,
    threshold_b: float = 0.0001,
    gap_g: int = 3,
) -> dict:
    """Decompose funding series into calm segments and stress clusters.

    A micro stress episode is a contiguous run where f_t < -b.
    A macro cluster merges micro episodes separated by gaps <= g intervals,
    including the gap (non-stress) intervals within the cluster.

    Parameters
    ----------
    funding_cf : 1-D funding CF array.
    threshold_b : Stress threshold (same as DAF's b parameter).
    gap_g : Maximum gap (in intervals) between micro episodes to merge
            into a single cluster. Default 3 (= 24h for 8h cadence).

    Returns
    -------
    Dict with 'clusters', 'calm_segments' (lists of arrays),
    per-cluster statistics, and metadata.
    """
    n = len(funding_cf)
    stress = funding_cf < -threshold_b

    micro_episodes = []
    ep_start = None
    for i in range(n):
        if stress[i] and ep_start is None:
            ep_start = i
        elif not stress[i] and ep_start is not None:
            micro_episodes.append((ep_start, i))
            ep_start = None
    if ep_start is not None:
        micro_episodes.append((ep_start, n))

    if len(micro_episodes) == 0:
        return {
            "clusters": [],
            "calm_segments": [funding_cf.copy()],
            "cluster_stats": [],
            "n_clusters": 0,
            "n_calm": 1,
            "n_total": n,
            "threshold_b": threshold_b,
            "gap_g": gap_g,
        }

    merged = []
    cur_start, cur_end = micro_episodes[0]
    for i in range(1, len(micro_episodes)):
        next_start, next_end = micro_episodes[i]
        if next_start - cur_end <= gap_g:
            cur_end = next_end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = next_start, next_end
    merged.append((cur_start, cur_end))

    clusters = []
    calm_segments = []
    cluster_stats = []
    prev_end = 0

    for cs, ce in merged:
        if cs > prev_end:
            calm_segments.append(funding_cf[prev_end:cs].copy())
        cluster_vec = funding_cf[cs:ce].copy()
        clusters.append(cluster_vec)
        losses = np.maximum(0.0, -cluster_vec)
        cluster_stats.append({
            "start": cs,
            "end": ce,
            "length": ce - cs,
            "total_loss": float(np.sum(losses)),
            "max_draw": float(np.max(losses)),
            "mean_severity": float(np.mean(losses[losses > 0])) if np.any(losses > 0) else 0.0,
        })
        prev_end = ce

    if prev_end < n:
        calm_segments.append(funding_cf[prev_end:n].copy())

    return {
        "clusters": clusters,
        "calm_segments": calm_segments,
        "cluster_stats": cluster_stats,
        "n_clusters": len(clusters),
        "n_calm": len(calm_segments),
        "n_total": n,
        "threshold_b": threshold_b,
        "gap_g": gap_g,
    }


def fit_cluster_tail(
    clusters: list[np.ndarray],
    quantile_threshold: float = 0.90,
) -> dict:
    """Fit GPD to the upper tail of cluster total losses.

    Parameters
    ----------
    clusters : List of cluster arrays (from extract_episodes_and_clusters).
    quantile_threshold : Quantile of cluster total losses for the POT threshold.
    """
    total_losses = np.array([float(np.sum(np.maximum(0.0, -c))) for c in clusters])

    if len(total_losses) < 10:
        return {"fit_success": False, "n_clusters": len(total_losses)}

    u = float(np.quantile(total_losses, quantile_threshold))
    exceedances = total_losses[total_losses > u] - u

    if len(exceedances) < 5:
        return {
            "fit_success": False,
            "threshold_u": u,
            "n_exceedances": len(exceedances),
            "n_clusters": len(total_losses),
        }

    shape, _, scale = genpareto.fit(exceedances, floc=0)

    return {
        "fit_success": True,
        "threshold_u": u,
        "n_exceedances": len(exceedances),
        "n_clusters": len(total_losses),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "total_losses": total_losses,
    }


def simulate_semi_markov(
    clusters: list[np.ndarray],
    calm_segments: list[np.ndarray],
    n_intervals: int = 10_000,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
    tail_params: dict | None = None,
    p_augment: float = 0.0,
    cap: float = 0.00375,
) -> np.ndarray:
    """Generate synthetic funding series by alternating calm/cluster segments.

    Samples whole segments with replacement from the empirical pools,
    preserving within-episode correlation and duration structure.

    Parameters
    ----------
    clusters : List of cluster arrays.
    calm_segments : List of calm-segment arrays.
    n_intervals : Length of each synthetic path.
    n_paths : Number of independent paths.
    rng : NumPy random Generator.
    tail_params : If provided and fit_success, occasionally generate
                  scaled-up clusters for tail augmentation.
    p_augment : Probability of tail-augmenting a cluster (0 = no augmentation).
    cap : Venue cap on per-interval loss magnitude.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_cl = len(clusters)
    n_ca = len(calm_segments)

    use_tail = (
        tail_params is not None
        and tail_params.get("fit_success", False)
        and p_augment > 0
    )

    paths = np.empty((n_paths, n_intervals))

    for p in range(n_paths):
        segments = []
        total = 0
        in_calm = rng.random() > (n_cl / (n_cl + n_ca))

        while total < n_intervals:
            if in_calm:
                seg = calm_segments[rng.integers(0, n_ca)].copy()
            else:
                seg = clusters[rng.integers(0, n_cl)].copy()
                if use_tail and rng.random() < p_augment:
                    seg = _augment_cluster(
                        seg, tail_params, rng, cap
                    )
            segments.append(seg)
            total += len(seg)
            in_calm = not in_calm

        path = np.concatenate(segments)[:n_intervals]
        paths[p] = path

    return paths


def _augment_cluster(
    cluster: np.ndarray,
    tail_params: dict,
    rng: np.random.Generator,
    cap: float,
) -> np.ndarray:
    """Scale a cluster's loss profile to a GPD-drawn target total loss."""
    xi = tail_params["shape_xi"]
    sigma = tail_params["scale_sigma"]
    u = tail_params["threshold_u"]

    exc = genpareto.rvs(xi, scale=sigma, random_state=rng)
    target_S = u + exc

    losses = np.maximum(0.0, -cluster)
    current_S = losses.sum()

    if current_S <= 0:
        return cluster.copy()

    scale_factor = target_S / current_S
    scaled_losses = losses * scale_factor
    scaled_losses = np.minimum(scaled_losses, cap)

    result = cluster.copy()
    stress_mask = losses > 0
    result[stress_mask] = -scaled_losses[stress_mask]

    return result
