"""Compute descriptive funding analytics and regime statistics.

Runs on a single venue's processed parquet. Execute once per venue.

Usage:
    python scripts/run_descriptives.py --data data/processed/bybit_btcusd.parquet --venue-label bybit
    python scripts/run_descriptives.py --data data/processed/bitmex_xbtusd.parquet --venue-label bitmex

Outputs to reports/figures/ and reports/tables/ with venue-specific naming.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ddx.data.io import load_processed
from ddx.risk.metrics import negative_episode_stats, extract_episodes, total_loss
from ddx.backtest.rolling import rolling_windows_regular, rolling_windows
from ddx.viz.plots import (
    plot_funding_timeline,
    plot_streak_distributions,
    plot_rolling_regime_markers,
    plot_distribution_analysis,
    plot_top_episodes,
)
from ddx.utils.config import load_analysis_config


def _streak_lengths(funding_cf: np.ndarray, threshold_b: float) -> np.ndarray:
    """Compute all completed negative-streak lengths at a given threshold."""
    bad = funding_cf < -threshold_b
    streaks: list[int] = []
    current = 0
    for b in bad:
        if b:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    return np.array(streaks) if streaks else np.array([], dtype=int)


def _markov_transitions(funding_cf: np.ndarray, threshold_b: float) -> dict:
    """Estimate empirical Markov transition probabilities for bad/good states."""
    bad = funding_cf < -threshold_b
    n = len(bad)
    bb = gg = bg = gb = 0
    for i in range(1, n):
        if bad[i - 1] and bad[i]:
            bb += 1
        elif bad[i - 1] and not bad[i]:
            bg += 1
        elif not bad[i - 1] and bad[i]:
            gb += 1
        else:
            gg += 1

    total_from_bad = bb + bg
    total_from_good = gb + gg
    p_bb = bb / total_from_bad if total_from_bad > 0 else 0.0
    p_bg = bg / total_from_bad if total_from_bad > 0 else 0.0
    p_gb = gb / total_from_good if total_from_good > 0 else 0.0
    p_gg = gg / total_from_good if total_from_good > 0 else 0.0

    expected_bad_run = 1.0 / p_bg if p_bg > 0 else float("inf")

    return {
        "threshold_b": threshold_b,
        "P(bad->bad)": round(p_bb, 4),
        "P(bad->good)": round(p_bg, 4),
        "P(good->bad)": round(p_gb, 4),
        "P(good->good)": round(p_gg, 4),
        "expected_bad_run_intervals": round(expected_bad_run, 2),
        "pct_time_in_bad": round(float(np.mean(bad)) * 100, 2),
    }


def _auto_discover_events(
    df: pd.DataFrame,
    cf: np.ndarray,
    window: int,
    n_worst: int = 5,
    n_longest: int = 5,
) -> list[dict]:
    """Find the worst 30d windows and longest negative streaks."""
    events = []

    # Worst windows by aggregate loss Lambda
    if "is_regular" in df.columns:
        wins, starts = rolling_windows_regular(cf, df["is_regular"].values, window)
    else:
        wins = rolling_windows(cf, window)
        starts = np.arange(len(wins))

    if len(wins) > 0:
        lambdas = np.array([total_loss(w) for w in wins])
        worst_idx = np.argsort(lambdas)[-n_worst:][::-1]

        for rank, idx in enumerate(worst_idx):
            si = starts[idx]
            ts_start = df["timestamp"].iloc[si]
            ts_end = df["timestamp"].iloc[min(si + window - 1, len(df) - 1)]
            ts_train = ts_start - pd.Timedelta(days=1)
            events.append({
                "name": f"Worst-Lambda-{rank+1} ({ts_start.strftime('%Y-%m-%d')})",
                "description": f"Auto-discovered: rank {rank+1} worst 30d window by aggregate loss (Lambda={lambdas[idx]:.6f})",
                "start": ts_start.isoformat(),
                "end": ts_end.isoformat(),
                "premium_train_end": ts_train.isoformat(),
                "metric_value": float(lambdas[idx]),
            })

    # Longest negative streaks (b=0)
    streaks_raw = _streak_lengths(cf, 0.0)
    if len(streaks_raw) > 0:
        is_neg = cf < 0
        streak_records = []
        current_start = None
        current_len = 0
        for i, neg in enumerate(is_neg):
            if neg:
                if current_start is None:
                    current_start = i
                current_len += 1
            else:
                if current_start is not None:
                    streak_records.append((current_start, current_len))
                    current_start = None
                    current_len = 0
        if current_start is not None:
            streak_records.append((current_start, current_len))

        streak_records.sort(key=lambda x: x[1], reverse=True)
        for rank, (si, dur) in enumerate(streak_records[:n_longest]):
            ts_start = df["timestamp"].iloc[si]
            ts_end = df["timestamp"].iloc[min(si + dur - 1, len(df) - 1)]
            ts_train = ts_start - pd.Timedelta(days=1)
            events.append({
                "name": f"Longest-Streak-{rank+1} ({ts_start.strftime('%Y-%m-%d')})",
                "description": f"Auto-discovered: rank {rank+1} longest negative streak ({dur} intervals)",
                "start": ts_start.isoformat(),
                "end": ts_end.isoformat(),
                "premium_train_end": ts_train.isoformat(),
                "metric_value": dur,
            })

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--venue-label", default=None,
                        help="Label for output file naming (default: inferred from filename)")
    args = parser.parse_args()

    venue = args.venue_label
    if venue is None:
        venue = Path(args.data).stem.split("_")[0]

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("configs").mkdir(parents=True, exist_ok=True)

    df = load_processed(args.data)
    cf = df["funding_cf"].values
    config = load_analysis_config()

    print(f"=== Descriptives for {venue} ({len(cf)} intervals) ===\n")

    # ---- Basic stats ----
    stats = {
        "venue": venue,
        "n_intervals": len(cf),
        "date_start": str(df["timestamp"].iloc[0]),
        "date_end": str(df["timestamp"].iloc[-1]),
        "mean_cf": float(np.mean(cf)),
        "std_cf": float(np.std(cf)),
        "min_cf": float(np.min(cf)),
        "max_cf": float(np.max(cf)),
        "median_cf": float(np.median(cf)),
        "pct_negative": float(np.mean(cf < 0) * 100),
    }
    if "is_regular" in df.columns:
        stats["n_irregular"] = int((~df["is_regular"]).sum())

    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ---- Negative episode stats ----
    ep_stats = negative_episode_stats(cf)
    print("\n  Episodes:")
    for k, v in ep_stats.items():
        print(f"    {k}: {v}")

    # ---- Loss quantiles ----
    losses = np.maximum(0.0, -cf)
    loss_q_levels = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    loss_quantiles = {
        f"l_i_p{int(q*100)}": float(np.quantile(losses, q))
        for q in loss_q_levels
    }

    # ---- Markov transitions ----
    thresholds = [0.0, 0.00002, 0.00005, 0.0001, 0.0003]
    transitions = [_markov_transitions(cf, b) for b in thresholds]
    for t in transitions:
        print(f"  b={t['threshold_b']}: E[bad run]={t['expected_bad_run_intervals']} intervals")

    # ---- Streak lengths ----
    streak_data: dict[float, np.ndarray] = {}
    streak_rows = []
    for b in thresholds:
        sl = _streak_lengths(cf, b)
        streak_data[b] = sl
        for length in sl:
            streak_rows.append({"threshold_b": b, "streak_length": int(length)})
        if len(sl) > 0:
            print(f"  Streaks b={b}: mean={np.mean(sl):.1f}, median={np.median(sl):.0f}, max={np.max(sl)}")

    # ---- Rolling 30d ----
    h30 = next((h for h in config["horizons"] if h["name"] == "30d"), None)
    lambda_quantiles = {}
    rolling_mean_arr = None
    rolling_pct_neg_arr = None
    rolling_p01_arr = None
    rolling_timestamps = None

    if h30:
        window = h30["intervals"]
        if "is_regular" in df.columns:
            wins, win_starts = rolling_windows_regular(
                cf, df["is_regular"].values, window
            )
        else:
            wins = rolling_windows(cf, window)
            win_starts = np.arange(len(wins))

        if len(wins) > 0:
            rolling_lambda = np.array([total_loss(w) for w in wins])
            for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
                lambda_quantiles[f"Lambda_30d_p{int(q*100)}"] = float(
                    np.quantile(rolling_lambda, q)
                )

            rolling_mean_arr = np.mean(wins, axis=1)
            rolling_pct_neg_arr = np.mean(wins < 0, axis=1)
            rolling_p01_arr = np.quantile(wins, 0.01, axis=1)
            rolling_timestamps = df["timestamp"].iloc[win_starts].values

    # ---- Auto-discover stress events ----
    auto_events = []
    if h30 and len(cf) > h30["intervals"]:
        auto_events = _auto_discover_events(df, cf, h30["intervals"])
        print(f"\n  Auto-discovered {len(auto_events)} events")

    # ---- Save tables ----
    all_quantiles = {**loss_quantiles, **lambda_quantiles}
    pd.DataFrame([{**stats, **ep_stats, **all_quantiles}]).to_csv(
        f"reports/tables/descriptives_{venue}.csv", index=False
    )
    pd.DataFrame(transitions).to_csv(
        f"reports/tables/markov_transitions_{venue}.csv", index=False
    )
    if streak_rows:
        pd.DataFrame(streak_rows).to_csv(
            f"reports/tables/streak_lengths_{venue}.csv", index=False
        )
    if all_quantiles:
        pd.DataFrame([all_quantiles]).to_csv(
            f"reports/tables/loss_quantiles_{venue}.csv", index=False
        )
    if rolling_mean_arr is not None:
        pd.DataFrame({
            "timestamp": rolling_timestamps,
            "rolling_mean": rolling_mean_arr,
            "rolling_pct_neg": rolling_pct_neg_arr,
            "rolling_p01": rolling_p01_arr,
        }).to_csv(f"reports/tables/rolling_regime_{venue}.csv", index=False)

    # ---- Save auto events ----
    if auto_events:
        auto_path = Path("configs/events_auto.yaml")
        existing = {}
        if auto_path.exists():
            with open(auto_path) as fh:
                existing = yaml.safe_load(fh) or {}
        existing[f"events_{venue}"] = auto_events
        with open(auto_path, "w") as fh:
            yaml.dump(existing, fh, default_flow_style=False, sort_keys=False)
        print(f"  Saved: configs/events_auto.yaml (section: events_{venue})")

    # ---- Plots ----
    plot_funding_timeline(
        df["timestamp"].values, cf,
        title=f"{venue} Funding Rate History",
        save_path=f"reports/figures/funding_timeline_{venue}.png",
    )
    plt_closed()

    plot_streak_distributions(
        streak_data, title_prefix=f"{venue}: ",
        save_path=f"reports/figures/streak_distributions_{venue}.png",
    )
    plt_closed()

    if rolling_mean_arr is not None:
        plot_rolling_regime_markers(
            rolling_timestamps, rolling_mean_arr, rolling_pct_neg_arr,
            rolling_p01_arr,
            title=f"{venue}: Rolling 30d Regime Markers",
            save_path=f"reports/figures/rolling_regime_markers_{venue}.png",
        )
        plt_closed()

    plot_distribution_analysis(
        cf, title_prefix=f"{venue}: ",
        save_path=f"reports/figures/distribution_analysis_{venue}.png",
    )
    plt_closed()

    episodes = extract_episodes(cf)
    if episodes:
        plot_top_episodes(
            episodes, df["timestamp"].values, metric="duration", n=20,
            title=f"{venue}: Top 20 Episodes by Duration",
            save_path=f"reports/figures/top_episodes_duration_{venue}.png",
        )
        plt_closed()
        plot_top_episodes(
            episodes, df["timestamp"].values, metric="total_loss", n=20,
            title=f"{venue}: Top 20 Episodes by Total Loss",
            save_path=f"reports/figures/top_episodes_severity_{venue}.png",
        )
        plt_closed()

    print(f"\n  All outputs saved for {venue}.")


def plt_closed():
    """Close all matplotlib figures to free memory."""
    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    main()
