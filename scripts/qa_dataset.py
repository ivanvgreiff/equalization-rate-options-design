"""Quality-assurance report for the processed BitMEX funding dataset.

Loads the processed parquet, computes summary statistics and diagnostics,
prints a report to stdout, and writes a markdown report to reports/markdown/.

Usage:
    python scripts/qa_dataset.py
    python scripts/qa_dataset.py --input data/processed/bitmex_xbtusd.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddx.data.io import load_processed
from ddx.data.schema import INTERVAL_HOURS, INTERVAL_TOLERANCE_HOURS


def compute_qa(df: pd.DataFrame) -> dict:
    """Compute all QA statistics. Returns a dict of results."""
    results: dict = {}

    ts = df["timestamp"]
    cf = df["funding_cf"].values

    # --- Basic shape ---
    results["total_rows"] = len(df)
    results["ts_min"] = str(ts.iloc[0])
    results["ts_max"] = str(ts.iloc[-1])

    span = ts.iloc[-1] - ts.iloc[0]
    results["span_days"] = span.total_seconds() / 86400

    # --- Interval analysis ---
    dt = df["dt_hours"].values

    unique_dt, dt_counts = np.unique(np.round(dt, 2), return_counts=True)
    results["dt_distribution"] = list(zip(unique_dt.tolist(), dt_counts.tolist()))

    is_reg = df["is_regular"].values if "is_regular" in df.columns else (
        np.abs(dt - INTERVAL_HOURS) < INTERVAL_TOLERANCE_HOURS
    )
    n_irreg = int(np.sum(~is_reg))
    results["n_irregular"] = n_irreg
    results["pct_irregular"] = 100.0 * n_irreg / len(df) if len(df) > 0 else 0.0

    # Detect gaps: intervals where dt > expected + tolerance
    gap_threshold = INTERVAL_HOURS + INTERVAL_TOLERANCE_HOURS
    gaps = dt > gap_threshold
    gap_indices = np.where(gaps)[0]
    results["n_gaps"] = int(len(gap_indices))

    if len(gap_indices) > 0:
        gap_details = []
        for idx in gap_indices[:20]:  # show at most 20
            gap_details.append({
                "index": int(idx),
                "timestamp": str(ts.iloc[idx]),
                "dt_hours": float(dt[idx]),
            })
        results["gap_details"] = gap_details

    # --- Funding cadence transition ---
    # Detect where interval changes from ~24h to ~8h
    daily_mask = np.abs(dt - 24.0) < 1.0
    eight_h_mask = np.abs(dt - INTERVAL_HOURS) < INTERVAL_TOLERANCE_HOURS

    n_daily = int(np.sum(daily_mask))
    n_8h = int(np.sum(eight_h_mask))
    results["n_daily_intervals"] = n_daily
    results["n_8h_intervals"] = n_8h

    if n_daily > 0 and n_8h > 0:
        last_daily_idx = int(np.where(daily_mask)[0][-1])
        first_8h_idx = int(np.where(eight_h_mask)[0][0])
        results["last_daily_timestamp"] = str(ts.iloc[last_daily_idx])
        results["first_8h_timestamp"] = str(ts.iloc[first_8h_idx])

    # --- Funding CF statistics ---
    results["mean"] = float(np.mean(cf))
    results["std"] = float(np.std(cf, ddof=1))
    results["min"] = float(np.min(cf))
    results["max"] = float(np.max(cf))
    results["median"] = float(np.median(cf))

    results["pct_negative"] = 100.0 * float(np.mean(cf < 0))
    results["pct_zero"] = 100.0 * float(np.mean(cf == 0))
    results["pct_positive"] = 100.0 * float(np.mean(cf > 0))

    # Skewness and kurtosis (excess)
    z = (cf - np.mean(cf)) / np.std(cf, ddof=0)
    results["skewness"] = float(np.mean(z ** 3))
    results["kurtosis_excess"] = float(np.mean(z ** 4) - 3.0)

    # Quantiles
    quantile_levels = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    results["quantiles"] = {
        f"{q*100:.1f}%": float(np.quantile(cf, q)) for q in quantile_levels
    }

    # --- Loss statistics ---
    losses = np.maximum(0.0, -cf)
    results["total_loss"] = float(np.sum(losses))
    results["mean_loss_per_interval"] = float(np.mean(losses))
    results["mean_loss_given_negative"] = (
        float(np.mean(-cf[cf < 0])) if np.any(cf < 0) else 0.0
    )

    return results


def generate_rolling_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a rolling 30d mean plot of funding_cf."""
    ts = df["timestamp"]
    cf = df["funding_cf"]

    window = 90  # 30d = 90 intervals of 8h
    rolling_mean = cf.rolling(window=window, min_periods=window).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: raw funding rate
    axes[0].plot(ts, cf, linewidth=0.3, alpha=0.6, color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Funding CF (per interval)")
    axes[0].set_title("BitMEX XBTUSD Funding Rate - Full History")

    neg_mask = cf < 0
    axes[0].fill_between(
        ts, cf, 0,
        where=neg_mask,
        color="red", alpha=0.3, label="Negative funding"
    )
    axes[0].legend(loc="upper right", fontsize=8)

    # Panel 2: rolling mean
    axes[1].plot(ts, rolling_mean, linewidth=1.0, color="darkblue")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Rolling 30d Mean")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Rolling 30-Day Mean Funding Rate")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_report(results: dict, title: str = "Funding Rate") -> str:
    """Format QA results as a markdown string."""
    lines = []
    lines.append(f"# {title} - Data QA Report\n")
    lines.append(f"*Generated: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}*\n")

    lines.append("## Dataset Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total rows | {results['total_rows']:,} |")
    lines.append(f"| First timestamp | {results['ts_min']} |")
    lines.append(f"| Last timestamp | {results['ts_max']} |")
    lines.append(f"| Span (days) | {results['span_days']:.1f} |")
    lines.append(f"| Span (years) | {results['span_days']/365.25:.2f} |")
    lines.append("")

    lines.append("## Interval Analysis\n")
    lines.append("### dt_hours Distribution\n")
    lines.append("| dt_hours (rounded) | Count |")
    lines.append("|-------------------|-------|")
    for dt_val, count in results["dt_distribution"]:
        lines.append(f"| {dt_val} | {count:,} |")
    lines.append("")

    lines.append(f"- **Daily (~24h) intervals**: {results['n_daily_intervals']:,}")
    lines.append(f"- **Standard (~8h) intervals**: {results['n_8h_intervals']:,}")

    if "last_daily_timestamp" in results:
        lines.append(f"- **Last daily interval**: {results['last_daily_timestamp']}")
        lines.append(f"- **First 8h interval**: {results['first_8h_timestamp']}")

    lines.append(f"- **Irregular intervals** (outside 8h +/- 0.5h): "
                 f"{results['n_irregular']:,} ({results['pct_irregular']:.2f}%)")
    lines.append(f"- **Gaps** (dt > 8.5h): {results['n_gaps']}")
    lines.append("")

    if results.get("gap_details"):
        lines.append("### Gap Details (first 20)\n")
        lines.append("| Index | Timestamp | dt_hours |")
        lines.append("|-------|-----------|----------|")
        for g in results["gap_details"]:
            lines.append(f"| {g['index']} | {g['timestamp']} | {g['dt_hours']:.2f} |")
        lines.append("")

    lines.append("## Funding CF Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean | {results['mean']:.8f} |")
    lines.append(f"| Std | {results['std']:.8f} |")
    lines.append(f"| Min | {results['min']:.8f} |")
    lines.append(f"| Max | {results['max']:.8f} |")
    lines.append(f"| Median | {results['median']:.8f} |")
    lines.append(f"| Skewness | {results['skewness']:.4f} |")
    lines.append(f"| Excess kurtosis | {results['kurtosis_excess']:.4f} |")
    lines.append("")

    lines.append("### Sign Breakdown\n")
    lines.append(f"| Direction | % of intervals |")
    lines.append(f"|-----------|----------------|")
    lines.append(f"| Positive (f > 0) | {results['pct_positive']:.2f}% |")
    lines.append(f"| Zero (f = 0) | {results['pct_zero']:.2f}% |")
    lines.append(f"| Negative (f < 0) | {results['pct_negative']:.2f}% |")
    lines.append("")

    lines.append("### Quantiles\n")
    lines.append("| Quantile | Value |")
    lines.append("|----------|-------|")
    for q_label, q_val in results["quantiles"].items():
        lines.append(f"| {q_label} | {q_val:.8f} |")
    lines.append("")

    lines.append("## Loss Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total cumulative loss | {results['total_loss']:.6f} |")
    lines.append(f"| Mean loss per interval (incl. zeros) | "
                 f"{results['mean_loss_per_interval']:.8f} |")
    lines.append(f"| Mean loss given negative | "
                 f"{results['mean_loss_given_negative']:.8f} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="QA report for processed funding data")
    parser.add_argument(
        "--input",
        default="data/processed/bitmex_xbtusd.parquet",
        help="Path to processed parquet",
    )
    parser.add_argument(
        "--report-output",
        default="reports/markdown/data_qa.md",
        help="Output path for the markdown report",
    )
    parser.add_argument(
        "--plot-output",
        default="reports/figures/funding_qa_overview.png",
        help="Output path for the rolling-mean plot",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title (default: inferred from input filename)",
    )
    args = parser.parse_args()

    print(f"Loading processed data from {args.input}")
    df = load_processed(args.input)
    print(f"  {len(df)} rows loaded")

    print("\nComputing QA statistics...")
    results = compute_qa(df)

    title = args.title
    if title is None:
        stem = Path(args.input).stem
        title_map = {
            "bitmex_xbtusd": "BitMEX XBTUSD Funding Rate",
            "deribit_btcperp": "Deribit BTC-PERPETUAL Funding Rate",
            "bybit_btcusd": "Bybit BTCUSD Inverse Perp Funding Rate",
            "binance_btcusd": "Binance COIN-M BTCUSD_PERP Funding Rate",
        }
        title = title_map.get(stem, f"{stem} Funding Rate")

    report = format_report(results, title=title)
    print("\n" + report)

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nMarkdown report saved to {report_path}")

    print(f"\nGenerating overview plot...")
    generate_rolling_plot(df, Path(args.plot_output))
    print(f"  Plot saved to {args.plot_output}")


if __name__ == "__main__":
    main()
