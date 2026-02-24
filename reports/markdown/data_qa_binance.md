# Binance COIN-M BTCUSD_PERP Funding Rate - Data QA Report

*Generated: 2026-02-23 19:08 UTC*

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 6,070 |
| First timestamp | 2020-08-10 16:00:00+00:00 |
| Last timestamp | 2026-02-23 16:00:00+00:00 |
| Span (days) | 2023.0 |
| Span (years) | 5.54 |

## Interval Analysis

### dt_hours Distribution

| dt_hours (rounded) | Count |
|-------------------|-------|
| 8.0 | 6,070 |

- **Daily (~24h) intervals**: 0
- **Standard (~8h) intervals**: 6,070
- **Irregular intervals** (outside 8h +/- 0.5h): 0 (0.00%)
- **Gaps** (dt > 8.5h): 0

## Funding CF Statistics

| Metric | Value |
|--------|-------|
| Mean | 0.00009193 |
| Std | 0.00018791 |
| Min | -0.00300000 |
| Max | 0.00186012 |
| Median | 0.00010000 |
| Skewness | 2.2529 |
| Excess kurtosis | 31.7397 |

### Sign Breakdown

| Direction | % of intervals |
|-----------|----------------|
| Positive (f > 0) | 82.29% |
| Zero (f = 0) | 0.00% |
| Negative (f < 0) | 17.71% |

### Quantiles

| Quantile | Value |
|----------|-------|
| 0.1% | -0.00080762 |
| 1.0% | -0.00022198 |
| 5.0% | -0.00009367 |
| 10.0% | -0.00004094 |
| 25.0% | 0.00002672 |
| 50.0% | 0.00010000 |
| 75.0% | 0.00010000 |
| 90.0% | 0.00012711 |
| 95.0% | 0.00037681 |
| 99.0% | 0.00094700 |
| 99.9% | 0.00159021 |

## Loss Statistics

| Metric | Value |
|--------|-------|
| Total cumulative loss | 0.088159 |
| Mean loss per interval (incl. zeros) | 0.00001452 |
| Mean loss given negative | 0.00008201 |
