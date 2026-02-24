# Deribit BTC-PERPETUAL Funding Rate - Data QA Report

*Generated: 2026-02-23 18:27 UTC*

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 7,474 |
| First timestamp | 2019-04-30 16:00:00+00:00 |
| Last timestamp | 2026-02-23 16:00:00+00:00 |
| Span (days) | 2491.0 |
| Span (years) | 6.82 |

## Interval Analysis

### dt_hours Distribution

| dt_hours (rounded) | Count |
|-------------------|-------|
| 8.0 | 7,474 |

- **Daily (~24h) intervals**: 0
- **Standard (~8h) intervals**: 7,474
- **Irregular intervals** (outside 8h +/- 0.5h): 0 (0.00%)
- **Gaps** (dt > 8.5h): 0

## Funding CF Statistics

| Metric | Value |
|--------|-------|
| Mean | 0.00007499 |
| Std | 0.00023734 |
| Min | -0.00488801 |
| Max | 0.00247490 |
| Median | 0.00001392 |
| Skewness | -2.4264 |
| Excess kurtosis | 87.6736 |

### Sign Breakdown

| Direction | % of intervals |
|-----------|----------------|
| Positive (f > 0) | 71.86% |
| Zero (f = 0) | 1.67% |
| Negative (f < 0) | 26.47% |

### Quantiles

| Quantile | Value |
|----------|-------|
| 0.1% | -0.00174076 |
| 1.0% | -0.00033466 |
| 5.0% | -0.00006341 |
| 10.0% | -0.00001886 |
| 25.0% | -0.00000006 |
| 50.0% | 0.00001392 |
| 75.0% | 0.00010192 |
| 90.0% | 0.00027569 |
| 95.0% | 0.00044498 |
| 99.0% | 0.00087692 |
| 99.9% | 0.00175037 |

## Loss Statistics

| Metric | Value |
|--------|-------|
| Total cumulative loss | 0.131881 |
| Mean loss per interval (incl. zeros) | 0.00001765 |
| Mean loss given negative | 0.00006667 |
