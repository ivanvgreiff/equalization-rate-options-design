# Bybit BTCUSD Inverse Perp Funding Rate - Data QA Report

*Generated: 2026-02-23 18:52 UTC*

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total rows | 7,971 |
| First timestamp | 2018-11-15 00:00:00+00:00 |
| Last timestamp | 2026-02-23 16:00:00+00:00 |
| Span (days) | 2657.7 |
| Span (years) | 7.28 |

## Interval Analysis

### dt_hours Distribution

| dt_hours (rounded) | Count |
|-------------------|-------|
| 8.0 | 7,968 |
| 16.0 | 3 |

- **Daily (~24h) intervals**: 0
- **Standard (~8h) intervals**: 7,968
- **Irregular intervals** (outside 8h +/- 0.5h): 3 (0.04%)
- **Gaps** (dt > 8.5h): 3

### Gap Details (first 20)

| Index | Timestamp | dt_hours |
|-------|-----------|----------|
| 511 | 2019-05-04 16:00:00+00:00 | 16.00 |
| 681 | 2019-06-30 16:00:00+00:00 | 16.00 |
| 749 | 2019-07-23 16:00:00+00:00 | 16.00 |

## Funding CF Statistics

| Metric | Value |
|--------|-------|
| Mean | 0.00010664 |
| Std | 0.00034353 |
| Min | -0.00375000 |
| Max | 0.00375000 |
| Median | 0.00010000 |
| Skewness | 1.4095 |
| Excess kurtosis | 37.3416 |

### Sign Breakdown

| Direction | % of intervals |
|-----------|----------------|
| Positive (f > 0) | 81.65% |
| Zero (f = 0) | 0.00% |
| Negative (f < 0) | 18.35% |

### Quantiles

| Quantile | Value |
|----------|-------|
| 0.1% | -0.00260727 |
| 1.0% | -0.00065812 |
| 5.0% | -0.00016975 |
| 10.0% | -0.00006479 |
| 25.0% | 0.00003740 |
| 50.0% | 0.00010000 |
| 75.0% | 0.00010000 |
| 90.0% | 0.00023256 |
| 95.0% | 0.00053327 |
| 99.0% | 0.00157451 |
| 99.9% | 0.00298810 |

## Loss Statistics

| Metric | Value |
|--------|-------|
| Total cumulative loss | 0.270587 |
| Mean loss per interval (incl. zeros) | 0.00003395 |
| Mean loss given negative | 0.00018495 |
