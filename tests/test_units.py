"""Tests for ddx.utils.units — unit conversion functions."""

import pytest
from ddx.utils.units import (
    INTERVALS_PER_YEAR,
    per_interval_to_apr,
    per_interval_to_apr_pct,
    window_sum_to_pct_notional,
    window_sum_to_annualized_pct,
    window_intervals_for_days,
    to_apr, to_apr_pct, to_pct_notional, to_ann_pct,
)


def test_intervals_per_year():
    assert INTERVALS_PER_YEAR == 1095.0


def test_per_interval_to_apr():
    assert per_interval_to_apr(0.0001) == pytest.approx(0.1095)
    assert per_interval_to_apr(0.0) == 0.0
    assert per_interval_to_apr(-0.001) == pytest.approx(-1.095)


def test_per_interval_to_apr_pct():
    assert per_interval_to_apr_pct(0.0001) == pytest.approx(10.95)
    assert per_interval_to_apr_pct(0.001) == pytest.approx(109.5)


def test_window_sum_to_pct_notional():
    assert window_sum_to_pct_notional(0.005) == pytest.approx(0.5)
    assert window_sum_to_pct_notional(0.0) == 0.0
    assert window_sum_to_pct_notional(0.1) == pytest.approx(10.0)


def test_window_sum_to_annualized_pct():
    lam = 0.005
    n = 90  # 30d window
    expected = (lam / n) * 1095 * 100
    assert window_sum_to_annualized_pct(lam, n) == pytest.approx(expected)
    assert window_sum_to_annualized_pct(0.0, 90) == 0.0
    assert window_sum_to_annualized_pct(1.0, 0) == 0.0


def test_window_intervals_for_days():
    assert window_intervals_for_days(7) == 21
    assert window_intervals_for_days(30) == 90
    assert window_intervals_for_days(90) == 270


def test_aliases_match():
    assert to_apr(0.0001) == per_interval_to_apr(0.0001)
    assert to_apr_pct(0.0001) == per_interval_to_apr_pct(0.0001)
    assert to_pct_notional(0.005) == window_sum_to_pct_notional(0.005)
    assert to_ann_pct(0.005, 90) == window_sum_to_annualized_pct(0.005, 90)


def test_annualized_pct_vs_raw_pct_notional():
    """Verify that annualizing a 30d Lambda and raw % notional are different."""
    lam = 0.005
    raw_pct = to_pct_notional(lam)       # 0.5%
    ann_pct = to_ann_pct(lam, 90)        # (0.005/90)*1095*100 ≈ 6.08%

    assert raw_pct == pytest.approx(0.5)
    assert ann_pct > raw_pct  # annualized should be larger
    assert ann_pct == pytest.approx(6.0833, rel=1e-3)


def test_never_multiply_cumsum_by_1095():
    """Demonstrate the anti-pattern: multiplying a cumulative sum by 1095
    gives a number that is ~n times too large.
    """
    lam = 0.005  # aggregate loss over 90 intervals
    wrong = to_apr_pct(lam)   # 547.5% — WRONG for a cumulative sum
    correct = to_pct_notional(lam)  # 0.5% of notional per 30d window
    assert wrong == pytest.approx(547.5)
    assert correct == pytest.approx(0.5)
    assert wrong / correct == pytest.approx(1095.0)  # off by INTERVALS_PER_YEAR
