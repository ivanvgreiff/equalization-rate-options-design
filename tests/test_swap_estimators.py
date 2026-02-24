"""Tests for swap rate estimators."""

import numpy as np
import pytest

from ddx.backtest.hedges import (
    estimate_swap_rate,
    estimate_swap_rate_mean,
    estimate_swap_rate_median,
    estimate_swap_rate_ewma,
)


class TestSwapRateMean:
    def test_trailing_mean(self):
        cf = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        rate = estimate_swap_rate_mean(cf, lookback=3, window_start=4)
        # mean of cf[1:4] = mean(0.002, 0.003, 0.004) = 0.003
        assert rate == pytest.approx(0.003)

    def test_lookback_exceeds_history(self):
        cf = np.array([0.001, 0.002])
        rate = estimate_swap_rate_mean(cf, lookback=100, window_start=2)
        assert rate == pytest.approx(0.0015)


class TestSwapRateMedian:
    def test_trailing_median(self):
        cf = np.array([0.001, 0.010, 0.002, 0.003])
        rate = estimate_swap_rate_median(cf, lookback=3, window_start=4)
        # median of cf[1:4] = median(0.010, 0.002, 0.003) = 0.003
        assert rate == pytest.approx(0.003)


class TestSwapRateEWMA:
    def test_ewma_weights_recent_more(self):
        cf = np.array([0.001, 0.001, 0.001, 0.010])
        rate_mean = estimate_swap_rate_mean(cf, lookback=4, window_start=4)
        rate_ewma = estimate_swap_rate_ewma(cf, lookback=4, window_start=4, halflife=1)
        # EWMA with short halflife should weight the recent 0.010 more
        assert rate_ewma > rate_mean


class TestSwapRateDispatcher:
    def test_dispatch_mean(self):
        cf = np.array([0.001, 0.002, 0.003])
        a = estimate_swap_rate(cf, 3, 3, method="mean")
        b = estimate_swap_rate_mean(cf, 3, 3)
        assert a == pytest.approx(b)

    def test_dispatch_median(self):
        cf = np.array([0.001, 0.002, 0.003])
        a = estimate_swap_rate(cf, 3, 3, method="median")
        b = estimate_swap_rate_median(cf, 3, 3)
        assert a == pytest.approx(b)

    def test_dispatch_ewma(self):
        cf = np.array([0.001, 0.002, 0.003])
        a = estimate_swap_rate(cf, 3, 3, method="ewma")
        b = estimate_swap_rate_ewma(cf, 3, 3)
        assert a == pytest.approx(b)

    def test_invalid_method_raises(self):
        cf = np.array([0.001])
        with pytest.raises(ValueError):
            estimate_swap_rate(cf, 1, 1, method="invalid")
