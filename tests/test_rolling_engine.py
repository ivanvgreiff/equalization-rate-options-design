"""Unit tests for the rolling-window engine."""

import numpy as np
import pytest

from ddx.backtest.rolling import rolling_windows, rolling_payoffs


class TestRollingWindows:
    def test_shape(self):
        cf = np.arange(100, dtype=float)
        windows = rolling_windows(cf, 10)
        assert windows.shape == (91, 10)

    def test_first_and_last_window(self):
        cf = np.arange(20, dtype=float)
        windows = rolling_windows(cf, 5)
        np.testing.assert_array_equal(windows[0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(windows[-1], [15, 16, 17, 18, 19])

    def test_too_short_raises(self):
        cf = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            rolling_windows(cf, 5)


class TestRollingPayoffs:
    def test_sum_payoff(self):
        cf = np.ones(20)
        payoffs = rolling_payoffs(cf, 5, lambda w: float(np.sum(w)))
        assert len(payoffs) == 16
        np.testing.assert_allclose(payoffs, 5.0)
