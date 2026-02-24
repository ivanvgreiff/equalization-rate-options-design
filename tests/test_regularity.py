"""Tests for regularity gate: preprocessing + rolling windows."""

import numpy as np
import pandas as pd
import pytest

from ddx.data.preprocess import normalize_bitmex, enforce_regular_grid
from ddx.backtest.rolling import rolling_windows_regular, rolling_payoffs


class TestIsRegularFlag:
    def test_all_regular_8h(self):
        raw = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="8h", tz="UTC"),
            "fundingRate": [0.0001] * 10,
        })
        df = normalize_bitmex(raw)
        assert df["is_regular"].all()

    def test_irregular_gap_flagged(self):
        ts = list(pd.date_range("2020-01-01", periods=5, freq="8h", tz="UTC"))
        ts.append(ts[-1] + pd.Timedelta(hours=24))  # big gap
        raw = pd.DataFrame({
            "timestamp": ts,
            "fundingRate": [0.0001] * 6,
        })
        df = normalize_bitmex(raw)
        assert df["is_regular"].sum() == 5
        assert not df["is_regular"].iloc[5]


class TestEnforceRegularGrid:
    def test_drops_irregular_rows(self):
        ts = list(pd.date_range("2020-01-01", periods=5, freq="8h", tz="UTC"))
        ts.append(ts[-1] + pd.Timedelta(hours=24))
        ts.append(ts[-1] + pd.Timedelta(hours=8))
        raw = pd.DataFrame({
            "timestamp": ts,
            "fundingRate": [0.0001] * 7,
        })
        df = normalize_bitmex(raw)
        regular = enforce_regular_grid(df)
        assert regular["is_regular"].all()
        assert len(regular) < len(df)


class TestRollingWindowsRegular:
    def test_skips_window_with_gap(self):
        cf = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        is_reg = np.array([True, True, True, False, True, True])
        windows, starts = rolling_windows_regular(cf, is_reg, 3)
        # Windows starting at 0,1 are valid (indices 0-2, 1-3 contains irregular at 3)
        # Actually: window [0,1,2] all regular, window [1,2,3] has index 3 irregular,
        # window [2,3,4] has 3 irregular, window [3,4,5] has 3 irregular.
        # Only window starting at 0 is fully valid.
        assert len(windows) == 1
        assert starts[0] == 0
        np.testing.assert_array_equal(windows[0], [1.0, 2.0, 3.0])

    def test_all_regular_gives_all_windows(self):
        cf = np.arange(10, dtype=float)
        is_reg = np.ones(10, dtype=bool)
        windows, starts = rolling_windows_regular(cf, is_reg, 3)
        assert len(windows) == 8
        assert len(starts) == 8

    def test_rolling_payoffs_with_regularity(self):
        cf = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        is_reg = np.array([True, True, True, True, True])
        payoffs = rolling_payoffs(cf, 3, lambda w: float(np.sum(w)), is_regular=is_reg)
        assert len(payoffs) == 3
        np.testing.assert_allclose(payoffs, [6.0, 9.0, 12.0])
