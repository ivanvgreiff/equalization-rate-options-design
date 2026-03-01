"""Unit tests for ddx.backtest.walkforward."""

import numpy as np
import pandas as pd
import pytest

from ddx.backtest.walkforward import walkforward_evaluate


def _make_synthetic_series(n=5000, seed=42):
    """Create a synthetic funding series long enough for walk-forward."""
    rng = np.random.default_rng(seed)
    cf = rng.normal(0.0001, 0.0005, size=n)
    is_reg = np.ones(n, dtype=bool)
    return cf, is_reg


class TestWalkforwardEvaluate:
    def test_returns_dataframe(self):
        cf, reg = _make_synthetic_series()
        strategies = [{"name": "Unhedged", "type": "unhedged"}]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_contains_expected_columns(self):
        cf, reg = _make_synthetic_series()
        strategies = [{"name": "Unhedged", "type": "unhedged"}]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        expected_cols = {
            "month_idx", "t_start", "strategy", "premium",
            "payoff", "reserve_draw_unhedged", "reserve_draw_hedged", "net_cf",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_unhedged_premium_is_zero(self):
        cf, reg = _make_synthetic_series()
        strategies = [{"name": "Unhedged", "type": "unhedged"}]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        assert (result["premium"] == 0.0).all()

    def test_unhedged_draw_equals_hedged_draw(self):
        cf, reg = _make_synthetic_series()
        strategies = [{"name": "Unhedged", "type": "unhedged"}]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        np.testing.assert_array_almost_equal(
            result["reserve_draw_unhedged"].values,
            result["reserve_draw_hedged"].values,
        )

    def test_swap_strategy(self):
        cf, reg = _make_synthetic_series()
        strategies = [
            {"name": "Swap EWMA", "type": "swap", "swap_method": "ewma",
             "swap_lookback": 90, "swap_halflife": 45},
        ]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        assert len(result) > 0
        assert (result["premium"] == 0.0).all()

    def test_option_strategy(self):
        from ddx.payoffs import vanilla_floor
        cf, reg = _make_synthetic_series()
        strategies = [
            {"name": "Floor d=0", "type": "option",
             "payoff_fn": vanilla_floor, "payoff_kwargs": {"deductible": 0.0}},
        ]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        assert len(result) > 0
        assert (result["premium"] > 0).any()

    def test_too_short_series_returns_empty(self):
        cf = np.random.default_rng(0).normal(0.0001, 0.0005, size=100)
        reg = np.ones(100, dtype=bool)
        strategies = [{"name": "Unhedged", "type": "unhedged"}]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=3.0,
        )
        assert len(result) == 0

    def test_multiple_strategies(self):
        from ddx.payoffs import vanilla_floor
        cf, reg = _make_synthetic_series()
        strategies = [
            {"name": "Unhedged", "type": "unhedged"},
            {"name": "Floor d=0", "type": "option",
             "payoff_fn": vanilla_floor, "payoff_kwargs": {"deductible": 0.0}},
        ]
        result = walkforward_evaluate(
            cf, reg, strategies,
            horizon_intervals=90, train_years=1.0, step_intervals=90,
        )
        assert set(result["strategy"].unique()) == {"Unhedged", "Floor d=0"}
