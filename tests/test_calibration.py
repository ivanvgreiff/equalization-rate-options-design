"""Tests for ddx.calibration — parameter calibration functions."""

import numpy as np
import pytest

from ddx.calibration import (
    conditional_loss_quantiles,
    lambda_quantiles_per_horizon,
    daf_activation_analysis,
    compute_asl_deductible,
    freeze_baseline_parameters,
)


def _make_series(n=1000, seed=42):
    """Generate a synthetic funding CF series for testing."""
    rng = np.random.default_rng(seed)
    cf = rng.normal(0.0001, 0.0003, n)
    return cf


class TestConditionalLossQuantiles:
    def test_basic(self):
        cf = np.array([0.001, -0.001, 0.002, -0.003, 0.0, -0.0005])
        result = conditional_loss_quantiles(cf, [0.50])
        assert result["n_negative"] == 3
        assert result["n_total"] == 6
        assert result["frac_negative"] == pytest.approx(0.5)
        assert result["q50"] == pytest.approx(0.001)

    def test_all_positive(self):
        cf = np.array([0.001, 0.002, 0.003])
        result = conditional_loss_quantiles(cf, [0.50, 0.90])
        assert result["n_negative"] == 0
        assert result["q50"] == 0.0
        assert result["q90"] == 0.0

    def test_all_negative(self):
        cf = np.array([-0.001, -0.002, -0.003])
        result = conditional_loss_quantiles(cf, [0.50])
        assert result["n_negative"] == 3
        assert result["frac_negative"] == pytest.approx(1.0)
        assert result["q50"] == pytest.approx(0.002)

    def test_quantile_ordering(self):
        cf = _make_series()
        result = conditional_loss_quantiles(cf, [0.25, 0.50, 0.75, 0.99])
        assert result["q25"] <= result["q50"]
        assert result["q50"] <= result["q75"]
        assert result["q75"] <= result["q99"]

    def test_has_mean_and_std(self):
        cf = _make_series()
        result = conditional_loss_quantiles(cf, [0.50])
        assert "mean" in result
        assert "std" in result
        assert result["mean"] > 0
        assert result["std"] > 0


class TestLambdaQuantiles:
    def test_basic(self):
        cf = _make_series(500)
        result = lambda_quantiles_per_horizon(cf, None, 21, [0.50, 0.90])
        assert result["n_windows"] == 500 - 21 + 1
        assert result["horizon_intervals"] == 21
        assert result["mean_lambda"] >= 0
        assert result["q50"] <= result["q90"]

    def test_with_regularity(self):
        cf = _make_series(200)
        is_reg = np.ones(200, dtype=bool)
        is_reg[50] = False  # one irregular
        result = lambda_quantiles_per_horizon(cf, is_reg, 21, [0.50])
        assert result["n_windows"] < 200 - 21 + 1

    def test_all_positive_cf(self):
        cf = np.abs(_make_series(100))
        result = lambda_quantiles_per_horizon(cf, None, 21, [0.50])
        assert result["mean_lambda"] == pytest.approx(0.0, abs=1e-10)

    def test_short_series(self):
        cf = _make_series(10)
        with pytest.raises(ValueError):
            lambda_quantiles_per_horizon(cf, None, 21, [0.50])


class TestDAFActivation:
    def test_never_activates_on_positive(self):
        cf = np.abs(_make_series(200))
        result = daf_activation_analysis(cf, None, 21, 0.0001, 3)
        assert result["frac_windows_activated"] == 0.0
        assert result["mean_payoff"] == 0.0

    def test_always_activates_on_deep_negative(self):
        cf = np.full(100, -0.005)
        result = daf_activation_analysis(cf, None, 21, 0.0001, 3)
        assert result["frac_windows_activated"] == 1.0
        assert result["mean_payoff"] > 0

    def test_higher_m_reduces_activation(self):
        cf = _make_series(500)
        r2 = daf_activation_analysis(cf, None, 90, 0.0001, 2)
        r3 = daf_activation_analysis(cf, None, 90, 0.0001, 3)
        r5 = daf_activation_analysis(cf, None, 90, 0.0001, 5)
        assert r2["frac_windows_activated"] >= r3["frac_windows_activated"]
        assert r3["frac_windows_activated"] >= r5["frac_windows_activated"]

    def test_d_equals_b_payoff(self):
        """When d=b, payoff should equal max(0, -f_i - b) for active intervals."""
        cf = np.array([-0.0005] * 10)
        b = 0.0001
        result = daf_activation_analysis(cf, None, 10, b, 3)
        expected_per_interval = max(0, 0.0005 - b)
        expected_total = expected_per_interval * (10 - 3 + 1)
        assert result["mean_payoff"] == pytest.approx(expected_total, rel=1e-6)


class TestComputeASLDeductible:
    def test_basic(self):
        cf = _make_series(500)
        D = compute_asl_deductible(cf, None, 90, 0.90)
        assert D >= 0

    def test_higher_quantile_gives_higher_D(self):
        cf = _make_series(500)
        D90 = compute_asl_deductible(cf, None, 90, 0.90)
        D95 = compute_asl_deductible(cf, None, 90, 0.95)
        assert D95 >= D90


class TestFreezeBaselineParameters:
    def test_structure(self):
        cf = _make_series(500)
        horizons = [
            {"name": "7d", "intervals": 21},
            {"name": "30d", "intervals": 90},
        ]
        params = freeze_baseline_parameters(cf, None, horizons)

        assert "conditional_loss_quantiles" in params
        assert "floor" in params
        assert "daf" in params
        assert "swap" in params
        assert "horizons" in params
        assert "7d" in params["horizons"]
        assert "30d" in params["horizons"]

        for hname in ["7d", "30d"]:
            hp = params["horizons"][hname]
            assert "lambda_quantiles" in hp
            assert "asl" in hp
            assert "baseline" in hp["asl"]
            assert "deductible_D" in hp["asl"]["baseline"]

    def test_daf_params(self):
        cf = _make_series(500)
        horizons = [{"name": "30d", "intervals": 90}]
        params = freeze_baseline_parameters(cf, None, horizons)
        daf = params["daf"]
        assert daf["baseline"]["threshold_b"] == 0.0001
        assert daf["baseline"]["streak_m"] == 3
        assert daf["baseline"]["deductible_d"] == 0.0001
        assert daf["sensitivity"]["streak_m"] == 2
