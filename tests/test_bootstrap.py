"""Tests for the block bootstrap module."""

import numpy as np
import pytest

from ddx.bootstrap import circular_block_bootstrap, bootstrap_premiums
from ddx.payoffs import vanilla_floor


class TestCircularBlockBootstrap:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        series = rng.standard_normal(100)
        out = circular_block_bootstrap(series, block_size=10, n_samples=50, rng=rng)
        assert out.shape == (50, 100)

    def test_values_come_from_original(self):
        series = np.arange(20, dtype=float)
        out = circular_block_bootstrap(series, block_size=5, n_samples=10,
                                       rng=np.random.default_rng(7))
        for row in out:
            assert set(row).issubset(set(series))

    def test_reproducibility(self):
        series = np.random.default_rng(1).standard_normal(50)
        a = circular_block_bootstrap(series, 10, 20, rng=np.random.default_rng(99))
        b = circular_block_bootstrap(series, 10, 20, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)

    def test_block_size_1_is_iid(self):
        series = np.arange(10, dtype=float)
        out = circular_block_bootstrap(series, block_size=1, n_samples=5,
                                       rng=np.random.default_rng(42))
        assert out.shape == (5, 10)
        for row in out:
            assert set(row).issubset(set(series))

    def test_block_size_equals_n(self):
        series = np.arange(8, dtype=float)
        out = circular_block_bootstrap(series, block_size=8, n_samples=3,
                                       rng=np.random.default_rng(0))
        assert out.shape == (3, 8)
        for row in out:
            assert len(row) == 8

    def test_block_size_gt_n_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            circular_block_bootstrap(np.arange(5.0), block_size=10, n_samples=1)

    def test_block_size_zero_raises(self):
        with pytest.raises(ValueError, match="block_size"):
            circular_block_bootstrap(np.arange(5.0), block_size=0, n_samples=1)


class TestBootstrapPremiums:
    @pytest.fixture
    def funding_cf(self):
        rng = np.random.default_rng(42)
        return rng.normal(0.0001, 0.0003, size=500)

    def test_basic_execution(self, funding_cf):
        result = bootstrap_premiums(
            funding_cf, window_size=21,
            payoff_fn=lambda w: vanilla_floor(w, deductible=0.0),
            n_bootstrap=10, block_size=21,
            rng=np.random.default_rng(0),
        )
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "mean" in result
        assert len(result["samples_pure"]) == 10

    def test_ci_ordering(self, funding_cf):
        result = bootstrap_premiums(
            funding_cf, window_size=21,
            payoff_fn=lambda w: vanilla_floor(w, deductible=0.0),
            n_bootstrap=50, block_size=21,
            rng=np.random.default_rng(1),
        )
        for key in ["pure", "total", "risk_load", "capital_charge"]:
            assert result["ci_lower"][key] <= result["mean"][key] + 1e-12
            assert result["mean"][key] <= result["ci_upper"][key] + 1e-12

    def test_constant_series_narrow_ci(self):
        cf = np.full(200, 0.0001)
        result = bootstrap_premiums(
            cf, window_size=21,
            payoff_fn=lambda w: vanilla_floor(w, deductible=0.0),
            n_bootstrap=20, block_size=10,
            rng=np.random.default_rng(5),
        )
        ci_width = result["ci_upper"]["pure"] - result["ci_lower"]["pure"]
        assert ci_width < 1e-10

    def test_n_bootstrap_1(self, funding_cf):
        result = bootstrap_premiums(
            funding_cf, window_size=21,
            payoff_fn=lambda w: vanilla_floor(w, deductible=0.0),
            n_bootstrap=1, block_size=21,
            rng=np.random.default_rng(0),
        )
        assert len(result["samples_pure"]) == 1
