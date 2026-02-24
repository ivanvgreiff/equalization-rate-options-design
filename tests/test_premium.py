"""Unit tests for premium calculations."""

import numpy as np
import pytest

from ddx.pricing.premium import (
    pure_premium,
    cvar_loaded_premium,
    target_sharpe_premium,
    full_premium,
    compute_premium,
)


class TestPurePremium:
    def test_mean_of_payoffs(self):
        payoffs = np.array([0.0, 0.01, 0.02, 0.0])
        assert pure_premium(payoffs) == pytest.approx(0.0075)


class TestCvarLoadedPremium:
    def test_exceeds_pure_premium(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        loaded = cvar_loaded_premium(payoffs, lam=0.35)
        assert loaded >= pp


class TestTargetSharpePremium:
    def test_exceeds_pure_premium(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        sharpe_p = target_sharpe_premium(payoffs, target_sharpe=0.75)
        assert sharpe_p > pp


class TestFullPremium:
    def test_total_is_sum_of_components(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        result = full_premium(payoffs)
        expected_total = result["pure"] + result["risk_load"] + result["capital_charge"]
        assert result["total"] == pytest.approx(expected_total)


class TestComputePremium:
    def test_pure_method(self):
        payoffs = np.array([0.0, 0.01, 0.02, 0.0])
        result = compute_premium(payoffs, method="pure")
        assert result["method"] == "pure"
        assert result["premium"] == pytest.approx(0.0075)

    def test_full_method_matches_full_premium(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        result = compute_premium(payoffs, method="full")
        fp = full_premium(payoffs)
        assert result["premium"] == pytest.approx(fp["total"])

    def test_all_method_has_all_keys(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        result = compute_premium(payoffs, method="all")
        assert "premium_pure" in result
        assert "premium_full" in result
        assert "premium_target_sharpe" in result
        assert result["premium_full"] >= result["premium_pure"]
