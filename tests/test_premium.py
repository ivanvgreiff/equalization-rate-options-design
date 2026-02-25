"""Unit tests for premium calculations."""

import numpy as np
import pytest

from ddx.pricing.premium import (
    pure_premium,
    cvar_loaded_premium,
    target_sharpe_premium,
    full_premium,
    compute_premium,
    wang_distortion_premium,
    esscher_premium,
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
        assert "premium_wang" in result
        assert "premium_esscher" in result
        assert result["premium_full"] >= result["premium_pure"]

    def test_wang_method(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        result = compute_premium(payoffs, method="wang", wang_theta=0.5)
        assert result["method"] == "wang"
        assert result["premium"] > result["pure"]

    def test_esscher_method(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        result = compute_premium(payoffs, method="esscher", esscher_theta=1.0)
        assert result["method"] == "esscher"
        assert result["premium"] > result["pure"]


class TestWangDistortionPremium:
    def test_exceeds_pure(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        wp = wang_distortion_premium(payoffs, theta=0.5)
        assert wp >= pp - 1e-12

    def test_theta_zero_equals_pure(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        wp = wang_distortion_premium(payoffs, theta=0.0)
        assert wp == pytest.approx(pp, rel=1e-4)

    def test_monotone_in_theta(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        w03 = wang_distortion_premium(payoffs, theta=0.3)
        w05 = wang_distortion_premium(payoffs, theta=0.5)
        w08 = wang_distortion_premium(payoffs, theta=0.8)
        assert w03 <= w05 + 1e-12
        assert w05 <= w08 + 1e-12

    def test_constant_payoffs(self):
        payoffs = np.full(100, 0.05)
        wp = wang_distortion_premium(payoffs, theta=0.5)
        assert wp == pytest.approx(0.05, abs=1e-10)


class TestEsscherPremium:
    def test_exceeds_pure(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        ep = esscher_premium(payoffs, theta=1.0)
        assert ep >= pp - 1e-12

    def test_theta_zero_equals_pure(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        pp = pure_premium(payoffs)
        ep = esscher_premium(payoffs, theta=0.0)
        assert ep == pytest.approx(pp, rel=1e-6)

    def test_monotone_in_theta(self):
        rng = np.random.default_rng(42)
        payoffs = np.abs(rng.normal(0.01, 0.005, 1000))
        e05 = esscher_premium(payoffs, theta=0.5)
        e10 = esscher_premium(payoffs, theta=1.0)
        e20 = esscher_premium(payoffs, theta=2.0)
        assert e05 <= e10 + 1e-12
        assert e10 <= e20 + 1e-12

    def test_constant_payoffs(self):
        payoffs = np.full(100, 0.05)
        ep = esscher_premium(payoffs, theta=1.0)
        assert ep == pytest.approx(0.05, abs=1e-10)
