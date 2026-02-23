"""Unit tests for premium calculations."""

import numpy as np
import pytest

from ddx.pricing.premium import (
    pure_premium,
    cvar_loaded_premium,
    target_sharpe_premium,
    full_premium,
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
