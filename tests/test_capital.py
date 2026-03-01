"""Unit tests for ddx.capital — capital efficiency metrics."""

import numpy as np
import pytest

from ddx.capital import (
    efficiency_A,
    reserve_requirement,
    swap_margin_proxy,
    total_economic_cost,
)


class TestReserveRequirement:
    def test_cvar_basic(self):
        losses = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10])
        r = reserve_requirement(losses, alpha=0.01, method="cvar")
        assert r == pytest.approx(0.10, abs=0.01)

    def test_var_basic(self):
        losses = np.arange(0, 1.01, 0.01)
        r = reserve_requirement(losses, alpha=0.01, method="var")
        assert r == pytest.approx(0.99, abs=0.02)

    def test_empty_returns_zero(self):
        assert reserve_requirement(np.array([]), method="cvar") == 0.0

    def test_constant_losses(self):
        losses = np.full(100, 0.05)
        r = reserve_requirement(losses, alpha=0.01, method="cvar")
        assert r == pytest.approx(0.05, abs=1e-6)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            reserve_requirement(np.array([1.0]), method="mad")


class TestEfficiencyA:
    def test_positive_reduction(self):
        eff = efficiency_A(R_unhedged=0.10, R_hedged=0.04, premium=0.02)
        assert eff == pytest.approx(3.0)

    def test_no_reduction(self):
        eff = efficiency_A(R_unhedged=0.10, R_hedged=0.10, premium=0.02)
        assert eff == pytest.approx(0.0)

    def test_zero_premium_returns_zero(self):
        assert efficiency_A(0.10, 0.04, 0.0) == 0.0

    def test_negative_premium_returns_zero(self):
        assert efficiency_A(0.10, 0.04, -0.01) == 0.0

    def test_hedge_increases_risk(self):
        eff = efficiency_A(R_unhedged=0.05, R_hedged=0.08, premium=0.01)
        assert eff < 0


class TestTotalEconomicCost:
    def test_basic(self):
        cost = total_economic_cost(premium=0.02, R_hedged=0.05, k=0.10, horizon_days=30)
        expected = 0.02 + 0.10 * (30 / 365) * 0.05
        assert cost == pytest.approx(expected)

    def test_zero_premium(self):
        cost = total_economic_cost(premium=0.0, R_hedged=0.10, k=0.20, horizon_days=30)
        assert cost == pytest.approx(0.20 * (30 / 365) * 0.10)

    def test_zero_reserve(self):
        cost = total_economic_cost(premium=0.03, R_hedged=0.0, k=0.10, horizon_days=30)
        assert cost == pytest.approx(0.03)

    def test_horizon_scaling(self):
        cost_30 = total_economic_cost(0.01, 0.05, 0.10, 30)
        cost_90 = total_economic_cost(0.01, 0.05, 0.10, 90)
        assert cost_90 > cost_30


class TestSwapMarginProxy:
    def test_all_positive_swap(self):
        cfs = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        assert swap_margin_proxy(cfs) == 0.0

    def test_mixed_swap(self):
        cfs = np.array([0.05, 0.03, -0.01, -0.04, -0.08])
        margin = swap_margin_proxy(cfs, alpha=0.01)
        assert margin > 0

    def test_all_negative_swap(self):
        cfs = np.full(100, -0.02)
        margin = swap_margin_proxy(cfs, alpha=0.01)
        assert margin == pytest.approx(0.02, abs=1e-6)
