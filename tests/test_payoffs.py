"""Unit tests for payoff functions."""

import numpy as np
import pytest

from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.distress import distress_activated_floor, soft_duration_cover
from ddx.payoffs.stoploss import aggregate_stop_loss


class TestVanillaFloor:
    def test_all_positive_funding_pays_zero(self):
        cf = np.array([0.001, 0.002, 0.0005, 0.001])
        assert vanilla_floor(cf) == 0.0

    def test_all_negative_funding_pays_full(self):
        cf = np.array([-0.001, -0.002, -0.003])
        expected = 0.001 + 0.002 + 0.003
        assert vanilla_floor(cf) == pytest.approx(expected)

    def test_deductible_reduces_payoff(self):
        cf = np.array([-0.005, -0.001, 0.002])
        no_ded = vanilla_floor(cf, deductible=0.0)
        with_ded = vanilla_floor(cf, deductible=0.002)
        assert with_ded < no_ded

    def test_cap_limits_payoff(self):
        cf = np.array([-0.01, -0.01, -0.01])
        uncapped = vanilla_floor(cf, cap=None)
        capped = vanilla_floor(cf, cap=0.01)
        assert uncapped == pytest.approx(0.03)
        assert capped == 0.01

    def test_mixed_sign_only_pays_on_negative(self):
        cf = np.array([0.002, -0.001, 0.003, -0.004])
        assert vanilla_floor(cf) == pytest.approx(0.005)


class TestDistressActivatedFloor:
    def test_no_streak_no_payoff(self):
        cf = np.array([-0.001, 0.002, -0.001, 0.002])
        assert distress_activated_floor(cf, threshold_b=0.0, streak_m=2) == 0.0

    def test_streak_activates(self):
        cf = np.array([-0.001, -0.002, -0.003, 0.001])
        payoff = distress_activated_floor(cf, threshold_b=0.0, streak_m=2)
        # Activated at i=1 (run=2) and i=2 (run=3); pays on those
        assert payoff > 0

    def test_higher_streak_reduces_payoff(self):
        cf = np.array([-0.001] * 10)
        m2 = distress_activated_floor(cf, threshold_b=0.0, streak_m=2)
        m5 = distress_activated_floor(cf, threshold_b=0.0, streak_m=5)
        assert m5 < m2


class TestAggregatStopLoss:
    def test_below_deductible_pays_zero(self):
        cf = np.array([-0.001, -0.001, 0.005])
        assert aggregate_stop_loss(cf, deductible_D=0.01) == 0.0

    def test_above_deductible_pays_excess(self):
        cf = np.array([-0.01, -0.01, -0.01])
        payoff = aggregate_stop_loss(cf, deductible_D=0.02)
        assert payoff == pytest.approx(0.01)

    def test_cap_applied(self):
        cf = np.array([-0.05])
        payoff = aggregate_stop_loss(cf, deductible_D=0.01, cap=0.02)
        assert payoff == 0.02


class TestSoftDurationCover:
    def test_below_m_pays_zero(self):
        cf = np.array([-0.001, -0.001, 0.001])
        assert soft_duration_cover(cf, threshold_b=0.0, streak_m=3, ramp_s=2) == 0.0

    def test_ramp_partial_weight(self):
        cf = np.array([-0.001] * 6)
        hard = distress_activated_floor(cf, threshold_b=0.0, streak_m=3)
        soft = soft_duration_cover(cf, threshold_b=0.0, streak_m=3, ramp_s=3)
        # Soft should pay less than hard activation (partial weights during ramp)
        assert soft < hard
