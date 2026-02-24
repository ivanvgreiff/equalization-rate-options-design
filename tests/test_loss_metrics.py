"""Tests for loss-only (reserve draw) metrics."""

import numpy as np
import pytest

from ddx.risk.metrics import total_loss, hedged_loss, extract_episodes
from ddx.backtest.hedges import option_hedge_loss, swap_hedge_loss
from ddx.payoffs.floor import vanilla_floor


class TestTotalLoss:
    def test_all_positive_is_zero(self):
        cf = np.array([0.001, 0.002, 0.003])
        assert total_loss(cf) == 0.0

    def test_all_negative(self):
        cf = np.array([-0.001, -0.002, -0.003])
        assert total_loss(cf) == pytest.approx(0.006)

    def test_mixed(self):
        cf = np.array([-0.001, 0.002, -0.003])
        assert total_loss(cf) == pytest.approx(0.004)

    def test_zeros(self):
        cf = np.array([0.0, 0.0, 0.0])
        assert total_loss(cf) == 0.0


class TestHedgedLoss:
    def test_perfect_floor_leaves_only_premium(self):
        cf = np.array([-0.001, -0.002, -0.003])
        premium = 0.005
        result = hedged_loss(cf, vanilla_floor, premium=premium)
        # Floor payoff = 0.006 = total_loss, so max(0, 0.006-0.006)+0.005 = 0.005
        assert result == pytest.approx(premium)

    def test_no_loss_pays_only_premium(self):
        cf = np.array([0.001, 0.002])
        premium = 0.001
        result = hedged_loss(cf, vanilla_floor, premium=premium)
        assert result == pytest.approx(premium)

    def test_zero_premium(self):
        cf = np.array([-0.01, 0.005])
        result = hedged_loss(cf, vanilla_floor, premium=0.0)
        # loss=0.01, floor pays 0.01, net = max(0,0)+0 = 0
        assert result == pytest.approx(0.0)


class TestOptionHedgeLoss:
    def test_matches_hedged_loss(self):
        cf = np.array([-0.002, -0.003, 0.001])
        prem = 0.004
        a = hedged_loss(cf, vanilla_floor, premium=prem)
        b = option_hedge_loss(cf, vanilla_floor, prem)
        assert a == pytest.approx(b)


class TestSwapHedgeLoss:
    def test_positive_rate_zero_loss(self):
        cf = np.array([0.001] * 10)
        assert swap_hedge_loss(cf, 0.001) == 0.0

    def test_negative_rate_produces_loss(self):
        cf = np.array([0.001] * 10)
        loss = swap_hedge_loss(cf, -0.0005)
        assert loss == pytest.approx(10 * 0.0005)


class TestExtractEpisodes:
    def test_basic_episodes(self):
        cf = np.array([0.001, -0.002, -0.003, 0.001, -0.001, 0.002])
        eps = extract_episodes(cf)
        assert len(eps) == 2
        assert eps[0]["start_idx"] == 1
        assert eps[0]["duration"] == 2
        assert eps[0]["total_loss"] == pytest.approx(0.005)
        assert eps[1]["start_idx"] == 4
        assert eps[1]["duration"] == 1

    def test_no_episodes(self):
        cf = np.array([0.001, 0.002, 0.003])
        eps = extract_episodes(cf)
        assert len(eps) == 0

    def test_all_negative(self):
        cf = np.array([-0.001, -0.002, -0.003])
        eps = extract_episodes(cf)
        assert len(eps) == 1
        assert eps[0]["duration"] == 3
        assert eps[0]["total_loss"] == pytest.approx(0.006)

    def test_mean_severity(self):
        cf = np.array([-0.002, -0.004, 0.001])
        eps = extract_episodes(cf)
        assert eps[0]["mean_severity"] == pytest.approx(0.003)
