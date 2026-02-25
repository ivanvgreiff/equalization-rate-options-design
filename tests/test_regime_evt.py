"""Tests for the regime-switching + EVT model."""

import numpy as np
import pytest

from ddx.models.regime_evt import fit_regime_model, fit_evt_tail, simulate_regime_evt


class TestFitRegimeModel:
    def test_state_assignment(self):
        cf = np.array([0.0002, 0.0001, -0.00005, -0.0002, 0.0001])
        model = fit_regime_model(cf, threshold_b=0.0001)
        np.testing.assert_array_equal(model["states"], [0, 0, 0, 1, 0])

    def test_transition_matrix_rows_sum_to_1(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0003, 1000)
        model = fit_regime_model(cf, threshold_b=0.0001)
        row_sums = model["transition_matrix"].sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-10)

    def test_stationary_distribution(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0003, 5000)
        model = fit_regime_model(cf, threshold_b=0.0001)
        pi = model["stationary_dist"]
        P = model["transition_matrix"]
        pi_check = pi @ P
        np.testing.assert_allclose(pi, pi_check, atol=1e-10)

    def test_known_series(self):
        cf = np.array([0.001, 0.001, -0.001, -0.001, -0.001, 0.001])
        model = fit_regime_model(cf, threshold_b=0.0001)
        assert model["n_normal"] == 3
        assert model["n_stress"] == 3
        assert model["transition_matrix"][0, 1] == pytest.approx(0.5)
        assert model["transition_matrix"][1, 0] == pytest.approx(1 / 3, abs=0.01)


class TestFitEvtTail:
    def test_returns_params(self):
        rng = np.random.default_rng(42)
        losses = np.abs(rng.normal(0.0003, 0.0002, 500))
        result = fit_evt_tail(losses, quantile_threshold=0.90)
        assert "shape_xi" in result
        assert "scale_sigma" in result
        assert "threshold_u" in result
        assert result["n_exceedances"] > 0

    def test_threshold_selection(self):
        losses = np.arange(1.0, 101.0)
        result = fit_evt_tail(losses, quantile_threshold=0.95)
        assert result["threshold_u"] == pytest.approx(95.05, abs=1.0)

    def test_few_exceedances_fallback(self):
        losses = np.array([0.1, 0.2, 0.3])
        result = fit_evt_tail(losses, quantile_threshold=0.95)
        assert result["fit_success"] is False


class TestSimulateRegimeEvt:
    @pytest.fixture
    def fitted(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0003, 2000)
        return fit_regime_model(cf, threshold_b=0.0001)

    def test_output_shape(self, fitted):
        out = simulate_regime_evt(fitted, n_intervals=500, n_paths=3,
                                   rng=np.random.default_rng(0))
        assert out.shape == (3, 500)

    def test_reproducibility(self, fitted):
        a = simulate_regime_evt(fitted, n_intervals=200, n_paths=1,
                                 rng=np.random.default_rng(99))
        b = simulate_regime_evt(fitted, n_intervals=200, n_paths=1,
                                 rng=np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)

    def test_fraction_negative_plausible(self, fitted):
        out = simulate_regime_evt(fitted, n_intervals=10000, n_paths=1,
                                   rng=np.random.default_rng(7))
        frac_neg = np.mean(out < 0)
        assert 0.05 < frac_neg < 0.60
