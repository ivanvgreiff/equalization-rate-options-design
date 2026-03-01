"""Tests for the episode-based semi-Markov simulator."""

import numpy as np
import pytest

from ddx.models.cluster_semi_markov import (
    extract_episodes_and_clusters,
    fit_cluster_tail,
    simulate_semi_markov,
)


class TestExtractEpisodesAndClusters:
    def test_reconstruction(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0003, 500)
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=3)

        reconstructed = np.concatenate(
            [seg for pair in zip(result["calm_segments"], result["clusters"])
             for seg in pair]
            + (result["calm_segments"][len(result["clusters"]):] if
               len(result["calm_segments"]) > len(result["clusters"]) else
               result["clusters"][len(result["calm_segments"]):])
        )

        all_segments = []
        ci, cli = 0, 0
        stats = result["cluster_stats"]
        prev_end = 0
        for s in stats:
            if s["start"] > prev_end:
                all_segments.append(result["calm_segments"][ci])
                ci += 1
            all_segments.append(result["clusters"][cli])
            cli += 1
            prev_end = s["end"]
        if ci < len(result["calm_segments"]):
            all_segments.append(result["calm_segments"][ci])

        reconstructed = np.concatenate(all_segments)
        np.testing.assert_array_almost_equal(reconstructed, cf)

    def test_episode_count_known_series(self):
        cf = np.array([0.001, -0.001, -0.001, 0.001, 0.001, -0.001, 0.001])
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=0)
        assert result["n_clusters"] == 2

    def test_gap_merging(self):
        cf = np.array([0.001, -0.001, 0.001, -0.001, 0.001, 0.001, 0.001, -0.001])
        no_merge = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=0)
        merged = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=1)
        assert no_merge["n_clusters"] == 3
        assert merged["n_clusters"] <= no_merge["n_clusters"]

    def test_starts_in_stress(self):
        cf = np.array([-0.001, -0.001, 0.001, 0.001])
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=3)
        assert result["n_clusters"] >= 1
        assert result["cluster_stats"][0]["start"] == 0

    def test_ends_in_stress(self):
        cf = np.array([0.001, 0.001, -0.001, -0.001])
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=3)
        assert result["n_clusters"] >= 1
        assert result["cluster_stats"][-1]["end"] == 4


class TestFitClusterTail:
    def test_returns_params(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0005, 5000)
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=3)
        if len(result["clusters"]) >= 10:
            tail = fit_cluster_tail(result["clusters"], quantile_threshold=0.90)
            assert "n_clusters" in tail


class TestSimulateSemiMarkov:
    @pytest.fixture
    def segments(self):
        rng = np.random.default_rng(42)
        cf = rng.normal(0.0001, 0.0003, 2000)
        result = extract_episodes_and_clusters(cf, threshold_b=0.0001, gap_g=3)
        return result["clusters"], result["calm_segments"]

    def test_output_shape(self, segments):
        clusters, calm = segments
        out = simulate_semi_markov(clusters, calm, n_intervals=500, n_paths=3,
                                    rng=np.random.default_rng(0))
        assert out.shape == (3, 500)

    def test_reproducibility(self, segments):
        clusters, calm = segments
        a = simulate_semi_markov(clusters, calm, n_intervals=200, n_paths=1,
                                  rng=np.random.default_rng(99))
        b = simulate_semi_markov(clusters, calm, n_intervals=200, n_paths=1,
                                  rng=np.random.default_rng(99))
        np.testing.assert_array_equal(a, b)

    def test_cap_bounds_no_augmentation(self, segments):
        clusters, calm = segments
        out = simulate_semi_markov(clusters, calm, n_intervals=5000, n_paths=1,
                                    rng=np.random.default_rng(7))
        src = np.concatenate(clusters + calm)
        assert out.min() >= src.min() - 1e-15
        assert out.max() <= src.max() + 1e-15

    def test_simulate_with_tail(self, segments):
        clusters, calm = segments
        if len(clusters) >= 10:
            tail = fit_cluster_tail(clusters)
            out = simulate_semi_markov(
                clusters, calm, n_intervals=2000, n_paths=1,
                rng=np.random.default_rng(0),
                tail_params=tail, p_augment=0.2, cap=0.004,
            )
            assert out.shape == (1, 2000)
