"""Tests for data normalization and preprocessing."""

import pandas as pd
import pytest

from ddx.data.preprocess import normalize_bitmex, funding_cf_to_apr, apr_to_funding_cf
from ddx.data.schema import validate


class TestNormalizeBitmex:
    def test_basic_normalization(self):
        raw = pd.DataFrame({
            "timestamp": [
                "2020-03-12T04:00:00.000Z",
                "2020-03-12T12:00:00.000Z",
                "2020-03-12T20:00:00.000Z",
            ],
            "fundingRate": [0.0001, -0.0005, 0.0003],
        })
        df = normalize_bitmex(raw)
        assert len(df) == 3
        assert list(df.columns) >= ["timestamp", "funding_rate", "funding_cf", "dt_hours"]
        assert df["funding_cf"].iloc[0] == pytest.approx(0.0001)
        assert df["funding_cf"].iloc[1] == pytest.approx(-0.0005)

    def test_sign_convention_no_flip(self):
        """Positive BitMEX rate = short-perp receives = positive CF."""
        raw = pd.DataFrame({
            "timestamp": ["2020-01-01T00:00:00Z"],
            "fundingRate": [0.001],
        })
        df = normalize_bitmex(raw)
        assert df["funding_cf"].iloc[0] > 0


class TestConversions:
    def test_roundtrip(self):
        import numpy as np
        dt = 8 / (24 * 365)
        cf = np.array([0.0001, -0.0005])
        apr = funding_cf_to_apr(cf, dt)
        cf2 = apr_to_funding_cf(apr, dt)
        np.testing.assert_allclose(cf, cf2)
