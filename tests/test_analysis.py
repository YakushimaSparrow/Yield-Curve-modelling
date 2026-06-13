import numpy as np
import pandas as pd

from analysis import attach_key_rate, correlation, linear_fit, level_vs_rate, inversion_stats


def test_correlation_perfect():
    assert abs(correlation([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9


def test_linear_fit_recovers_line():
    x = np.arange(20)
    y = 2.5 * x + 1.0
    slope, intercept = linear_fit(x, y)
    assert abs(slope - 2.5) < 1e-6
    assert abs(intercept - 1.0) < 1e-6


def test_attach_key_rate_uses_last_known():
    factors = pd.DataFrame({
        "trade_date": ["2022-01-10", "2022-03-01"],
        "level": [8.0, 9.0],
        "slope": [-1.0, -2.0],
        "curvature": [1.0, 1.5],
    })
    key_rate = pd.DataFrame({
        "effective_date": ["2022-01-01", "2022-02-28"],
        "rate": [8.5, 20.0],
    })
    merged = attach_key_rate(factors, key_rate)
    assert merged.iloc[0]["rate"] == 8.5
    assert merged.iloc[1]["rate"] == 20.0


def test_level_tracks_rate():
    factors = pd.DataFrame({
        "trade_date": pd.date_range("2022-01-01", periods=10, freq="D").astype(str),
        "level": np.linspace(8, 12, 10),
        "slope": np.zeros(10),
        "curvature": np.zeros(10),
    })
    key_rate = pd.DataFrame({
        "effective_date": pd.date_range("2022-01-01", periods=10, freq="D").astype(str),
        "rate": np.linspace(8, 12, 10),
    })
    merged = attach_key_rate(factors, key_rate)
    res = level_vs_rate(merged)
    assert res["corr"] > 0.99


def test_inversion_more_common_at_high_rates():
    merged = pd.DataFrame({
        "trade_date": pd.date_range("2022-01-01", periods=6, freq="D"),
        "level": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        "slope": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        "rate": [6.0, 6.0, 6.0, 16.0, 16.0, 16.0],
    })
    res = inversion_stats(merged)
    assert res["inverted_high"] > res["inverted_low"]
    assert res["corr_rate_slope"] > 0
