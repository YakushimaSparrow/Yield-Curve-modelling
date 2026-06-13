import numpy as np

from curve import cashflow_schedule, price_from_yield, solve_ytm, nelson_siegel, fit_nelson_siegel


def test_cashflow_schedule_ends_at_maturity():
    flows = cashflow_schedule("2021-01-01", "2024-01-01", 35.0, 182, 1000.0)
    assert flows
    assert flows[-1][1] > 1000
    assert all(t > 0 for t, _ in flows)


def test_ytm_round_trip():
    flows = cashflow_schedule("2021-01-01", "2026-01-01", 35.0, 182, 1000.0)
    target = 0.08
    price = price_from_yield(flows, target)
    recovered = solve_ytm(price, flows)
    assert abs(recovered - target) < 1e-4


def test_nelson_siegel_level_at_long_end():
    short = nelson_siegel([0.1], 6.0, -2.0, 1.0, 0.6)[0]
    long = nelson_siegel([30.0], 6.0, -2.0, 1.0, 0.6)[0]
    assert abs(long - 6.0) < 0.5
    assert short < long


def test_fit_recovers_known_curve():
    maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20])
    true = nelson_siegel(maturities, 7.0, -1.5, 2.0, 0.6)
    fit = fit_nelson_siegel(maturities, true)
    assert fit is not None
    assert abs(fit["level"] - 7.0) < 0.3
    assert abs(fit["slope"] - (-1.5)) < 0.5
    assert fit["r2"] > 0.99


def test_fit_returns_none_when_too_few_points():
    assert fit_nelson_siegel([1, 2], [5, 6]) is None
