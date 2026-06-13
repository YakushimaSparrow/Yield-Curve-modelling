import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares


def cashflow_schedule(trade_date, maturity_date, coupon_value, coupon_period, nominal):
    trade_date = pd.Timestamp(trade_date)
    maturity_date = pd.Timestamp(maturity_date)
    step = pd.Timedelta(days=coupon_period)

    dates = []
    d = maturity_date
    while d > trade_date:
        dates.append(d)
        d = d - step
    dates = sorted(dates)

    flows = []
    for d in dates:
        amount = coupon_value
        if d == dates[-1]:
            amount += nominal
        years = (d - trade_date).days / 365.0
        flows.append((years, amount))
    return flows


def price_from_yield(flows, y):
    total = 0.0
    for t, cf in flows:
        total += cf / (1.0 + y) ** t
    return total


def solve_ytm(dirty_price, flows):
    if not flows:
        return np.nan
    f = lambda y: price_from_yield(flows, y) - dirty_price
    try:
        return brentq(f, -0.5, 2.0, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


def nelson_siegel(tau_grid, b0, b1, b2, lam):
    t = np.asarray(tau_grid, dtype=float)
    t = np.where(t <= 0, 1e-6, t)
    term = (1 - np.exp(-lam * t)) / (lam * t)
    return b0 + b1 * term + b2 * (term - np.exp(-lam * t))


def fit_nelson_siegel(maturities, yields, lam=0.6):
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    mask = np.isfinite(maturities) & np.isfinite(yields)
    maturities, yields = maturities[mask], yields[mask]
    if len(maturities) < 4:
        return None

    def residuals(p):
        return nelson_siegel(maturities, p[0], p[1], p[2], lam) - yields

    long_level = yields[maturities >= np.median(maturities)].mean()
    short_level = yields[maturities < np.median(maturities)].mean()
    x0 = [long_level, short_level - long_level, 0.0]

    res = least_squares(residuals, x0, max_nfev=2000)
    b0, b1, b2 = res.x
    fitted = nelson_siegel(maturities, b0, b1, b2, lam)
    ss_res = np.sum((yields - fitted) ** 2)
    ss_tot = np.sum((yields - yields.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "level": float(b0),
        "slope": float(b1),
        "curvature": float(b2),
        "lam": float(lam),
        "r2": float(r2),
        "n_points": int(len(maturities)),
    }


def build_curve_points(slice_rows, trade_date):
    points = []
    for row in slice_rows:
        flows = cashflow_schedule(
            trade_date,
            row["maturity_date"],
            row["coupon_value"] or 0.0,
            int(row["coupon_period"] or 182),
            row["nominal"] or 1000.0,
        )
        if not flows:
            continue
        dirty = row.get("dprice")
        if dirty is None or not np.isfinite(dirty):
            dirty = (row["clean_price_pct"] * (row["nominal"] or 1000.0) / 100.0) + (row["accint"] or 0.0)
        ytm = solve_ytm(dirty, flows)
        ttm = flows[-1][0]
        if np.isfinite(ytm) and 0 < ttm < 35 and -0.2 < ytm < 1.0:
            points.append({"secid": row["secid"], "ttm": ttm, "ytm": ytm * 100})
    return sorted(points, key=lambda p: p["ttm"])
