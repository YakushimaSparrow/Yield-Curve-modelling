import numpy as np
import pandas as pd


def attach_key_rate(factors, key_rate):
    factors = factors.copy()
    key_rate = key_rate.copy()
    factors["trade_date"] = pd.to_datetime(factors["trade_date"])
    key_rate["effective_date"] = pd.to_datetime(key_rate["effective_date"])
    factors = factors.sort_values("trade_date")
    key_rate = key_rate.sort_values("effective_date")
    merged = pd.merge_asof(
        factors,
        key_rate,
        left_on="trade_date",
        right_on="effective_date",
        direction="backward",
    )
    return merged


def correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def level_vs_rate(merged):
    beta = correlation(merged["level"], merged["rate"])
    slope, intercept = linear_fit(merged["rate"], merged["level"])
    return {"corr": beta, "slope": slope, "intercept": intercept}


def end_sensitivity(merged):
    d = merged.dropna(subset=["level", "slope", "rate"]).copy()
    d["short_end"] = d["level"] + d["slope"]
    d["long_end"] = d["level"]
    short_beta = linear_fit(d["rate"], d["short_end"])[0]
    long_beta = linear_fit(d["rate"], d["long_end"])[0]
    return {"short_beta": short_beta, "long_beta": long_beta}


def curvature_summary(merged):
    c = merged["curvature"].dropna()
    return {
        "mean": float(c.mean()),
        "std": float(c.std()),
        "share_positive": float((c > 0).mean()),
    }


def inversion_stats(merged):
    d = merged.dropna(subset=["level", "slope", "rate"]).copy()
    d["inverted"] = (d["level"] + d["slope"]) > d["level"]
    threshold = d["rate"].median()
    high = d[d["rate"] >= threshold]
    low = d[d["rate"] < threshold]
    return {
        "threshold": float(threshold),
        "inverted_overall": float(d["inverted"].mean()),
        "inverted_high": float(high["inverted"].mean()),
        "inverted_low": float(low["inverted"].mean()),
        "corr_rate_slope": correlation(d["rate"], d["slope"]),
    }
