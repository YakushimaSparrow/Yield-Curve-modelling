import json
import sys

from database import DatabaseManager, MoexClient
from curve import build_curve_points, fit_nelson_siegel
from key_rate_history import CBR_KEY_RATE


def load_market(manager, client, years):
    for year in years:
        data = client.fetch_full_year_dataframe(year)
        if data.empty:
            print(f"{year}: empty, skipping")
            continue
        instruments = manager.prepare_instruments_data(data)
        market = manager.prepare_market_data(data)
        manager.save_instruments(instruments)
        manager.save_market_data(market)
        print(f"{year}: {len(instruments)} instruments, {len(market)} quotes")


def load_key_rate(manager):
    manager.save_key_rate(CBR_KEY_RATE)
    print(f"Key rate: {len(CBR_KEY_RATE)} points")


def compute_factors(manager):
    dates = [r["trade_date"] for r in json.loads(
        manager.select("SELECT DISTINCT trade_date FROM market_data ORDER BY trade_date")
    )]
    rows = []
    for d in dates:
        raw = manager.select(
            """
            SELECT m.secid, m.clean_price_pct, m.accint, m.dprice,
                   i.coupon_value, i.coupon_period, i.maturity_date, i.nominal
            FROM market_data m
            JOIN instruments i ON m.secid = i.secid
            WHERE m.trade_date = ?
            """,
            [d],
        )
        points = build_curve_points(json.loads(raw), d)
        fit = fit_nelson_siegel([p["ttm"] for p in points], [p["ytm"] for p in points])
        if fit:
            rows.append((d, fit["level"], fit["slope"], fit["curvature"],
                         fit["lam"], fit["r2"], fit["n_points"]))
    manager.save_factors(rows)
    print(f"Factors computed for {len(rows)} trading days")


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2021, 2022, 2023, 2024]
    manager = DatabaseManager()
    client = MoexClient()
    load_market(manager, client, years)
    load_key_rate(manager)
    compute_factors(manager)
    print("Done.")
