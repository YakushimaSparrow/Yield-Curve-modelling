import json
from datetime import date
from typing import Optional

from fastapi import FastAPI, Response, Query, HTTPException
from pydantic import BaseModel

from database import DatabaseManager
from curve import build_curve_points, fit_nelson_siegel

manager = DatabaseManager()
app = FastAPI(title="Vertex Yield Curve API", version="1.0.0")


class KeyRatePoint(BaseModel):
    effective_date: str
    rate: float


@app.get("/api/v1/instruments")
def get_instruments():
    data = manager.select("SELECT * FROM instruments ORDER BY maturity_date")
    return Response(content=data, media_type="application/json")


@app.get("/api/v1/market-data")
def get_market_data(
    secid: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(1000, le=20000),
):
    query = "SELECT * FROM market_data WHERE 1=1"
    params = []
    if secid:
        query += " AND secid = ?"
        params.append(secid)
    if start_date:
        query += " AND trade_date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND trade_date <= ?"
        params.append(str(end_date))
    query += " ORDER BY trade_date ASC, secid ASC LIMIT ?"
    params.append(limit)
    data = manager.select(query, params)
    return Response(content=data, media_type="application/json")


@app.get("/api/v1/key-rate")
def get_key_rate():
    data = manager.select("SELECT * FROM key_rate ORDER BY effective_date")
    return Response(content=data, media_type="application/json")


@app.post("/api/v1/key-rate")
def add_key_rate(point: KeyRatePoint):
    manager.save_key_rate([(point.effective_date, point.rate)])
    return {"status": "ok", "saved": point.model_dump()}


@app.get("/api/v1/dates")
def get_dates():
    data = manager.select("SELECT DISTINCT trade_date FROM market_data ORDER BY trade_date")
    return Response(content=data, media_type="application/json")


@app.get("/api/v1/curve")
def get_curve(trade_date: date = Query(...)):
    raw = manager.select(
        """
        SELECT m.trade_date, m.secid, m.clean_price_pct, m.accint, m.dprice, m.volume,
               i.coupon_value, i.coupon_period, i.maturity_date, i.nominal
        FROM market_data m
        JOIN instruments i ON m.secid = i.secid
        WHERE m.trade_date = ?
        """,
        [str(trade_date)],
    )
    rows = json.loads(raw)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {trade_date}")

    points = build_curve_points(rows, str(trade_date))
    fit = fit_nelson_siegel([p["ttm"] for p in points], [p["ytm"] for p in points])
    return {"trade_date": str(trade_date), "points": points, "fit": fit}


@app.get("/api/v1/factors")
def get_factors(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
):
    q = "SELECT * FROM factors WHERE 1=1"
    params = []
    if start_date:
        q += " AND trade_date >= ?"
        params.append(str(start_date))
    if end_date:
        q += " AND trade_date <= ?"
        params.append(str(end_date))
    q += " ORDER BY trade_date"
    return Response(content=manager.select(q, params), media_type="application/json")
