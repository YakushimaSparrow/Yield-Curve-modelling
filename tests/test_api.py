import json

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_instruments_endpoint():
    r = client.get("/api/v1/instruments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_market_data_filter():
    r = client.get("/api/v1/market-data", params={"limit": 5})
    assert r.status_code == 200
    assert len(r.json()) <= 5


def test_curve_missing_date():
    r = client.get("/api/v1/curve", params={"trade_date": "1990-01-01"})
    assert r.status_code == 404


def test_post_key_rate_roundtrip():
    r = client.post("/api/v1/key-rate", json={"effective_date": "2099-01-01", "rate": 1.23})
    assert r.status_code == 200
    rates = client.get("/api/v1/key-rate").json()
    assert any(x["effective_date"] == "2099-01-01" for x in rates)
