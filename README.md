# Yield Curve Modelling

A study of the Russian government bond (OFZ) yield curve. We pull the full trading history from the
MOEX ISS API, store it in SQLite, expose it through our own API and present everything in Streamlit.

## Hypotheses

The shape of the yield curve is driven by a few separate forces:

1. **Parallel shift.** The whole curve moves up and down with the key rate.
2. **Short vs long end.** The short end reacts to monetary-policy expectations far more than the long end.
3. **Curvature.** The medium-term segment carries a persistent convexity that bends the curve.
4. **Our own hypothesis.** A sharp tightening cycle inverts the curve: when the key rate is high, the
   short end climbs above the long end.

For every trading day we fit a Nelson-Siegel curve and split it into three coefficients: level, slope
and curvature. We then line them up against the Bank of Russia key rate and check each claim against four
years of data.

## Layout

- `database.py`: SQLite access and the MOEX ISS client.
- `curve.py`: yield-to-maturity solving and Nelson-Siegel calibration.
- `analysis.py`: joining the factors with the key rate and the hypothesis checks.
- `api.py`: FastAPI service on top of the database.
- `streamlit_app.py`: the narrated report with charts.
- `build_database.py`: data download and factor computation.
- `tests/`: pytest.

## Running

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The Streamlit app starts the FastAPI service on its own, so a single command is enough. The database with
2021-2024 data is already in `database.db`, so there is no need to download anything first.

To rebuild the data from scratch:

```
python build_database.py 2021 2022 2023 2024
```

## Sharing a public link

To open the running app from another machine without a localhost address, expose the local Streamlit
port through a Cloudflare quick tunnel:

```
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8501
```

It prints a public `https://...trycloudflare.com` link that works from anywhere while your machine and
the tunnel stay running. The `http2` protocol is used because some networks block the default QUIC
transport.
