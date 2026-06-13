import sqlite3 as sql
import json
import time
import pandas as pd
import requests


class DatabaseManager:

    def __init__(self, db_name="database.db"):
        self.db_name = db_name
        self.create_tables()

    def create_tables(self):
        connection = sql.connect(self.db_name)
        try:
            with connection:
                cursor = connection.cursor()
                cursor.execute("PRAGMA foreign_keys = ON;")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS instruments(
                    secid TEXT PRIMARY KEY,
                    shortname TEXT,
                    coupon_value REAL,
                    coupon_period INTEGER,
                    maturity_date TEXT,
                    nominal REAL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS market_data(
                    trade_date DATE NOT NULL,
                    secid TEXT,
                    clean_price_pct REAL,
                    accint REAL,
                    dprice REAL,
                    volume REAL,
                    PRIMARY KEY (secid, trade_date),
                    FOREIGN KEY (secid) REFERENCES instruments(secid)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS key_rate(
                    effective_date TEXT PRIMARY KEY,
                    rate REAL
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS factors(
                    trade_date TEXT PRIMARY KEY,
                    level REAL,
                    slope REAL,
                    curvature REAL,
                    lam REAL,
                    r2 REAL,
                    n_points INTEGER
                    )
                """)
        finally:
            connection.close()

    def _execute_query_(self, query, params=None, is_many=False):
        connection = sql.connect(self.db_name)
        try:
            with connection:
                connection.execute("PRAGMA foreign_keys = ON;")
                cursor = connection.cursor()
                if is_many:
                    cursor.executemany(query, params)
                else:
                    cursor.execute(query, params or ())
        finally:
            connection.close()

    def prepare_market_data(self, df_raw):
        df = df_raw.copy()
        df['CLEAN_PRICE_PCT'] = df['LEGALCLOSEPRICE'].fillna(df['CLOSE'])
        df['DIRTY_PRICE'] = (df['CLEAN_PRICE_PCT'] * 10) + df['ACCINT']
        final_cols = ['TRADEDATE', 'SECID', 'CLEAN_PRICE_PCT', 'ACCINT', 'DIRTY_PRICE', 'VOLUME']
        return df[final_cols].dropna(subset=['DIRTY_PRICE'])

    def prepare_instruments_data(self, df_raw):
        df = df_raw.copy()
        df = df.rename(columns={
            'SECID': 'secid',
            'SHORTNAME': 'shortname',
            'MATDATE': 'maturity_date'
        })
        df['coupon_value'] = df['COUPONVALUE'] if 'COUPONVALUE' in df.columns else 0.0
        df['coupon_period'] = df['COUPONPERIOD'] if 'COUPONPERIOD' in df.columns else 182
        df['nominal'] = df['FACEVALUE'] if 'FACEVALUE' in df.columns else 1000.0
        df_unique = df.drop_duplicates(subset=['secid'])
        final_cols = ['secid', 'shortname', 'coupon_value', 'coupon_period', 'maturity_date', 'nominal']
        return df_unique[final_cols].dropna(subset=['secid'])

    def save_instruments(self, df_instruments):
        data_tuples = list(df_instruments.itertuples(index=False, name=None))
        query = """
            INSERT OR REPLACE INTO instruments (secid, shortname, coupon_value, coupon_period, maturity_date, nominal)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self._execute_query_(query, data_tuples, is_many=True)

    def save_market_data(self, df_market):
        data_tuples = list(df_market.itertuples(index=False, name=None))
        query = """
            INSERT OR REPLACE INTO market_data (trade_date, secid, clean_price_pct, accint, dprice, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self._execute_query_(query, data_tuples, is_many=True)

    def save_key_rate(self, rows):
        query = "INSERT OR REPLACE INTO key_rate (effective_date, rate) VALUES (?, ?)"
        self._execute_query_(query, rows, is_many=True)

    def save_factors(self, rows):
        query = """
            INSERT OR REPLACE INTO factors (trade_date, level, slope, curvature, lam, r2, n_points)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query_(query, rows, is_many=True)

    def select(self, query, params=None):
        connection = sql.connect(self.db_name)
        try:
            cursor = connection.cursor()
            cursor.execute(query, params or ())
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
        finally:
            connection.close()
        return json.dumps(data, ensure_ascii=False)

    def frame(self, query, params=None):
        connection = sql.connect(self.db_name)
        try:
            return pd.read_sql_query(query, connection, params=params)
        finally:
            connection.close()


class MoexClient:
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"
        self.session = requests.Session()

    def _send_request(self, url):
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_full_year_dataframe(self, year):
        from_date = f"{year}-01-01"
        till_date = f"{year}-12-31"
        if year == pd.Timestamp.now().year:
            till_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        print(f"Fetching MOEX for {year}...")
        init_url = (
            f"{self.base_url}/history/engines/stock/markets/bonds/boards/TQOB/securities.json"
            f"?iss.meta=off&from={from_date}&till={from_date}"
        )
        init_json = self._send_request(init_url)

        block = init_json.get('history')
        if not block or not block['data']:
            print("Nothing returned for the first day, trying the first trading week...")
            init_url = (
                f"{self.base_url}/history/engines/stock/markets/bonds/boards/TQOB/securities.json"
                f"?iss.meta=off&from={from_date}&till={year}-01-15"
            )
            init_json = self._send_request(init_url)
            block = init_json.get('history')
            if not block or not block['data']:
                return pd.DataFrame()

        df_init = pd.DataFrame(block['data'], columns=block['columns'])
        ofz_tickers = df_init[df_init['SECID'].str.startswith('SU26')]['SECID'].unique()
        print(f"Found {len(ofz_tickers)} OFZ-PD issues.")

        all_dfs = []
        for i, ticker in enumerate(ofz_tickers, 1):
            start = 0
            ticker_pages = []
            while True:
                url = (
                    f"{self.base_url}/history/engines/stock/markets/bonds/boards/TQOB/securities/{ticker}.json"
                    f"?iss.meta=off&from={from_date}&till={till_date}&start={start}&limit=100"
                )
                try:
                    raw_json = self._send_request(url)
                    h_block = raw_json.get('history')
                    if not h_block or not h_block['data']:
                        break
                    df_page = pd.DataFrame(h_block['data'], columns=h_block['columns'])
                    if df_page.empty:
                        break
                    ticker_pages.append(df_page)
                    start += 100
                except Exception as e:
                    print(f"Failed on {ticker}: {e}")
                    break

            if ticker_pages:
                df_single = pd.concat(ticker_pages, ignore_index=True)
                all_dfs.append(df_single)
                print(f"[{i}/{len(ofz_tickers)}] {ticker} | rows: {len(df_single)}")
            else:
                print(f"[{i}/{len(ofz_tickers)}] {ticker} empty, skipping")
            time.sleep(0.02)

        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)
