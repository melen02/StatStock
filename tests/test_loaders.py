import pandas as pd

from loadData import load_yfinance
from loadData.mongo_helpers import is_missing, merge_quarterly_rows, replace_collection


class FakeCollection:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.updates = []
        self.dropped = False

    def find(self, query=None, projection=None):
        return list(self.rows)

    def find_one(self, query=None, sort=None):
        return self.rows[-1] if self.rows else None

    def insert_one(self, record):
        self.rows.append(record)

    def insert_many(self, records):
        self.rows.extend(records)

    def update_one(self, query, update):
        self.updates.append(update["$set"])

    def drop(self):
        self.dropped = True
        self.rows = []


def prices(start, values):
    return pd.DataFrame({"Date": pd.bdate_range(start, periods=len(values)), "Adj Close": values, "Ticker": "A"})


def stored_rows(frame):
    return frame[["Date", "Adj Close"]].to_dict("records")


def test_is_missing_covers_none_nan_and_nat():
    assert is_missing(None) and is_missing(float("nan")) and is_missing(pd.NaT)
    assert not is_missing(0) and not is_missing("x")


def test_unchanged_prices_are_not_flagged():
    frame = prices("2024-01-01", [100.0, 101.0])
    assert not load_yfinance.prices_were_readjusted(FakeCollection(stored_rows(frame)), "A", frame)


def test_a_split_after_the_last_fetch_is_flagged():
    stored = prices("2024-01-01", [1000.0, 1010.0])
    fresh = prices("2024-01-01", [100.0, 101.0])
    assert load_yfinance.prices_were_readjusted(FakeCollection(stored_rows(stored)), "A", fresh)


def test_incremental_fetch_reloads_everything_after_a_split(monkeypatch):
    stored = prices("2024-01-01", [1000.0, 1010.0])
    full_history = prices("2020-01-01", [100.0] * 5)
    starts = []

    def fake_download(ticker, start):
        starts.append(start)
        return full_history if start == load_yfinance.HISTORY_START else prices("2024-01-01", [100.0, 101.0])

    monkeypatch.setattr(load_yfinance, "download_history", fake_download)
    result = load_yfinance.fetch_stock_history("A", FakeCollection(stored_rows(stored)))

    assert starts[-1] == load_yfinance.HISTORY_START
    assert len(result) == len(full_history)


def test_incremental_fetch_keeps_only_new_rows_when_nothing_changed(monkeypatch):
    stored = prices("2024-01-01", [100.0, 101.0])
    fresh = prices("2024-01-01", [100.0, 101.0, 102.0])
    monkeypatch.setattr(load_yfinance, "download_history", lambda ticker, start: fresh)

    result = load_yfinance.fetch_stock_history("A", FakeCollection(stored_rows(stored)))
    assert result is fresh


def test_merge_fills_a_missing_date_but_keeps_old_values_outside_refresh_quarters():
    existing = {"Ticker": "A", "Quarter": "2024-Q2", "QuarterDate": pd.NaT, "Revenue": 10.0}
    col = FakeCollection([existing])
    new = pd.DataFrame([{"Ticker": "A", "Quarter": "2024-Q2", "QuarterDate": pd.Timestamp("2024-06-30"), "Revenue": 99.0}])

    merge_quarterly_rows(col, new, ["Ticker", "Quarter"], "test")

    assert col.updates == [{"QuarterDate": pd.Timestamp("2024-06-30")}]


def test_replace_collection_stores_none_and_text_dates():
    db = {"Results": FakeCollection([{"old": 1}])}
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-31"]), "Value": [float("nan")]})

    replace_collection(db, "Results", df)

    assert db["Results"].dropped
    assert db["Results"].rows == [{"Date": "2024-01-31", "Value": None}]
