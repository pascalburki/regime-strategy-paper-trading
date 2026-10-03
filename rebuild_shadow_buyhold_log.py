"""
rebuild_shadow_buyhold_log.py -- one-time repair of shadow_buyhold_log.csv
==========================================================================
The old tracker appended a row on EVERY run (manual runs, holidays, weekends)
and re-applied the latest available return each time, so some days were
compounded twice and some non-trading days were compounded at all. See the
FIX note in run_shadow_buyhold.py.

This script re-derives the benchmark from actual prices: exactly one row per
real trading day from START_DATE onward, equal-weighted, compounded from
BASE equity. It writes a NEW file and never overwrites the old log, so you can
compare before replacing anything:

    python3 rebuild_shadow_buyhold_log.py
    # review, then:
    mv shadow_buyhold_log.rebuilt.csv shadow_buyhold_log.csv

START_DATE is the date of the first return the live 5-asset strategy earned
(the first row of the old log). BASE matches STARTING_EQUITY in the tracker.

Needs internet (yfinance). The core functions are pure so they can be tested
offline with synthetic prices.
"""

import datetime
import os

import pandas as pd

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
START_DATE = "2026-08-25"
BASE = 100000.0
OLD_LOG = "shadow_buyhold_log.csv"
NEW_LOG = "shadow_buyhold_log.rebuilt.csv"


def trading_day_returns(closes):
    """closes: {symbol: Series of closes}. Equal-weighted daily return on days
    every asset has a price -- non-trading days simply do not exist here."""
    rets = pd.concat({s: c.pct_change(fill_method=None) for s, c in closes.items()}, axis=1)
    return rets.dropna().mean(axis=1)


def rebuild(daily, start, base):
    d = daily.loc[start:]
    equity = base * (1 + d).cumprod()
    return pd.DataFrame({
        "date": [x.date() for x in d.index],
        "daily_portfolio_return": d.values,
        "equity": equity.values,
    })


def fetch_closes(start):
    import yfinance as yf
    lo = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(datetime.date.today()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    closes = {}
    for s in ASSETS:
        df = yf.download(s, start=lo, end=hi)
        df.columns = df.columns.get_level_values(0)
        closes[s] = df["Close"]
    return closes


def main():
    new = rebuild(trading_day_returns(fetch_closes(START_DATE)), START_DATE, BASE)
    new.to_csv(NEW_LOG, index=False)

    print(f"Rebuilt {len(new)} trading days from {new['date'].iloc[0]} to {new['date'].iloc[-1]}.")
    print(f"Rebuilt benchmark: ${new['equity'].iloc[-1]:,.2f} "
          f"({new['equity'].iloc[-1] / BASE - 1:+.2%} since start)")
    if os.path.isfile(OLD_LOG):
        old = pd.read_csv(OLD_LOG)
        dup = int(old["date"].duplicated().sum())
        wk = int((pd.to_datetime(old["date"]).dt.dayofweek >= 5).sum())
        print(f"Old log: {len(old)} rows ({dup} duplicate dates, {wk} on weekends), "
              f"final ${old['equity'].iloc[-1]:,.2f} ({old['equity'].iloc[-1] / BASE - 1:+.2%})")
    print(f"\nWrote {NEW_LOG}. Review it, then replace {OLD_LOG} if it looks right.")


if __name__ == "__main__":
    main()
