"""
Shadow Tracker — Buy-and-Hold Benchmark (No Real Trades, No Regime Timing)
Simplest possible comparison: what would $100k be worth just holding the
same 5-asset universe (SPY, QQQ, GLD, TLT, XOM) equally-weighted, always
at 1.0x exposure, with no regime-based timing at all?

This is the honest baseline every "smart" strategy has to actually beat.
No HMM, no regime detection needed -- that's the whole point of buy-and-hold.

No ALPACA_KEY needed. No orders submitted, ever. Pure computation + a log file.

SETUP: pip install yfinance numpy pandas --break-system-packages
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import csv

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]   # same universe as the real 5-asset strategy
LOG_FILE = "shadow_buyhold_log.csv"
STARTING_EQUITY = 100000.0   # same baseline as the real account and other shadow tracker


def get_latest_trading_day_return():
    """Equal-weighted close-to-close return for the most recent TRADING day in
    the downloaded data, together with that day's date taken from the price
    index -- never from the calendar.

    FIX (2026-10-03): the old version took the latest available return and
    labelled it with today's calendar date. Any run on a weekend, on a market
    holiday, or a second run on the same day re-applied the same return and
    compounded it into the equity curve (e.g. 2026-09-07, Labor Day, repeated
    Friday's -0.51%; 2026-09-26, a Saturday, repeated Thursday's -0.23%).
    """
    today = pd.Timestamp(datetime.date.today())
    start = (today - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    pieces = []
    for symbol in ASSETS:
        df = yf.download(symbol, start=start, end=end)
        df.columns = df.columns.get_level_values(0)
        pieces.append(df["Close"].pct_change(fill_method=None).rename(symbol))
    rets = pd.concat(pieces, axis=1).dropna()      # only days every asset traded
    return rets.index[-1].date(), float(rets.iloc[-1].mean())


def already_logged(trading_date):
    """True if this trading day is already in the log (repeat run, weekend, holiday)."""
    if not os.path.isfile(LOG_FILE):
        return False
    df = pd.read_csv(LOG_FILE)
    return len(df) > 0 and str(trading_date) in set(df["date"].astype(str))


def load_previous_equity():
    if not os.path.isfile(LOG_FILE):
        return STARTING_EQUITY
    df = pd.read_csv(LOG_FILE)
    if len(df) == 0:
        return STARTING_EQUITY
    return float(df.iloc[-1]["equity"])


def log_result(date, daily_return, equity):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "daily_portfolio_return", "equity"])
        writer.writerow([date, daily_return, equity])


def main():
    print("=== SHADOW TRACKER (buy-and-hold, 5-asset, no timing) -- simulation only ===\n")

    print("Fetching the latest trading day's portfolio return across SPY, QQQ, GLD, TLT, XOM...")
    trading_date, daily_return = get_latest_trading_day_return()
    print(f"Latest trading day: {trading_date}")
    if already_logged(trading_date):
        print(f"{trading_date} is already logged -- nothing to do "
              "(weekend, holiday, or a repeat run).")
        return
    print(f"Equal-weighted daily return: {daily_return:.5f}")

    previous_equity = load_previous_equity()
    new_equity = previous_equity * (1 + daily_return)   # always 1.0x, no regime adjustment

    print(f"Previous equity: ${previous_equity:,.2f}")
    print(f"New equity: ${new_equity:,.2f}")

    log_result(trading_date, daily_return, new_equity)
    print(f"\nLogged to {LOG_FILE}")


if __name__ == "__main__":
    main()
