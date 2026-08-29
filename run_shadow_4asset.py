"""
Shadow Tracker — 4-Asset Portfolio (No Real Trades)
Runs the same regime signal as the live strategy, but instead of submitting
real Alpaca orders, it simulates what a 4-asset portfolio (SPY, QQQ, GLD,
XOM -- no TLT) would be worth, starting from the same $100k baseline as
the real account.

This exists purely to compare, day by day going forward, whether the
5-asset (real, live) or 4-asset (shadow, simulated) version actually
performs better -- not just in the historical backtest, but forward in
real market conditions from today onward.

No ALPACA_KEY needed. No orders submitted, ever. Pure computation + a log file.

SETUP: pip install yfinance numpy pandas hmmlearn --break-system-packages
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import datetime
import csv

# ============================================================
# CONFIG
# ============================================================
SIGNAL_SYMBOL = "SPY"
ASSETS = ["SPY", "QQQ", "GLD", "XOM"]     # the 4-asset comparison portfolio, no TLT
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
DATA_HISTORY_START = '2010-01-01'
exposure = {0: 1.2, 1: 1.0, 2: 0.0}
regime_names = {0: "calm", 1: "moderate", 2: "stress"}
LOG_FILE = "shadow_4asset_log.csv"
STARTING_EQUITY = 100000.0   # matches the real account's baseline


def run_walkforward(df_clean, start_date):
    """Identical walk-forward logic to the live strategy -- same signal, fair comparison."""
    current_date = start_date
    all_states = []

    while current_date < df_clean.index[-1]:
        next_date = current_date + pd.DateOffset(months=1)
        train_data = df_clean.loc[:current_date, ["returns", "vol"]]
        test_data = df_clean.loc[current_date:next_date, ["returns", "vol"]]

        if len(train_data) < 100 or len(test_data) == 0:
            current_date = next_date
            continue

        model = GaussianHMM(
            n_components=3, covariance_type="diag", random_state=3,
            n_iter=200, tol=0.01, init_params=""
        )
        model.startprob_ = np.array([0.34, 0.33, 0.33])
        model.transmat_ = np.array([
            [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]
        ])
        model.means_ = np.array([[0.0, 0.01], [0.001, 0.02], [-0.001, 0.03]])
        model.covars_ = np.array([[0.0001, 0.0001], [0.0001, 0.0001], [0.0001, 0.0001]])

        try:
            model.fit(train_data)
        except Exception:
            current_date = next_date
            continue

        predicted_states = model.predict(test_data)
        all_states.append(pd.Series(predicted_states, index=test_data.index))
        current_date = next_date

    if not all_states:
        return None
    all_predicted_states = pd.concat(all_states)
    return all_predicted_states[~all_predicted_states.index.duplicated(keep='first')]


def get_current_regime():
    today = pd.Timestamp(datetime.date.today())
    df = yf.download(SIGNAL_SYMBOL, start=DATA_HISTORY_START, end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    returns = np.log(close / close.shift(1)).dropna()
    vol = returns.rolling(window=20).std().dropna()
    df["returns"] = returns
    df["vol"] = vol
    df_clean = df.dropna().copy()

    regime_signal = run_walkforward(df_clean, WALKFORWARD_HMM_START)
    if regime_signal is None or len(regime_signal) == 0:
        raise RuntimeError("Walk-forward model produced no regime signal.")

    current_state = regime_signal.iloc[-1]
    current_date = regime_signal.index[-1]
    return current_state, current_date


def get_todays_portfolio_return():
    """Equal-weighted daily simple return across the 4 assets, most recent close-to-close."""
    today = pd.Timestamp(datetime.date.today())
    start = (today - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    daily_returns = []
    for symbol in ASSETS:
        df = yf.download(symbol, start=start, end=end)
        df.columns = df.columns.get_level_values(0)
        close = df["Close"]
        pct_change = close.pct_change(fill_method=None).dropna()
        daily_returns.append(pct_change.iloc[-1])

    return float(np.mean(daily_returns))


def load_previous_equity():
    if not os.path.isfile(LOG_FILE):
        return STARTING_EQUITY
    df = pd.read_csv(LOG_FILE)
    if len(df) == 0:
        return STARTING_EQUITY
    return float(df.iloc[-1]["equity"])


def log_result(date, regime, exposure_pct, daily_return, equity):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "regime", "target_exposure", "daily_portfolio_return", "equity"])
        writer.writerow([date, regime, exposure_pct, daily_return, equity])


def main():
    print("=== SHADOW TRACKER (4-asset, no TLT) -- simulation only, no real trades ===\n")

    print("Determining current regime from real SPY data...")
    current_state, current_date = get_current_regime()
    regime_label = regime_names.get(current_state, "unknown")
    target_exposure = exposure.get(current_state, 1.0)
    print(f"Date: {current_date.date()}, Regime: {regime_label}, Target exposure: {target_exposure}x")

    print("Fetching today's actual portfolio return across SPY, QQQ, GLD, XOM...")
    daily_return = get_todays_portfolio_return()
    print(f"Equal-weighted daily return: {daily_return:.5f}")

    previous_equity = load_previous_equity()
    new_equity = previous_equity * (1 + target_exposure * daily_return)

    print(f"Previous equity: ${previous_equity:,.2f}")
    print(f"New equity: ${new_equity:,.2f}")

    log_result(datetime.date.today(), regime_label, target_exposure, daily_return, new_equity)
    print(f"\nLogged to {LOG_FILE}")


if __name__ == "__main__":
    main()
