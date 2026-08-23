"""
Regime Strategy — Alpaca Paper Trading Runner
Uses a validated HMM walk-forward exposure strategy (calm 1.2x /
moderate 1.0x / stress 0x) on SPY, submitting real rebalancing
orders to an Alpaca PAPER account — so Alpaca's own engine handles
fills, position tracking, and portfolio value, instead of manually
simulating it in Python.

SETUP:
1. Free Alpaca account: https://alpaca.markets
2. Get your paper trading API key + secret from the Alpaca dashboard
3. pip install alpaca-py yfinance numpy pandas hmmlearn --break-system-packages
4. Set ALPACA_KEY and ALPACA_SECRET as environment variables (or GitHub Secrets)
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import datetime
import csv

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "SPY"
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
DATA_HISTORY_START = '2010-01-01'
exposure = {0: 1.2, 1: 1.0, 2: 0.0}
# NOTE: 1.2x amplifies downside too — a 2% SPY drop becomes roughly a 2.4%
# equity drop while in the "calm" regime. Confirm this is a risk level you've
# deliberately chosen for this exposure level, not just a number that felt
# reasonable in isolation.
regime_names = {0: "calm", 1: "moderate", 2: "stress"}
LOG_FILE = "strategy_log.csv"


def run_walkforward(df_clean, start_date):
    """Same validated walk-forward logic as the original script."""
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
    """Fetch real SPY data and run the walk-forward HMM to get today's regime."""
    today = pd.Timestamp(datetime.date.today())
    df = yf.download(SYMBOL, start=DATA_HISTORY_START, end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
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


def rebalance_to_target(trading_client, target_exposure_pct):
    """Adjust the Alpaca paper position to match the target exposure
    (e.g. 1.2 = 120% of equity in SPY, 0.0 = fully out).

    Uses dollar-based (notional) orders rather than calculating share counts
    manually — Alpaca supports fractional shares, so this removes an entire
    class of price-fetching and rounding bugs. Verified via current Alpaca
    documentation (checked in a separate research session): all Alpaca
    accounts open as margin accounts, and any account with $2,000+ equity
    gets standard Reg T margin (2x overnight buying power) automatically —
    paper accounts start well above that threshold, so 1.2x exposure is
    comfortably within normal limits. No separate margin check needed.
    """
    account = trading_client.get_account()
    equity = float(account.equity)
    target_value = equity * target_exposure_pct

    try:
        position = trading_client.get_open_position(SYMBOL)
        current_value = float(position.market_value)
    except Exception:
        current_value = 0.0  # no current position

    gap = target_value - current_value

    if abs(gap) < 1:  # skip dust-sized trades
        return "no change needed", equity, current_value, target_value

    side = OrderSide.BUY if gap > 0 else OrderSide.SELL
    order_request = MarketOrderRequest(
        symbol=SYMBOL,
        notional=round(abs(gap), 2),  # dollar amount, not share count
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order_data=order_request)
    action = f"{side.value} ${abs(gap):.2f} notional"
    return action, equity, current_value, target_value


def log_result(date, regime, exposure_pct, action, equity, current_value, target_value):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "regime", "target_exposure", "action", "equity", "position_value_before", "position_value_target"])
        writer.writerow([date, regime, exposure_pct, action, equity, current_value, target_value])


def main():
    api_key = os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_KEY and ALPACA_SECRET environment variables before running.")

    trading_client = TradingClient(api_key, api_secret, paper=True)

    print("Determining current regime from real SPY data...")
    current_state, current_date = get_current_regime()
    regime_label = regime_names.get(current_state, "unknown")
    target_exposure = exposure.get(current_state, 1.0)

    print(f"Date: {current_date.date()}, Regime: {regime_label}, Target exposure: {target_exposure}x")

    print("Rebalancing Alpaca paper account to match target exposure...")
    action, equity, current_value, target_value = rebalance_to_target(trading_client, target_exposure)

    print(f"Action taken: {action}")
    print(f"Account equity: ${equity:,.2f}")
    print(f"Position value before: ${current_value:,.2f}, target: ${target_value:,.2f}")

    log_result(current_date.date(), regime_label, target_exposure, action, equity, current_value, target_value)
    print(f"\nLogged to {LOG_FILE}")


if __name__ == "__main__":
    main()
