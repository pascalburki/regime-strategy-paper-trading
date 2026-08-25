"""
Regime Strategy — Multi-Asset Alpaca Paper Trading Runner
Extends the original single-asset (SPY-only) version to your real,
validated 5-asset portfolio (SPY, QQQ, GLD, TLT, XOM), using SPY's
regime signal to drive equal-weighted exposure across all 5 - matching
the approach you already validated (beat buy-and-hold on both return
and Sharpe ratio: 0.57 vs 0.44 return, 0.84 vs 0.69 Sharpe).

IMPORTANT SAFETY NOTE, learned the hard way tonight:
Every run of this script submits REAL orders to your Alpaca paper
account - including manual "Run workflow" triggers, not just the
scheduled automatic runs. A manual trigger used for testing during
setup caused a real double-buy that had to be corrected.

This version adds a genuine DRY_RUN mode so you can safely test
without touching your real account:
    DRY_RUN=true python3 run_strategy_multi_asset.py
This prints exactly what WOULD happen, with no real orders submitted.
Only remove DRY_RUN (or set it to false) once you're ready for the
scheduled workflow to run for real - don't manually trigger the real
version again unless you specifically intend to execute a real trade.

SETUP: same as before - pip install alpaca-py yfinance numpy pandas hmmlearn --break-system-packages
Requires ALPACA_KEY and ALPACA_SECRET as environment variables.
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
REGIME_SIGNAL_SYMBOL = "SPY"  # the asset whose regime drives exposure
PORTFOLIO_SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]  # your real validated 5-asset list
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
DATA_HISTORY_START = '2010-01-01'
exposure = {0: 1.2, 1: 1.0, 2: 0.0}
regime_names = {0: "calm", 1: "moderate", 2: "stress"}
LOG_FILE = "strategy_log_multi_asset.csv"

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"


def run_walkforward(df_clean, start_date):
    """Same validated walk-forward logic as the original single-asset script."""
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
    df = yf.download(REGIME_SIGNAL_SYMBOL, start=DATA_HISTORY_START, end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
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


def rebalance_asset(trading_client, symbol, target_value, dry_run=False):
    """Adjust one asset's position toward its target dollar value.
    Returns (action_description, current_value_before)."""
    try:
        current_position = trading_client.get_open_position(symbol)
        current_value = float(current_position.market_value)
    except Exception:
        current_value = 0.0

    gap = target_value - current_value

    if abs(gap) < 1:
        return "no change needed", current_value

    side = OrderSide.BUY if gap > 0 else OrderSide.SELL
    action_desc = f"{side.value} ${abs(gap):.2f} notional of {symbol}"

    if dry_run:
        return f"[DRY RUN - not executed] {action_desc}", current_value

    order_request = MarketOrderRequest(
        symbol=symbol,
        notional=round(abs(gap), 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order_data=order_request)
    return action_desc, current_value


def log_result(date, regime, exposure_pct, actions, equity):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "regime", "target_exposure", "equity", "actions"])
        writer.writerow([date, regime, exposure_pct, equity, " | ".join(actions)])


def main():
    if DRY_RUN:
        print("=" * 60)
        print("DRY RUN MODE - no real orders will be submitted")
        print("=" * 60)

    api_key = os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_KEY and ALPACA_SECRET environment variables before running.")

    trading_client = TradingClient(api_key, api_secret, paper=True)

    print(f"Determining current regime from real {REGIME_SIGNAL_SYMBOL} data...")
    current_state, current_date = get_current_regime()
    regime_label = regime_names.get(current_state, "unknown")
    target_exposure = exposure.get(current_state, 1.0)

    print(f"Date: {current_date.date()}, Regime: {regime_label}, Target exposure: {target_exposure}x")

    account = trading_client.get_account()
    equity = float(account.equity)
    # Equal-weighted across all 5 assets, matching your validated approach
    per_asset_target = (equity * target_exposure) / len(PORTFOLIO_SYMBOLS)

    print(f"Account equity: ${equity:,.2f}")
    print(f"Target per asset (equal-weighted across {len(PORTFOLIO_SYMBOLS)}): ${per_asset_target:,.2f}")
    print()

    actions = []
    for symbol in PORTFOLIO_SYMBOLS:
        action, before_value = rebalance_asset(trading_client, symbol, per_asset_target, dry_run=DRY_RUN)
        print(f"{symbol}: {action} (was ${before_value:,.2f})")
        actions.append(f"{symbol}: {action}")

    if not DRY_RUN:
        log_result(current_date.date(), regime_label, target_exposure, actions, equity)
        print(f"\nLogged to {LOG_FILE}")
    else:
        print("\nDry run complete - nothing was logged or executed for real.")


if __name__ == "__main__":
    main()
