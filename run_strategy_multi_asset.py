"""
Regime Strategy — Multi-Asset Alpaca Paper Trading Runner
Extends the validated single-asset (SPY-only) version to an equal-weighted
5-asset portfolio (SPY, QQQ, GLD, TLT, XOM), using SPY's regime signal to
drive exposure across all 5 equally -- matching the originally validated
approach (0.57 vs 0.44 return, 0.84 vs 0.69 Sharpe vs buy-and-hold).

KNOWN TRADEOFF, kept deliberately for portfolio coherence: per-asset
contribution analysis (Aug 25 2026) showed TLT as a real drag of -0.033
within that +0.57 total, caused by a genuine multi-year bond bear market
(2020-2023 rate hikes) that an equity-vol-based signal has no way to see.
Every other asset (SPY, QQQ, GLD, XOM) contributed positively. TLT stays
in for now rather than breaking from the validated 5-asset structure; a
properly backtested, rate-aware TLT-specific signal remains a real future
improvement, not something to build and deploy same-day.

SAFETY: every run of this script submits REAL orders to your Alpaca paper
account, including manual "Run workflow" triggers. Use DRY_RUN to preview
without touching the account:
    DRY_RUN=true python3 run_strategy_multi_asset.py
Only unset DRY_RUN once you've confirmed the preview looks right and you're
ready for it to actually trade.

SETUP: pip install alpaca-py yfinance numpy pandas hmmlearn --break-system-packages
Requires ALPACA_KEY and ALPACA_SECRET as environment variables.
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
import datetime
import csv

# ============================================================
# CONFIG
# ============================================================
SIGNAL_SYMBOL = "SPY"                       # drives the regime signal
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]  # equal-weighted portfolio, matches validated backtest (see note above re: TLT drag)
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
DATA_HISTORY_START = '2010-01-01'
exposure = {0: 1.2, 1: 1.0, 2: 0.0}
regime_names = {0: "calm", 1: "moderate", 2: "stress"}
LOG_FILE = "strategy_log.csv"
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"


def run_walkforward(df_clean, start_date):
    """Same validated walk-forward logic as the original SPY-only script."""
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


def has_open_order(trading_client, symbol):
    """Check whether this symbol already has a pending order, so we never
    submit a second reduction/order while the first one is still settling --
    this is exactly what caused the original double-buy incident.
    """
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    open_orders = trading_client.get_orders(filter=request)
    return len(open_orders) > 0


def get_gap(trading_client, symbol, target_value):
    """Compute how far a position is from its target, without trading yet."""
    try:
        position = trading_client.get_open_position(symbol)
        current_value = float(position.market_value)
    except Exception:
        current_value = 0.0
    gap = target_value - current_value
    return gap, current_value


def reduce_position(trading_client, symbol, current_value, target_value):
    """Reduce an existing position toward target using close_position with a
    percentage. This is a risk-reducing action (deleveraging), so it isn't
    blocked by the same buying-power checks a regular BUY/SELL order goes
    through -- important for an account currently sitting at $0 buying power.
    """
    pct_to_close = (1 - (target_value / current_value)) * 100
    action = f"close {pct_to_close:.2f}% of {symbol} (${current_value:,.2f} -> target ${target_value:,.2f})"

    if DRY_RUN:
        return f"[DRY RUN] would {action}"

    trading_client.close_position(
        symbol_or_asset_id=symbol,
        close_options=ClosePositionRequest(percentage=str(round(pct_to_close, 2)))
    )
    return action


def increase_position(trading_client, symbol, gap):
    """Buy into a position using notional (dollar) amount. Requires real
    buying power, so this should only run after any reductions above have
    had a chance to free up cash.
    """
    action = f"buy ${gap:.2f} notional {symbol}"

    if DRY_RUN:
        return f"[DRY RUN] would {action}"

    order_request = MarketOrderRequest(
        symbol=symbol,
        notional=round(gap, 2),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    trading_client.submit_order(order_data=order_request)
    return action


def log_result(date, regime, exposure_pct, actions_summary, equity, total_before, total_target):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "regime", "target_exposure", "action", "equity", "position_value_before", "position_value_target"])
        writer.writerow([date, regime, exposure_pct, actions_summary, equity, total_before, total_target])


def main():
    api_key = os.environ.get("ALPACA_KEY")
    api_secret = os.environ.get("ALPACA_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Set ALPACA_KEY and ALPACA_SECRET environment variables before running.")

    trading_client = TradingClient(api_key, api_secret, paper=True)

    if DRY_RUN:
        print("=== DRY RUN MODE -- no real orders will be submitted ===\n")

    print("Determining current regime from real SPY data...")
    current_state, current_date = get_current_regime()
    regime_label = regime_names.get(current_state, "unknown")
    target_exposure = exposure.get(current_state, 1.0)
    print(f"Date: {current_date.date()}, Regime: {regime_label}, Target exposure: {target_exposure}x\n")

    account = trading_client.get_account()
    equity = float(account.equity)
    per_asset_target = (equity * target_exposure) / len(ASSETS)
    print(f"Account equity: ${equity:,.2f}")
    print(f"Target per asset (equal-weighted across {len(ASSETS)}): ${per_asset_target:,.2f}\n")

    # Compute every asset's gap first, before touching anything
    gaps = {}
    for symbol in ASSETS:
        gap, current_value = get_gap(trading_client, symbol, per_asset_target)
        gaps[symbol] = (gap, current_value)

    actions = []
    total_before = sum(v for _, v in gaps.values())

    # Phase 1: reductions first (deleveraging, not blocked by buying-power checks)
    print("--- Phase 1: reducing oversized positions ---")
    for symbol, (gap, current_value) in gaps.items():
        if gap < -1:  # needs to shrink
            if not DRY_RUN and has_open_order(trading_client, symbol):
                print(f"{symbol}: already has a pending order -- skipping to avoid a duplicate. Wait for it to settle.")
                continue
            action = reduce_position(trading_client, symbol, current_value, per_asset_target)
            print(f"{symbol}: {action}")
            actions.append(action)

    # Phase 2: re-check real buying power before attempting any buys
    if not DRY_RUN:
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        print(f"\nBuying power after reductions: ${buying_power:,.2f}")
        if buying_power <= 0:
            print("Buying power is still $0 -- reductions likely haven't settled yet.")
            print("Stopping here. Re-run this script in a few minutes once the sale settles,")
            print("it will skip the completed reduction and proceed straight to the buys.")
            log_result(current_date.date(), regime_label, target_exposure, "; ".join(actions), equity, total_before, per_asset_target * len(ASSETS))
            print(f"\nLogged partial progress to {LOG_FILE}")
            return

    print("\n--- Phase 2: buying into underweight positions ---")
    for symbol, (gap, current_value) in gaps.items():
        if gap > 1:  # needs to grow
            if not DRY_RUN and has_open_order(trading_client, symbol):
                print(f"{symbol}: already has a pending order -- skipping to avoid a duplicate.")
                continue
            action = increase_position(trading_client, symbol, gap)
            print(f"{symbol}: {action}")
            actions.append(action)

    total_target = per_asset_target * len(ASSETS)
    actions_summary = "; ".join(actions)

    if not DRY_RUN:
        log_result(current_date.date(), regime_label, target_exposure, actions_summary, equity, total_before, total_target)
        print(f"\nLogged to {LOG_FILE}")
    else:
        print("\n[DRY RUN] Nothing logged, nothing traded.")


if __name__ == "__main__":
    main()
