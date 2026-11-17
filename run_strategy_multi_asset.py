"""
Regime Strategy — Multi-Asset Alpaca Paper Trading Runner
Extends the validated single-asset (SPY-only) version to an equal-weighted
5-asset portfolio (SPY, QQQ, GLD, TLT, XOM), using SPY's regime signal to
drive exposure across all 5 equally -- matching the originally validated
approach (0.57 vs 0.44 return, 0.84 vs 0.69 Sharpe vs buy-and-hold).

VERSION 2 (Nov 17 2026), each change backtested first in evaluate_honest.py
and research/backtest_bonds_bil.py:
  * No leverage: calm exposure is 1.0x (was 1.15x). Leverage added about
    +0.1%/yr while lowering the Sharpe ratio.
  * Bonds only when they protect: TLT is held only while its 63-day
    correlation with SPY is negative. Otherwise its share goes to BIL.
  * T-bills instead of cash: whatever is not invested is held in BIL.
  Backtest 2015-now with costs: Sharpe 0.987 -> 1.030, worst drawdown
  -15.6% -> -14.1%. Block bootstrap: gain above 0 in 81.7% of paths
  (encouraging, not statistically significant).

SAFETY: every run of this script submits REAL orders to your Alpaca paper
account, including manual "Run workflow" triggers. Use DRY_RUN to preview
without touching the account:
    DRY_RUN=true python3 run_strategy_multi_asset.py
Only unset DRY_RUN once you've confirmed the preview looks right and you're
ready for it to actually trade.

REBALANCING BAND (added Aug 26 2026): only trades a position if it's drifted
more than 3% off target. Before this, the script chased every tiny daily
price move -- e.g. a real observed trade of "buy $23.08 notional SPY" on
Aug 26, which is pure noise, not a meaningful rebalance. This band doesn't
change the validated regime signal or exposure logic at all, just cuts
noise trades around it.

STOP LOSS (added Aug 26 2026): each position closes if it's down more than
STOP_LOSS_PCT from its average entry price, regardless of what the regime
signal says. HONEST LIMITATION: this script runs once a day (scheduled
workflow), not continuously -- so this is a once-per-day check, not a true
intraday stop. It won't catch a flash crash and recover before the next
run; a real intraday stop would need a process running 24/7, not a daily
cron job. What it does do: prevents holding a badly losing position all
the way to the next regime classification, which previously had zero
downside protection below the portfolio level.

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

SIGNAL_SYMBOL = "SPY"
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
DATA_HISTORY_START = '2010-01-01'
exposure = {0: 1.0, 1: 1.0, 2: 0.0}
CASH_SYMBOL = "BIL"
BOND_SYMBOL = "TLT"
CORR_WINDOW = 63
HOLDINGS = ASSETS + [CASH_SYMBOL]
regime_names = {0: "calm", 1: "moderate", 2: "stress"}
LOG_FILE = "strategy_log.csv"
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
REBALANCE_BAND = 0.03
STOP_LOSS_PCT = 0.08

DRAWDOWN_LOOKBACK_DAYS = 5
DRAWDOWN_CAP_THRESHOLD_PCT = 0.01

TREND_WINDOW = 50


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
    """Fetch real SPY data, run the walk-forward HMM to get today's regime,
    and also check today's directional trend (price vs. its own 50-day
    moving average -- see TREND_WINDOW comment above for why)."""
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

    ma50 = close.rolling(window=TREND_WINDOW).mean()
    is_above_trend = bool(close.loc[current_date] > ma50.loc[current_date])

    return current_state, current_date, is_above_trend


def bonds_protect(current_date):
    """True when TLT's 63-day return correlation with SPY is negative,
    i.e. bonds are currently moving against stocks and act as a hedge."""
    start = (current_date - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    end = (current_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    px = yf.download([SIGNAL_SYMBOL, BOND_SYMBOL], start=start, end=end, auto_adjust=True)["Close"]
    rets = px[[SIGNAL_SYMBOL, BOND_SYMBOL]].pct_change().dropna().tail(CORR_WINDOW)
    corr = rets[SIGNAL_SYMBOL].corr(rets[BOND_SYMBOL])
    return bool(corr < 0), float(corr)


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


def check_stop_losses(trading_client):
    """Close any position down more than STOP_LOSS_PCT from its average entry
    price. Runs before the regime-based rebalance so a stopped-out position
    isn't immediately bought back in the same run -- that decision waits for
    tomorrow's fresh regime check instead.
    Returns the set of symbols stopped out this run.
    """
    stopped_out = set()
    for symbol in ASSETS:
        try:
            position = trading_client.get_open_position(symbol)
        except Exception:
            continue

        entry_price = float(position.avg_entry_price)
        current_price = float(position.current_price)
        pct_change = (current_price - entry_price) / entry_price

        if pct_change < -STOP_LOSS_PCT:
            print(f"STOP LOSS TRIGGERED: {symbol} is {pct_change:.1%} from entry (${entry_price:.2f} -> ${current_price:.2f})")
            if not DRY_RUN:
                if has_open_order(trading_client, symbol):
                    print(f"  {symbol} already has a pending order -- skipping stop-loss close this run, will retry next run.")
                    continue
                trading_client.close_position(symbol_or_asset_id=symbol)
                print(f"  Closed {symbol} entirely.")
            else:
                print(f"  [DRY RUN] would close {symbol} entirely.")
            stopped_out.add(symbol)
    return stopped_out


def check_drawdown_override(current_equity, lookback_days=DRAWDOWN_LOOKBACK_DAYS,
                              threshold_pct=DRAWDOWN_CAP_THRESHOLD_PCT):
    """
    Independent safety check, separate from the regime signal entirely.
    Caps exposure at 1.0x (never levered up) if account equity has dropped
    more than threshold_pct over the last lookback_days trading days,
    regardless of what the regime says. Returns (triggered: bool,
    recent_pct_change: float or None -- None only if there isn't enough
    log history yet to evaluate).
    """
    if not os.path.isfile(LOG_FILE):
        return False, None

    with open(LOG_FILE, "r") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < lookback_days:
        return False, None

    past_equity = float(rows[-lookback_days]["equity"])
    pct_change = (current_equity - past_equity) / past_equity

    return (pct_change < -threshold_pct), pct_change


POSITIONS_LOG = "positions_log.csv"


def already_ran(trading_date):
    if not os.path.isfile(POSITIONS_LOG):
        return False
    with open(POSITIONS_LOG, "r") as f:
        return str(trading_date) in {row["date"] for row in csv.DictReader(f)}


def log_positions(date, equity, gaps):
    file_exists = os.path.isfile(POSITIONS_LOG)
    with open(POSITIONS_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "equity"] + HOLDINGS)
        writer.writerow([date, equity] + [round(gaps[s][1], 2) for s in HOLDINGS])


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
    current_state, current_date, is_above_trend = get_current_regime()
    if already_ran(current_date.date()):
        print(f"Trading day {current_date.date()} was already handled. Stopping to avoid trading twice.")
        return
    regime_label = regime_names.get(current_state, "unknown")
    target_exposure = exposure.get(current_state, 1.0)

    if current_state == 0 and not is_above_trend:
        print(f"TREND FILTER: regime is 'calm' but SPY is below its {TREND_WINDOW}-day average "
              f"(calm-but-declining) -- capping exposure at 1.0x instead of {target_exposure}x.\n")
        target_exposure = 1.0
    else:
        trend_desc = "above" if is_above_trend else "below"
        print(f"Date: {current_date.date()}, Regime: {regime_label} (SPY {trend_desc} its {TREND_WINDOW}-day average), "
              f"Target exposure: {target_exposure}x\n")

    print("Checking stop losses...")
    stopped_out = check_stop_losses(trading_client)
    if stopped_out:
        print(f"Stopped out this run: {', '.join(stopped_out)} -- will not re-buy until tomorrow's regime check.\n")
    else:
        print("No positions breached the stop loss.\n")

    account = trading_client.get_account()
    equity = float(account.equity)

    drawdown_triggered, recent_pct_change = check_drawdown_override(equity)
    if drawdown_triggered:
        print(f"DRAWDOWN CIRCUIT-BREAKER TRIGGERED: equity down {recent_pct_change:.2%} "
              f"over the last {DRAWDOWN_LOOKBACK_DAYS} trading days.")
        print(f"Capping exposure at 1.0x regardless of regime (regime said {target_exposure}x from '{regime_label}').\n")
        target_exposure = min(target_exposure, 1.0)
    elif recent_pct_change is not None:
        print(f"Drawdown check: {recent_pct_change:+.2%} over last {DRAWDOWN_LOOKBACK_DAYS} trading days -- no override needed.\n")

    per_asset_target = (equity * target_exposure) / len(ASSETS)
    hedge_ok, bond_corr = bonds_protect(current_date)
    targets = {symbol: per_asset_target for symbol in ASSETS}
    targets[CASH_SYMBOL] = equity * (1 - target_exposure)
    if not hedge_ok:
        targets[CASH_SYMBOL] += targets[BOND_SYMBOL]
        targets[BOND_SYMBOL] = 0.0
    print(f"Account equity: ${equity:,.2f}")
    print(f"Target per asset (equal-weighted across {len(ASSETS)}): ${per_asset_target:,.2f}")
    print(f"Bond check: {CORR_WINDOW}-day SPY/TLT correlation {bond_corr:+.2f} -> "
          f"{'hold TLT' if hedge_ok else 'TLT moved to ' + CASH_SYMBOL}")
    print(f"{CASH_SYMBOL} target: ${targets[CASH_SYMBOL]:,.2f}\n")

    gaps = {}
    for symbol in HOLDINGS:
        gap, current_value = get_gap(trading_client, symbol, targets[symbol])
        gaps[symbol] = (gap, current_value)

    band_usd = (equity / len(ASSETS)) * REBALANCE_BAND
    print(f"Rebalancing band: +/-${band_usd:,.2f} ({REBALANCE_BAND:.0%} of target) -- smaller drifts are skipped\n")

    actions = []
    skipped = []
    total_before = sum(gaps[s][1] for s in ASSETS)

    print("--- Phase 1: reducing oversized positions ---")
    for symbol, (gap, current_value) in gaps.items():
        if symbol in stopped_out:
            continue
        if gap < -band_usd:
            if not DRY_RUN and has_open_order(trading_client, symbol):
                print(f"{symbol}: already has a pending order -- skipping to avoid a duplicate. Wait for it to settle.")
                continue
            action = reduce_position(trading_client, symbol, current_value, targets[symbol])
            print(f"{symbol}: {action}")
            actions.append(action)
        elif gap < 0:
            skipped.append(f"{symbol} (${-gap:,.2f} under, within band)")

    if not DRY_RUN:
        account = trading_client.get_account()
        buying_power = float(account.buying_power)
        print(f"\nBuying power after reductions: ${buying_power:,.2f}")
        if buying_power <= 0:
            print("Buying power is still $0 -- reductions likely haven't settled yet.")
            print("Stopping here. Re-run this script in a few minutes once the sale settles,")
            print("it will skip the completed reduction and proceed straight to the buys.")
            log_result(current_date.date(), regime_label, target_exposure, "; ".join(actions), equity, total_before, sum(targets[s] for s in ASSETS))
            print(f"\nLogged partial progress to {LOG_FILE}")
            return

    print("\n--- Phase 2: buying into underweight positions ---")
    for symbol, (gap, current_value) in gaps.items():
        if symbol in stopped_out:
            print(f"{symbol}: skipping buy -- stopped out this run, waiting for tomorrow's regime check.")
            continue
        if gap > band_usd:
            if not DRY_RUN and has_open_order(trading_client, symbol):
                print(f"{symbol}: already has a pending order -- skipping to avoid a duplicate.")
                continue
            if not DRY_RUN:
                available = float(trading_client.get_account().buying_power) * 0.99
                if available < band_usd:
                    print(f"{symbol}: not enough buying power yet (${available:,.2f}) -- will buy on the next run.")
                    continue
                gap = min(gap, available)
            action = increase_position(trading_client, symbol, gap)
            print(f"{symbol}: {action}")
            actions.append(action)
        elif gap > 0:
            skipped.append(f"{symbol} (${gap:,.2f} over, within band)")

    if skipped:
        print(f"\nSkipped (within {REBALANCE_BAND:.0%} band, not worth trading): {'; '.join(skipped)}")

    total_target = sum(targets[s] for s in ASSETS)
    actions_summary = "; ".join(actions)

    if not DRY_RUN:
        log_result(current_date.date(), regime_label, target_exposure, actions_summary, equity, total_before, total_target)
        log_positions(current_date.date(), equity, gaps)
        print(f"\nLogged to {LOG_FILE} and {POSITIONS_LOG}")
    else:
        print("\n[DRY RUN] Nothing logged, nothing traded.")


if __name__ == "__main__":
    main()
