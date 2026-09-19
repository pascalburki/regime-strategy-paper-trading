"""
Backtest: Regime + Directional Trend Filter
==============================================
Purpose: test whether adding a simple trend check on top of the existing
volatility regime fixes the real, live problem diagnosed Sept 2026 --
the regime read "calm" on 18 of 19 trading days from Aug 24-Sep 18 2026
while equity quietly drifted down ~1.2%, because "calm" measures
volatility, not direction.

THE RULE BEING TESTED:
  calm + price above its own 50-day moving average -> 1.15x (unchanged)
  calm + price BELOW its own 50-day moving average -> 1.0x  (capped down)
  moderate -> 1.0x (unchanged)
  stress   -> 0x   (unchanged)

Only the ambiguous "calm" case changes. Moderate and stress already
imply enough caution on their own -- calm is the one regime that says
nothing about direction at all, which is exactly the gap this closes.

Run this yourself (needs real internet access -- same limitation as the
other two scripts, the sandbox I write this in can't reach Yahoo
Finance):

    pip install yfinance numpy pandas hmmlearn --break-system-packages
    python3 backtest_directional_filter.py

What it does:
1. Pulls real historical SPY data (the signal driving the whole
   portfolio, same as the live strategy)
2. Runs the SAME walk-forward HMM as always -- unchanged
3. Computes SPY's own 50-day moving average, using only data up to
   each day (a rolling average is naturally walk-forward safe -- day T
   only ever uses days T-49 through T, never the future)
4. Builds two equity curves: CURRENT (pure regime, existing rule) vs
   REGIME+TREND (the new rule above), plus a buy-and-hold benchmark
5. Prints total return, Sharpe ratio, and max drawdown for all three

HONEST LIMITATIONS, same as before:
- This isolates just the regime+trend question -- no trading costs,
  slippage, rebalancing band, or stop loss included, so don't treat
  these exact numbers as what live trading would have produced.
- 50 days is a reasonable, common choice for a trend window, not
  something exhaustively tuned -- if this shows promise, testing other
  windows (20-day, 100-day) would be a reasonable next step, not
  something to over-optimize on the first pass.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

SIGNAL_SYMBOL = "SPY"
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
DATA_HISTORY_START = '2010-01-01'
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
TREND_WINDOW = 50
STARTING_EQUITY = 100000.0


def run_walkforward(df_clean, start_date):
    """Identical to the live strategy -- kept the same everywhere so
    every comparison is apples-to-apples."""
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


def get_signal_data():
    """Pull SPY data, compute regime AND the 50-day trend check."""
    today = pd.Timestamp.today()
    df = yf.download(SIGNAL_SYMBOL, start=DATA_HISTORY_START,
                      end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    returns = np.log(close / close.shift(1)).dropna()
    vol = returns.rolling(window=20).std().dropna()
    df["returns"] = returns
    df["vol"] = vol
    df_clean = df.dropna().copy()

    regime = run_walkforward(df_clean, WALKFORWARD_HMM_START)

    # Trend check: is price above or below its own 50-day moving average?
    # This uses only past data at each point (T-49 through T), so it's
    # automatically walk-forward safe -- no future leakage.
    ma50 = close.rolling(window=TREND_WINDOW).mean()
    trend_up = (close > ma50)

    return close, returns, regime, trend_up


def get_asset_returns(symbol):
    today = pd.Timestamp.today()
    df = yf.download(symbol, start=DATA_HISTORY_START,
                      end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df.columns = df.columns.get_level_values(0)
    return df["Close"].pct_change()


def build_equity_curve(daily_returns_by_asset, exposure_series, shared_dates):
    """Equal-weighted portfolio: same exposure applied across all 5
    assets each day (matches the live strategy's shared-signal design)."""
    n_assets = len(daily_returns_by_asset)
    equity = STARTING_EQUITY
    curve = []
    for date in shared_dates:
        daily_port_return = 0.0
        exp = exposure_series.get(date, 1.0)
        for symbol in daily_returns_by_asset:
            r = daily_returns_by_asset[symbol].get(date, 0.0)
            daily_port_return += (r * exp) / n_assets
        equity *= (1 + daily_port_return)
        curve.append(equity)
    return pd.Series(curve, index=shared_dates)


def compute_metrics(equity_curve):
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else float('nan')
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    return total_return, sharpe, max_dd


def main():
    print("Pulling SPY data, computing regime and 50-day trend...")
    close, spy_returns, regime, trend_up = get_signal_data()

    print("Pulling data for the other 4 assets...")
    daily_returns = {"SPY": spy_returns}
    for symbol in ["QQQ", "GLD", "TLT", "XOM"]:
        print(f"  {symbol}...")
        daily_returns[symbol] = get_asset_returns(symbol)

    shared_dates = regime.index.intersection(trend_up.index)
    for symbol in ASSETS:
        shared_dates = shared_dates.intersection(daily_returns[symbol].dropna().index)
    print(f"\nBacktesting over {len(shared_dates)} shared trading days\n")

    regime = regime.loc[shared_dates]
    trend_up = trend_up.loc[shared_dates]

    # --- CURRENT rule: pure regime, no trend consideration ---
    EXPOSURE = {0: 1.15, 1: 1.0, 2: 0.0}
    exposure_current = regime.map(EXPOSURE)

    # --- NEW rule: calm gets capped to 1.0x if price is below its own trend ---
    exposure_new = []
    for date in shared_dates:
        r = regime[date]
        if r == 0:  # calm
            exposure_new.append(1.15 if trend_up[date] else 1.0)
        else:
            exposure_new.append(EXPOSURE[r])
    exposure_new = pd.Series(exposure_new, index=shared_dates)

    # --- Buy & hold: always 1.0x ---
    exposure_bh = pd.Series(1.0, index=shared_dates)

    curve_current = build_equity_curve(daily_returns, exposure_current, shared_dates)
    curve_new = build_equity_curve(daily_returns, exposure_new, shared_dates)
    curve_bh = build_equity_curve(daily_returns, exposure_bh, shared_dates)

    print("=== RESULTS ===\n")
    for name, curve in [("CURRENT (pure regime)", curve_current),
                         ("REGIME + TREND FILTER (new)", curve_new),
                         ("BUY & HOLD (no timing)", curve_bh)]:
        total_return, sharpe, max_dd = compute_metrics(curve)
        print(f"{name}:")
        print(f"  Total return: {total_return:+.2%}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Max drawdown: {max_dd:.2%}")
        print()

    # How often did the trend filter actually change anything?
    calm_days = (regime == 0).sum()
    calm_but_declining = ((regime == 0) & (~trend_up)).sum()
    print(f"Out of {calm_days} 'calm' days, {calm_but_declining} "
          f"({calm_but_declining/calm_days:.1%}) were also below the 50-day trend "
          f"-- these are the days the new rule actually changes.")

    out = pd.DataFrame({
        "current_pure_regime": curve_current,
        "regime_plus_trend": curve_new,
        "buy_and_hold": curve_bh,
    })
    out.to_csv("backtest_directional_filter.csv")
    print("\nSaved full equity curves to backtest_directional_filter.csv")


if __name__ == "__main__":
    main()
