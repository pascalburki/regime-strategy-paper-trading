"""
Backtest: Asset-Specific Regime Signals vs Shared SPY Signal
===============================================================
Purpose: now that asset_regime_comparison.py proved QQQ/GLD/TLT/XOM
disagree with SPY's regime 7-36% of the time, this actually tests
whether giving each asset its OWN regime signal produces a better
portfolio than the current approach (SPY's regime driving all 5).

Run this yourself (needs real internet access -- same limitation as
asset_regime_comparison.py, the sandbox I write this in can't reach
Yahoo Finance):

    pip install yfinance numpy pandas hmmlearn --break-system-packages
    python3 backtest_asset_specific_signals.py

What it does:
1. Pulls real historical data for all 5 assets
2. Runs each asset's OWN independent walk-forward HMM (same as
   asset_regime_comparison.py -- reused directly, not rebuilt)
3. Builds THREE parallel equity curves over the same historical period:
   a) CURRENT approach: SPY's regime sets exposure for all 5 assets
   b) ASSET-SPECIFIC: each asset's OWN regime sets its own exposure
   c) BUY & HOLD: no timing at all, equal-weighted, always 1.0x
4. Prints total return, annualized Sharpe ratio, and max drawdown for
   all three, so you can see directly whether asset-specific signals
   actually help or not -- not just assume they would.

HONEST LIMITATIONS, stated up front:
- This is a backtest, not a live result -- same walk-forward
  discipline as the live strategy (never trains on future data), but
  still historical simulation, not proof of future performance.
- Trading costs, slippage, and the rebalancing-band/stop-loss logic
  from the live script are NOT included here -- this isolates just the
  regime-signal question, deliberately kept simple so the comparison
  is clean. Don't treat these exact return numbers as what live
  trading would have produced.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
DATA_HISTORY_START = '2010-01-01'
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')
EXPOSURE = {0: 1.15, 1: 1.0, 2: 0.0}  # calm, moderate, stress -- same as live strategy
STARTING_EQUITY = 100000.0


def run_walkforward(df_clean, start_date):
    """Identical to the live strategy and asset_regime_comparison.py --
    kept the same everywhere so every comparison is apples-to-apples."""
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


def get_price_and_regime(symbol):
    """Pull real data, compute daily returns, and this asset's own regime."""
    today = pd.Timestamp.today()
    df = yf.download(symbol, start=DATA_HISTORY_START,
                      end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    daily_return = close.pct_change()
    log_return = np.log(close / close.shift(1)).dropna()
    vol = log_return.rolling(window=20).std().dropna()
    df["returns"] = log_return
    df["vol"] = vol
    df_clean = df.dropna().copy()

    regime = run_walkforward(df_clean, WALKFORWARD_HMM_START)
    return daily_return, regime


def build_equity_curve(daily_returns_by_asset, exposures_by_asset, shared_dates):
    """Equal-weighted portfolio: each asset's daily return, scaled by that
    day's target exposure, averaged across all 5 assets, compounded."""
    n_assets = len(daily_returns_by_asset)
    equity = STARTING_EQUITY
    curve = []
    for date in shared_dates:
        daily_port_return = 0.0
        for symbol in daily_returns_by_asset:
            r = daily_returns_by_asset[symbol].get(date, 0.0)
            exp = exposures_by_asset[symbol].get(date, 1.0)
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
    print("Pulling data and computing regimes for all 5 assets...")
    print("(Reuses the same walk-forward HMM as the live strategy)\n")

    daily_returns = {}
    regimes = {}
    for symbol in ASSETS:
        print(f"  Processing {symbol}...")
        ret, reg = get_price_and_regime(symbol)
        daily_returns[symbol] = ret
        regimes[symbol] = reg

    # Shared dates across everything
    regime_df = pd.DataFrame(regimes).dropna()
    shared_dates = regime_df.index
    print(f"\nBacktesting over {len(shared_dates)} shared trading days\n")

    # --- Approach A: CURRENT (SPY's regime drives all 5) ---
    spy_regime = regime_df["SPY"]
    exposures_current = {
        symbol: spy_regime.map(EXPOSURE) for symbol in ASSETS
    }
    curve_current = build_equity_curve(daily_returns, exposures_current, shared_dates)

    # --- Approach B: ASSET-SPECIFIC (each asset's own regime) ---
    exposures_specific = {
        symbol: regime_df[symbol].map(EXPOSURE) for symbol in ASSETS
    }
    curve_specific = build_equity_curve(daily_returns, exposures_specific, shared_dates)

    # --- Approach C: BUY & HOLD (always 1.0x, no timing) ---
    exposures_bh = {symbol: pd.Series(1.0, index=shared_dates) for symbol in ASSETS}
    curve_bh = build_equity_curve(daily_returns, exposures_bh, shared_dates)

    print("=== RESULTS ===\n")
    for name, curve in [("CURRENT (shared SPY signal)", curve_current),
                         ("ASSET-SPECIFIC (each asset's own regime)", curve_specific),
                         ("BUY & HOLD (no timing)", curve_bh)]:
        total_return, sharpe, max_dd = compute_metrics(curve)
        print(f"{name}:")
        print(f"  Total return: {total_return:+.2%}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Max drawdown: {max_dd:.2%}")
        print()

    out = pd.DataFrame({
        "current_shared_signal": curve_current,
        "asset_specific_signal": curve_specific,
        "buy_and_hold": curve_bh,
    })
    out.to_csv("backtest_comparison.csv")
    print("Saved full equity curves to backtest_comparison.csv")


if __name__ == "__main__":
    main()
