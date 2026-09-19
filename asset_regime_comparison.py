"""
Independent Regime Comparison: SPY, QQQ, GLD, TLT, XOM
=========================================================
Purpose: check whether each asset's OWN regime behavior actually
matches what the live strategy currently assumes -- that SPY's
regime signal is a fair stand-in for all 5 assets.

Run this yourself (needs real internet access to Yahoo Finance, which
the sandbox I write code in doesn't have -- see setup steps below):

    pip install yfinance numpy pandas hmmlearn matplotlib --break-system-packages
    python3 asset_regime_comparison.py

What it does:
1. Pulls real historical data for all 5 assets independently
2. Runs the SAME walk-forward HMM on each one separately (the exact
   same method your live strategy already uses for SPY -- just
   applied to every asset on its own, not borrowed from SPY)
3. Compares how often each asset's OWN regime disagrees with SPY's
   regime on the same trading day
4. Plots all 5 regime timelines stacked, so you can see visually
   where they diverge
5. Saves the raw regime data to a CSV for further digging
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
DATA_HISTORY_START = '2010-01-01'
WALKFORWARD_HMM_START = pd.Timestamp('2015-01-01')


def run_walkforward(df_clean, start_date):
    """Same validated walk-forward logic as the main strategy script --
    kept identical so this comparison is apples-to-apples."""
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


def get_regime_series(symbol):
    """Pull real data for one asset and run its OWN walk-forward HMM."""
    today = pd.Timestamp.today()
    df = yf.download(symbol, start=DATA_HISTORY_START,
                      end=(today + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    df.columns = df.columns.get_level_values(0)
    close = df["Close"]
    returns = np.log(close / close.shift(1)).dropna()
    vol = returns.rolling(window=20).std().dropna()
    df["returns"] = returns
    df["vol"] = vol
    df_clean = df.dropna().copy()

    regime_signal = run_walkforward(df_clean, WALKFORWARD_HMM_START)
    return regime_signal


def main():
    print("Pulling data and running independent walk-forward HMM per asset...")
    print("(This takes a while -- 5 assets, each refitting monthly across ~10 years)\n")

    regimes = {}
    for symbol in ASSETS:
        print(f"  Running {symbol}...")
        regimes[symbol] = get_regime_series(symbol)

    # Align all 5 on shared dates
    combined = pd.DataFrame(regimes).dropna()
    print(f"\nShared trading days across all 5 assets: {len(combined)}\n")

    # How often does each asset's OWN regime differ from SPY's?
    print("=== Disagreement with SPY's regime (the signal currently driving all 5) ===")
    for symbol in ["QQQ", "GLD", "TLT", "XOM"]:
        disagreement = (combined[symbol] != combined["SPY"]).mean()
        print(f"{symbol} disagrees with SPY's regime on {disagreement:.1%} of days")

    # Visual: stacked regime timelines
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    regime_names = {0: "calm", 1: "moderate", 2: "stress"}
    for ax, symbol in zip(axes, ASSETS):
        ax.plot(combined.index, combined[symbol], drawstyle='steps-post')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels([regime_names[i] for i in [0, 1, 2]])
        ax.set_title(f"{symbol} regime (its own independent HMM)")
    plt.tight_layout()
    plt.savefig("asset_regime_comparison.png", dpi=150)
    print("\nSaved chart to asset_regime_comparison.png")

    combined.to_csv("asset_regime_comparison.csv")
    print("Saved raw regime data to asset_regime_comparison.csv")


if __name__ == "__main__":
    main()
