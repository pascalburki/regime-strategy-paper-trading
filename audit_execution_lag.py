"""
audit_execution_lag.py -- offline audit of backtest_directional_filter.csv
===========================================================================
Needs only pandas + numpy and the committed CSV. No internet, no HMM.

WHY THIS EXISTS
The published backtest applies the exposure decided from day T's data to day
T's OWN return. In live trading the script runs after the close and orders
fill the next session, so exposure decided on day T earns day T+1's return.
That one-day timing gap flatters a backtest, because the regime signal is
built from returns (and a 20-day volatility that includes day T) -- a big
down day pushes the signal toward "stress" and the backtest dodges that very
day.

HOW IT WORKS WITHOUT RAW PRICES
The strategy applies one exposure to all five assets, so each day:
    strategy_return = exposure x buy_and_hold_return
Dividing the two saved equity curves' daily returns therefore recovers the
exposure the backtest actually used. Sanity check: only 1.15 / 1.00 / 0.00
should appear. Then we rebuild the curves with exposure lagged one day.

WHAT IT PRINTS
1. Published (same-day) vs 1-day-lag results, per period and full sample
2. A leverage-matched buy-and-hold baseline (same average exposure), because
   the strategy sits above 1.0x on most days -- some "edge" is just leverage
3. Block-bootstrap ranges for the Sharpe differences that matter

LIMITS (stated plainly)
- Close-to-close lag is a proxy; live fills happen at the next open.
- No trading costs or financing on the extra 0.15x (see evaluate_honest.py).
- The period labels are chronological blocks, NOT a clean holdout: the rules
  were chosen with the full sample in view.

Run:  python3 audit_execution_lag.py [path/to/backtest_directional_filter.csv]
"""

import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "backtest_directional_filter.csv"
BLOCKS = {
    "TRAIN 2015-2019": ("2015-01-01", "2019-12-31"),
    "VALID 2020-2022": ("2020-01-01", "2022-12-31"),
    "TEST  2023-2026": ("2023-01-01", "2026-12-31"),
    "FULL": ("2015-01-01", "2026-12-31"),
}
rng = np.random.default_rng(7)


def recover_exposure(strategy_ret, bh_ret):
    """exposure_t = strategy_ret_t / buy_and_hold_ret_t, where bh is not ~0."""
    ok = bh_ret.abs() > 1e-4
    e = pd.Series(np.nan, index=bh_ret.index)
    e[ok] = (strategy_ret[ok] / bh_ret[ok]).round(2)
    return e.ffill().bfill()


def stats(ret):
    curve = (1 + ret).cumprod()
    years = len(ret) / 252
    return {
        "Total": curve.iloc[-1] - 1,
        "CAGR": curve.iloc[-1] ** (1 / years) - 1,
        "Sharpe": ret.mean() / ret.std() * np.sqrt(252),
        "MaxDD": (curve / curve.cummax() - 1).min(),
    }


def block_bootstrap_sharpe_diff(a, b, block=21, n=2000):
    """Paired moving-block bootstrap of Sharpe(a) - Sharpe(b)."""
    d = pd.concat([a, b], axis=1).dropna().values
    T = len(d)
    starts_max = T - block
    diffs = []
    for _ in range(n):
        idx = np.concatenate([
            np.arange(s, s + block)
            for s in rng.integers(0, starts_max, size=T // block + 1)
        ])[:T]
        x, y = d[idx, 0], d[idx, 1]
        diffs.append(x.mean() / x.std() * np.sqrt(252) - y.mean() / y.std() * np.sqrt(252))
    return np.percentile(diffs, [5, 50, 95])


def main():
    curves = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date")
    r = curves.pct_change().dropna()
    bh = r["buy_and_hold"]

    e_reg = recover_exposure(r["current_pure_regime"], bh)
    e_trd = recover_exposure(r["regime_plus_trend"], bh)

    allowed = {0.0, 1.0, 1.15}
    assert set(e_reg.unique()) <= allowed and set(e_trd.unique()) <= allowed, \
        "Recovered exposures are not the expected set -- reconstruction invalid"
    print("Reconstruction check passed: exposures recovered exactly "
          f"{sorted(set(e_reg.unique()))}\n")

    lag = lambda e: e.shift(1).fillna(1.0)
    variants = {
        "Buy & hold 1.0x": bh,
        "Buy & hold, matched leverage": e_reg.mean() * bh,
        "Regime only - published (same-day)": e_reg * bh,
        "Regime only - 1-day lag": lag(e_reg) * bh,
        "Regime+trend - published (same-day)": e_trd * bh,
        "Regime+trend - 1-day lag": lag(e_trd) * bh,
    }

    print(f"Average exposure: regime-only {e_reg.mean():.3f}x, "
          f"regime+trend {e_trd.mean():.3f}x. "
          f"Days above 1.0x: {(e_reg > 1).mean():.0%} / {(e_trd > 1).mean():.0%}. "
          f"Days at 0x: {(e_reg == 0).sum()} of {len(e_reg)}.\n")

    pd.options.display.float_format = "{:,.3f}".format
    for name, (a, b) in BLOCKS.items():
        table = pd.DataFrame({k: stats(v.loc[a:b]) for k, v in variants.items()}).T
        print(f"=== {name} ({len(bh.loc[a:b])} days) ===")
        print(table.to_string(), "\n")

    print("=== Sharpe differences, FULL sample, 1-day lag (5th / median / 95th pct) ===")
    reg_l, trd_l = variants["Regime only - 1-day lag"], variants["Regime+trend - 1-day lag"]
    matched = variants["Buy & hold, matched leverage"]
    for label, x, y in [
        ("Regime-only  minus  matched-leverage buy&hold", reg_l, matched),
        ("Trend filter minus  regime-only", trd_l, reg_l),
    ]:
        lo, mid, hi = block_bootstrap_sharpe_diff(x, y)
        print(f"  {label}: {lo:+.2f} / {mid:+.2f} / {hi:+.2f}")


if __name__ == "__main__":
    main()
