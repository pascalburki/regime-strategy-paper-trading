"""
Test of two Strategy v2 ideas, written down before running:

1. BIL in stress: money the strategy does not invest earns the T-bill rate (BIL)
   instead of sitting in cash at 0%.
2. Bonds only when they protect: TLT is held only while its 63-day correlation
   with SPY is negative. Otherwise its share goes to BIL.

Expectation before testing: (1) adds a little return with no extra risk.
(2) lowers losses in 2021-2026, when stocks and bonds fell together.

Everything else matches the live strategy (v1.1): same HMM regimes, live-replica
decode, calm 1.0x, moderate 1.0x, stress 0x, one-day lag, 5 bps costs.
The 63-day window is fixed in advance and is not tuned.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_honest as ev

ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
CASH = "BIL"
CORR_WINDOW = 63
COST_BPS = 5.0
BLOCK = 21
N_BOOT = 5000


def load_prices(synthetic):
    if synthetic:
        px = ev.synthetic_prices()
        px[CASH] = 100 * np.exp(np.cumsum(np.full(len(px), 0.0001)))
        return px
    import yfinance as yf
    px = yf.download(ASSETS + [CASH], start=ev.DATA_START, auto_adjust=True)["Close"]
    return px[ASSETS + [CASH]].dropna()


def weights(exposure, bonds_ok, use_bil, use_bond_rule):
    n = len(ASSETS)
    w = pd.DataFrame(0.0, index=exposure.index, columns=ASSETS + [CASH])
    for a in ASSETS:
        w[a] = exposure / n
    if use_bond_rule:
        tlt_off = ~bonds_ok
        w.loc[tlt_off, CASH] += w.loc[tlt_off, "TLT"]
        w.loc[tlt_off, "TLT"] = 0.0
    if use_bil:
        w[CASH] += 1.0 - exposure
    return w


def run(w, rets):
    w_live = w.shift(1).reindex(rets.index).fillna(0.0)
    gross = (w_live * rets).sum(axis=1)
    turnover = w_live.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * COST_BPS / 1e4


def sharpe(r):
    sd = r.std()
    return r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan


def block_bootstrap_sharpe_gain(new, old, seed=7):
    new, old = new.align(old, join="inner")
    a, b = new.to_numpy(), old.to_numpy()
    n = len(a)
    starts_per_path = int(np.ceil(n / BLOCK))
    rng = np.random.default_rng(seed)
    gains = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK, starts_per_path)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()[:n]
        gains[i] = sharpe(pd.Series(a[idx])) - sharpe(pd.Series(b[idx]))
    return sharpe(new) - sharpe(old), gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--wf-start", default="2015-01-01")
    args = ap.parse_args()

    ev.EXPOSURE = {0: 1.0, 1: 1.0, 2: 0.0}
    prices = load_prices(args.synthetic)
    if args.synthetic:
        print("*** SYNTHETIC DATA: smoke test only, ignore the numbers ***\n")
    rets = prices.pct_change()
    spy = prices["SPY"]

    print("Walk-forward HMM, one fit per month ...")
    labels, _ = ev.walkforward_regimes(ev.build_features(spy), args.wf_start)
    exposure = ev.exposures(labels["causal"], spy)[0]

    corr = rets["SPY"].rolling(CORR_WINDOW).corr(rets["TLT"])
    bonds_ok = (corr < 0).reindex(exposure.index).fillna(True)

    idx = exposure.index.intersection(rets.dropna().index)
    exposure = exposure.reindex(idx)
    bonds_ok = bonds_ok.reindex(idx)
    rets = rets.reindex(idx)

    variants = {
        "Buy & hold (5 ETFs)": rets[ASSETS].mean(axis=1),
        "v1.1 now (stress = cash)": run(weights(exposure, bonds_ok, False, False), rets),
        "+ BIL in stress": run(weights(exposure, bonds_ok, True, False), rets),
        "+ bonds only when they protect": run(weights(exposure, bonds_ok, False, True), rets),
        "+ both": run(weights(exposure, bonds_ok, True, True), rets),
    }

    print(f"\nTLT held on {bonds_ok.mean():.0%} of days under the bond rule\n")
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200
    for name, (a, b) in ev.BLOCKS.items():
        rows = {k: ev.stats(r.loc[a:b].dropna()) for k, r in variants.items() if len(r.loc[a:b].dropna()) > 20}
        if rows:
            print(f"=== {name} ===")
            print(pd.DataFrame(rows).T.to_string(), "\n")

    start = ev.BLOCKS["FULL"][0]
    new = variants["+ both"].loc[start:].dropna()
    old = variants["v1.1 now (stress = cash)"].loc[start:].dropna()
    print(f"Block bootstrap ({N_BOOT} paths, {BLOCK}-day blocks): Sharpe gain of '+ both' over 'v1.1 now', 2015-now")
    gain, gains = block_bootstrap_sharpe_gain(new, old)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    print(f"  Measured gain:          {gain:+.3f}")
    print(f"  95% range of the gain:  {lo:+.3f} to {hi:+.3f}")
    print(f"  Share of paths with a gain above 0: {(gains > 0).mean():.1%}")


if __name__ == "__main__":
    main()
