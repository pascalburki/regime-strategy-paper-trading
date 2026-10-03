"""
evaluate_honest.py -- honest re-evaluation of the regime strategy
=================================================================
Same HMM, same exposure map, same five assets as the live strategy. What
changes is HOW the backtest is scored. Each step removes one way a backtest
can look better than live trading could be:

  A. PUBLISHED   decode each whole month at once (Viterbi over the month),
                 apply the exposure to the SAME day's return.
  B. + LAG       exposure decided at the close of day t earns day t+1's
                 return (live orders go in after the close).
  C. LIVE REPLICA  the label for day t decodes only the current month's data
                 through day t, restarting every month -- exactly what
                 get_current_regime() does live.
  D. FULL-HISTORY  the label for day t decodes ALL history through day t with
                 the fitted model. Still causal, and consistent with how the
                 model was trained.
  E. D + RELABEL   as D, but states are relabelled each refit by fitted
                 volatility (lowest = calm, highest = stress).

WHY D AND E EXIST (found on 2026-10-03)
  * EM collapses the fitted start-state probabilities to one-hot (the state of
    the first 2010 observation). Restarting the decode every month therefore
    forces each month to START in that one state, whatever the market was
    actually doing -- so early-month labels in C are largely an artifact.
  * With fixed initial parameters, the fitted states are not guaranteed to
    keep the order calm < moderate < stress. The script counts refits where
    the fitted volatility order differs from the state index (a "swap").
    Relabelling (E) removes that dependence.

BASELINES (the strategy sits above 1.0x on most days, so plain buy-and-hold
is not the only yardstick):
  - Buy & hold 1.0x
  - Buy & hold at the live replica's AVERAGE exposure ("matched leverage")

FRICTIONS (scenario "with frictions"): turnover cost in basis points per unit
of exposure traded, plus financing on exposure above 1.0x.
  ASSUMPTIONS, not measurements: 5 bps, 4% a year. Edit SCENARIOS to taste.

PERIODS are chronological blocks. They are NOT a clean holdout: the rules
were chosen with the full sample in view, so TEST is "later and untouched by
re-tuning from here on", not "never seen". The only genuinely out-of-sample
evidence is the live paper account from 2026-08-24 onward.

USAGE
  pip install yfinance numpy pandas hmmlearn
  python3 evaluate_honest.py                 # real data (needs internet)
  python3 evaluate_honest.py --synthetic     # smoke test, fake data, no internet
Writes results/metrics.csv and results/daily_returns.csv. Each month is fit
once and decoded every way, so runtime is roughly the cost of ~140 HMM fits.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

SIGNAL = "SPY"
ASSETS = ["SPY", "QQQ", "GLD", "TLT", "XOM"]
EXPOSURE = {0: 1.15, 1: 1.0, 2: 0.0}          # calm / moderate / stress
TREND_WINDOW = 50
DATA_START = "2010-01-01"
BLOCKS = {
    "TRAIN 2015-2019": ("2015-01-01", "2019-12-31"),
    "VALID 2020-2022": ("2020-01-01", "2022-12-31"),
    "TEST  2023-now":  ("2023-01-01", "2100-01-01"),
    "FULL":            ("2015-01-01", "2100-01-01"),
}
SCENARIOS = {
    "frictionless":   dict(cost_bps=0.0, financing=0.00),
    "with frictions": dict(cost_bps=5.0, financing=0.04),
}


# ---------------------------------------------------------------- HMM ------
def make_model():
    """Identical configuration to the live strategy."""
    m = GaussianHMM(n_components=3, covariance_type="diag", random_state=3,
                    n_iter=200, tol=0.01, init_params="")
    m.startprob_ = np.array([0.34, 0.33, 0.33])
    m.transmat_ = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    m.means_ = np.array([[0.0, 0.01], [0.001, 0.02], [-0.001, 0.03]])
    m.covars_ = np.array([[0.0001, 0.0001]] * 3)
    return m


def build_features(spy_close):
    ret = np.log(spy_close / spy_close.shift(1)).dropna()
    vol = ret.rolling(20).std().dropna()
    return pd.concat({"returns": ret, "vol": vol}, axis=1).dropna()


DECODINGS = ("smoothed", "causal", "causal_full")


def walkforward_regimes(feat, start):
    """Monthly refit on past data only. Each month is fit ONCE and decoded
    three ways from that same fit, plus a volatility-relabelled copy of each:
      smoothed     whole month decoded at once (published backtest)
      causal       day t decoded from this month's data through t (live replica)
      causal_full  day t decoded from ALL history through t
    Returns ({name: labels}, diagnostics)."""
    X = feat.values
    cur = pd.Timestamp(start)
    out = {d + sfx: [] for d in DECODINGS for sfx in ("", "+relabel")}
    fits = swaps = 0
    start_max = []
    while cur < feat.index[-1]:
        nxt = cur + pd.DateOffset(months=1)
        train, test = feat.loc[:cur], feat.loc[cur:nxt]
        if len(train) < 100 or len(test) == 0:
            cur = nxt
            continue
        model = make_model()
        try:
            model.fit(train)
        except Exception:
            cur = nxt
            continue
        fits += 1
        vol_order = np.argsort(model.means_[:, 1])            # states, calmest -> most volatile
        swaps += int(not np.array_equal(vol_order, [0, 1, 2]))
        start_max.append(float(model.startprob_.max()))
        rank = np.empty(3, dtype=int)
        rank[vol_order] = np.arange(3)                         # state -> volatility rank
        pos = feat.index.get_indexer(test.index)
        raw = {
            "smoothed": model.predict(test),
            "causal": np.array([model.predict(test.iloc[: i + 1])[-1] for i in range(len(test))]),
            "causal_full": np.array([model.predict(X[: p + 1])[-1] for p in pos]),
        }
        for d, lab in raw.items():
            out[d].append(pd.Series(lab, index=test.index))
            out[d + "+relabel"].append(pd.Series(rank[lab], index=test.index))
        cur = nxt
    labels = {k: pd.concat(v) for k, v in out.items()}
    labels = {k: v[~v.index.duplicated(keep="first")] for k, v in labels.items()}
    return labels, {"refits": fits, "state_order_swaps": swaps,
                    "startprob_max_median": float(np.median(start_max))}


# ------------------------------------------------------------ exposures ----
def exposures(regime, spy_close):
    """Pure regime exposure, and regime + trend filter (calm is capped to 1.0x
    when SPY is at or below its own 50-day average)."""
    ma = spy_close.rolling(TREND_WINDOW).mean()
    up = (spy_close > ma).reindex(regime.index)
    e_reg = regime.map(EXPOSURE).astype(float)
    e_trd = e_reg.copy()
    e_trd[(regime == 0) & (~up)] = 1.0
    return e_reg, e_trd


def strategy_returns(e_decided, bh, cost_bps, financing, lag):
    e = e_decided.reindex(bh.index)
    e = e.shift(1) if lag else e
    e = e.fillna(1.0)
    turnover = e.diff().abs().fillna(0.0)
    r = e * bh - turnover * cost_bps / 1e4 - (e - 1).clip(lower=0) * financing / 252
    return r


def stats(ret):
    curve = (1 + ret).cumprod()
    years = max(len(ret) / 252, 1e-9)
    sd = ret.std()
    return {
        "Total": curve.iloc[-1] - 1,
        "CAGR": curve.iloc[-1] ** (1 / years) - 1,
        "Sharpe": ret.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "MaxDD": (curve / curve.cummax() - 1).min(),
    }


# ----------------------------------------------------------------- data ----
def load_real_prices():
    import yfinance as yf
    px = yf.download(ASSETS, start=DATA_START, auto_adjust=True)["Close"]
    return px[ASSETS].dropna()


def synthetic_prices(start="2011-01-03", end="2019-12-31", seed=1):
    """SMOKE-TEST DATA ONLY. Regime-switching volatility, five correlated
    assets. Proves the pipeline runs; says nothing about real performance."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    T = len(idx)
    P = np.array([[.985, .012, .003], [.02, .96, .02], [.03, .05, .92]])
    state = np.zeros(T, int)
    for t in range(1, T):
        state[t] = rng.choice(3, p=P[state[t - 1]])
    sig = np.array([0.006, 0.011, 0.026])[state]
    mkt = rng.normal(0.0005, 1, T) * sig
    betas = {"SPY": 1.0, "QQQ": 1.2, "GLD": 0.1, "TLT": -0.2, "XOM": 0.8}
    cols = {a: 100 * np.exp(np.cumsum(b * mkt + rng.normal(0, 0.006, T))) for a, b in betas.items()}
    return pd.DataFrame(cols, index=idx)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="smoke test on fake data")
    ap.add_argument("--wf-start", default="2015-01-01", help="first walk-forward month")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    prices = synthetic_prices() if args.synthetic else load_real_prices()
    if args.synthetic:
        print("*** SYNTHETIC DATA: pipeline smoke test only -- ignore the numbers ***\n")
    spy = prices[SIGNAL]
    bh = prices[ASSETS].pct_change().mean(axis=1)
    feat = build_features(spy)

    print("Walk-forward HMM: one fit per month, decoded every way ...")
    L, diag = walkforward_regimes(feat, args.wf_start)
    idx = bh.dropna().index
    for k in L:
        idx = idx.intersection(L[k].index)

    print(f"\nDiagnostics: {diag}")
    dis = (L["smoothed"].reindex(idx) != L["causal"].reindex(idx))
    print(f"Published-style vs live-replica labels disagree on {dis.mean():.1%} of days "
          f"(first 7 calendar days of a month: {dis[idx.day <= 7].mean():.1%}; "
          f"rest of month: {dis[idx.day > 7].mean():.1%}).")
    print(f"State-order swaps: {diag['state_order_swaps']} of {diag['refits']} refits "
          f"(0 means the exposure map always hit the intended regime).")
    print("startprob_max_median near 1.0 means each month's decode is forced to start in one state.\n")

    e = {k: exposures(L[k], spy) for k in L}                     # (regime-only, +trend)
    avg_exposure = e["causal"][0].reindex(idx).mean()
    print(f"Average regime-only exposure (live replica): {avg_exposure:.3f}x\n")

    daily, rows = {}, []
    for scen, p in SCENARIOS.items():
        c, f = p["cost_bps"], p["financing"]
        sr = lambda key, which, lag: strategy_returns(e[key][which], bh, c, f, lag=lag)
        v = {
            "Buy & hold 1.0x": bh,
            "Buy & hold, matched leverage (to C)":
                avg_exposure * bh - max(avg_exposure - 1, 0) * f / 252,
            "Regime only | A published (whole-month decode, same-day)": sr("smoothed", 0, False),
            "Regime only | B + 1-day lag": sr("smoothed", 0, True),
            "Regime only | C live replica (month-restart decode)": sr("causal", 0, True),
            "Regime only | D full-history decode": sr("causal_full", 0, True),
            "Regime only | E full-history + vol-relabel": sr("causal_full+relabel", 0, True),
            "Regime+trend | C live replica": sr("causal", 1, True),
            "Regime+trend | E full-history + vol-relabel": sr("causal_full+relabel", 1, True),
        }
        for name, r in v.items():
            daily[(scen, name)] = r.reindex(idx)
        for bname, (a, b) in BLOCKS.items():
            for name, r in v.items():
                rr = r.reindex(idx).loc[a:b].dropna()
                if len(rr) > 20:
                    rows.append({"scenario": scen, "block": bname, "variant": name, **stats(rr)})

    metrics = pd.DataFrame(rows)
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200
    for scen in SCENARIOS:
        for bname in BLOCKS:
            t = metrics[(metrics.scenario == scen) & (metrics.block == bname)]
            if len(t):
                print(f"=== {scen.upper()} | {bname} ===")
                print(t.drop(columns=["scenario", "block"]).set_index("variant").to_string(), "\n")

    os.makedirs(args.out, exist_ok=True)
    metrics.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    pd.DataFrame({f"{s_} | {n}": r for (s_, n), r in daily.items()}).to_csv(
        os.path.join(args.out, "daily_returns.csv"))
    print(f"Wrote {args.out}/metrics.csv and {args.out}/daily_returns.csv")


if __name__ == "__main__":
    main()
