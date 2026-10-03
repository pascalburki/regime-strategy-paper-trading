# regime-strategy-paper-trading

An HMM (Hidden Markov Model) walk-forward regime-timing strategy, running live
on a real Alpaca paper trading account. Not financial advice, not live with
real money — this is a personal research and learning project.

## What it does

A Gaussian HMM is fit on SPY's daily returns and 20-day rolling volatility to
classify each trading day into one of three regimes:

| Regime | Target exposure |
|---|---|
| Calm | 1.15x |
| Moderate | 1.0x |
| Stress | 0x |

That single SPY-driven signal sets exposure equally across an equal-weighted
5-asset portfolio: **SPY, QQQ, GLD, TLT, XOM**. The model refits monthly on
past data only.

Two overlays can reduce exposure further:

- **Directional trend filter** — "calm" only measures low volatility, not
  direction. If the regime is calm but SPY is below its own 50-day average,
  exposure is capped at 1.0x.
- **Drawdown circuit-breaker** — if account equity is down more than 1% over
  the last 5 trading days, exposure is capped at 1.0x.

## Why 5 assets, and why TLT stays in despite a real drag

The backtest was run once on the full 5-asset portfolio and once with TLT
excluded. TLT is a genuine drag on its own (-0.033 contribution to the total
return), driven by the 2020–2023 bond bear market — a rate-driven move the
SPY-vol-based signal has no way to see coming. But TLT also meaningfully
reduced the portfolio's worst drawdown (-0.265 vs. -0.359 without it). TLT
stays in deliberately, for the diversification benefit — not because the drag
was missed.

## Evidence: what the backtest does and doesn't show

Two sources: `audit_execution_lag.py` (offline, from the committed backtest
CSV) and `evaluate_honest.py` (fresh adjusted prices through 2026-10-02, so
its numbers differ slightly from the CSV). Equal-weight 5 assets, daily
rebalanced, 2015-01 onward. Each row removes one way a backtest can beat live
trading:

- **A, published:** each whole month decoded at once; the signal is applied to
  the *same* day's return.
- **B:** A with a one-day execution lag (live orders go in after the close).
- **C, live replica:** day t is decoded from this month's data through day t
  only, restarting every month — what `get_current_regime()` does — with the lag.
- **D, full-history decode:** day t is decoded from all history through day t
  with the fitted model. Still causal, and consistent with how it was trained.

| | Total return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & hold 1.0x | +265% | 11.7% | 1.00 | -22.2% |
| Buy & hold, matched leverage (1.07x) | +299% | 12.5% | 1.00 | -23.7% |
| Regime only, A published | +331% | 13.3% | 1.18 | -15.6% |
| Regime only, B (+ lag) | +309% | 12.8% | 1.14 | -15.6% |
| Regime only, **C live replica** | +267% | 11.7% | 1.03 | -15.6% |
| Regime only, **D full-history decode** | +281% | 12.1% | 1.05 | -15.6% |
| Regime + trend filter, C live replica | +253% | 11.4% | 1.01 | -15.6% |
| Regime only, D, with costs and financing (assumed 5 bps, 4%/yr) | +261% | 11.6% | 1.01 | -15.6% |

What this says:

1. **Every honest step lowers the result.** From A to C, total return goes
   +331% → +309% → +267% and Sharpe 1.18 → 1.14 → 1.03. The decoding step costs
   more than the timing step: published-style and live-replica labels differ on
   7.5% of days, concentrated early in the month (20.1% in the first 7
   calendar days vs 3.9% for the rest).
2. **The start-state artifact is real and partly fixable.** EM collapses the
   fitted start probabilities to exactly one-hot (median 1.0 across 142
   refits), so decoding from the start of each month forces that month to begin
   in a single state regardless of the market. Decoding the full history (D)
   recovers part of the loss: +267% → +281%, Sharpe 1.03 → 1.05, equal or better
   than C in every period.
3. **Even so, the strategy is roughly buy-and-hold on return and
   risk-adjusted return.** D: +281% vs +265%, Sharpe 1.05 vs 1.00; with costs
   and financing +261% vs +265%, 1.01 vs 1.00. By period (D vs buy-and-hold
   Sharpe): 2015–19 0.87 vs 0.91; 2020–22 0.64 vs 0.63; 2023–26 1.70 vs 1.68.
   Differences this small are within the noise of 11 years of data — even the
   earlier, more favourable lag-only comparison had a bootstrap range of -0.10
   to +0.41 on its Sharpe gain. What persists is a shallower worst drawdown
   (-15.6% vs -22.2%), coming from 2020–22 (-15.6% vs -22.2%) and 2023–26
   (-8.2% vs -10.1%), not 2015–19 (-12.7% vs -11.9%). In the committed backtest
   only 67 of 2,944 days are at 0x exposure, so that benefit rests on a few
   episodes.
4. **The trend filter does not help.** In the lag-only audit its Sharpe
   difference vs regime-only has a bootstrap range of -0.04 to -0.00, and under
   the live replica it scores lower (1.01 vs 1.03). The apparent gain came from
   same-day alignment: the filter compares price to a moving average that
   includes day T's own move.
5. **Relabelling states by volatility makes results worse, not better.** In 28
   of 142 monthly refits the fitted states came out in a different volatility
   order than the exposure map assumes (0 = calm, 2 = stress). Relabelling by
   fitted volatility lowered full-sample Sharpe from 1.05 to 0.93, and all of
   the damage is in 2015–19 (0.87 → 0.53); from 2020 on the results are
   identical. So the swaps are a stability caveat — what each state means can
   shift between refits — not a proven bug. Why the original labels do better
   in 2015–19 has not been investigated; one untested possibility is that state
   2 sometimes captures down-move clusters rather than pure volatility.

## Evaluation protocol

Chronological blocks, used for every comparison: **TRAIN 2015–2019**,
**VALID 2020–2022**, **TEST 2023–now**.

Honest caveat: the rules (3 regimes, the exposure map, the 5 assets, the
trend filter) were chosen with the whole sample in view, so TEST is "later
data, not re-tuned from here on", **not** a clean holdout. The only truly
out-of-sample evidence is the live paper account from 2026-08-24 onward.

Rule for any new signal: design and tune on TRAIN/VALID only, score once on
TEST and on live, and report what it did even if it is disappointing.

## Files

| File | Purpose |
|---|---|
| `run_strategy_multi_asset.py` | The live strategy — submits real orders to the Alpaca paper account |
| `run_shadow_4asset.py` | Comparison tracker simulating the same strategy *without* TLT — no real trades. **Stale config, see Known issues** |
| `run_shadow_buyhold.py` | Comparison tracker simulating plain buy-and-hold on the same 5 assets — no real trades |
| `audit_execution_lag.py` | Offline audit: published vs 1-day-lag results, matched-leverage baseline, bootstrap |
| `evaluate_honest.py` | Full re-evaluation: lag, live-replica and full-history decoding, vol-relabel, baselines, costs, chronological blocks (`--synthetic` for a smoke test) |
| `rebuild_shadow_buyhold_log.py` | One-time repair of `shadow_buyhold_log.csv` from real trading-day returns |
| `backtest_directional_filter.py` | The original backtest (kept for reproducibility; see Evidence for its limits) |
| `dashboard.py` | Streamlit dashboard, reads the log files and renders the equity curves |
| `check_open_orders.py` / `check_account.py` / `cancel_stuck_order.py` | Diagnostics |
| `strategy_log.csv` / `shadow_4asset_log.csv` / `shadow_buyhold_log.csv` | Daily logs, committed automatically by the scheduled workflows |

## Dashboard

`dashboard.py` deploys for free on Streamlit Community Cloud, connected
directly to this repo. It auto-redeploys on every push to `main`. Shows
current regime, target exposure, account equity, the equity curves
overlaid, a regime history table, and a drawdown chart. Note that dollar
levels on the chart start from different bases; compare percentage returns.

## Safety features

Built up over the course of actually running this, after real incidents —
each one exists because something specific went wrong first.

- **DRY_RUN mode** — `DRY_RUN=true python3 run_strategy_multi_asset.py`
  previews every action with zero real orders submitted. Always test here
  before running for real.
- **Concurrency guard** on every GitHub Actions workflow — a manual trigger
  can never run in parallel with the scheduled one. Added after a manual
  test run collided with the scheduled run and bought SPY twice in one day.
- **Duplicate-order protection** — checks for an existing open order on a
  symbol before submitting a new one.
- **Rebalancing band (3%)** — only trades a position if it's drifted more
  than 3% from target, to avoid noise trades.
- **Stop loss (8%)** — closes a position entirely if it's down more than 8%
  from its average entry price. **Honest limitation:** the script runs once a
  day, so this is a once-per-day check, not a true intraday stop.

## Setup

```bash
pip install -r requirements.txt
```

Requires `ALPACA_KEY` and `ALPACA_SECRET` as environment variables, pointing
at an Alpaca **paper trading** account — never live.

## Running it

```bash
# Preview only, no real orders
DRY_RUN=true python3 run_strategy_multi_asset.py

# The real thing -- submits real (paper) orders
python3 run_strategy_multi_asset.py

# Shadow trackers -- never touch Alpaca, safe to run any time
python3 run_shadow_4asset.py
python3 run_shadow_buyhold.py

# Evaluation
python3 audit_execution_lag.py
python3 evaluate_honest.py
```

In production, all three run automatically on weekdays via GitHub Actions
(`.github/workflows/`), committing their own log updates.

## Known issues and what's still open

- **Shadow-tracker logging bug (fixed 2026-10-03).** Both trackers appended a
  row on every run and re-applied the latest return, so weekend, holiday and
  repeat runs were compounded into the benchmark curves (e.g. 2026-09-07,
  Labor Day, repeated Friday's -0.51%). They now log one row per real trading
  day, dated from the price data. Rows before the fix are unreliable; run
  `rebuild_shadow_buyhold_log.py` to repair the buy-and-hold log.
  `shadow_4asset_log.csv` is not repaired and should be treated as invalid.
- **The 4-asset shadow tracker is stale.** It still uses 1.2x calm exposure
  and has none of the live strategy's trend filter, circuit-breaker, band or
  stop loss, so it is no longer a like-for-like comparison. Sync it or retire
  it.
- **Live comparison is too short to mean anything.** Since 2026-08-24 the live
  account is about -2.5%. The benchmark with its two impossible rows removed is
  about -1.5% (-2.2% as originally logged); a full repair needs the rebuild
  script. About five weeks, with the strategy
  levered 1.15–1.2x through a slow decline — consistent with the Evidence
  section, and not evidence for or against the strategy. Compare percentage
  returns from a common start; the strategy's dollar level started higher.
- **Backtest limits:** no trading costs, financing, or slippage; close-to-close
  timing is a proxy for next-open fills; one signal asset (SPY) drives all five.
- **Live strategy still uses month-restart decoding (C).** Full-history
  decoding (D) was equal or better in every period and removes a demonstrated
  artifact, but the live code has not been changed. If it is changed, record
  the date: it ends the clean out-of-sample record of the current logic. Paper
  account only.
- A rate-aware, TLT-specific regime signal is still unbuilt.
- The strategy reduces risk; it does not yet generate alpha. A directional
  signal layered on the regime filter is the next real project, to be
  evaluated under the protocol above.
