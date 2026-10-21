# regime-strategy-paper-trading

This is a regime-timing strategy based on a Hidden Markov Model (HMM). It runs every weekday on a real Alpaca paper trading account, so no real money is involved. It is a personal research and learning project, not financial advice.

## What it does

A Gaussian HMM is fit on SPY's daily returns and its 20-day rolling volatility. It sorts each trading day into one of three regimes:

| Regime | Target exposure |
|---|---|
| Calm | 1.15x |
| Moderate | 1.0x |
| Stress | 0x |

That one SPY signal sets the exposure for an equal-weight portfolio of five assets: SPY, QQQ, GLD, TLT and XOM. The model is refit every month using past data only.

Two extra rules can lower the exposure:

- **Trend filter.** "Calm" only means low volatility, not that prices are going up. If the regime is calm but SPY is below its 50-day average, exposure is capped at 1.0x.
- **Drawdown circuit breaker.** If the account is down more than 1% over the last 5 trading days, exposure is capped at 1.0x.

## Why 5 assets, and why TLT stays in

I ran the backtest with all five assets and again without TLT. On its own, TLT hurt the total return (a contribution of -0.033), mostly because of the 2020-2023 bond bear market. That was a rate-driven move, and a signal based on SPY volatility has no way to see it coming. But TLT also made the worst drawdown much smaller: -0.265 with it, -0.359 without it. I kept TLT on purpose for that protection, not because I missed the drag.

## Evidence: what the backtest shows and what it doesn't

There are two sources. `audit_execution_lag.py` works offline from the saved backtest CSV. `evaluate_honest.py` uses fresh adjusted prices up to 2026-10-02, so its numbers are slightly different from the CSV. Both use the five assets with equal weights, rebalanced daily, from January 2015 on. Each row below removes one way a backtest can look better than live trading:

- **A, published:** each month is decoded all at once, and the signal is applied to the same day's return.
- **B:** A, plus a one-day delay before trading (live orders go in after the close).
- **C, live replica:** day t is decoded only from this month's data up to day t, starting over every month, plus the delay. This is what `get_current_regime()` does live.
- **D, full-history decode:** day t is decoded from all data up to day t with the fitted model. It still only uses the past, and it matches how the model was trained.

| | Total return | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy and hold 1.0x | +265% | 11.7% | 1.00 | -22.2% |
| Buy and hold, same leverage (1.07x) | +299% | 12.5% | 1.00 | -23.7% |
| Regime only, A published | +331% | 13.3% | 1.18 | -15.6% |
| Regime only, B (with delay) | +309% | 12.8% | 1.14 | -15.6% |
| Regime only, C live replica | +267% | 11.7% | 1.03 | -15.6% |
| Regime only, D full-history decode | +281% | 12.1% | 1.05 | -15.6% |
| Regime + trend filter, C live replica | +253% | 11.4% | 1.01 | -15.6% |
| Regime only, D, with costs and financing (5 bps, 4% a year) | +261% | 11.6% | 1.01 | -15.6% |

What this shows:

- **Every honest step lowers the result.** From A to C, the total return goes from +331% to +309% to +267%, and the Sharpe ratio from 1.18 to 1.14 to 1.03. The decoding step costs more than the delay. The published labels and the live labels are different on 7.5% of days, mostly early in the month (20.1% of the first 7 days, 3.9% of the rest).
- **The start-of-month problem is real, and decoding the full history fixes part of it.** The fitting method pushes the starting probabilities to exactly one state (the median is 1.0 across 142 refits). So decoding each month from its first day forces the month to start in one state, whatever the market is doing. Decoding the full history (D) gets some of the loss back: +267% becomes +281%, and the Sharpe ratio goes from 1.03 to 1.05. D is equal to or better than C in every period.
- **Even so, the strategy is about the same as buy and hold on return and Sharpe.** D: +281% against +265%, Sharpe 1.05 against 1.00. With costs and financing: +261% against +265%, and 1.01 against 1.00. By period (Sharpe of D against buy and hold): 2015-19 0.87 vs 0.91, 2020-22 0.64 vs 0.63, 2023-26 1.70 vs 1.68. Differences this small are within the noise of 11 years of data. Even the earlier comparison, which looked better, had a bootstrap range of -0.10 to +0.41 for the Sharpe gain. What does hold up is a smaller worst drawdown (-15.6% against -22.2%). It comes from 2020-22 (-15.6% vs -22.2%) and 2023-26 (-8.2% vs -10.1%), not from 2015-19 (-12.7% vs -11.9%). Only 67 of 2,944 days in the backtest are at 0x exposure, so that benefit depends on a few episodes.
- **The trend filter does not help.** In the delay test, its Sharpe difference against the regime-only version has a bootstrap range of -0.04 to -0.00. In the live replica it scores lower (1.01 vs 1.03). The gain it seemed to have came from timing: the filter compared the price to a moving average that already included that same day's move.
- **Relabelling the states by volatility makes the results worse.** In 28 of 142 monthly refits, the states came out in a different volatility order than the exposure rules expect (0 = calm, 2 = stress). Relabelling them by volatility lowered the full-sample Sharpe from 1.05 to 0.93, and all of the damage is in 2015-19 (0.87 to 0.53). From 2020 on, the results are the same. So the swaps mean that what each state stands for can change between refits. That is a stability warning, not a proven bug. I haven't looked into why the original labels do better in 2015-19. One idea I haven't tested is that state 2 sometimes picks up groups of down days instead of pure volatility.

## How I evaluate

I split the data by time and use the same blocks for every comparison: TRAIN 2015-2019, VALID 2020-2022, TEST 2023 to now.

To be honest about it: I chose the rules (3 regimes, the exposure levels, the 5 assets, the trend filter) while looking at the whole sample. So TEST is "later data that I don't tune on anymore", not a clean holdout. The only real out-of-sample evidence is the live paper account since 2026-08-24.

My rule for any new signal: design and tune it on TRAIN and VALID only, score it once on TEST and live, and report what it did even if the result is disappointing.

## Files

| File | What it does |
|---|---|
| `run_strategy_multi_asset.py` | The live strategy. Sends real orders to the Alpaca paper account |
| `run_shadow_4asset.py` | Tracker that simulates the same strategy without TLT. No real trades. Its settings are out of date (see Known issues) |
| `run_shadow_buyhold.py` | Tracker that simulates plain buy and hold on the same 5 assets. No real trades |
| `audit_execution_lag.py` | Offline audit: published vs one-day-delay results, same-leverage baseline, bootstrap |
| `evaluate_honest.py` | Full re-evaluation: delay, live-replica and full-history decoding, relabelling, baselines, costs, time blocks (`--synthetic` for a quick test) |
| `rebuild_shadow_buyhold_log.py` | One-time repair of `shadow_buyhold_log.csv` from real trading-day returns |
| `backtest_directional_filter.py` | The original backtest, kept so it can be reproduced (see Evidence for its limits) |
| `dashboard.py` | Streamlit dashboard that reads the logs and draws the equity curves |
| `check_open_orders.py` / `check_account.py` / `cancel_stuck_order.py` | Diagnostics |
| `strategy_log.csv` / `shadow_4asset_log.csv` / `shadow_buyhold_log.csv` | Daily logs, saved automatically by the scheduled workflows |

## Dashboard

`dashboard.py` runs for free on Streamlit Community Cloud and is connected to this repo, so it updates every time I push to main. It shows the current regime, the target exposure, the account value, the equity curves, a regime history table and a drawdown chart. The dollar values on the chart start from different amounts, so compare the percentage returns.

## Safety features

I added each of these after something actually went wrong while running it.

- **DRY_RUN mode.** `DRY_RUN=true python3 run_strategy_multi_asset.py` shows every action without sending any orders. I always test here first.
- **No overlapping runs.** A manual run can never run at the same time as the scheduled one. I added this after a manual test overlapped with the scheduled run and bought SPY twice in one day.
- **Duplicate order check.** Before sending an order, it checks whether there is already an open order for that symbol.
- **Rebalancing band (3%).** A position is only traded if it has moved more than 3% away from its target, to avoid tiny trades.
- **Stop loss (8%).** A position is closed if it is down more than 8% from its average entry price. The script only runs once a day, so this is a daily check, not a real intraday stop.

## Setup

```
pip install -r requirements.txt
```

You need `ALPACA_KEY` and `ALPACA_SECRET` as environment variables, pointing at an Alpaca paper trading account. Never a live account.

## Running it

```
# Preview only, no real orders
DRY_RUN=true python3 run_strategy_multi_asset.py

# The real thing (sends real paper orders)
python3 run_strategy_multi_asset.py

# Shadow trackers (never touch Alpaca, safe to run any time)
python3 run_shadow_4asset.py
python3 run_shadow_buyhold.py

# Evaluation
python3 audit_execution_lag.py
python3 evaluate_honest.py
```

In production, all three run automatically on weekdays through GitHub Actions (`.github/workflows/`) and save their own log updates.

## Known issues and what's still open

- **Logging bug in the shadow trackers (fixed 2026-10-03).** Both trackers added a row every time they ran and applied the latest return again, so weekends, holidays and repeat runs got counted extra in the benchmark curves. For example, on 2026-09-07 (Labor Day) Friday's -0.51% was counted again. Now they log one row per real trading day. Rows from before the fix are unreliable. Run `rebuild_shadow_buyhold_log.py` to repair the buy-and-hold log. `shadow_4asset_log.csv` is not repaired and should be treated as invalid.
- **The 4-asset shadow tracker is out of date.** It still uses 1.2x in calm regimes and has none of the live strategy's trend filter, circuit breaker, band or stop loss, so it is no longer a fair comparison. I need to update it or remove it.
- **The live comparison is too short to mean anything.** Since 2026-08-24 the live account is down about 2.5%. The benchmark, with its two impossible rows removed, is down about 1.5% (2.2% as first logged), and a full repair needs the rebuild script. That is about five weeks, with the strategy at 1.15-1.2x through a slow decline. It matches the Evidence section, and it is not evidence for or against the strategy. Compare percentage returns from the same start date, because the strategy's dollar value started higher.
- **Backtest limits:** no trading costs, financing or slippage. Close-to-close timing stands in for next-morning fills. One signal asset (SPY) drives all five.
- **The live strategy still decodes month by month (C).** Full-history decoding (D) was equal or better in every period and removes a known problem, but I haven't changed the live code yet. If I change it, I'll record the date, because it ends the clean out-of-sample record of the current version. Paper account only.
- I still haven't built a regime signal for TLT that takes interest rates into account.
- **The strategy lowers risk, but it does not create alpha yet.** My next real project is a directional signal on top of the regime filter, tested with the rules above.

## How I worked

I chose the project, the questions and what to test, ran the code and checked the results. I used an AI tutor (Claude) to help me learn the methods and to debug the code.
