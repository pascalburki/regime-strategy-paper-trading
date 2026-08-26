# regime-strategy-paper-trading

An HMM (Hidden Markov Model) walk-forward regime-timing strategy, running live
on a real Alpaca paper trading account. Not financial advice, not live with
real money — this is a personal research and learning project.

## What it does

A Gaussian HMM is fit on SPY's daily returns and 20-day rolling volatility to
classify each trading day into one of three regimes:

| Regime | Target exposure |
|---|---|
| Calm | 1.2x |
| Moderate | 1.0x |
| Stress | 0x |

That single SPY-driven signal sets exposure equally across an equal-weighted
5-asset portfolio: **SPY, QQQ, GLD, TLT, XOM**. The model refits on a
walk-forward basis (monthly), never seeing future data — the same approach
used in the historical backtest that validated this idea before it went live
(0.57 vs. 0.44 return, 0.84 vs. 0.69 Sharpe, vs. a plain buy-and-hold
baseline over the same period).

## Why 5 assets, and why TLT stays in despite a real drag

The backtest was run once on the full 5-asset portfolio and once with TLT
excluded. TLT is a genuine drag on its own (-0.033 contribution to the total
+0.57 return), driven by the 2020–2023 bond bear market — a rate-driven move
the SPY-vol-based signal has no way to see coming. But TLT also meaningfully
reduced the portfolio's worst drawdown (-0.265 vs. -0.359 without it). TLT
stays in deliberately, for that diversification benefit and for consistency
with the originally validated structure — not because the drag was missed.

## Files

| File | Purpose |
|---|---|
| `run_strategy_multi_asset.py` | The live strategy — submits real orders to the Alpaca paper account |
| `run_shadow_4asset.py` | Comparison tracker simulating the same strategy *without* TLT — no real trades |
| `run_shadow_buyhold.py` | Comparison tracker simulating plain buy-and-hold on the same 5 assets, no regime timing — no real trades |
| `dashboard.py` | Streamlit dashboard, reads the three log files and renders all three equity curves |
| `check_open_orders.py` | Diagnostic — lists any currently open/pending orders |
| `check_account.py` | Diagnostic — prints current equity, cash, buying power, and SPY position |
| `cancel_stuck_order.py` | Diagnostic — cancels a specific order by ID, for when an order needs manual clearing |
| `strategy_log.csv` / `shadow_4asset_log.csv` / `shadow_buyhold_log.csv` | Daily logs, committed automatically by the scheduled workflows |

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
  symbol before submitting a new one, so a slow-settling order can't get
  stacked on top of itself.
- **Rebalancing band (3%)** — only trades a position if it's drifted more
  than 3% from target. Without this, the script chased every tiny daily
  price move with real orders (a real observed trade: "buy $23.08 notional
  SPY") — noise, not a meaningful rebalance.
- **Stop loss (8%)** — closes a position entirely if it's down more than 8%
  from its average entry price, regardless of what the regime signal says.
  **Honest limitation:** this script runs once a day via the scheduled
  workflow, not continuously — so this is a once-per-day check, not a true
  intraday stop. It won't catch and react to a flash crash before the next
  scheduled run; that would need a process running 24/7, a different
  architecture than a daily cron job.

## Setup

```bash
pip install alpaca-py yfinance numpy pandas hmmlearn streamlit altair --break-system-packages
```

Requires `ALPACA_KEY` and `ALPACA_SECRET` as environment variables, pointing
at an Alpaca **paper trading** account — never live.

```bash
export ALPACA_KEY="your paper key"
export ALPACA_SECRET="your paper secret"
```

## Running it

```bash
# Preview only, no real orders
DRY_RUN=true python3 run_strategy_multi_asset.py

# The real thing -- submits real (paper) orders
python3 run_strategy_multi_asset.py

# Shadow trackers -- never touch Alpaca, safe to run any time
python3 run_shadow_4asset.py
python3 run_shadow_buyhold.py
```

In production, all three run automatically on the same weekday schedule via
GitHub Actions (`.github/workflows/`), committing their own log updates.
Manual runs are for testing/dry-run only — the scheduled workflow is the
source of truth.

## Dashboard

`dashboard.py` deploys for free on Streamlit Community Cloud, connected
directly to this repo. It auto-redeploys on every push to `main`. Shows
current regime, target exposure, account equity, all three equity curves
overlaid, a regime history table, and a drawdown chart.

## What's genuinely still open

- A rate-aware, TLT-specific regime signal — the real fix for the TLT drag,
  not yet built or backtested. Not something to add without validating it
  first, same standard as everything else here.
- True intraday risk protection would require a continuously-running
  process rather than a daily scheduled script — a bigger architectural
  change, not a quick addition.
