"""
Regime Strategy Dashboard — reads strategy_log.csv (committed by the
GitHub Actions workflow) and displays the running paper-trading results.

Deploy for free at https://share.streamlit.io by connecting this repo.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Regime Strategy — Paper Trading", layout="wide")

st.title("Regime Strategy — Live Paper Trading")
st.caption("HMM walk-forward exposure strategy on SPY, running on a real Alpaca paper account.")


@st.cache_data(ttl=300)  # re-read the file every 5 minutes, don't cache forever
def load_log():
    return pd.read_csv("strategy_log.csv", parse_dates=["date"])


try:
    log = load_log()
except FileNotFoundError:
    st.warning("No log file yet — the strategy hasn't run for the first time.")
    st.stop()

if len(log) == 0:
    st.warning("Log file is empty — waiting for the first scheduled run.")
    st.stop()

latest = log.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Current Regime", latest["regime"].capitalize())
col2.metric("Target Exposure", f"{latest['target_exposure']}x")
col3.metric("Account Equity", f"${latest['equity']:,.2f}")

import altair as alt

st.subheader("Equity Over Time")

try:
    shadow_4asset = pd.read_csv("shadow_4asset_log.csv", parse_dates=["date"])
except FileNotFoundError:
    shadow_4asset = None

try:
    shadow_buyhold = pd.read_csv("shadow_buyhold_log.csv", parse_dates=["date"])
except FileNotFoundError:
    shadow_buyhold = None

frames = []
real_clean = log.drop_duplicates(subset="date", keep="last")[["date", "equity"]].copy()
real_clean["Series"] = "5-asset (real, live)"
frames.append(real_clean)

missing = []
if shadow_4asset is not None and len(shadow_4asset) > 0:
    s4 = shadow_4asset.drop_duplicates(subset="date", keep="last")[["date", "equity"]].copy()
    s4["Series"] = "4-asset (shadow, simulated)"
    frames.append(s4)
else:
    missing.append("4-asset shadow")

if shadow_buyhold is not None and len(shadow_buyhold) > 0:
    sbh = shadow_buyhold.drop_duplicates(subset="date", keep="last")[["date", "equity"]].copy()
    sbh["Series"] = "Buy & Hold (shadow, no timing)"
    frames.append(sbh)
else:
    missing.append("buy-and-hold shadow")

chart_df = pd.concat(frames, ignore_index=True)
total_days = chart_df["date"].nunique()

if total_days < 5:
    st.info(
        f"Only {total_days} trading day(s) of data so far — the chart will look sparse until more "
        "history builds up. Each series gets one new point per weekday close; check back in a "
        "couple of weeks for a readable trend."
    )

base = alt.Chart(chart_df).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("equity:Q", title="Equity (USD)", scale=alt.Scale(zero=False), axis=alt.Axis(format="$,.0f")),
    color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="bottom")),
    tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Series:N"), alt.Tooltip("equity:Q", title="Equity", format="$,.2f")],
)
lines = base.mark_line(strokeWidth=2.5)
points = base.mark_circle(size=70)
st.altair_chart((lines + points).properties(height=420).interactive(), use_container_width=True)

if missing:
    st.caption(f"Still waiting on data for: {', '.join(missing)}.")

# ============================================================
# Performance Calculator
# ============================================================
st.subheader("Performance Calculator")
st.caption("Since-inception return for each strategy. All three start from a $100,000 baseline, so this is a true apples-to-apples comparison.")

STARTING_EQUITY = 100000.0

def pct_return(current_equity):
    return (current_equity - STARTING_EQUITY) / STARTING_EQUITY * 100

perf_rows = []

real_latest_equity = real_clean.sort_values("date")["equity"].iloc[-1]
perf_rows.append({
    "Strategy": "5-asset (real, live)",
    "Current Equity": real_latest_equity,
    "Return (%)": pct_return(real_latest_equity),
})

if shadow_4asset is not None and len(shadow_4asset) > 0:
    s4_latest = shadow_4asset.drop_duplicates(subset="date", keep="last").sort_values("date")["equity"].iloc[-1]
    perf_rows.append({
        "Strategy": "4-asset (shadow)",
        "Current Equity": s4_latest,
        "Return (%)": pct_return(s4_latest),
    })

if shadow_buyhold is not None and len(shadow_buyhold) > 0:
    sbh_latest = shadow_buyhold.drop_duplicates(subset="date", keep="last").sort_values("date")["equity"].iloc[-1]
    perf_rows.append({
        "Strategy": "Buy & Hold (shadow)",
        "Current Equity": sbh_latest,
        "Return (%)": pct_return(sbh_latest),
    })

perf_df = pd.DataFrame(perf_rows).sort_values("Return (%)", ascending=False).reset_index(drop=True)
perf_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(perf_df))])

perf_cols = st.columns(len(perf_df))
for col, (_, row) in zip(perf_cols, perf_df.iterrows()):
    col.metric(
        row["Strategy"],
        f"${row['Current Equity']:,.2f}",
        f"{row['Return (%)']:+.3f}%",
    )

with st.expander("Full comparison table"):
    display_df = perf_df.copy()
    display_df["Current Equity"] = display_df["Current Equity"].apply(lambda x: f"${x:,.2f}")
    display_df["Return (%)"] = display_df["Return (%)"].apply(lambda x: f"{x:+.4f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

if len(perf_df) > 1:
    leader = perf_df.iloc[0]
    laggard = perf_df.iloc[-1]
    gap = leader["Return (%)"] - laggard["Return (%)"]
    st.caption(f"Currently leading: **{leader['Strategy']}** by {gap:.3f} percentage points over {laggard['Strategy']}. Early days — this gap will move around a lot until more history builds up.")

st.subheader("Regime History")
st.dataframe(log[["date", "regime", "target_exposure", "action"]].sort_values("date", ascending=False), use_container_width=True)

# Simple drawdown calculation
log["running_max"] = log["equity"].cummax()
log["drawdown"] = (log["equity"] - log["running_max"]) / log["running_max"]
st.subheader("Drawdown")
st.line_chart(log.set_index("date")["drawdown"])

st.caption(f"Last updated: {latest['date']}")
