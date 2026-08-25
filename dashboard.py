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

st.subheader("Equity Over Time")

try:
    shadow_4asset = pd.read_csv("shadow_4asset_log.csv", parse_dates=["date"])
except FileNotFoundError:
    shadow_4asset = None

try:
    shadow_buyhold = pd.read_csv("shadow_buyhold_log.csv", parse_dates=["date"])
except FileNotFoundError:
    shadow_buyhold = None

series = {"5-asset (real, live)": log.set_index("date")["equity"]}
missing = []

if shadow_4asset is not None and len(shadow_4asset) > 0:
    series["4-asset (shadow, simulated)"] = shadow_4asset.set_index("date")["equity"]
else:
    missing.append("4-asset shadow")

if shadow_buyhold is not None and len(shadow_buyhold) > 0:
    series["Buy & Hold (shadow, no timing)"] = shadow_buyhold.set_index("date")["equity"]
else:
    missing.append("buy-and-hold shadow")

st.line_chart(pd.DataFrame(series))
if missing:
    st.caption(f"Still waiting on data for: {', '.join(missing)}.")

st.subheader("Regime History")
st.dataframe(log[["date", "regime", "target_exposure", "action"]].sort_values("date", ascending=False), use_container_width=True)

# Simple drawdown calculation
log["running_max"] = log["equity"].cummax()
log["drawdown"] = (log["equity"] - log["running_max"]) / log["running_max"]
st.subheader("Drawdown")
st.line_chart(log.set_index("date")["drawdown"])

st.caption(f"Last updated: {latest['date']}")
