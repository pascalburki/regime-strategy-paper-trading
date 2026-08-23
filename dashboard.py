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
st.line_chart(log.set_index("date")["equity"])

st.subheader("Regime History")
st.dataframe(log[["date", "regime", "target_exposure", "action"]].sort_values("date", ascending=False), use_container_width=True)

# Simple drawdown calculation
log["running_max"] = log["equity"].cummax()
log["drawdown"] = (log["equity"] - log["running_max"]) / log["running_max"]
st.subheader("Drawdown")
st.line_chart(log.set_index("date")["drawdown"])

st.caption(f"Last updated: {latest['date']}")
