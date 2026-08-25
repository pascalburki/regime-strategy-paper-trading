import os
from alpaca.trading.client import TradingClient

api_key = os.environ.get("ALPACA_KEY")
api_secret = os.environ.get("ALPACA_SECRET")
trading_client = TradingClient(api_key, api_secret, paper=True)

account = trading_client.get_account()
print(f"Equity: ${float(account.equity):,.2f}")
print(f"Cash: ${float(account.cash):,.2f}")
print(f"Buying power: ${float(account.buying_power):,.2f}")

try:
    position = trading_client.get_open_position("SPY")
    print(f"\nSPY position: {position.qty} shares, ${float(position.market_value):,.2f}")
except Exception as e:
    print(f"\nSPY position check failed: {e}")
