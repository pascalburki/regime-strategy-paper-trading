import os
from alpaca.trading.client import TradingClient

api_key = os.environ.get("ALPACA_KEY")
api_secret = os.environ.get("ALPACA_SECRET")
trading_client = TradingClient(api_key, api_secret, paper=True)

order_id = "15220bcf-3fb2-451b-8c16-0db6ca7d8ffc"
trading_client.cancel_order_by_id(order_id)
print(f"Cancel request sent for order {order_id}")
