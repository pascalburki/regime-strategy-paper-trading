import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

api_key = os.environ.get("ALPACA_KEY")
api_secret = os.environ.get("ALPACA_SECRET")
trading_client = TradingClient(api_key, api_secret, paper=True)

request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
orders = trading_client.get_orders(filter=request)

if not orders:
    print("No open orders.")
else:
    print(f"{len(orders)} open order(s):\n")
    for o in orders:
        print(f"Symbol: {o.symbol}")
        print(f"  Side: {o.side.value}")
        print(f"  Qty: {o.qty}, Notional: {o.notional}")
        print(f"  Status: {o.status.value}")
        print(f"  Submitted: {o.submitted_at}")
        print(f"  Order ID: {o.id}")
        print()
