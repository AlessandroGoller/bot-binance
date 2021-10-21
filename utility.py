# Utilities, support functions
from decouple import config
from binance.client import Client

def read_data():
    print('Under construction')

def plot_data():
    print('Under construction')


api_key = config('binance_key')
api_secret = config('binance_secret')

#client = Client(api_key, api_secret)
client = Client("", "")
# btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
# print(btc_price)