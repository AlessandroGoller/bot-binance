# Utilities, support functions
#from decouple import config
from binance.client import Client
from function import *
import numpy as np

def read_data():
    print('Under construction')

def plot_data():
    print('Under construction')

#api_key = config('binance_key')
#api_secret = config('binance_secret')

#client = Client(api_key, api_secret)
client = Client("", "")
# btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
# print(btc_price)


symbol = "ETHUSDT"
start = "21 Oct, 2021"
end = ""

client = Client("", "")

interval = Client.KLINE_INTERVAL_30MINUTE #permette di cambiare l'intervallo

klines = client.get_historical_klines(symbol, interval,  start)
klines=np.array(klines)
klines=np.delete(klines,[2,3,5,6,7,8,9,10,11],1)

save_datafile(klines,symbol, interval,  start)





print("ciao")








