# Utilities, support functions
#from decouple import config
from binance.client import Client
from function import *
import numpy as np
import matplotlib.pyplot as plt

def read_data():
    print('Under construction')

def plot_data():
    print('Under construction')

#api_key = config('binance_key')
#api_secret = config('binance_secret')

#client = Client(api_key, api_secret)
#client = Client("", "")
# btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
# print(btc_price)




def dati():
    symbol = "ETHUSDT"
    start = "1 Oct, 2021"
    end = ""
    client = Client("", "")
    interval = Client.KLINE_INTERVAL_30MINUTE #permette di cambiare l'intervallo
    klines = client.get_historical_klines(symbol, interval,  start)
    klines=np.array(klines)
    klines=np.delete(klines,[2,3,5,6,7,8,9,10,11],1)
    # Change data in every element of the first column
    for i in klines:#sarebbe da vettorizzare
        i[0]=milliseconds_to_date(int(i[0]))
    #klines =(i[0]=milliseconds_to_date(i[0]) for i in klines)

    return klines



#klines=(klines[1:],'float64')
#klines=klines[0]

def suddividi(klines):
    start_prices = klines[:,1]
    close_prices = klines[:,2]
    #mid_prices = (start_prices+close_prices)/2.0
    return start_prices,close_prices
#save_datafile(klines,symbol, interval,  start)

def plotta(klines):
    x=klines[:,0]
    y=klines[:,1].astype(float)

    plt.figure(figsize = (18,9))
    plt.plot(x,y) 
    #plt.plot(range(klines.shape[0]),(klines[1]+klines[2])/2.0)
    #plt.xticks(range(0,klines.shape[0],500),klines[0].loc[::500],rotation=45)
    #plt.title(symbol) 
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

def plotta_singolo(value):
    plt.figure(figsize = (18,9))
    plt.plot(value)
    plt.show()







