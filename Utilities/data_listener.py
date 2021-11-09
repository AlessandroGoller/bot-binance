from binance.client import Client
from datetime import datetime
import time
import sys
import os

# Forever
def listen_data(symbols,sleep_factor=60):
    print('Listening..')
    # symbols : list of symbols for which we want data: es [BTCUSDT,ETHUSDT..]
    # sleep_factor : time in seconds between two measures 
    # Initialize variables 
    path_to_data = os.path.join(os.path.dirname(os.getcwd()),'data')
    client = Client("", "")
    # Forever
    while True:
        print('Read!')
        for symbol in symbols:
            # Path to symbol data file
            path_symbol = os.path.join(path_to_data,symbol+'_'+str(sleep_factor)+'.txt')

            # Get price and timestamp 
            timestamp = client.get_server_time() 
            timestamp = timestamp['serverTime']/1000
            timestamp = round(timestamp,0)
            date = datetime.fromtimestamp(timestamp) 
            price = client.get_symbol_ticker(symbol=symbol)
            newPrice = price["price"]

            # Write data to file
            with open(path_symbol,'a') as f:
                f.write(str(date)+' '+str(newPrice)+'\n')

        # Wait 
        time.sleep(sleep_factor)

if __name__ == "__main__":
    n = len(sys.argv[0])
    a = sys.argv[1][1:n]
    a = a.split(',')
    symbols = []
    for piece in a:
        symbols.append(piece)

    sleep_factor = int(sys.argv[2])
    
    listen_data(symbols,sleep_factor)