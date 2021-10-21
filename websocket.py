#https://livedataframe.com/live-cryptocurrency-data-python-tutorial/

#The Binance WebSocket requires us to only send a command once to open up a stream, and then data will automatically stream over as prices get updated.

import time
from binance.client import Client # Import the Binance Client
from binance.websockets import BinanceSocketManager # Import the Binance Socket Manager

client = Client("", "")

# Instantiate a BinanceSocketManager, passing in the client that you instantiated
bm = BinanceSocketManager(client)

# This is our callback function. 

def handle_message(msg):
    
    # If the message is an error, print the error
    if msg['e'] == 'error':    
        print(msg['m'])
    
    # If the message is a trade: print time, symbol, price, and quantity
    else:
        print("Time: {} Symbol: {} Price: {} Quantity: {} ".format(msg['T'],
                                                                   msg['s'],
                                                                   msg['p'],
                                                                   msg['q']))



# Start trade socket with 'ETHBTC' and use handle_message to.. handle the message.
conn_key = bm.start_trade_socket('ETHBTC', handle_message)

print('Start')
# then start the socket manager
bm.start()

# let some data flow..
time.sleep(1)
print('wait')

# stop the socket manager
bm.stop_socket(conn_key)
print('End')



