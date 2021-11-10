# Machine learning algos for buying and selling
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import os 
import csv

datapath = r'C:\Users\Riccardo\Desktop\TRADING BOT\bot-binance\ETH'

prices = []

for file in os.listdir(datapath):
    if(file.endswith('.csv')):
        # Read file 
        with open(os.path.join(datapath,file), mode ='r') as csvFile:
            data = csv.reader(csvFile)
            for line in data:
                pieces = line[0].split('\t')
                if(pieces[2]!='CLOSE'):
                    prices.append(float(pieces[2]))
                


# FUNCTIONS FOR CONVENTIONAL (NON AI) ANALYSIS

def macd_indicator(sequence,short_period=12,long_period=26,intermediate_period=9):
    # Calculate the mean average convergence divergence indicator
    # Calculating EMAshort and EMAlong
    k_short = 2/(short_period+1)
    k_long = 2/(long_period+1)
    EMA_short=[]
    EMA_long=[]
    EMA_short.append(sum(sequence[0:short_period])/short_period)
    EMA_long.append(sum(sequence[0:long_period])/long_period)
    for t in range(short_period,len(sequence)):
        EMA_short.append(sequence[t]*k_short+EMA_short[t-short_period]*(1-k_short))
    for t in range(long_period,len(sequence)):
        EMA_long.append(sequence[t]*k_long+EMA_long[t-long_period]*(1-k_long))
    
    # Inserting zeros at the beginning of the series 
    for i in range(long_period-1):
        EMA_long.insert(0,0)
    for i in range(short_period-1):
        EMA_short.insert(0,0)
    
    diff = [i-j for i,j in zip(EMA_short, EMA_long)]

    EMA_diff = []
    EMA_diff.append(sum(diff[0:intermediate_period])/intermediate_period)
    k_diff = 2/(intermediate_period+1)
    for t in range(intermediate_period,len(diff)):
        EMA_diff.append(diff[t]*k_diff+EMA_diff[t-intermediate_period]*(1-k_diff))
    
    for i in range(intermediate_period-1):
        EMA_diff.insert(0,0)

    macd = [i-j for i,j in zip(diff, EMA_diff)]

    return macd

def rsi_indicator(sequence, lookback_window=14):
    # Implement the relative strength index for a give sequence
    print('Under construction')
    shifted_sequence = [i for i in sequence]
    sequence = sequence[1:]
    difference = [i-j for i, j in zip(sequence,shifted_sequence)]
    # print(sequence[0:2])
    # print(shifted_sequence[0:2])
    # print(difference[0:2])
    positive_gain = 0
    negative_gain = 0 
    for i in range(lookback_window):
        if(difference[i]>0):
            positive_gain += difference[i]*100/shifted_sequence[i]
        else:
            negative_gain += abs(difference[i])*100/shifted_sequence[i]
    rsi = []
    rsi.append(100-(100/(1+((positive_gain/lookback_window))/(negative_gain/lookback_window))))
    
    # for i in range(lookback_window,len(sequence)):
    #     print('Calculate RSI step two')

rsi_values = rsi_indicator(prices) 
# figure = plt.figure()
# ax = figure.subplots()
# ax.plot(prices[50:200],label='Price')
# figure2 = plt.figure()
# ax2 = figure2.subplots()
# ax2.plot(macd_values[50:200],label='macd')
# plt.legend()
# plt.show()

# FUNCTIONS FOR NN ANALYSIS

def create_data(price_now, price_past):
    # Creates a target based on the price now and the past price difference
    # 1 mean buy
    # 0 mean sell
    if(price_now-price_past>0):
        return 1
    else:
        return 0

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Network, self).__init__()
        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    