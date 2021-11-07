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

def macd(sequence,short_period,long_period):
    # Calculate the mean average convergence divergence indicator
    
    # Calculating EMAshort and EMAlong
    k_short = 2/(short_period+1)
    k_long = 2/(long_period+1)
    EMA_short=[]
    EMA_long=[]
    EMA_short.append(sum(prices[0:short_period])/short_period)
    EMA_long.append(sum(prices[0:long_period])/long_period)
    for t in range(short_period,len(sequence)):
        EMA_short.append(prices[t]*k_short+EMA_short[t-short_period]*(1-k_short))
    for t in range(long_period,len(sequence)):
        EMA_long.append(prices[t]*k_long+EMA_long[t-long_period]*(1-k_long))

    return EMA_long, EMA_short

emalong, emashort = macd(prices,12,26)

plt.plot(prices[:1000],label='Price')
plt.plot(emashort[:1000],label='EMA short')
plt.plot(emalong[:1000],label='EMA long')
plt.legend()
plt.show()

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
    
    