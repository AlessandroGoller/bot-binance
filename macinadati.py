# Machine learning algos for buying and selling

from torch import nn


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
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    