import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layers, output_size, bidirectional, linear_layers):
        super().__init__()        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=number_layers, 
                            bidirectional=bidirectional, batch_first=True)
        linear_layers.insert(0, hidden_size)
        linear_layers = [nn.Linear(linear_layers[i], linear_layers[i+1]) for i in range(len(linear_layers)-1)]
        self.linear_stack = nn.Sequential(*linear_layers)
    
    def forward(self, input):
        x = self.lstm(input)
        output = self.linear_stack(x)
        return output


