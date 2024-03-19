import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, number_layers, output_size, bidirectional, linear_layers):
        super().__init__()        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=number_layers, 
                            bidirectional=bidirectional, batch_first=True)
        if bidirectional==True:
            linear_layers.insert(0, sequence_length * hidden_size * 2)
        else : 
            linear_layers.insert(0, sequence_length * hidden_size)
        linear_layers.append(output_size)
        linear_layers = [nn.Linear(linear_layers[i], linear_layers[i+1]) for i in range(len(linear_layers)-1)]
        self.linear_stack = nn.Sequential(*linear_layers)
    
    def forward(self, input):
        x, _ = self.lstm(input)
        x = x.reshape((x.shape[0], -1))
        output = self.linear_stack(x)
        return output


