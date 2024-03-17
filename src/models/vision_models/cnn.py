import torch
import torch.nn as nn
import torch.nn.functional as F
import src.utils.utils_vision as utils_vision


class CNN(nn.Module):
    def __init__(self, linear_layers, conv_layers, kernel_size, stride, input_size, output_size):
        super().__init__()   
        conv_layers.insert(0, 3)
        conv_layers = [nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_size=kernel_size, stride=stride, padding="same") 
                       for i in range(len(conv_layers)-1)]
        
        number_of_maxpools = 0
        pooling_kernel_size = (2,2)
        pooling_stride = 2
        for i in range(1, len(conv_layers)+1, 2):
            conv_layers.insert(i, nn.MaxPool2d((2,2),2))
            number_of_maxpools+=1
        self.conv_stack = nn.Sequential(*conv_layers)

        conv_output_H, conv_output_W = utils_vision.calculate_maxpool_output(input_size=input_size, 
                                                                    number_of_maxpools=number_of_maxpools, 
                                                                    pooling_kernel_size=pooling_kernel_size,
                                                                    pooling_stride=pooling_stride)
        
        linear_layers.insert(0, conv_output_H * conv_output_W * conv_layers[-1].out_channels)
        linear_layers.append(output_size)
        linear_layers = [nn.Linear(linear_layers[i], linear_layers[i+1]) for i in range(len(linear_layers)-1)]
        for i in range(1, len(linear_layers)+1, 2):
            linear_layers.insert(i, nn.ReLU())
        self.linear_stack = nn.Sequential(*linear_layers)
         
        
    def forward(self, input):
        x = self.conv_stack(input)
        x = torch.flatten(x)
        output = self.linear_stack(x)
        return output