import torch
from torch import nn

class MultimodalModel(nn.Module):
    
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.text_model = self._build_text_model()
        self.vision_model = self._build_vision_model()
        self.linear_model = self._build_linear_model()

    def _build_text_model(self):
        text_model_type = self.configuration['text_model_type']
        return 1

    def _build_vision_model(self):
        return 1

    def _build_linear_model(self):
        layers = self.configuration['linear_model']["layers"]
        input_size = self.configuration['text_model']['output_size'] + self.configuration['vision_model']['output_size']
        output_size = int(self.configuration['target_variable'][0])
        layers.insert(0, input_size)
        layers.insert(-1, output_size)
        linear_layers = [nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        for i in range(0,len(linear_layers), 2):
            linear_layers.insert(i, nn.ReLu())
        linear_stack = nn.Sequential(*linear_layers)
        return linear_stack

    def forward(self, input):
        x1, x2 = input
        x1 = self.text_model(x1)
        x2 = self.vision_model(x2)
        x3 = torch.concat([x1, x2])
        x3 = self.linear_model(x3)
        output = nn.Softmax(x3)
        return output

