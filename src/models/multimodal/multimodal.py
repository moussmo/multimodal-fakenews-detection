import torch
from torch import nn
from src.models.text_models.lstm import LSTM
from src.models.vision_models.cnn import CNN

class MultimodalModel(nn.Module):
    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self._build_text_model()
        self._build_vision_model()
        self._build_linear_model()

    def _build_text_model(self):
        try : 
            text_model_type = self.configuration['text_model']['model']
        except : 
            raise KeyError("Text model not specified. Add key 'model' to 'text_model'")
        if text_model_type.lower() == "lstm":
            self.text_model = LSTM(input_size=self.configuration['word_embedding_model']['vector_size'], 
                                   hidden_size=self.configuration['text_model']['hidden_size'], 
                                   number_layers=self.configuration['text_model']['number_layers'], 
                                   output_size=self.configuration['text_model']['output_size'], 
                                   bidirectional=self.configuration['text_model']['bidirectional'],
                                   linear_layers=self.configuration['text_model']['linear_layers'])
        else : 
            raise Exception('Text model "{}" not recognized'.format(text_model_type))

    def _build_vision_model(self):
        try : 
            vision_model_type = self.configuration['vision_model']['model']
        except : 
            raise KeyError("Vision model not specified. Add key 'model' to 'vision_model'")
        if vision_model_type.lower() == 'cnn' :
            self.vision_model = CNN(linear_layers=self.configuration['vision_model']['linear_layers'],
                                    conv_layers=self.configuration['vision_model']['conv_layers'], 
                                    kernel_size=self.configuration['vision_model']['kernel_size'], 
                                    stride=self.configuration['vision_model']['stride'],
                                    input_size=self.configuration['vision_model']['input_size'], 
                                    output_size=self.configuration['text_model']['output_size'])
        else : 
            raise Exception('Vision model "{}" not recognized'.format(vision_model_type))


    def _build_linear_model(self):
        layers = self.configuration['linear_model']["layers"]
        input_size = self.configuration['text_model']['output_size'] + self.configuration['vision_model']['output_size']
        output_size = int(self.configuration['target_variable'][0])
        layers.insert(0, input_size)
        layers.append(output_size)
        linear_layers = [nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        for i in range(1, len(linear_layers)+1, 2):
            linear_layers.insert(i, nn.ReLU())
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

