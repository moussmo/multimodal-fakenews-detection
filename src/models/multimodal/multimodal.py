import torch

class MultimodalModel(torch.nn.Module):
    
    def __init__(self, configuration):
        self.configuration = configuration
        self.text_model = self._build_text_model()
        self.vision_model = self._build_vision_model()

    def _build_text_model(self):
        text_model_type = self.configuration['text_model_type']
        pass

    def _build_vision_model():
        pass

    def forward(self, input):
        pass