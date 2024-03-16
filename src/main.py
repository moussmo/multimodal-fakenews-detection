import sys
sys.path.append('.')

import os
import utils.utils as utils
from dataset.multimodal_dataset import MultimodalDataset

if __name__=="__main__":
    utils.create_dirs()
    configuration_path = "src/config/default_configuration.json"
    configuration = utils.read_configuration_file(configuration_path)
    multimodal_dataset = MultimodalDataset(configuration)
    multimodal_dataset[0]

    