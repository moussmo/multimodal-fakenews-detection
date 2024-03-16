import sys
sys.path.append('.')

import os
import utils.utils as utils
from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal.multimodal import MultimodalModel

if __name__=="__main__":
    configuration_path = r"src/config/default_configuration.json"
    configuration = utils.read_configuration_file(configuration_path)

    utils.setup_directories(configuration)
    #utils.check_data_directory(configuration)

    multimodal_dataset = MultimodalDataset(configuration)

    multimodal_model = MultimodalModel(configuration)

    