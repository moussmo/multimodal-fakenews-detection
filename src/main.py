import sys
sys.path.append('.')

import os
from dataset.multimodal_dataset import MultimodalDataset

if __name__=="__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    data_path = 'data'
    multimodal_dataset = MultimodalDataset(data_path, force_embedding_training=True)
    multimodal_dataset[0]