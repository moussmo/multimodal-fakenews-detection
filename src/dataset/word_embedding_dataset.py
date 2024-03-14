import os
import torch 
import pandas as pd
from torch.utils.data import Dataset


class WordEmbeddingDataset(Dataset):
    TSV_FILES_NAMES = {'train' : "multimodal_train.tsv",
                       "validate" : "multimodal_validate.tsv",
                       "test" : "multimodal_test_public.tsv"}
    TITLE_COLUMN = "clean_title"
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.datasets = {'train' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['train']), delimiter='\t'),
                         'validate': pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['validate']), delimiter='\t'),
                         'test' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['test']), delimiter='\t')}

