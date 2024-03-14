import os
import torch 
import cv2
import pandas as pd
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    TSV_FILES_NAMES = {'train' : "multimodal_train.tsv",
                       "validate" : "multimodal_validate.tsv",
                       "test" : "multimodal_test_public.tsv"}
    IMAGES_SUBPATH = "images"

    ID_COLUMN = 'id'
    TITLE_COLUMN = "clean_title"
    TARGET_COLUMN = "2_way_label"

    def __init__(self, data_path):
        self.data_path = data_path
        self.datasets = {'train' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['train']), delimiter='\t'),
                         'validate': pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['validate']), delimiter='\t'),
                         'test' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['test']), delimiter='\t')}


    def _fetch_image(self, id):
        image_path = os.path.join(self.data_path, self.IMAGES_SUBPATH, "{}.jpg".format(id))
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        else : 
            return None
        
    def _preprocess_title():
        return 1
    
    def _preprocess_image():
        return 1
        
    def __getitem__(self, index, type='train'):
        sample = self.datasets[type].iloc[index]
        sample_image = self._fetch_image(sample[self.ID_COLUMN]) 
        
        if sample_image == None : 
            return (None, None)
        else : 
            sample_title_preprocessed = self._preprocess_title(sample[self.TITLE_COLUMN])
            sample_image_preprocessed = self._preprocess_image(sample_image)

            x = (sample_title_preprocessed, sample_image_preprocessed)
            y = sample[self.TARGET_COLUMN]
            # to_tensor
            return (x, y)


        

