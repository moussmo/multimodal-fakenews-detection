import os
import torch 
import cv2
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.models.word_embeddings.word2vec import Word2VecEmbedding
import src.utils.utils_text as utils_text

class MultimodalDataset(Dataset):

    def __init__(self, configuration):
        self.configuration = configuration
        data_dir_path = self.configuration['data_dir_path']
        self.datasets = {'train' : pd.read_csv(os.path.join(data_dir_path, 'multimodal_train.tsv'), delimiter='\t'),
                         'validate': pd.read_csv(os.path.join(data_dir_path, 'multimodal_validate.tsv'), delimiter='\t'),
                         'test' : pd.read_csv(os.path.join(data_dir_path, 'multimodal_test_public.tsv'), delimiter='\t')}
        self._load_word_embedding_model()

    def _fetch_image(self, id):
        image_path = os.path.join(self.configuration['data_dir_path'], "images", "{}.jpg".format(id))
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        else : 
            return 1
    
    def _load_word_embedding_model(self):
        try : 
            word_embedding_type = self.configuration['word_embedding_model']['model']
        except : 
            raise KeyError("Word embedding model not specified. Add key 'model' to 'word_embedding_model'")
        
        if word_embedding_type=='word2vec':
            self.embedding_model = Word2VecEmbedding(self.configuration)
            embedding_model_path = self.embedding_model.get_model_path()
            force_retraining = self.configuration['word_embedding_model']['force_retraining']
            if os.path.exists(embedding_model_path) and  force_retraining == False:
                self.embedding_model.load_model()
            else : 
                training_text_data = pd.concat([dataset.clean_title for dataset in self.datasets.values()]).reset_index(drop=True)
                self.embedding_model.train(training_text_data)
        else : 
            raise Exception('Word embedding model "{}" not recognized'.format(word_embedding_type))

    def _preprocess_title(self, title):
        title = utils_text.remove_stopwords(title)
        title = utils_text.tokenize(title)
        title = utils_text.cut_or_pad(title, self.configuration['text_model']['sequence_length'])
        title = self.embedding_model.predict_tokenized_text(title)
        return np.array(title)
    
    def _preprocess_image(self, image):
        #TODO
        return 1
        
    def __getitem__(self, index, type='train'):
        sample = self.datasets[type].iloc[index]
        sample_title = sample['clean_title']
        sample_image = self._fetch_image(sample['id'])
        sample_label = sample[self.configuration['target_variable']] 

        if sample_image == None : 
            return (None, None)
        else : 
            sample_title_preprocessed = self._preprocess_title(sample_title)
            sample_image_preprocessed = self._preprocess_image(sample_image)
        
            x = (sample_title_preprocessed, sample_image_preprocessed)
            y = sample_label
            
            return x, y


        

