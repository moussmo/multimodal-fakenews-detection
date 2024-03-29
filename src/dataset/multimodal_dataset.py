import os
import torch 
import cv2
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.models.word_embeddings.word2vec import Word2VecEmbedding
import src.utils.utils_text as utils_text
import src.utils.utils_vision as utils_vision

class MultimodalDataset(Dataset):

    def __init__(self, configuration, mode='train'):
        self.configuration = configuration
        data_dir_path = self.configuration['data_dir_path']
        self.dataset = pd.read_csv(os.path.join(data_dir_path, 'multimodal_{}.tsv').format(mode), delimiter='\t')
        self._load_word_embedding_model()

    def __len__(self):
        return self.dataset.shape[0]
    
    def _fetch_image(self, id):
        image_path = os.path.join(self.configuration['data_dir_path'], "images", "{}.jpg".format(id))
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        else : 
            return 1
    
    def _clean_dataset(self):
        #TODO 
        # clean titles, 
        # clean rows with no images
        pass

    def _load_all_datasets(self,):
        data_dir_path = self.configuration['data_dir_path']
        return {'train' : pd.read_csv(os.path.join(data_dir_path, 'multimodal_train.tsv'), delimiter='\t'),
                'validate': pd.read_csv(os.path.join(data_dir_path, 'multimodal_validate.tsv'), delimiter='\t'),
                'test' : pd.read_csv(os.path.join(data_dir_path, 'multimodal_test_public.tsv'), delimiter='\t')}
    
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
                all_datasets = self._load_all_datasets()
                training_text_data = pd.concat([dataset.clean_title for dataset in all_datasets.values()]).reset_index(drop=True)
                self.embedding_model.train(training_text_data)
        else : 
            raise Exception('Word embedding model "{}" not recognized'.format(word_embedding_type))

    def _preprocess_title(self, title):
        title = utils_text.tokenize(title)

        sequence_length = self.configuration['text_model']['sequence_length']
        if len(title) > sequence_length : 
            title = utils_text.remove_stopwords_tokenized(title)
            if len(title) > sequence_length : 
                title = title[:sequence_length]

        title = self.embedding_model.predict_tokenized_text(title)

        if len(title) < sequence_length : 
            title = utils_text.zero_pad(title, sequence_length)

        return np.array(title)
    
    def _preprocess_image(self, image):
        image = cv2.resize(image, self.configuration['vision_model']['input_size'])
        image = utils_vision.min_maxer(image)
        image = image.transpose(2, 0, 1)
        return image
        
    def __getitem__(self, index):
        sample = self.dataset.iloc[index]
        sample_title = sample['clean_title']
        sample_image = self._fetch_image(sample['id'])
        sample_label = sample[self.configuration['target_variable']] 

        if sample_image is None :
            random_other_sample_index = np.random.choice(self.dataset.index, 1) 
            return self[random_other_sample_index]
        else : 
            sample_title_preprocessed = self._preprocess_title(sample_title)
            sample_image_preprocessed = self._preprocess_image(sample_image)
        
            x = (sample_title_preprocessed, sample_image_preprocessed)
            y = sample_label
            
            return x, y 


        

