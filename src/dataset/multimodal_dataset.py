import os
import torch 
import cv2
import logging
import pandas as pd
from torch.utils.data import Dataset
from src.models.word_embeddings.word2vec import Word2VecEmbedding


class MultimodalDataset(Dataset):
    TSV_FILES_NAMES = {'train' : "multimodal_train.tsv",
                       "validate" : "multimodal_validate.tsv",
                       "test" : "multimodal_test_public.tsv"}
    IMAGES_SUBPATH = "images"

    ID_COLUMN = 'id'
    TITLE_COLUMN = "clean_title"
    TARGET_COLUMN = "2_way_label"

    SAVED_MODELS_PATH = "saved_models"

    def __init__(self, data_path, word_embedding_type='word2vec', force_embedding_training=False):
        self.data_path = data_path
        self.datasets = {'train' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['train']), delimiter='\t'),
                         'validate': pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['validate']), delimiter='\t'),
                         'test' : pd.read_csv(os.path.join(data_path, self.TSV_FILES_NAMES['test']), delimiter='\t')}
        self.word_embedding_type = word_embedding_type
        self.embedding_model = None
        self.force_embedding_training=force_embedding_training

    def _fetch_image(self, id):
        image_path = os.path.join(self.data_path, self.IMAGES_SUBPATH, "{}.jpg".format(id))
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        else : 
            return None
    
    def _load_word_embedding_model(self):
        if self.word_embedding_type=='word2vec':
            self.embedding_model = Word2VecEmbedding()
            embedding_model_path = os.path.join(self.SAVED_MODELS_PATH, "word2vec.wordvectors")
            if os.path.exists(embedding_model_path) and self.force_embedding_training == False:
                self.embedding_model.load_model()
            else : 
                self.embedding_model.train()

    def _preprocess_title(self, documents, number_words_per_text=15):
        if self.embedding_model is None :
            self._load_word_embedding_model()
        documents = documents.apply(self.embedding_model.remove_stopwords)
        documents = documents.apply(lambda x : self.embedding_model.cut(x, number_words_per_text))
        documents = documents.apply(self.embedding_model.tokenize)
        documents = documents.apply(self.embedding_model.predict_tokenized_text)
        return documents
    
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


        

