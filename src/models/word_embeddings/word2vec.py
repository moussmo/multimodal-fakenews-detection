import os
from gensim.models import Word2Vec, KeyedVectors
from src.models.word_embeddings.word_embedding import WordEmbedding
import src.utils.utils as utils

class Word2VecEmbedding(WordEmbedding):
    SAVED_MODEL_NAME = 'word2vec.wordvectors'

    def __init__(self, configuration):
        self.configuration = configuration

    def load_model(self):
        self.word_vectors = KeyedVectors.load(self.get_model_path(), mmap='r')

    def get_model_path(self):
        return os.path.join(self.configuration['saved_models_dir_path'],'word_embeddings', self.SAVED_MODEL_NAME)
    
    def train(self, text_data):
        vector_size = self.configuration['word_embedding_model']["vector_size"]
        window = self.configuration['word_embedding_model']["window"]
        min_count = self.configuration['word_embedding_model']["min_count"]
        remove_stopwords = self.configuration['word_embedding_model']["remove_stopwords"]

        text_data_preprocessed = text_data.apply(utils.remove_stopwords) if remove_stopwords else text_data
        text_data_preprocessed = text_data.apply(utils.tokenize)

        word2vec = Word2Vec(text_data_preprocessed, vector_size=vector_size, min_count=min_count, window=window)
        self.word_vectors = word2vec.wv
        self.word_vectors.save(self.get_model_path())
    
    def predict(self, word):
        return self.word_vectors[word]
    
    def predict_tokenized_text(self, tokenized_text):
        return[self.predict(word) for word in tokenized_text]
    
