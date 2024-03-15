import os
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors
from word_embeddings.word_embedding import WordEmbedding

class Word2VecEmbedding(WordEmbedding):
    SAVED_MODEL_NAME = 'word2vec.wordvectors'

    def __init__(self):
        super.__init__()

    def load_model(self):
        self.word_vectors = KeyedVectors.load(self.SAVED_MODEL_PATH)

    def train(self, text_data, vector_size=300, window=2, remove_stopwords=True):
        text_data_preprocessed = text_data.apply(self.remove_stopwords) if remove_stopwords else text_data
        text_data_preprocessed = text_data.apply(self.tokenize)
        word2vec = Word2Vec(text_data_preprocessed, vector_size=vector_size, window=window)
        self.word_vectors = word2vec.wv
        self.word_vectors.save(os.path.join(self.SAVED_MODELS_PATH, self.SAVED_MODEL_NAME))
    
    def predict(self, word):
        return self.word_vectors[word]
    
    def predict_tokenized_text(self, tokenized_text):
        return[self.predict(word) for word in simple_preprocess(tokenized_text)]
    
