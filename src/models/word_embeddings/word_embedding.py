import os
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec, KeyedVectors
from gensim.parsing.preprocesing import remove_stopwords

class WordEmbedding():
    SAVED_MODELS_PATH = os.path.join('..', '..', 'saved_models')

    def __init__():
        pass

    def remove_stopwords(self, text):
        return remove_stopwords(text)
    
    def tokenize(self, text):
        return simple_preprocess(text)
    