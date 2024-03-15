import os
import gensim

class WordEmbedding():
    SAVED_MODELS_PATH = os.path.join('saved_models', 'word_embeddings')

    def __init__():
        pass

    def remove_stopwords(self, text):
        return gensim.parsing.preprocessing.remove_stopwords(text)
    
    def tokenize(self, text):
        return gensim.utils.simple_preprocess(text)
    
    def cut_or_pad(self, tokenized_text, limit):
        if len(tokenized_text)>=limit:
            return tokenized_text[:limit]
        else : 
            #TODO padding
            return tokenized_text
    