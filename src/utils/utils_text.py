import gensim
import numpy as np

def remove_stopwords(text):
    return gensim.parsing.preprocessing.remove_stopwords(text)

def tokenize(text):
    return list(gensim.utils.tokenize(text))

def zero_pad(input, sequence_length):
    input.extend([np.zeros(input[0].shape) for i in range(len(input), sequence_length)])
    return input