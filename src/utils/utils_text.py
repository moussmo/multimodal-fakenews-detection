import gensim
import numpy as np

def tokenize(text):
    return list(gensim.utils.tokenize(text))

def detokenize(tokenized_text):
    return ' '.join(tokenized_text)

def remove_stopwords(text):
    return gensim.parsing.preprocessing.remove_stopwords(text)

def remove_stopwords_tokenized(tokenized_text):
    return tokenize(remove_stopwords(detokenize(tokenized_text)))

def zero_pad(input, sequence_length):
    input.extend([np.zeros(input[0].shape) for i in range(len(input), sequence_length)])
    return input