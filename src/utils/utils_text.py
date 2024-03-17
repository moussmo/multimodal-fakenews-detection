import gensim

def remove_stopwords(text):
    return gensim.parsing.preprocessing.remove_stopwords(text)

def tokenize(text):
    return gensim.utils.simple_preprocess(text)

def cut_or_pad(tokenized_text, sequence_length):
    if len(tokenized_text)>=sequence_length:
        return tokenized_text[:sequence_length]
    else : 
        #TODO padding
        return tokenized_text