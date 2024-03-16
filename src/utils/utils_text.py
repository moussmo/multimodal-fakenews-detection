import gensim

def remove_stopwords(text):
    return gensim.parsing.preprocessing.remove_stopwords(text)

def tokenize(text):
    return gensim.utils.simple_preprocess(text)

def cut_or_pad(tokenized_text, limit):
    if len(tokenized_text)>=limit:
        return tokenized_text[:limit]
    else : 
        #TODO padding
        return tokenized_text