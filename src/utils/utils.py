import os
import gensim
import json

def read_configuration_file(configuration_file_path):
    with open(configuration_file_path, 'r') as file:
        configuration = json.load(file)
    return configuration

def create_dirs():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

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