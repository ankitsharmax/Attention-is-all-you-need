import enum
import os
import sys
import warnings
import re
import math

from nltk.tokenize import word_tokenize
warnings.filterwarnings("ignore")
from dataclasses import dataclass
import re
import nltk
from nltk.stem import WordNetLemmatizer
import string
import torch
import torch.nn as nn
from configs import get_config
import numpy as np


@dataclass
class DataConfigs:
    corpus_path = os.path.abspath("../../data/corpus_text.txt")
    sample_path = os.path.abspath("../../data/sample_text.txt")

class DataPreprocessing:
    def __init__(self):
        self.data_config = DataConfigs()

    def read_data(self,file_path):
        with open(file_path) as file_obj:
            self.data = file_obj.readlines()
            self.data = [word.replace("\n","") for word in self.data]
        return self.data

    def clean_data(self,data):

        # lower words
        self.data = data.lower()
        
        # remove punctuations/special characters
        translator = str.maketrans('','',string.punctuation)
        self.data = self.data.translate(translator)

        # word lemmatization
        lemmatizer = WordNetLemmatizer()
        word_tokens = nltk.word_tokenize(self.data)
        self.data = [lemmatizer.lemmatize(word) for word in word_tokens]
        self.data = " ".join(self.data)

        return self.data

    def create_data_dict(self,data):
        data_dict = {}
        self.data = set(data.split())
        for i,word in enumerate(self.data):
            data_dict[word] = i
        return data_dict

    def vocab_indexing(self,input,data_dict):
        input = input.split()
        for i,word in enumerate(input):
            if word in data_dict:
                input[i] = data_dict[word]
        return input

    def input_embedding(self,vocab_size,d_model,data):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)

        return self.embedding(data) * math.sqrt(self.d_model)

    def positional_encoding(self,d_model,seq_len):
        self.seq_len = seq_len
        self.d_model = d_model
        pe = np.zeros((self.seq_len,self.d_model))
        for pos in range(seq_len):
            for i in range(int(d_model/2)):
                denominator = np.power(10000,2*i/d_model)
                pe[pos,2*i] = np.sin(pos/denominator)
                pe[pos,2*i+1] = np.cos(pos/denominator)

        return pe


if __name__ == '__main__':
    data_obj = DataPreprocessing()
    # corpus data
    corpus_data = data_obj.read_data(data_obj.data_config.corpus_path)
    corpus_clean_data = data_obj.clean_data(*corpus_data)
    data_dict = data_obj.create_data_dict(corpus_clean_data)
    # print(data_dict)

    # sample data
    sample_data = data_obj.read_data(data_obj.data_config.sample_path)
    sample_clean_data = data_obj.clean_data(*sample_data)
    input_x = data_obj.vocab_indexing(sample_clean_data,data_dict)
    input_x = torch.tensor(input_x)
    config = get_config()
    src_embedding = data_obj.input_embedding(config['vocab_size'],config['d_model'],input_x)
    # print(src_embedding)
    print(sample_clean_data)
    print(input_x)
    pos_encodding = data_obj.positional_encoding(config['d_model'],len(input_x))
    print(pos_encodding)

    