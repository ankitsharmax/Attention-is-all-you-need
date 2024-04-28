import os
import sys
import warnings
import re

from nltk import text, translate
from nltk.tokenize import word_tokenize
warnings.filterwarnings("ignore")
from dataclasses import dataclass
import re
import nltk
from nltk.stem import WordNetLemmatizer
import string


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
        pass

        
    
if __name__ == '__main__':
    data_obj = DataPreprocessing()
    data = data_obj.read_data(data_obj.data_config.corpus_path)
    data_obj.clean_data(*data)