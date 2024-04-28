import os
import sys
import warnings
import re
warnings.filterwarnings("ignore")
from dataclasses import dataclass

@dataclass
class DataPreparationConfig:
    corpus_text = os.path.join('data',"corpus_tex.txt")

class DataPreparation:
    def __init__(self):
        self.data_config = DataPreparationConfig()

    def read_data(self,file_path:str):
        with open(file_path) as file_obj:
            self.data = file_obj.readlines()
            

    
if __name__ == '__main__':
    data_obj = DataPreparation()