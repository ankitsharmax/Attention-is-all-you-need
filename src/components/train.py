import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer #the class that will train the tokenizer, create the vocabolory given the list of sentences
from tokenizers.pre_tokenizers import Whitespace #split the word according to the whitespace

from pathlib import Path

def get_all_sentences(dataset, language):
  for item in dataset:
    yield item['translation'][language]

def get_or_build_tokenizer(config,dataset, language):
  # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.josn'
  tokenizer_path = Path(config['tokenizer_file'].format(language))
  # Create path if not exist
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # if the word is not recognized by the tokenizer in its vocablary, it replace by the unknown token
    tokenizer.pre_tokenizers = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2) # min_fequency = 2 for a word to appear in our vocablary it has to have a frequency of at least 2
    tokenizer.train_from_iterator(get_all_sentences(dataset,language),trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer


def get_dataset(config):
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
  # Build tokenizers
  tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

  # keep 90% for training and 10% for validation
  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds , val_ds = random_split(ds_raw,[train_ds_size,val_ds_size])
