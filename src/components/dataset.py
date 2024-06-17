import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
  def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
    super().__init__()

    self.dataset = dataset
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.seq_leng = seq_len

    self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype = torch.int64)
    self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype = torch.int64)
    self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype = torch.int64)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index: Any) -> Any:
    src_target_pair = self.dataset[index]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.tgt_lang]

    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

    # add padding token to fill the sentence until it reaches the seq_len (not all sentence will be of same size)
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # leaving 2 spaces for SOS and EOS
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # leaving 1 spaces for SOS

    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError('Sentence is too long')

    # Add SOS, EOS and PAD (if needed) to the source text
    encoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ]
    )

    # Add SOS to the decoder input
    decoder_input = torch.cat(
        [
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ]
    )

    # ADD EOS to the label (what we expect as output from the decoder)
    label = torch.cat(
        [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtypes=torch.int64)
        ]
    )

    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
        "encoder_input" : encoder_input, # (size = seq_len)
        "decoder_input" : decoder_input, # (size = seq_len)
        # encoder_mask remove all pad_token which we don't want the trainer to see.
        "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len)
        # .unsqueeze(0): This operation adds a dimension of size 1 at the specified position (index 0 in this case) of the tensor.
        # .unsqueeze(0) (again): This adds another dimension of size 1 at index 0. So, after this operation, the tensor would have a shape of (1, 1, N), where N is the original size of encoder_input.
        "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1,seq_len, seq_len)
        "label" : label, # (seq_len)
        "src_text" : src_text,
        "tgt_text" : tgt_text
    }


def causal_mask(size):
  # torch.triu is a function provided by PyTorch that computes the upper triangular part of a matrix (2-D tensor)
  # torch.triu(input, diagonal=0, *, out=None)
  mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int) # every value above the diagonal multiply by 1 and everythings else by 0
  '''
     [[1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]]

      [[0, 2, 3],
       [0, 0, 6],
       [0, 0, 0]]
  '''
  return mask == 0 # this will convert all the 0 values as true and return

