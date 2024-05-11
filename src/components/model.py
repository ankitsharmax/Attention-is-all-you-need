import torch
import torch.nn as nn
import math

from torch.nn.modules import linear

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model) for the positional encoding
        pe = torch.zeros(seq_len,d_model)
        for pos in range(seq_len):
            for i in range(int(d_model/2)):
                denominator = math.pow(10000.0,2*i/d_model)
                pe[pos,2*i] = math.sin(pos/denominator)
                pe[pos,2*i+1] = math.cos(pos/denominator)

        self.register_buffer('pe',pe) # storing the value of pe in buffer to use or store later if needed

    def forward(self,x):
        # adding positional embedding to every word in the sentence and setting the gradient to False to not learn the pe as
        # they always be the same
        x = x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        # applying dropout to the sums of the embeddings and the positional encodings
        return self.dropout(x)

class LayerNormilization(nn.Module):
    # check the formula of layer normalization 
    # y_i = alpha * (x_i - meu) / (sqrt(sigma^2 + epsilon) + beta) 
    # here meu is for mean and sigma^2 is for variance, alpha and beta are learnable parameters
    # if sigma happens to be zero or close to 0 then numerator will become very big. And the cpu/gpu will not be able to handl
    # also this is to avoid division by zero as well
    def __init__(self,epsilon:float = 10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim=-1,keepdims=True) # -1 here refers to the last dimension
        std = x.std(dim=-1,keepdims=True)
        return self.alpha * (x - mean) / math.sqrt(self.std + self.epsilon) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int, dff:int, dropout:float):
        super().__init__()
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        # 1st linear layer xW1 + b1
        # relu activation
        # 2nd liner layer (relu(1st liner layer))W2 + b2
        self.linear_1 = nn.Linear(d_model,dff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff,d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



