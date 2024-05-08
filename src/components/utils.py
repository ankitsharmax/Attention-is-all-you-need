import torch
import torch.nn as nn

d_model = 5 #512 in paper

inp = torch.tensor([0, 39, 24, 35, 10, 1, 59])
vocab_size = len(inp)
vec = nn.Embedding(10000,d_model)
print(vec(inp))