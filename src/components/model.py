from numpy import mat
import torch
import torch.nn as nn
import math

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

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormilization()

    def forward(self,x,sublayer):
        return x+self.dropout(self.norm(sublayer(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)

    @staticmethod
    def attention(self,query,key,value,mask,dropout:nn.Dropout):
        d_k = query.shape[-1] #last dimension of query, key and value

        # @ is for matrix multiplication in pytorch
        # Q * K^T
        # K^T (Batch, h, d_k, seq_len)
        # (Batch, h, seq_len, d_k) -> (Batch,h,seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch,h, seq_len,seq_len) <- (seq_len * d_k) * (d_k * seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (Batch,h,seq_len,seq_len) * (Batch,h,seq_len,d_k) -> (Batch,h,seq_len,d_k)
        return (attention_scores @ value), attention_scores


    def forward(self,q,k,v,mask):
        # mask is to hide the attentions of some scores (words in our case)
        # softmax(Q.K^T/sqrt(d_k)) this gives us an attention score

        # (Batch,seq_len,d_model) -> (Batch, seq_len,d_model)
        query = self.w_q(q) #q_prime = q * w_q
        key = self.w_k(k) #k_prime = k * w_k
        value = self.w_v(v) #v_prime = v * w_v

        # divide the query, key and value to smaller matrix to give to each head
        # (Batch,seq_len,d_model) -> (Batch,seq_len,h,d_k) -> (Batch,h,seq_len,d_k)
        # each h (head) to have (seq_len,d_k)
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        # calculate the attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        # (Batch,h,seq_len,d_k) -> (Batch,seq_len,h,d_k) -> (Batch,seq_len,seq_len)
        x = x.transpose(1,2) # (Batch,h,seq_len,d_k) -> (Batch,seq_len,h,d_k)
        x = x.contiguous().view(x.shape[0],-1,self.h*self.d_k) # (Batch,seq_len,seq_len)

        # (Batch,seq_len,seq_len) -> (Batch,seq_len,d_model)
        return self.w_o(x)
        
class EncoderBlock(nn.Module):
    def __init__(self,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

        def forward(self,x,src_mask):
            # we have two residual connections
            # 1st take the input x and the output of multi head attention and Add and Norm
            # 2nd take the output of 1st residual connection and input of feed forward block and Add ad Norm
            x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
            x = self.feed_forward_block[1](x, self.feed_forward_block)
            return x

# According to the paper there are many N EncoderBlocks
class Encoder(nn.Module):
    def __init__(self,layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormilization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                feed_forward_block: FeedForwardBlock,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # src_mask for encoder
        # tgt_mask for decoder
        # why src_mask and tgt_mask, we are dealing with translation task, so we have a source language and a target language
        # decoder self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        # encoder key, value pair to the cross_attention_block with src_mask
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# According to the paper there are many N DecoderBlocks
class Decoder(nn.Module):
    def __init__(self,layers: nn.Module) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        
class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len,d_model) --> (Batch, seq_len, vocab_size)
        # apply log softmax for numerical stability (something to learn)
        return torch.log_softmax(self.proj(x),dim = -1) # to the last dimension

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return  self.decoder(tgt,encoder_output,src_mask, tgt_mask)

    def project(self,x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder_block = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters (to make the training faster so it doesn't initialize randomly)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer




