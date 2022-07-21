import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


class PositionEncodeing(nn.Module):
    """Positional encoding"""
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super(PositionEncodeing, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a Long enough
        self.P = torch.zeros(1, max_len, num_hiddens)
        X = torch.arange(max_len, dtype= torch.float32).reshape(-1,1) \
            / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype= torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionwiseFNN(nn.Module):
    """Positionwise feed-forward network"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,**kwargs):
        super(PositionwiseFNN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))


class AddNorm(nn.Module):
    """residual connection followed by layer normalization"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)

# module

class EncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias = False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionwiseFNN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, x, valid_lens):
        Y = self.addnorm1(x, self.attention(x,x,x, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    """transformer encoder"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias = False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_enconding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+ str(i),
            EncoderBlock(key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_enconding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention_weights
        return X


class DecoderBlock(nn.Module): # problem!!!!!
    """the i -th block in de decoder"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, i , **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionwiseFNN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed at the same time,
        # so state[2][sekf.i] is "None" as initialized.
        # When decoding any output sequence token by token during prediction
        # state[2][self.i] contains representations of the decoded output at the i _th block up to
        # the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis = 1) # accumulate key-value pairs
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # shape of dec_valid_lens:(batch_size, num_steps), where every row is [1,2,..., num_steps]   : for masked
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # self-attention
        x2 = self.attention1(X,key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, x2)
        # Encoder_decoder attention. Shape of 'enc_outputs':
        # ('batcg_size', num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3((Z, self.ffn(Z)), state)


class TransformerDecoder(d2l.AttentionDecoder):
    def __int__(self,vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, i , **kwargs):
        super(TransformerDecoder, self).__int__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), DecoderBlock(key_size, query_size,
                                value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enco_outputs, enc_valid_lens, *args):
        return [enco_outputs, enc_valid_lens, [None] * self.num_layers]  # state from each layer is none firstly

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attntion_weights = [[None] * len(self.blks) for _ in range(2)] # drawing
        for i, blk in enumerate(X, state):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attntion_weights[0][i] = blk.attention1.attention.attention_weights
            #Encoder-decider attention weights
            self._attntion_weights[1][i] = blk.attention2.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attntion_weights


# training
num_hiddens , num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads
    , num_layers, dropout
)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
    dropout
)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
