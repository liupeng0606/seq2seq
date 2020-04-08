import torch
from torch import nn
import math
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embed_dim, layer_size, drop=0.5, embed=None):
        super(Encoder, self).__init__()
        self.input_size = vocab_size
        self.embed_dim = embed_dim
        self.layer_size = layer_size
        self.hidden_size = hidden_size
        self.drop = drop
        if embed==None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = embed
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_size,
                          dropout=drop, num_layers=layer_size, bidirectional=True)
    def forward(self, x):
        embed = self.embed(x)
        out, hidden = self.gru(embed)
        out = out[:, :, :self.hidden_size]+out[:, :, self.hidden_size:]
        return out, hidden                              # out [T, B, H]    hidden [L*2, B, H]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.zeros(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
    def forward(self, encoder_outputs, last_hidden):
        # expect encoder_outputs shape [T, B, H],  last_hidden [B, H]
        # torch.tensor()
        encoder_outputs = encoder_outputs.transpose(0, 1)                                          # [B, T, H]
        step = encoder_outputs.size(1)
        batch = encoder_outputs.size(0)
        last_hidden = last_hidden.unsqueeze(1).repeat(1, step, 1)                                  # [B, T, H]
        en = nn.functional.relu(self.linear(torch.cat([encoder_outputs, last_hidden], dim=-1)))    # [B, T, H]
        en = en.transpose(1, 2)                                                                    # [B, H, T]
        v = self.v.unsqueeze(0).repeat(batch, 1).unsqueeze(1)                                      # [B, 1, H]
        attn_weights = self.score(en, v)
        return attn_weights                                                                        # [B, T]
    def score(self, en, v):
        weights = v.bmm(en).squeeze(1)                                                             # [B, T]
        weights = nn.functional.softmax(weights, dim=-1)
        return weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, layer_size, drop=0.5, embed=None):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.drop = drop
        self.embed_dim = embed_dim
        self.attention = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size * 2, vocab_size)
        if embed==None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = embed
        self.gru = nn.GRU(input_size=hidden_size + embed_dim, hidden_size=hidden_size, num_layers=layer_size, dropout=drop)
    def forward(self, input, encoder_outs, last_hidden):
        # expect input shape [1, B], last_hidden [L, B, H], encoder_outs [T, B, H]
        # torch.tensor()
        embed = self.embed(input)                                                 # [1, B, E_D]
        att_weights = self.attention(encoder_outs, last_hidden[-1])                   # [B, T]
        context = att_weights.unsqueeze(1).\
            bmm(encoder_outs.transpose(0, 1)).transpose(0, 1)                     # [1, B, H]
        gru_input = torch.cat([embed, context], dim=-1)
        out, hidden = self.gru(gru_input, last_hidden)
        out_context = torch.cat([out, context], dim=-1)
        y_hat = self.linear(out_context).squeeze(0)                               # y_hat [B, vocab_size]
        output = nn.functional.log_softmax(y_hat, dim=1)
        return output, hidden, att_weights                                         # out [1, B, H], hidden [L, B, H]


dev = torch.device("cuda:0")

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg):
        encoder_out, encoder_hidden = self.encoder(src)
        batch_size = trg.size(1)
        max_step = trg.size(0)
        decoder_input = trg[:1]             # [1, B]
        last_hidden = encoder_hidden[:self.decoder.layer_size]
        outputs = Variable(torch.zeros(max_step, batch_size, self.decoder.vocab_size)).to(dev)
        for step in range(max_step):
            out, last_hidden, att_weights = self.decoder(decoder_input, encoder_out, last_hidden)
            decoder_input = trg[step:step+1]
            outputs[step] = out
        return outputs                                                      # [T, B, vocab_size]


# encoder = Encoder(vocab_size=1000, hidden_size=300, embed_dim=50, layer_size=5)
# decoder = Decoder(vocab_size=2000, embed_dim=100, hidden_size=300, layer_size=3)
#
# model = Seq2SeqModel(encoder, decoder)
#
# src = torch.LongTensor(torch.randint(10, 1000, [16, 100]))
# trg = torch.LongTensor(torch.randint(10, 1000, [32, 100]))
#
# r = model(src, trg)
#
# print(r.shape)