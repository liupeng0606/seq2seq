from Seq2Seq import Seq2SeqModel, Encoder, Decoder
from train_util import get_batch_data
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam
import torch.nn.functional as F
import math
import tqdm
import torch

dev = torch.device("cuda:0")

train_iter, test_iter, TEXT_EN, TEXT_CH = get_batch_data(100)

encoder = Encoder(vocab_size=len(TEXT_EN.vocab), hidden_size=300, embed_dim=300, layer_size=4, drop=0.2)
decoder = Decoder(vocab_size=len(TEXT_CH.vocab), embed_dim=300, hidden_size=300, layer_size=4, drop=0.2)

seq2seq = Seq2SeqModel(encoder, decoder).to(dev)

opt = Adam(seq2seq.parameters(), lr=5e-4)


def train(epochs):
    for epoch in range(epochs):
        total_loss = 0
        pad = TEXT_CH.vocab.stoi['<pad>']
        for b, batch_data in tqdm.tqdm(enumerate(train_iter)):
            src, src_len = batch_data.src
            trg, trg_len = batch_data.trg
            output = seq2seq(src.to(dev), trg.to(dev))
            loss = F.nll_loss(output[1:].view(-1, len(TEXT_CH.vocab)),
                              trg[1:].to(dev).contiguous().view(-1),
                              ignore_index=pad)
            opt.zero_grad()
            loss.backward()
            clip_grad_norm(seq2seq.parameters(), 8.0)
            opt.step()
            total_loss += loss.item()
            if b % 50 == 0 and b != 0:
                total_loss = total_loss / 50
                print("epoch: [%d],[%d][loss:%5.2f][pp:%5.2f]" % (epoch, b, total_loss, math.exp(total_loss)))
                total_loss = 0
        for batch_data in test_iter:
            src, src_len = batch_data.src
            trg, trg_len = batch_data.trg
            y_hat = seq2seq(src.to(dev), trg.to(dev))
            y_hat = torch.argmax(y_hat[1:], dim=-1)
            en = TEXT_EN.reverse(src)
            ch = TEXT_CH.reverse(y_hat)

            for index, item in enumerate(en):
                print(item)
                print(ch[index])
                if index > 1:
                    break
                break



if __name__ == '__main__':
    train(10000)


