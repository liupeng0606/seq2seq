from torchtext import data
from torchtext.data import Iterator, BucketIterator
import pandas as pd
import random


TEXT_EN = data.ReversibleField(sequential=True, use_vocab=True,
                     fix_length=None, unk_token="<unk>",
                               lower=True, include_lengths=True, pad_token="<pad>")

TEXT_CH = data.ReversibleField(sequential=True, use_vocab=True,
                     fix_length=None, eos_token="<eos>", unk_token="<unk>", init_token="<bos>",
                               include_lengths=True, pad_token="<pad>")

ch = pd.read_csv("./chinese.txt", header=None, sep="\t\t\t\t").values
en = pd.read_csv("./english.txt", header=None, sep="\t\t\t\t").values

en_ch = list(zip(en, ch))

random.shuffle(en_ch)

train_data = en_ch[:int(len(en_ch)*0.999)]
test_data = en_ch[int(len(en_ch)*0.999):]


def get_dataset(list_data):
    fields = [("src", TEXT_EN), ("trg", TEXT_CH)]
    examples = []
    for src, trg in list_data:
        examples.append(data.Example.fromlist([src[0], trg[0]], fields))
    return examples, fields



def get_batch_data(batch_size):
    examples_trn, fields = get_dataset(train_data)
    trn = data.Dataset(examples_trn, fields)
    examples_tst, fields = get_dataset(test_data)
    tst = data.Dataset(examples_tst, fields)
    train_iter = BucketIterator(trn, batch_size=batch_size)
    test_iter = BucketIterator(tst, batch_size=batch_size)
    TEXT_EN.build_vocab(trn, min_freq=2)
    TEXT_CH.build_vocab(trn, max_size=10000)
    return train_iter, test_iter, TEXT_EN, TEXT_CH

#
# train_iter, test_iter, TEXT_EN, TEXT_CH = get_batch_data(10)
#
# for item in train_iter:
#     src, src_len = item.src
#     trg, trg_len = item.trg
#     r = TEXT_CH.reverse(trg)
#     print(r)