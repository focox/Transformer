import torch
import torchtext
from torchtext.data import Field, Example, TabularDataset, BucketIterator
import pandas as pd

train_data = './data/train.csv'
test_data = './data/test.csv'
dev_data = './data/dev.csv'

tokenize = lambda x: [int(i) for i in x.split()]
src_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
data_fields = [('English', src_language), ('Chinese', tgt_language)]

# Todo: method 1
train_set = TabularDataset(path='./data/train.csv', format='csv', fields=data_fields, skip_header=True)
# Todo: method 2
train_set = torchtext.datasets.TranslationDataset(path='./data/train.', exts=('en', 'zh'), fields=data_fields)

train_iterator = BucketIterator(dataset=train_set, batch_size=100, repeat=True, shuffle=True)
for i in train_iterator:
    # 这里的i就是一个batch数据，包含长度。
    break

