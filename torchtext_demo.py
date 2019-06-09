import torch
import torchtext
from torchtext.data import Field, Example, TabularDataset, BucketIterator
import pandas as pd


tokenize = lambda x: [int(i) for i in x.split()]


src_path = './train.en'
tgt_path = './train.zh'


with open(src_path, 'r') as f:
    src_data = f.readlines()
src_data = [i[:-1] for i in src_data]
src_data = [[int(i) for i in s.split()] for s in src_data]


with open(tgt_path, 'r') as f:
    tgt_data = f.readlines()

tgt_data = [i[:-1] for i in tgt_data]
tgt_data = [[int(i) for i in s.split()] for s in tgt_data]


corpus = {'English': src_data, 'Chinese': tgt_data}
df_corpus = pd.DataFrame(corpus, columns=['English', 'Chinese'])

df_corpus['eng_len'] = df_corpus['English'].apply(lambda x: len(x))
df_corpus['zh_len'] = df_corpus['Chinese'].apply(lambda x: len(x))

df_corpus = df_corpus.query('eng_len < 80 & zh_len < 80')
df_corpus = df_corpus.query('eng_len > 5 & zh_len > 5')


src_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0)
tgt_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0)

