import torchtext
from torchtext.data import Field, Dataset, TabularDataset, Iterator, BPTTIterator
import jieba

with open('./data/train_raw.zh', 'r') as f:
    raw_corpus = f.readlines()

raw_corpus = [i[:-1] for i in raw_corpus]

tokenize = lambda x: [i for i in jieba.cut(x)]
Text = Field(sequential=True, use_vocab=False, init_token=None, eos_token=None, batch_first=True, tokenize=tokenize)

tst =TabularDataset(path='./data/train_raw.zh', format='csv', fields=('Text', Text))


dataset = Dataset(examples=raw_corpus, fields=('Text', Text))
Text.build_vocab(Text)


print(dataset)
