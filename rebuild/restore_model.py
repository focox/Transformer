from rebuild.re_transform import *
from torchtext.data import Field, TabularDataset, BucketIterator
import torch
import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = '../data/train.csv'
test_data = '../data/test.csv'
dev_data = '../data/dev.csv'

tokenize = lambda x: [int(i) for i in x.split()[:-1]]

src_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_input = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_output = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
data_fields = [('English', src_language), ('Chinese_input', tgt_input), ('Chinese_output', tgt_output)]
test_data = TabularDataset(path=test_data, format='csv', fields=data_fields, skip_header=True)


test_iterator = BucketIterator(dataset=test_data, batch_size=100, repeat=True, shuffle=True)

checkpoint_path = '../model/checkpoint'
model = torch.load(checkpoint_path)

print()

src = '65 2 5 2 270 2 926 2 25 2 14 2 9740 4 2 1'
src = [int(i) for i in src.split()]
src_size = len(src)
src = torch.tensor(src).unsqueeze(dim=0)
src_size = torch.tensor([src_size])

target = '0 47 483 3 2 8 9 10298 5 2 1'

result = model.predict(src, src_size)
print(result)