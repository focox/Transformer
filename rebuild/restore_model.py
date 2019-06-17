from re_transform import *
from torchtext.data import Field, TabularDataset, BucketIterator
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = '../data/train.csv'
test_data = '../data/test.csv'
dev_data = '../data/dev.csv'

tokenize = lambda x: [int(i) for i in x.split()[:-1]]

src_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_input = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_output = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
data_fields = [('English', src_language), ('Chinese_input', tgt_input), ('Chinese_output', tgt_output)]
test_data = TabularDataset(path=test_data, format='csv', fields=data_fields, skip_header=True)


test_iterator = BucketIterator(dataset=test_data, batch_size=1, repeat=False, shuffle=True)

checkpoint_path = '../model/checkpoint'

model = EncoderDecoder(
    src_vocab_size=10000,
    tgt_vocab_size=4950,
    h=8,
    dim_model=512,
    dim_ff=2048,
    num_sub_decoder=6,
    num_sub_encoder=6
).to(device)

model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

print(model)

src = '65 2 5 2 270 2 926 2 25 2 14 2 9740 4 2 1'
src = [int(i) for i in src.split()]
src_size = len(src)
src = torch.tensor(src).unsqueeze(dim=0).to(device)
src_size = torch.tensor([src_size]).to(device)

target = '0 47 483 3 2 8 9 10298 5 2 1'

for i in test_iterator:
    src = i.English[0].to(device)
    src_size = i.English[1].to(device)
    tgt = i.Chinese_input[0]

    result = model.predict(src, src_size)
    print('tgt:', tgt.tolist(), '\nresult:', result, '\n\n\n' + '*'*100)

print(result)