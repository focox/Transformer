from re_transform import *
import torch
import torchtext
from torchtext.data import Field, Example, TabularDataset, BucketIterator
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = '../data/train.csv'
test_data = '../data/test.csv'
dev_data = '../data/dev.csv'



tokenize = lambda x: [int(i) for i in x.split()[:-1]]

src_language = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_input = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
tgt_output = Field(sequential=True, use_vocab=False, tokenize=tokenize, include_lengths=True, pad_token=0, batch_first=True)
data_fields = [('English', src_language), ('Chinese_input', tgt_input), ('Chinese_output', tgt_output)]
train_set = TabularDataset(path=train_data, format='csv', fields=data_fields, skip_header=True)


train_iterator = BucketIterator(dataset=train_set, batch_size=100, repeat=True, shuffle=True)


model = EncoderDecoder(
    src_vocab_size=10003,
    tgt_vocab_size=20003,
    h=8,
    dim_model=512,
    dim_ff=2048,
    num_sub_decoder=6,
    num_sub_encoder=6
)

print(model)

EPOCHS = 5
d_model = 512
warmup_steps = 4000
tgt_vocab_size = 20003
steps = 1
PATH = '../model/checkpoint'

model = model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=0)
criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.9, 0.98], eps=1e-9)


# lr_lambda = lambda steps: (1/d_model)**0.5 * min((1/steps)**0.5, steps*(1/4000)**1.5)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# lr_lambda = lambda steps: (1/d_model)**0.5 * min((1/steps)**0.5, steps*(1/warmup_steps)**1.5)
# lr_lambda = lambda steps: 1/d_model
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(EPOCHS):
    for i in train_iterator:
        # print(i)
        output = model(i.English[0].cuda(), i.English[1].cuda(), i.Chinese_input[0].cuda(), i.Chinese_input[1].cuda())
        output = output.view(-1, tgt_vocab_size).cuda()
        # print(output.shape)
        label = i.Chinese_output[0].view(-1, 1).squeeze(dim=-1).cuda()
        # label_len = i.Chinese_output[1]

        loss = criterion(input=output, target=label).cuda()
        print(steps, '==>>', loss)

        optimizer.zero_grad()
        loss.backward()
        # scheduler.step()
        optimizer.step()

        steps += 1
        if steps % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, PATH)

