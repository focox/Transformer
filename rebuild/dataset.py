import torch
import torch.utils.data as Data
# from torchtext import data, datasets


src_path = '../train.en'
tgt_path = '../train.zh'

with open(src_path, 'r') as f:
    src_data = f.readlines()
src_data = [i[:-1] for i in src_data]
src_data = [[int(i) for i in s.split()] for s in src_data]


with open(tgt_path, 'r') as f:
    tgt_data = f.readlines()

tgt_data = [i[:-1] for i in tgt_data]
tgt_data = [[int(i) for i in s.split()] for s in tgt_data]

print()
