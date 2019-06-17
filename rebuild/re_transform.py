import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='talk')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sequence_mask(raw_input, bidirectional=True):
    """
    raw_input为embedding前的batch_size，形如[[3,2,1,0,0],[6,7,8,5,1]]。其中的1为eos, 0为padding的值，因些可以根据raw_input计算出mask的张量。
    不能仅仅只根据0来判断padding的位置，因为在解码器的输入序列中，开头是以sos=0开始的。
    :param raw_input: batch_size input before embedding.
    :param bidirectional: bool, 当为True时，现在时刻可以利用未来时刻信息，否则不可以利用。
    :return:
    """
    batch_size, max_sequence = list(raw_input.shape)
    # 这里将raw_input中每个样本的第一个index替换成1，以防当句子开头是sos=0时被误mask
    remolded_input = copy.deepcopy(raw_input)
    remolded_input[:, 0] = 1
    mask_vertical = remolded_input.unsqueeze(dim=-1).expand(batch_size, max_sequence, max_sequence)
    mask_vertical = (mask_vertical != 0)
    # 此处仅做替零处理
    mask_vertical = mask_vertical.masked_fill(mask_vertical == 0, 0)
    mask_horizontal = remolded_input.unsqueeze(dim=-2).expand(batch_size, max_sequence, max_sequence)
    mask_horizontal = (mask_horizontal != 0).type_as(torch.tensor(np.inf))
    # 此处仅做替零处理，具体替为-inf在attention中进行。
    mask_horizontal = mask_horizontal.masked_fill(mask_horizontal == 0, 0)
    if not bidirectional:
        mask_unidirectional = torch.tensor([[1]*i + [0]*(max_sequence-i) for i in range(1, max_sequence+1)])
        mask_horizontal.masked_fill_(mask_unidirectional == 0, 0)
    return (mask_horizontal.to(device), mask_vertical.to(device))


def attention(query, key, value, mask=None, dropout=None):
    """
    参考论文<Attention is All You Need>3.2.1,
    :param torch.tensor, query: shape=[batch_size, num_word, word_vec], 若从batch中选择一个样本，则这个样本的形状：
    [word_vec1; word_vec2; ...], 其中每一个word_vec占用一行，列的维度固定为word_vec的维度。因此：第一维是batch的size数，第二维是句子中词的个数，第三维是词向量的维度。
    :param key: shape=[batch_size, num_word, word_vec]
    :param value: shape=[batch_size, num_word, word_vec]
    :param mask: mask可根据embedding前的batch_size input得到，不为0的地方置为1,为0的置为-inf。这里的mask为一个tuple, 第一个用于softmax之前，将padding位置的权重置为-inf；第二个用于softmax之后，将padding位置与非padding位置的点积值置为0。这里为什么要设置两个mask？因为如果在softmax之前，将所有与padding位置相关的权值全部置为-inf, 那么padding位置的权值向量将全为-inf, 即[-inf, -inf, -inf, ...], 这个向量再经过softmax后，将全部变成nan, nan与任何值的运算将仍然为nan. 这将不便于后续计算。
    :return:
    """
    # 获取query中最后一个维度，即词向量的维度，来源于论文第3页倒数第三段最后一行。
    d_k = query.size(-1)
    # 因为是batch数据，所以计算torch.bmm时，要对第二个tensor的后两维进行transpose。
    # torch.bmm之后的维度为[batch_size, num_word, num_word], 单独看后二维的矩阵A为:
    # [<word_vec1, word_vec1>, <word_vec1, word_vec2>, ... <word_vec1, word_vecn>(第一行结束);
    # <word_vec2, word_vec1>, <word_vec2, word_vec2>, ... <word_vec2, word_vecn>(第二行结束);...;
    # <word_vecn, word_vec1>, <word_vecn, word_vec2>, ... <word_vecn, word_vecn>(最后一行结束)]。因此第i行就代表第i个词的query_i
    # 与其它词的key向量进行点积计算相似度。
    scores = torch.bmm(query, key.transpose(dim0=1, dim1=2)) / math.sqrt(d_k)
    if mask is not None:
        # scores = torch.mul(scores, mask[0].type_as(scores))
        scores.masked_fill_(mask[0] == 0, -np.inf)
    scores = F.softmax(scores, dim=-1)
    if mask is not None:
        scores = torch.mul(scores, mask[1].type_as(scores))
    # output的维度为[batch_size, num_word, value最后一维的维度]
    # 同样取一个样本的后两维构成的矩阵B进行分析：第i行为第i个词要计算的value加权和。
    output = torch.bmm(scores, value)
    if dropout is not None:
        output = dropout(output)
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dim_model, dropout=0.1):
        """
        :param h: multi_head_attention的个数， 该参数名取自论文第4页
        :param dim_model: 词向量的维度。
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert dim_model % h == 0
        d_k = dim_model // h  # 该变量名取自论文第4页
        self.h = h
        self.dim_model = dim_model
        # 这里取query, key, value的维度相同。
        # 利用nn.ModuleList创建h个线性映射，nn.ModuleList中各Module之间没有联系，而nn.Sequential中前一个Module的输出是后一个Module的输入
        self.linear_query = nn.ModuleList([copy.deepcopy(nn.Linear(dim_model, d_k)) for _ in range(h)])
        self.linear_key = nn.ModuleList([copy.deepcopy(nn.Linear(dim_model, d_k)) for _ in range(h)])
        self.linear_value = nn.ModuleList([copy.deepcopy(nn.Linear(dim_model, d_k)) for _ in range(h)])
        # 最后一层的输入维度应该是 h*d_k=dim_model
        self.linear_last = nn.Linear(h*d_k, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        query_list, key_list, value_list = [], [], []
        # 将query, key, value分别进行h次线性映射，并将结果进行分别保存。
        for linear in self.linear_query:
            query_list.append(linear(query))
        for linear in self.linear_key:
            key_list.append(linear(key))
        for linear in self.linear_value:
            value_list.append(linear(value))
        # 这里的实现流程参考自论文第4页右上图
        attention_output_list, attention_scores_list = [], []
        for query_i, key_i, value_i in zip(query_list, key_list, value_list):
            attention_output, attention_scores = attention(query_i, key_i, value_i, dropout=self.dropout, mask=mask)
            attention_output_list.append(attention_output)
            attention_scores_list.append(attention_scores)
        # concatenate
        # attention的output维度为[batch_size, num_word, dim_value_vec], 同样将最后两个维度数据提取出来可以得到矩阵C,
        # 矩阵C的每一行代表一组加权value之和。根据论文第5页最上面的公式，可以从MultiHead(Q,K,V)公式推断出：h个head拼接后的最后一维的
        # 大小为h*dim_value_vec.因此，attention_output_list中的各个张量应该沿着value_vec向量的维度方向进行拼接，
        # 即对各张量在最后一维上进行拼接，从而得到维度大于为h*dim_value_vec的结果。
        multi_head_attention_output = torch.cat(attention_output_list, dim=-1)
        # 根据论文第4页的右上图得知：Multi-Head Attention的最后一层有Linear层
        return self.linear_last(multi_head_attention_output)


class PositionWiseFeedForward(nn.Module):
    """
    论文3.3节
    """
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.inner_layer = nn.Linear(dim_model, dim_ff)
        self.output = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        inner_layer_output = self.inner_layer(x)
        # 原文：'This consists of two linear transformations with a ReLU activation in between.'
        inner_layer_relu_output = F.relu(inner_layer_output)
        dropout = self.dropout(inner_layer_relu_output)
        output = self.output(dropout)
        return output


class SubEncoder(nn.Module):
    """
    single encoder block
    """
    def __init__(self, h, dim_model, dim_ff):
        super(SubEncoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(h, dim_model)
        # Normalize over last dimension of size dim_model
        # 这里要实例化两个LayerNorm，不然应该会出现参数共享的情况。
        self.layer_norm = nn.ModuleList([copy.deepcopy(nn.LayerNorm(dim_model)) for _ in range(2)])
        self.ffn = PositionWiseFeedForward(dim_model, dim_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None, dropout=False):
        """
        sub-encoder, the complete encoder block.
        :param x: batch输入数据
        :return:
        """
        # *********sub-layer1************
        multi_head_attention_output = self.multi_head_attention(x, x, x, mask=mask)
        # residual connection
        if dropout:
            multi_head_attention_output = self.dropout(multi_head_attention_output)
        residual_layer = multi_head_attention_output + x
        layer_norm_output = self.layer_norm[0](residual_layer)
        # *********sub-layer2************
        ffn_output = self.ffn(layer_norm_output)
        # residual connection
        residual_layer = ffn_output + layer_norm_output
        layer_norm_output = self.layer_norm[1](residual_layer)
        return layer_norm_output, mask


class SubDecoder(nn.Module):
    def __init__(self, h, dim_model, dim_ff):
        super(SubDecoder, self).__init__()
        # 因为一个子decoder要用到2次MultiHeadAttention, 所以要实例化2次
        self.multi_head_attention = nn.ModuleList([copy.deepcopy(MultiHeadAttention(h, dim_model)) for _ in range(2)])
        # Normalize over last dimension of size dim_model
        self.layer_norm = nn.ModuleList([copy.deepcopy(nn.LayerNorm(dim_model)) for _ in range(3)])
        self.ffn = PositionWiseFeedForward(dim_model, dim_ff)
        self.dropout = nn.Dropout(0.1)

    # Todo: 这里还是有问题，因为没有处理现在会利用将来信息的问题。
    def forward(self, x, encoder_output, mask, dropout=False):
        """
        sub-decoder, the complete decoder block.
        :param x: batch输入数据
        :param encoder_output: encoder输出
        :return:
        """
        # *********sub-layer1************
        # Multi-Head Attention
        multi_head_attention_output = self.multi_head_attention[0](x, x, x, mask)
        # Add, residual connection 1
        if dropout:
            multi_head_attention_output = self.dropout(multi_head_attention_output)
        residual_layer = multi_head_attention_output + x
        # Norm 1
        layer_norm_output = self.layer_norm[0](residual_layer)
        # *********sub-layer2************
        # encoder-decoder attention, 该层不需要mask
        encoder_decoder_attention_output = self.multi_head_attention[1](layer_norm_output, encoder_output, encoder_output)
        # Add, residual connection 2
        residual_layer = encoder_decoder_attention_output + layer_norm_output
        # Norm 2
        layer_norm_output = self.layer_norm[1](residual_layer)
        # *********sub-layer3************
        # feed forward
        ffn_output = self.ffn(layer_norm_output)
        # Add, residual connection 3
        residual_layer = ffn_output + layer_norm_output
        # Norm 3
        layer_norm_output = self.layer_norm[2](residual_layer)
        return layer_norm_output


class Encoder(nn.Module):
    def __init__(self, h, dim_model, dim_ff, num_sub_encoder):
        super(Encoder, self).__init__()
        # 论文第3页中，编码器中有6个子层
        self.encoder = nn.ModuleList([copy.deepcopy(SubEncoder(h, dim_model, dim_ff)) for _ in range(num_sub_encoder)])

    def forward(self, x, mask, dropout=False):
        for sub_encoder in self.encoder:
            x, mask = sub_encoder(x, mask=mask, dropout=dropout)
        return x


class Decoder(nn.Module):
    def __init__(self, h, dim_model, dim_ff, num_sub_decoder):
        super(Decoder, self).__init__()
        # 论文第3页中，解码器中有6个子层
        self.decoder = nn.ModuleList([copy.deepcopy(SubDecoder(h, dim_model, dim_ff)) for _ in range(num_sub_decoder)])

    def forward(self, x, encoder_output, mask, dropout=False):
        for sub_decoder in self.decoder:
            x = sub_decoder(x, encoder_output, mask, dropout=dropout)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len=5000):
        # 长度过长的句子，已经在数据集上去除了。
        # 这里max_seq_len=5000是为了生成一个参考位置编码，这样以后就只需要从这个位置编码中截取所需即可。
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        pe = [[pos / np.power(10000, 2*i/dim_model) for i in range(dim_model)] for pos in range(max_seq_len)]
        pe = np.array(pe)
        # 偶数位置上为sin, 奇数位置上为cos
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        # 这里的pe为单独一个样本的位置编码，因此后续需要根据输入的batch将其调整为batch
        self.pe = pe
        self.dropout = nn.Dropout(0.1)
        # self.position_encoding = pe

    def forward(self, x, x_len, dropout=False):
        """
        compute positional encoding. x_len 为 batch_size 数据，形状为：[batch_size, 1], 行为样本，列为样本长度。
        :param x: batch_size data
        :param x_len: shape=[batch_size, 1]
        :return:
        """
        # 计算出当前batch下最长的序列长度。
        max_seq_len = max(x_len)
        # position_encoding = torch.tensor(self.pe[:max_seq_len], dtype=torch.float32)
        position_encoding = torch.tensor(self.pe[:max_seq_len], dtype=torch.float32).to(device)
        batch_size = x_len.size(0)
        position_encoding = position_encoding.expand([batch_size, max_seq_len, self.dim_model])
        for row_idx, seq_len in enumerate(x_len):
            # 将序列中补零位置的位置编码设置成0
            position_encoding[row_idx, int(seq_len):] = 0
        if not dropout:
            return x + position_encoding
        else:
            return x + self.dropout(position_encoding)

class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, dropout=False, maskable=True):
        """
        将batch_size数据转换成 dim_ model
        :param x: batch_size input, [[idx11, idx12, ...], [idx21, idx22, ...], ...]
        :param dropout: 是否dropout
        :param maskable: 是否将padding通过embedding后的词向量全部置0，这里也不能仅仅只根据是不是0来判断padding的位置
        :return:
        """
        embedding = self.embedding(x)
        if dropout:
            embedding = self.dropout(embedding)
        if maskable:
            # 这里将x中每个样本的第一个index替换成1，以防当句子开头是sos=0时被误mask
            remolded_x = copy.deepcopy(x)
            remolded_x[:, 0] = 1
            # 这里的embedding包含了padding=0的lookup后的embedding向量，因此要将这些embedding向量的值人为全部置为0。这里x!=0，即：x不等于padding的值。之后将x!=0得到的张量在最后一维进行unsqueeze, 从而将2阶张量变为3阶张量。
            mask_matrix = (remolded_x != 0).unsqueeze(dim=-1).type_as(x)
            return embedding.masked_fill(mask_matrix==0, 0)
            # 如果不用masked_fill, 则用以下方式也可达到相同目的。
            # 这里再将mask_matrix expand到与embedding同形的张量。
            # mask_matrix = mask_matrix.expand(list(embedding))
            # # 最后再将mask_matrix与embedding进行对应点相乘，从而达到mask padding embedding值。
            # return torch.mul(embedding, mask_matrix)

        else:
            return embedding


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, h, dim_model, dim_ff, num_sub_encoder, num_sub_decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(h, dim_model, dim_ff, num_sub_encoder)
        self.decoder = Decoder(h, dim_model, dim_ff, num_sub_decoder)
        self.positional_encoding = PositionalEncoding(dim_model)
        self.src_embedding = Embedding(src_vocab_size, dim_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, dim_model)
        # 自定义可训练的bias
        # self.softmax_biases = torch.randn(size=[tgt_vocab_size], requires_grad=True)
        # nn.Parameter是自动在Module注册变量的，其它变量需要注册时使用self.register_buffer()
        self.softmax_biases = nn.Parameter(data=torch.randn(size=[tgt_vocab_size]))
        # self.register_buffer('softmax_biases', self.softmax_biases)

    def forward(self, src, src_size, tgt, tgt_size):
        # 这里的src, 认为已经通过了padding
        # Todo: 暂时不考虑mask
        x = self.src_embedding(src, dropout=True)
        x = self.positional_encoding(x, src_size, dropout=True)
        mask = sequence_mask(src)
        encoder_output = self.encoder(x, mask=mask, dropout=True)
        y = self.tgt_embedding(tgt, dropout=True)
        y = self.positional_encoding(y, tgt_size, dropout=True)
        # Todo: 这里的mask应该要与encoder层不一样，因为decoder要处理现在会利用将来信息的情况。
        mask = sequence_mask(tgt, bidirectional=False)
        y = self.decoder(y, encoder_output, mask=mask, dropout=True)
        # 这里第二个参数自带转置
        output = F.linear(y, self.tgt_embedding.embedding.weight, bias=self.softmax_biases)
        return output

    def predict(self, src, src_size):
        x = self.src_embedding(src)
        x = self.positional_encoding(x, src_size)
        mask = sequence_mask(src, bidirectional=False)
        encoder_output = self.encoder(x, mask=mask)
        tgt = torch.tensor([[0]]).type_as(src)
        tgt_size = torch.tensor([1]).type_as(src_size)
        target = [0]
        while target[-1] != 1:
            y = self.tgt_embedding(tgt)
            y = self.positional_encoding(y, tgt_size)
            # mask = sequence_mask(tgt)
            y = self.decoder(y, encoder_output, mask=None)
            output = F.linear(y, self.tgt_embedding.embedding.weight, bias=self.softmax_biases)
            target = [0] + output.squeeze().tolist()
            tgt = torch.tensor(target).unsqueeze(dim=0).type_as(src)
        return target


# def make_model(vocab_size, h, dim_model, dim_ff, num_sub_encoder, num_sub_decoder):
#     model = EncoderDecoder(vocab_size, h, dim_model, dim_ff, num_sub_encoder, num_sub_decoder)

# model = EncoderDecoder(src_vocab_size, h, dim_model, dim_ff, num_sub_encoder, num_sub_decoder)





















