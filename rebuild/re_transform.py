import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='talk')


def attention(query, key, value, mask=None, dropout=None):
    """
    参考论文<Attention is All You Need>3.2.1,
    :param torch.tensor, query: shape=[batch_size, num_word, word_vec], 若从batch中选择一个样本，则这个样本的形状：
    [word_vec1; word_vec2; ...], 其中每一个word_vec占用一行，列的维度固定为word_vec的维度。因此：第一维是batch的size数，
    第二维是句子中词的个数，第三维是词向量的维度。
    :param key: shape=[batch_size, num_word, word_vec]
    :param value: shape=[batch_size, num_word, word_vec]
    :param mask:
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
        # Todo: 需要进行分析, 补零的地方再点积之后中，值仍然为零。因此在这里将值为零的地方代替为-np.inf，从而起到mask的作用。
        scores = scores.masked_fill(mask==0, value=-np.inf)
    # 因此softmax要沿着torch.bmm之后结果的最后一维进行，即沿矩阵A的列进行softmax, 即对矩阵A的每一行进行softmax
    # 总的来看就是可以得到每个样本中，每句话的每一个词的query与同句话中其它词key的相似度概率分布。
    scores = F.softmax(scores, dim=-1)
    # output的维度为[batch_size, num_word, value最后一维的维度]
    # 同样取一个样本的后两维构成的矩阵B进行分析：第i行为第i个词要计算的value加权和。
    output = torch.bmm(scores, value)
    if dropout is not None:
        output = dropout(output)
    return output, scores


def sequence_padding_mask(input_len):
    """
    解码器中在softmax中对补零的位置进行乘-inf，其它位置乘1。（softmax之后的值为正小数）
    :param input_len: shape = [batch_size, 1], 可以理解为是一个行向量或列向量。
    :return:
    """
    max_len = max(input_len)
    mask = [[[1]*int(i) + [-np.inf] * int(max_len-i)] * int(max_len) for i in input_len]
    mask = torch.tensor(mask)
    return mask


def decoder_sequence_mask(src_input):
    """
    解码器中解决左侧会依赖右侧的情况，因些这里要设计一个下三角矩阵（准确讲应该是方阵）与softmax前的矩阵进行对应点相乘。
    :param scr_input:
    :return:
    """
    batch_size = src_input.size(0)
    max_len = int(src_input.size(1))
    mask = [[1] * i + [-np.inf] * (max_len - i) for i in range(1, max_len+1)]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(dim=0).expand([batch_size, max_len, max_len])

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
        self.linear_query = nn.ModuleList([nn.Linear(dim_model, d_k) for _ in range(h)])
        self.linear_key = nn.ModuleList([nn.Linear(dim_model, d_k) for _ in range(h)])
        self.linear_value = nn.ModuleList([nn.Linear(dim_model, d_k) for _ in range(h)])
        # 最后一层的输入维度应该是 h*d_k=dim_model
        self.linear_last = nn.Linear(h*d_k, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Todo: mask还没有加，暂时没有想明白
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
        self.layer_norm = [nn.LayerNorm(dim_model) for _ in range(2)]
        self.ffn = PositionWiseFeedForward(dim_model, dim_ff)

    def forward(self, x):
        """
        sub-encoder, the complete encoder block.
        :param x: batch输入数据
        :return:
        """
        # *********sub-layer1************
        multi_head_attention_output = self.multi_head_attention(x, x, x)
        # residual connection
        residual_layer = multi_head_attention_output + x
        layer_norm_output = self.layer_norm[0](residual_layer)
        # *********sub-layer2************
        ffn_output = self.ffn(layer_norm_output)
        # residual connection
        residual_layer = ffn_output + layer_norm_output
        layer_norm_output = self.layer_norm[1](residual_layer)
        return layer_norm_output


class SubDecoder(nn.Module):
    def __init__(self, h, dim_model, dim_ff):
        super(SubDecoder, self).__init__()
        # 因为一个子decoder要用到2次MultiHeadAttention, 所以要实例化2次
        self.multi_head_attention = [MultiHeadAttention(h, dim_model) for _ in range(2)]
        # Normalize over last dimension of size dim_model
        self.layer_norm = [nn.LayerNorm(dim_model) for _ in range(3)]
        self.ffn = PositionWiseFeedForward(dim_model, dim_ff)

    def forward(self, x, encoder_output, mask):
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
        residual_layer = multi_head_attention_output + x
        # Norm 1
        layer_norm_output = self.layer_norm[0](residual_layer)
        # *********sub-layer2************
        # encoder-decoder attention
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
        self.encoder = nn.Sequential([SubEncoder(h, dim_model, dim_ff) for _ in range(num_sub_encoder)])

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, h, dim_model, dim_ff, num_sub_decoder):
        super(Decoder, self).__init__()
        # 论文第3页中，解码器中有6个子层
        self.decoder = nn.ModuleList([SubDecoder(h, dim_model, dim_ff) for _ in range(num_sub_decoder)])

    def forward(self, x, encoder_output, mask):
        for sub_decoder in self.decoder:
            x = sub_decoder(x, encoder_output, mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len=50000):
        # 长度过长的句子，已经在数据集上去除了。
        # 这里max_seq_len=50000是为了生成一个参考位置编码，这样以后就只需要从这个位置编码中截取所需即可。
        super(PositionalEncoding, self).__init__()
        self.dim_model = dim_model
        self.max_seq_len = max_seq_len
        pe = [[pos / np.power(10000, 2*i/dim_model) for i in range(dim_model)] for pos in range(max_seq_len)]
        pe = np.array(pe)
        # 偶数位置上为sin, 奇数位置上为cos
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        # 这里的pe为单独一个样本的位置编码，因此后续需要根据输入的batch将其调整为batch
        self.position_encoding = pe

    def forward(self, x, x_len):
        """
        compute positional encoding. x_len 为 batch_size 数据，形状为：[batch_size, 1], 行为样本，列为样本长度。
        :param x: batch_size data
        :param x_len: shape=[batch_size, 1]
        :return:
        """
        # 计算出当前batch下最长的序列长度。
        max_seq_len = max(x_len)
        self.position_encoding = self.position_encoding[:max_seq_len]
        batch_size = x_len.size(0)
        self.position_encoding = self.position_encoding.expand([batch_size, max_seq_len, self.dim_model])
        for row_idx, seq_len in enumerate(x_len):
            # 将序列中补零位置的位置编码设置成0
            self.position_encoding[row_idx, seq_len:] = 0
        return x + self.position_encoding


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)

    def forward(self, x):
        """
        将batch_size数据转换成 dim_ model
        :param x: batch_size input, [[idx11, idx12, ...], [idx21, idx22, ...], ...]
        :return:
        """
        return self.embedding(x)


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, h, dim_model, dim_ff, num_sub_encoder, num_sub_decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(h, dim_model, dim_ff, num_sub_encoder)
        self.decoder = Decoder(h, dim_model, dim_ff, num_sub_decoder)
        self.positional_encoding = PositionalEncoding(dim_model)
        self.embedding = Embedding(vocab_size, dim_model)

    def forward(self, src, src_size, tgt, tgt_size):
        # 这里的src, 认为已经通过了padding
        x = self.embedding(src)
        x = self.positional_encoding(x)
        encoder_output = self.encoder(x)
        y = self.embedding(tgt)
        y = self.positional_encoding(y)
        y = self.decoder(y, encoder_output, mask=None)
        output = torch.bmm(y, torch.transpose(self.embedding.weight, dim0=1, dim1=2))
        return output




























