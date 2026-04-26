# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html


import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

# num_hiddens: 模型的特征维度（隐藏层大小）。这里设为 100，意味着每个词或特征向量的长度是 100。
# num_heads: 多头注意力的“头”数。这里设为 5，意味着 100 维的特征会被分成 5 组（每组 20 维）并行处理，最后再拼接起来
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))


class PositionalEncoding(nn.Module):  # @save
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        '''
        num_hiddens: 隐藏层维度（也是位置编码的维度）
        dropout: dropout比率，用于正则化
        max_len: 预计算的最大序列长度，默认为1000
        创建Dropout层用于防止过拟合
        '''
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        # 创建位置编码张量P，形状为(1, max_len, num_hiddens) 第一维为1是为批量维度预留
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X) #偶数维度（0,2,4,...）使用正弦函数
        self.P[:, :, 1::2] = torch.cos(X) #奇数维度（1,3,5,...）使用余弦函数

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
plt.show()