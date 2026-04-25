# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html


import math
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
