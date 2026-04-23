# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html


import math
import torch
from torch import nn
from d2l import torch as d2l


class MultiHeadAttention(d2l.Module):  # @save
    """Multi-head attention."""

    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
        # num_hiddens并不是单个注意力头的维度，而是整个多头注意力模块输入和输出的维度（相当于标准Transformer论文中的dmodel）。
        # 单个注意力头的维度实际上是num_hiddens / num_heads。
        '''
        为什么输入/输出投影要保持num_hiddens 维度？
        
        原因一：残差连接（Residual Connection）
        在Transformer架构中，多头注意力的输出会与原始输入queries进行相加（Add & Norm）。
        这就要求MultiHeadAttention的输出形状必须与输入queries的形状严格一致：
        输入queries形状：(batch_size, num_queries, num_hiddens)
        输出形状：(batch_size, num_queries, num_hiddens)
        如果W_o输出的维度不是num_hiddens，后续的残差加法将无法执行。

        原因二：多头维度的拆分是在投影之后进行的
        流程如下：
        线性投影（扩大 / 变换特征）：
        self.W_q(queries)  # 输出形状：(batch, queries, num_hiddens)
        分割成多头（通过reshape）：
        X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # 形状变为：(batch, queries, num_heads, num_hiddens/num_heads)
        这种做法的好处是：所有头的参数共享在同一个大矩阵中计算（W_q是一个num_hiddens × num_hiddens的矩阵），而不是分别用
        num_heads个独立的小矩阵去投影。这在 PyTorch 中是高效且标准的实现方式。
        '''

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        '''
        将线性变换后的查询张量送入transpose_qkv函数进行维度重排，以便并行计算多个注意力头。
        输出的形状变为(batch_size * num_heads, num_queries, um_hiddens / num_heads)。
        同样的操作分别应用于键和值张量
        '''

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)
        '''
        torch.repeat_interleave 沿第0维（批次维度）将 valid_lens 中的每个元素重复 num_heads 次。
        例如，如果原始 valid_lens 是 [3, 2]，且 num_heads=5，则会变成 [3,3,3,3,3, 2,2,2,2,2]。
        这确保了掩码可以正确应用于 (batch_size * num_heads) 这个“批次”维度上的每个头
        '''

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # 调用缩放点积注意力，得到多头注意力的原始输出，形状为(batch_size * num_heads, no.of queries, num_hiddens / num_heads)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        # 执行与transpose_qkv 相反的操作，将输出形状恢复为(batch_size, no.of queries, num_hiddens)，为最终投影做准备
        return self.W_o(output_concat)
        # 通过线性层 W_o 对拼接后的多头输出进行最终投影，得到最终的多头注意力输出，形状仍为 (batch_size, no. of queries, num_hiddens)

@d2l.add_to_class(MultiHeadAttention)  # @save
def transpose_qkv(self, X): #维度变换
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input X: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). Shape of output X: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Shape of output X: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Shape of output: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])
'''
X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)：
将原始形状 (batch_size, seq_len, num_hiddens) 重塑为 (batch_size, seq_len, num_heads, head_dim)，其中 head_dim = num_hiddens / num_heads。
X = X.permute(0, 2, 1, 3)：
交换第1和第2维度，得到 (batch_size, num_heads, seq_len, head_dim)。这使得每个头的数据在维度上连续，便于独立计算。
return X.reshape(-1, X.shape[2], X.shape[3])：
将 batch_size 和 num_heads 合并为一个维度，最终得到 (batch_size * num_heads, seq_len, head_dim)。
这样，DotProductAttention 就可以像处理一个大“批次”一样，并行计算所有头的注意力。
'''

@d2l.add_to_class(MultiHeadAttention)  # @save
def transpose_output(self, X): #逆向变换
    """Reverse the operation of transpose_qkv."""
    # 接收注意力计算后的输出X，其形状为(batch_size * num_heads, num_queries, num_hiddens / num_heads)
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    # 将合并的批次维重新拆分回(batch_size, num_heads, num_queries, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # 交换维度1和维度2，得到形状(batch_size, num_queries, num_heads, num_hiddens / num_heads)
    return X.reshape(X.shape[0], X.shape[1], -1)
    # 将最后两维合并，从而将各头的输出拼接在一起，最终形状恢复为 (batch_size, num_queries, num_hiddens)


num_hiddens, num_heads = 100, 5 # 设置隐藏层维度为 100，注意力头数为 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5) # 实例化一个多头注意力模块，dropout 概率设为 0.5
batch_size, num_queries, num_kvpairs = 2, 4, 6 # 设置批次大小为 2，查询数量为 4，键值对数量为 6
valid_lens = torch.tensor([3, 2]) # 指定第一个样本的有效长度为 3，第二个样本的有效长度为 2
X = torch.ones((batch_size, num_queries, num_hiddens)) # 创建全 1 的查询张量，形状为 (2, 4, 100)
Y = torch.ones((batch_size, num_kvpairs, num_hiddens)) # 创建全 1 的查询张量，形状为 (2, 6, 100)
d2l.check_shape(attention(X, Y, Y, valid_lens), # 调用多头注意力模块
                (batch_size, num_queries, num_hiddens))
