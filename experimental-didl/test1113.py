# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html


import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4, 8)
ffn.eval()
result = ffn(torch.ones((2, 3, 5)))[0]

print(f"输出形状: {result.shape}")
print(f"输出内容:")
print(result)

print(f"dense1 权重形状: {ffn.dense1.weight.shape}")  # torch.Size([4, 5])
print(f"dense2 权重形状: {ffn.dense2.weight.shape}")  # torch.Size([8, 4])

ln = nn.LayerNorm(2) #创建一个层归一化层 对每个样本的所有特征进行归一化（横向归一化）
bn = nn.LazyBatchNorm1d() #创建一个批归一化层（懒加载版本，不需要预先指定特征数，会在第一次前向传播时自动推断） 对每个特征在所有样本上进行归一化（纵向归一化）
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from X in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        """
        X：原始输入（残差连接的“直连”路径）
        Y：经过子层（如自注意力或前馈网络）处理后的输出
        self.dropout(Y) + X：将子层输出（经过dropout后）与原始输入相加，这就是残差连接
        """
        return self.ln(self.dropout(Y) + X)

add_norm = AddNorm(4, 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(torch.ones(shape), torch.ones(shape)), shape)


class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        # 1. 多头自注意力层：学习序列内部的依赖关系
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        # 2. 第一个残差连接和层归一化（用于注意力层之后）
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        # 3. 逐位前馈网络：对每个位置的特征进行非线性变换
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        # 4. 第二个残差连接和层归一化（用于前馈网络之后）
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        """
        首先计算自注意力：使用相同的X作为查询、键和值，valid_lens控制有效长度。输出形状与X相同(batch_size, seq_len, num_hiddens)。
        然后通过addnorm1：将注意力输出与原始输入X做残差连接，然后进行层归一化和可能的dropout，输出Y。
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        #计算前馈网络：self.ffn(Y)将Y变换为相同形状。然后通过addnorm2：将前馈网络输出与Y做残差连接，再归一化。返回最终输出，形状不变。
        return self.addnorm2(Y, self.ffn(Y))

X = torch.ones((2, 100, 24)) # 全一张量，表示 batch_size=2，序列长度=100，嵌入维度=24（即 num_hiddens=24）。
valid_lens = torch.tensor([3, 2]) # 有效长度张量 [3, 2]，表示第一个样本只有前 3 个位置有效，第二个样本只有前 2 个位置有效，其余为填充。
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval() # 将模块切换到评估模式，主要影响 Dropout 和 BatchNorm 等层，例如 Dropout 在 eval 模式下不生效。
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)

class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        # 1. 词嵌入层：将单词索引转换为连续向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 2. 位置编码：为向量注入位置信息（Transformer 本身不识别顺序）
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 3. 堆叠多个编码器块
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        # 第一步：词嵌入，并乘以特征维度的平方根进行缩放（为了和位置编码匹配）
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 准备存储每一层的注意力权重
        self.attention_weights = [None] * len(self.blks)
        # 逐层通过编码器块
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 保存该层计算出的注意力权重
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

# 词汇表大小 200，特征/隐藏维度 24，前馈内部维度 48，头数 8，块数 2，dropout 0.5。
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))