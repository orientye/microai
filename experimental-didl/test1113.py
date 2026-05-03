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