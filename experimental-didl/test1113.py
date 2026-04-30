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
result = ffn(torch.ones((2, 3, 4)))[0]

print(f"输出形状: {result.shape}")
print(f"输出内容:")
print(result)