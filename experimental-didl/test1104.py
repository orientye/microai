# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#masked-softmax-operation

import matplotlib.pyplot as plt
import math
import torch
from torch import nn

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        # X形状通常为(batch_size, num_queries, num_keys)
        # 在masked_softmax的实现里，它执行了X.reshape(-1, shape[-1])。这意味着传入_sequence_mask的X实际上被展平成了二维：(batch_size * num_queries, num_keys)。
        # 因此，maxlen = X.size(1) 获取的是num_keys（即最后一维的长度），而不是num_queries。
        # 创建掩码矩阵
        # torch.arange(maxlen): 生成[0, 1, 2, ..., maxlen - 1]
        # [None,:]: 添加维度，形状变为(1, maxlen)
        # valid_len[:, None]: 将valid_len变为(batch_size, 1)
        # 广播比较，生成形状(batch_size, maxlen)的布尔掩码
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
#print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))