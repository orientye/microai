# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#exercises
'''
Implement distance-based attention by modifying the DotProductAttention code. Note that you only need the squared norms of the keys
 ||k_i||^2 for an efficient implementation.
'''

import math
import matplotlib.pyplot as plt
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

def check_shape(a, shape):
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'


class DistanceBasedAttention(nn.Module):  # @save
    """Distance-based attention using squared norms of keys."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]

        # 1. 计算点积部分: queries @ keys^T
        # queries: (batch_size, 查询个数, d)
        # keys.transpose(1, 2): (batch_size, d, 键值对个数)
        # dot_product: (batch_size, 查询个数, 键值对个数)
        dot_product = torch.bmm(queries, keys.transpose(1, 2))

        # 2. 计算 keys 的平方范数 ||k_i||^2
        # keys: (batch_size, 键值对个数, d)
        # keys_sq_norms: (batch_size, 1, 键值对个数)
        # 方法: 对每个 key 向量的各个维度求平方和
        keys_sq_norms = torch.sum(keys ** 2, dim=-1, keepdim=True)  # (batch_size, 键值对个数, 1)
        keys_sq_norms = keys_sq_norms.transpose(1, 2)  # (batch_size, 1, 键值对个数)

        # 3. 计算基于距离的分数 (负平方欧氏距离的未归一化形式)
        # score(q, k) = 2 * (q·k) - ||k||^2
        # 注意: 忽略了 -||q||^2 项，因为它对所有键相同，不影响 softmax
        scores = 2 * dot_product - keys_sq_norms

        # 4. 可选: 缩放 (距离注意力通常不需要缩放，但保留以保持灵活性)
        # 如果要缩放，使用 1/sqrt(d) 或其他因子
        # scores = scores / math.sqrt(d)  # 取消注释以启用缩放

        # 5. 应用 masked softmax (与原始代码相同)
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 6. 加权求和并返回
        return torch.bmm(self.dropout(self.attention_weights), values)

'''正态分布，均值0, 标准差为1'''
queries = torch.normal(0, 1, (2, 1, 2)) # 2个batch，每个batch有1个query，维度2
keys = torch.normal(0, 1, (2, 10, 2)) # 2个batch，每个batch有10个key，维度2
values = torch.normal(0, 1, (2, 10, 4)) # 2个batch，每个batch有10个value，维度4
valid_lens = torch.tensor([2, 6]) # 第1个batch的有效长度为2，第2个为6

attention = DistanceBasedAttention(dropout=0.5)
attention.eval()
check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

#plt.tight_layout()
plt.show()