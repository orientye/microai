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
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

def check_shape(a, shape):
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

class FlexibleDotProductAttention(nn.Module):  #@save
    """Scaled dot product attention with projection for different dimensions."""
    def __init__(self, dropout, query_dim=None, key_dim=None, common_dim=None):
        """
        Args:
            dropout: dropout rate
            query_dim: dimensionality of queries (required if common_dim specified)
            key_dim: dimensionality of keys (required if common_dim specified)
            common_dim: common dimensionality to project both to (if None, no projection)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create projection matrices if dimensions are specified
        self.query_proj = None
        self.key_proj = None
        self.common_dim = common_dim

        if common_dim is not None:
            if query_dim is not None:
                self.query_proj = nn.Linear(query_dim, common_dim, bias=False)
            if key_dim is not None:
                self.key_proj = nn.Linear(key_dim, common_dim, bias=False)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        Args:
            queries: (batch_size, num_queries, query_dim)
            keys: (batch_size, num_kv_pairs, key_dim)
            values: (batch_size, num_kv_pairs, value_dim)
            valid_lens: (batch_size,) or (batch_size, num_queries)

        Returns:
            output: (batch_size, num_queries, value_dim)
        """
        # Project to common dimension if needed
        if self.query_proj is not None:
            queries = self.query_proj(queries)
        if self.key_proj is not None:
            keys = self.key_proj(keys)

        # Get the dimension for scaling
        d = queries.shape[-1]  # should equal keys.shape[-1] after projection

        # Compute scaled dot product attention scores
        # scores shape: (batch_size, num_queries, num_kv_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # Apply masked softmax
        self.attention_weights = masked_softmax(scores, valid_lens)

        # Compute weighted sum of values
        # output shape: (batch_size, num_queries, value_dim)
        return torch.bmm(self.dropout(self.attention_weights), values)

'''
关键特性：
可学习的投影矩阵：使用 nn.Linear 将查询和键投影到共同维度
维度灵活性：允许查询和键具有不同的输入维度
高效计算：投影后使用标准的缩放点积注意力
无偏置投影：使用 bias=False 保持简单性（可根据需要修改）
'''

# ============= 测试代码 =============

# 创建数据
batch_size = 2
num_queries = 1
num_keys = 10
query_dim = 20  # queries 维度是 20
key_dim = 2     # keys 维度是 2
value_dim = 4   # values 维度是 4
common_dim = 8  # 投影到共同维度 8

# 生成随机数据
torch.manual_seed(42)  # 设置随机种子以确保可重复性
keys = torch.normal(0, 1, (batch_size, num_keys, key_dim))
values = torch.normal(0, 1, (batch_size, num_keys, value_dim))
valid_lens = torch.tensor([2, 6])  # 第1个batch的有效长度为2，第2个为6
queries = torch.normal(0, 1, (batch_size, num_queries, query_dim))

print("=" * 60)
print("FlexibleDotProductAttention 测试")
print("=" * 60)
print(f"Queries shape: {queries.shape} (dim={query_dim})")
print(f"Keys shape: {keys.shape} (dim={key_dim})")
print(f"Values shape: {values.shape} (dim={value_dim})")
print(f"Valid lengths: {valid_lens}")
print()

# 创建注意力模块
attention = FlexibleDotProductAttention(
    dropout=0.1,
    query_dim=query_dim,
    key_dim=key_dim,
    common_dim=common_dim
)
attention.eval()

# 前向传播
output = attention(queries, keys, values, valid_lens)
check_shape(output, (batch_size, num_queries, value_dim))

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention.attention_weights.shape}")
print()

# 检查投影矩阵
if attention.query_proj:
    print(f"Query projection: Linear({query_dim} -> {common_dim})")
if attention.key_proj:
    print(f"Key projection: Linear({key_dim} -> {common_dim})")
print()

# 显示注意力权重热力图
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
    fig.colorbar(pcm, ax=axes, shrink=0.6)

# 重塑注意力权重以便显示
# attention.attention_weights shape: (2, 1, 10) -> (1, 1, 2, 10)
attention_weights_reshaped = attention.attention_weights.reshape((1, 1, batch_size, num_keys))
print("Attention weights (first batch):", attention.attention_weights[0, 0].detach().numpy())
print("Attention weights (second batch):", attention.attention_weights[1, 0].detach().numpy())
print()

# 验证掩码是否生效
print("验证掩码效果:")
print(f"第一个batch的有效长度: {valid_lens[0]}")
print(f"前{valid_lens[0]}个位置的权重: {attention.attention_weights[0, 0, :valid_lens[0]].detach().numpy()}")
print(f"第{valid_lens[0]+1}个位置之后的权重应该接近0: {attention.attention_weights[0, 0, valid_lens[0]:].detach().numpy()}")
print()

# 显示热力图
show_heatmaps(attention_weights_reshaped, xlabel='Keys', ylabel='Queries',
              titles=[f'Batch {i+1}' for i in range(batch_size)], figsize=(5, 2.5))

plt.suptitle('FlexibleDotProductAttention - Attention Weights Heatmap', fontsize=12)
#plt.tight_layout()
plt.show()

# # ============= 额外测试：不使用投影的情况 =============
# print("\n" + "=" * 60)
# print("额外测试：不使用投影（查询和键维度相同）")
# print("=" * 60)
#
# # 当查询和键维度相同时，可以不使用投影
# queries_same_dim = torch.normal(0, 1, (batch_size, num_queries, key_dim))  # 使用与keys相同的维度
# attention_no_proj = FlexibleDotProductAttention(dropout=0.1, common_dim=None)
# attention_no_proj.eval()
#
# output_no_proj = attention_no_proj(queries_same_dim, keys, values, valid_lens)
# check_shape(output_no_proj, (batch_size, num_queries, value_dim))
# print(f"Queries shape (same as keys): {queries_same_dim.shape}")
# print(f"Output shape (no projection): {output_no_proj.shape}")
# print(f"Query projection exists: {attention_no_proj.query_proj is not None}")
# print(f"Key projection exists: {attention_no_proj.key_proj is not None}")
#
# # 显示不使用投影的注意力权重
# attention_no_proj.eval()
# _ = attention_no_proj(queries_same_dim, keys, values, valid_lens)
# attention_weights_no_proj_reshaped = attention_no_proj.attention_weights.reshape((1, 1, batch_size, num_keys))
#
# show_heatmaps(attention_weights_no_proj_reshaped, xlabel='Keys', ylabel='Queries',
#               titles=[f'Batch {i+1} (no proj)' for i in range(batch_size)], figsize=(5, 2.5))
#
# plt.suptitle('FlexibleDotProductAttention (No Projection) - Attention Weights', fontsize=12)
# plt.tight_layout()
# plt.show()
#
# # ============= 比较不同投影维度的影响 =============
# print("\n" + "=" * 60)
# print("比较不同投影维度的影响")
# print("=" * 60)
#
# common_dims = [4, 8, 16, 32]
# fig, axes = plt.subplots(1, len(common_dims), figsize=(12, 3))
#
# for idx, cdim in enumerate(common_dims):
#     attn = FlexibleDotProductAttention(
#         dropout=0.0,
#         query_dim=query_dim,
#         key_dim=key_dim,
#         common_dim=cdim
#     )
#     attn.eval()
#     with torch.no_grad():
#         _ = attn(queries, keys, values, valid_lens)
#         weights = attn.attention_weights[0, 0].detach().numpy()
#
#         ax = axes[idx]
#         ax.bar(range(len(weights)), weights)
#         ax.set_title(f'common_dim={cdim}')
#         ax.set_xlabel('Key index')
#         ax.set_ylabel('Attention weight')
#         ax.set_ylim(0, 1)
#
# plt.suptitle('Effect of Different Projection Dimensions on Attention Weights', fontsize=12)
# plt.tight_layout()
# plt.show()