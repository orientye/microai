# https://en.d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#additive-attention
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

class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs) #** kwargs：传递给父类的其他参数
        self.W_k = nn.LazyLinear(num_hiddens, bias=False) #将Key映射到隐藏维度num_hiddens。LazyLinear会根据输入自动推断输入特征维数。
        self.W_q = nn.LazyLinear(num_hiddens, bias=False) #将Query映射到相同的num_hiddens维度。
        self.w_v = nn.LazyLinear(1, bias=False) #最后的线性层。它将加权后的特征压缩成一个标量（分数），作为注意力的原始得分。
        self.dropout = nn.Dropout(dropout) #对注意力权重进行随机丢弃，防止过拟合。

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 将queries和keys投影到相同的维度num_hiddens
        # 投影后形状：
        # queries: (batch_size, num_queries, num_hiddens)
        # keys: (batch_size, num_keys, num_hiddens)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # 通过维度扩展实现所有query - key对的组合
        # queries.unsqueeze(2)在索引2的位置插入新维度
        # 形状变化：(2, 1, 8) → (2, 1, 1, 8) 含义：(batch, num_queries, 1, num_hiddens)
        # keys.unsqueeze(1)在索引1的位置插入新维度
        # 形状变化：(2, 10, 8) → (2, 1, 10, 8) 含义：(batch, 1, num_keys, num_hiddens)

        #相加时触发广播机制，生成一个(batch, query_count, key_count, num_hiddens)的张量。这代表了每一个Query和每一个Key的特征结合。
        #相加后features：(2, 1, 10, 8)这样每个query与每个key在最后一维（num_hiddens）上逐元素相加，实现了所有query - key对的组合。

        features = torch.tanh(features)
        # tanh（双曲正切）：将输出压缩到[-1, 1]区间 引入非线性，增强表达能力

        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        # w_v将最后一维从num_hiddens降维到1。
        # squeeze(-1)移除最后一维（大小为1），得到(batch, query_count, key_count)的得分矩阵。 得到(2, 1, 10)

        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        # 通过之前的 masked_softmax 函数将得分转为概率，并屏蔽掉无效长度之外的部分。

        # torch.bmm：批量矩阵乘法
        # 注意力权重：(2, 1, 10)
        # values：(2, 10, 4)
        # 结果：(2, 1, 4) — 每个query的上下文向量
        return torch.bmm(self.dropout(self.attention_weights), values)

'''正态分布，均值0, 标准差为1'''
#queries = torch.normal(0, 1, (2, 1, 2)) # 2个batch，每个batch有1个query，维度2
keys = torch.normal(0, 1, (2, 10, 2)) # 2个batch，每个batch有10个key，维度2
values = torch.normal(0, 1, (2, 10, 4)) # 2个batch，每个batch有10个value，维度4
valid_lens = torch.tensor([2, 6]) # 第1个batch的有效长度为2，第2个为6

queries = torch.normal(0, 1, (2, 1, 20)) #queries 维度是 20，与 keys 的维度（2）不同！这正是加性注意力的优势：query 和 key 的维度可以不同
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
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