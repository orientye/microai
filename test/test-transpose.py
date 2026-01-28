'''
PyTorch:
tensor.T：快捷的2D转置
torch.transpose(tensor, dim0, dim1)：交换两个指定维度
tensor.permute(*dims)：重新排列所有维度
tensor.t()：只用于2D张量的转置

NumPy:
ndarray.T：转置所有维度（逆序）
np.transpose(a, axes)：按指定顺序排列轴
'''

import torch

# 对于2D张量，等价于矩阵转置
# torch.transpose(input, dim0, dim1)  # 交换 dim0 和 dim1 两个轴

# 对于更高维，可以多次交换或使用 permute
x = torch.randn(2, 3, 4, 5)  # 形状 (2, 3, 4, 5)

# 交换第0维和第2维
y = torch.transpose(x, 0, 2)  # 新形状: (4, 3, 2, 5)

# 使用 permute 可以一次性重新排列所有维度
z = x.permute(2, 0, 3, 1)     # 新形状: (4, 2, 5, 3)