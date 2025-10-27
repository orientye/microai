import numpy as np
import matplotlib.pyplot as plt
from microai import Variable, Parameter
import microai.funcs as F

import torch

# 创建一个 1x6 的张量
x = torch.tensor([[1, 2, 3, 4, 5, 6]])
print(x)
print(f"Original shape: {x.shape}")
# 输出：
# tensor([[1, 2, 3, 4, 5, 6]])
# Original shape: torch.Size([1, 6])

# 重塑为 2x3 的张量
y = x.reshape(2, 3)
print(y)
print(f"Reshaped shape: {y.shape}")
# 输出：
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# Reshaped shape: torch.Size([2, 3])

# 重塑为 3x2 的张量
z = x.reshape(3, 2)
print(z)
# 输出：
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p
print(isinstance(p, Parameter))
print(isinstance( x, Parameter))
print(isinstance(y , Parameter))