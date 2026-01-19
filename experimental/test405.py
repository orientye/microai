import numpy as np
import microai.conv as conv


# im2col
x1 = np.random.rand(1, 3, 7, 7)
col1 = conv.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)  # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)  # 10个数据
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = conv.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)  # (90, 75)

