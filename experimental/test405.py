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

# 输入：1个样本，3个通道，7×7大小
# 卷积核：5×5
# 步长：1
# 填充：0
# 输出特征图大小：(7-5+0)/1 + 1 = 3×3（9个位置）
# 每个位置的元素数：5×5×3 = 75（卷积核覆盖的5×5区域 × 3个通道）
# 输出形状：(9, 75) = (输出位置数, 每个位置的展开元素数)

# 输入：10个样本，每个样本3个通道，7×7大小
# 每个样本产生9个输出位置（与第一个例子相同）
# 总共输出位置数：10 × 9 = 90
# 每个位置的元素数仍然是：75
# 输出形状：(90, 75)

# 当 to_matrix=True 时：
# 行数 = 批次大小 × 输出特征图高度 × 输出特征图宽度
# 列数 = 卷积核高度 × 卷积核宽度 × 输入通道数

