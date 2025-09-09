import numpy as np

# 创建一个一维数组
arr = np.array([1, 2, 3])
print("Original array:", arr)
print("Original shape:", arr.shape)

# 将其广播到一个 2x3 的形状
# (3,) -> (2, 3)
# 规则：原始形状 (3,) 与目标形状 (2, 3) 的尾部维度都是 3，匹配。
#        原始数组在新增的轴0上“复制”了2次。
broadcasted = np.broadcast_to(arr, (2, 3))
print("\nBroadcasted to (2, 3):\n", broadcasted)
print("New shape:", broadcasted.shape)

# 创建一个 2x1 的数组
arr_2d = np.array([[4], [5]])
print("Original array:\n", arr_2d)
print("Original shape:", arr_2d.shape)

# 将其广播到一个 2x3 的形状
# (2, 1) -> (2, 3)
# 规则：轴0维度相等 (2==2)。
#       轴1维度，原始是1，目标是3。1可以广播到3。
broadcasted_2 = np.broadcast_to(arr_2d, (2, 3))
print("\nBroadcasted to (2, 3):\n", broadcasted_2)
print("New shape:", broadcasted_2.shape)

# 创建一个 3x1x2 的数组
arr_3d = np.arange(6).reshape(3, 1, 2)
print("Original array:\n", arr_3d)
print("Original shape:", arr_3d.shape)

# 将其广播到一个 (3, 4, 2) 的形状
# (3, 1, 2) -> (3, 4, 2)
# 规则：轴0: 3==3
#       轴1: 1 -> 4 (1可以广播到4)
#       轴2: 2==2
broadcasted_3 = np.broadcast_to(arr_3d, (3, 4, 2))
print("\nBroadcasted to (3, 4, 2):\n", broadcasted_3)
print("New shape:", broadcasted_3.shape)
