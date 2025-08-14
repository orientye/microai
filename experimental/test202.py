import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("原始数组:\n", arr)
print("转置后的数组:\n", np.transpose(arr))

# 等价写法
print("使用.T属性:\n", arr.T)

print("\n\n\n")

arr_3d = np.arange(8).reshape(2, 2, 2)
print("原始3D数组:\n", arr_3d)
print("默认转置(相当于反转轴顺序):\n", np.transpose(arr_3d))
print("指定轴顺序(0,2,1):\n", np.transpose(arr_3d, (0, 2, 1)))