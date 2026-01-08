# 方法	文件扩展名	优点	缺点	适用场景
# save()/load()	.npy	快速，支持所有dtype	只能存一个数组	单个数组
# savez()	.npz	可存多个数组	稍慢于.npy	多个相关数组
# savez_compressed()	.npz	文件小	保存/加载慢	大型数组
# savetxt()/loadtxt()	.txt/.csv	可读性好	慢，只支持1D/2D	文本交换
# tofile()	任意	紧凑	无元数据	原始二进制

import numpy as np

# 创建示例数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 保存单个数组到 .npy 文件
np.save('array.npy', arr)  # 自动添加 .npy 扩展名

# 加载数组
loaded_arr = np.load('array.npy')
print(loaded_arr)

# 创建多个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([[1, 2], [3, 4]])

# 保存多个数组到 .npz 文件（压缩格式）
np.savez('arrays.npz', a=arr1, b=arr2, c=arr3)

# 保存为压缩格式（文件更小）
np.savez_compressed('compressed_arrays.npz',
                    array1=arr1,
                    array2=arr2,
                    matrix=arr3)

# 加载
data = np.load('arrays.npz')
print(data.files)  # ['a', 'b', 'c']
print(data['a'])  # [1 2 3]
print(data['b'])  # [4 5 6]
print(data['c'])  # [[1 2] [3 4]]