import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.elementwise import ElementwiseKernel

# 1. 创建一个简单的元素级加法内核
add_kernel = ElementwiseKernel(
    "float *a, float *b, float *c",  # 参数列表
    "c[i] = a[i] + b[i]",            # 操作表达式
    "add"                            # 内核名称
)

# 2. 准备数据
n = 1000000
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.empty_like(a)

# 3. 传输到GPU
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty_like(a_gpu)

# 4. 执行内核
add_kernel(a_gpu, b_gpu, c_gpu)

# 5. 验证结果
c_result = c_gpu.get()
print("前5个结果:", c_result[:5])
print("CPU验证:", np.allclose(a + b, c_result))