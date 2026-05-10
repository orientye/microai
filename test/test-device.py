import torch

import torch

# Check if CUDA (NVIDIA GPU) is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name()}")
else:
    print("GPU is not available. Using CPU instead.")
    print(torch.__version__)
    # 查看是否支持 CUDA
    print(torch.version.cuda)  # 如果返回 None，说明是 CPU 版本

# 查看安装类型
print(f"安装位置: {torch.__file__}")
print(f"PyTorch 版本: {torch.__version__}")

# 检查库文件
import sys
print(f"Python 路径: {sys.executable}")
print(f"Python 环境: {sys.prefix}")