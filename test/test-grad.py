import torch

x = torch.arange(5, requires_grad=True, dtype=torch.float32)

# 第一次反向传播
y1 = x ** 2
y1.sum().backward()
print(x.grad)  # tensor([0., 2., 4., 6., 8.])

# 第二次反向传播（梯度会累积）
y2 = x ** 3
y2.sum().backward()
print(x.grad)  # tensor([0., 5., 16., 33., 56.])
# 第一次梯度: [0,2,4,6,8]
# 第二次梯度: [0,3,12,27,48]
# 累积结果: [0,5,16,33,56]