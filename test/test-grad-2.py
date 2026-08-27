import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2  # y = [1.0, 4.0]
v = torch.tensor([0.5, 0.3])  # 传入的 gradient 参数

# 反向传播
y.backward(gradient=v)

# 实际计算过程（PyTorch 内部）：
# 1. 接收 v = [0.5, 0.3] 作为 y 的梯度
# 2. 计算 y 对 x 的局部导数：dy/dx = 2*x = [2.0, 4.0]
# 3. 将 v 与局部导数逐元素相乘并累加：
#    grad_x1 = v1 * (dy1/dx1) = 0.5 * 2.0 = 1.0
#    grad_x2 = v2 * (dy2/dx2) = 0.3 * 4.0 = 1.2
# 4. 得到 x.grad = [1.0, 1.2]

print(x.grad)  # tensor([1.0000, 1.2000])