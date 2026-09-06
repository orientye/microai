import torch


def lib_momentum_example():
    print("=" * 50)

    # 方法1: 使用PyTorch内置的SGD优化器（推荐）
    print("方法1: 使用torch.optim.SGD")

    # 定义可训练参数
    theta = torch.tensor([3.0], requires_grad=True)

    # 使用PyTorch的SGD优化器（带动量）
    optimizer = torch.optim.SGD([theta], lr=0.1, momentum=0.9)

    for i in range(5):
        # 前向传播：计算损失 J(θ) = θ²
        loss = theta ** 2

        # 清零梯度
        optimizer.zero_grad()

        # 反向传播：计算梯度
        loss.backward()

        # 保存当前梯度值（用于显示）
        current_grad = theta.grad.item()

        # 执行优化步骤（包含动量更新）
        optimizer.step()

        print(f"Iter {i + 1}: θ = {theta.item():.4f}, Gradient = {current_grad:.4f}")


def manual_momentum_correct():
    print("\n方法2: 手动实现动量")
    print("=" * 50)

    # 初始参数
    theta = torch.tensor([3.0], requires_grad=True)
    velocity = torch.tensor([0.0])
    lr = 0.1
    momentum = 0.9

    for i in range(5):
        # 计算损失和梯度
        loss = theta ** 2

        # 手动清零梯度（重要！）
        if theta.grad is not None:
            theta.grad.zero_()

        loss.backward()
        current_grad = theta.grad.item()

        # 手动动量更新
        with torch.no_grad():
            velocity = momentum * velocity + theta.grad
            theta -= lr * velocity

        print(f"Iter {i + 1}: θ = {theta.item():.4f}, Gradient = {current_grad:.4f}, Velocity = {velocity.item():.4f}")

lib_momentum_example()
manual_momentum_correct()

import torch
import torch.nn as nn
import torch.optim as optim
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def neural_net_momentum_example():
    print("\n神经网络中的动量法示例:")
    print("=" * 50)

    # 创建模型和数据
    model = SimpleNet()
    # 设置特定权重以便观察：y = 2x
    with torch.no_grad():
        model.linear.weight.data = torch.tensor([[2.0]])
        model.linear.bias.data = torch.tensor([0.0])

    # 创建优化器（带动量）
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    # 简单的训练数据：尝试拟合 y = x
    x = torch.tensor([[1.0]])
    y_true = torch.tensor([[1.0]])

    for epoch in range(10):
        # 前向传播
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y_true)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 保存梯度信息（用于观察）
        weight_grad = model.linear.weight.grad.item()

        # 更新参数
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, Weight Grad = {weight_grad:.4f}")
        print(f"         Weight = {model.linear.weight.item():.4f}")


# 运行神经网络示例
neural_net_momentum_example()
