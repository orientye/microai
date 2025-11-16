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
