import torch

def demonstrate_mean_operation():
    # 1. 创建一个模拟的 3D 张量，例如形状为 (batch_size=2, seq_len=3, hidden_dim=4)
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0], [0.0,  2.0,  4.0,  6.0]]
    ])
    
    print("--- 原始张量 ---")
    print(f"形状 (Shape): {x.shape}")
    print(x)
    print("\n" + "="*40 + "\n")

    # 2. 沿着最后一维 (-1) 求均值，并保持维度 (keepdim=True)
    # 计算逻辑：对最内层长度为 4 的数组求平均（如 [1, 2, 3, 4] 的平均值是 2.5）
    result_keepdim = x.mean(dim=-1, keepdim=True)

    print("--- 执行 x.mean(dim=-1, keepdim=True) ---")
    print(f"计算后形状: {result_keepdim.shape}  <- 注意：最后一维变成了 1，但张量依然是 3D 的")
    print(result_keepdim)
    print("\n" + "="*40 + "\n")

    # 3. 典型应用演示：利用广播机制（Broadcasting）进行数据中心化
    # 原始张量 (2, 3, 4) 减去 均值张量 (2, 3, 1)，均值会自动复制填充到 4 个特征上
    x_centered = x - result_keepdim
    print("--- 应用：数据中心化（x - mean） ---")
    print("减去均值后的每个元素（每组最内层数据的均值现在都变成了 0）：")
    print(x_centered)

if __name__ == "__main__":
    demonstrate_mean_operation()
