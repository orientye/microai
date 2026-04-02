import torch

# 训练数据
x_train = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])  # y = x²

# 查询点
x_val = torch.tensor([2.2])

# 高斯核（σ=1）
def gaussian(x):
    return torch.exp(-x**2 / 2)

# 计算距离和权重
dists = x_train - x_val  # [-1.2, -0.2, 0.8, 1.8, 2.8]
k = gaussian(dists)      # [0.4868, 0.9802, 0.7261, 0.1979, 0.0198]
attention_w = k / k.sum() # [0.2020, 0.4066, 0.3012, 0.0821, 0.0082]
#总和 = 0.4868 + 0.9802 + 0.7261 + 0.1979 + 0.0198 = 2.4108
#attention_w = [0.4868/2.4108, 0.9802/2.4108, 0.7261/2.4108, 0.1979/2.4108, 0.0198/2.4108] = [0.2020, 0.4066, 0.3012, 0.0821, 0.0082]

print("距离:", [f"{d:.1f}" for d in dists.numpy()])
print("核值:", [f"{v:.4f}" for v in k.numpy()])
print("权重:", [f"{w:.4f}" for w in attention_w.numpy()])
#       ↑ 在 x=2.0 处权重最大（0.4066），因为离查询点2.2最近

# 预测值
y_hat = y_train @ attention_w
# y_hat = 1.0×0.2020 + 4.0×0.4066 + 9.0×0.3012 + 16.0×0.0821 + 25.0×0.0082
      # = 0.2020 + 1.6264 + 2.7108 + 1.3136 + 0.2050
      # = 6.0578
print(f"真实值: {x_val.item()**2:.2f}") #4.84
print(f"预测值: {y_hat.item():.2f}")