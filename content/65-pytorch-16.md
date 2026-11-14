# 输入尺度为什么影响训练

看得懂的 PyTorch · 第 16 讲

同一项信息用米表示，数值是 1、2、3；改成毫米，就成了 1000、2000、3000。信息没变，梯度的规模却会改变。

把这两列同时交给线性模型，能清楚看到尺度带来的影响。两列故意完全相关，只为隔离单位差异；这不是值得保留两个冗余特征的理由。

## 一次更新为什么会冲出去

输入为 `[1,1000]、[2,2000]、[3,3000]`，目标为 2、4、6。模型为：

```text
预测 = w₁×x₁ + w₂×x₂ + b
```

参数从零开始，误差为 −2、−4、−6，均方误差为 `56/3≈18.666667`。

```text
w₁ 梯度 = (2/3)×[−2×1−4×2−6×3] = −56/3
w₂ 梯度 = (2/3)×[−2×1000−4×2000−6×3000] = −56000/3
b 梯度 = (2/3)×[−2−4−6] = −8
```

学习率取 0.01，更新后约为 `w₁=0.186667、w₂=186.666667、b=0.08`。

第一条样本的新预测约为：

```text
0.186667×1 + 186.666667×1000 + 0.08 = 186666.933333
```

目标只有 2。第二列既产生了更大的梯度，又在前向计算时乘上很大的输入，导致一步跨得太远。

## 每列减去均值，再除以标准差

标准化使用：

```text
z = (x−均值) / 标准差
```

第一列均值为 2。这里用除以样本数 N 的方差，而不是除以 N−1：

```text
方差 = [(1−2)²+(2−2)²+(3−2)²]/3 = 2/3
标准差 = √(2/3) ≈ 0.816497
```

第二列均值为 2000，标准差约为 816.496581。两列变换后都成为：

```text
[−1.224745, 0, 1.224745]
```

`torch.std` 这里明确使用 `correction=0`。若某列训练数据完全相同，标准差为零，代码将除数替换为 1，避免除零；这列在训练集上变为零，模型无法从它学出变化关系。

标准化后，从零参数出发，两个权重的梯度都约为 −3.265986，偏置梯度仍为 −8。同样用学习率 0.01，一步后的损失从 18.666667 降到 17.824。

## 用代码比较，再训练到新输入

下面先比较相同学习率下的一步，再重新初始化模型，在标准化数据上用学习率 0.1 训练 100 步：

```python
import torch
from torch import nn

X = torch.tensor([[1., 1000.], [2., 2000.], [3., 3000.]],
                 dtype=torch.float64)
y = torch.tensor([[2.], [4.], [6.]], dtype=torch.float64)
mean = X.mean(dim=0)
scale = X.std(dim=0, correction=0)
scale = torch.where(scale > 0, scale, torch.ones_like(scale))
Z = (X - mean) / scale
print("mean:", mean.tolist())
print("scale:", [round(v, 6) for v in scale.tolist()])
print("Z:", [[round(v, 6) for v in row] for row in Z.tolist()])


def make_model():
    model = nn.Linear(2, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


for name, inputs in [("raw", X), ("scaled", Z)]:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    before = (model(inputs) - y).square().mean()
    before.backward()
    print(name, "gradient:", model.weight.grad.tolist())
    optimizer.step()
    with torch.no_grad():
        after = (model(inputs) - y).square().mean()
    print(f"{name}: {before.item():.6f} -> {after.item():.6f}")

model = make_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = (model(Z) - y).square().mean()
    loss.backward()
    optimizer.step()

new_X = torch.tensor([[4., 4000.]], dtype=torch.float64)
with torch.no_grad():
    prediction = model((new_X - mean) / scale)
    raw_weight = model.weight / scale
    raw_bias = model.bias - (raw_weight * mean).sum(dim=1)
    equivalent = new_X @ raw_weight.T + raw_bias
print("prediction:", round(prediction.item(), 6))
torch.testing.assert_close(prediction, equivalent)
```

PyTorch 2.13.0、CPU 的关键输出：

```text
raw: 18.666667 -> 162604307925.882080
scaled: 18.666667 -> 17.824000
prediction: 8.0
```

原始尺度也可以训练，例如选用足够小的学习率，但可能让其他方向进展缓慢。标准化改善了本例的数值尺度，不保证任意学习率都稳定，也不能消除冗余特征或错误标签。

## 新样本不能重新计算自己的均值

输入 `[4,4000]`，必须使用训练集得到的均值和标准差：

```text
z₁ = (4−2)/0.816497 ≈ 2.449490
z₂ = (4000−2000)/816.496581 ≈ 2.449490
```

如果用这一条新样本重新求均值，两列减去各自均值都会变成零，真实输入变化被抹掉。

同样，验证集和测试集也只能应用训练集的统计量。先用全部数据估计均值再划分，会把留出数据的信息带入训练过程。流程应是：先划分，再在训练集拟合变换，最后应用到各份数据。

## 标准化没有偷偷换掉模型类型

对线性模型，代入变换后仍可以还原为原单位下的直线：

```text
预测 = Σ wⱼ(xⱼ−μⱼ)/sⱼ + b
     = Σ (wⱼ/sⱼ)xⱼ + [b−Σ(wⱼμⱼ/sⱼ)]
```

代码末尾计算了原单位下的权重和偏置，并核对两种写法的预测一致。标准化改变了参数坐标和训练过程，没有让线性模型凭空变成非线性模型。

本例两列完全相关，只能约束它们组合后的作用，不能唯一确定每列应该分到多少权重。输入尺度处理与参数可解释性，是不同的问题。
