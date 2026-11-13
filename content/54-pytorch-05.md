# 把一次完整训练算出来

看得懂的 PyTorch · 第 5 讲

一次更新后，三套房屋的均方误差从 25 降到约 22.59，但其中一套的预测反而更差了。

共享参数需要兼顾多条样本。训练降低的是所定义的整体损失，并不保证每条预测都同时改善。把完整的一步算出来，就能看到这个取舍怎样发生。

## 更新前的状态

仍用前三讲的虚构数据：面积以 10 平方米为单位，价格以万元为单位。

```python
import torch

X = torch.tensor([[6., 2.], [8., 3.], [10., 3.]], dtype=torch.float64)
y = torch.tensor([95., 110., 140.], dtype=torch.float64)
w = torch.tensor([10., 5.], dtype=torch.float64, requires_grad=True)
b = torch.tensor(20., dtype=torch.float64, requires_grad=True)
lr = 0.005

pred = X @ w + b
assert pred.shape == y.shape
loss = ((pred - y) ** 2).mean()
loss.backward()
```

此时各项数值为：

| 项目 | 数值 |
|---|---|
| 权重 `w` | `[10, 5]` |
| 偏置 `b` | `20` |
| 预测 | `[90, 115, 135]` |
| 误差 | `[−5, 5, −5]` |
| 均方误差 | `25` |
| `w.grad` | `[−80/3, −20/3]` |
| `b.grad` | `−10/3` |

梯度由第四讲的链式法则算出。它们对应的是当前参数位置，`backward()` 执行完毕后，参数还没有变化。

## 用同一组梯度更新参数

学习率取 0.005，更新规则为“参数减去学习率乘梯度”：

```text
面积权重 = 10 − 0.005 × (−80/3) = 10.133333…
房间权重 =  5 − 0.005 × (−20/3) =  5.033333…
偏置     = 20 − 0.005 × (−10/3) = 20.016667…
```

对应代码：

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

两行代码先后执行，但使用的梯度都来自更新前的同一次计算。更新完 `w` 后，不会重新计算 `b.grad`。

第一套房屋的新预测为：

```text
6 × 10.133333… + 2 × 5.033333… + 20.016667… = 90.883333…
```

重新计算全部预测和损失：

```python
with torch.no_grad():
    pred_after = X @ w + b
    loss_after = ((pred_after - y) ** 2).mean()

print([round(v, 4) for v in pred_after.tolist()])
# [90.8833, 116.1833, 136.45]
print(round(loss_after.item(), 6))  # 22.594352
```

| 房屋 | 正确价格 | 更新前预测 | 更新后预测 | 更新后误差平方 |
|---|---|---|---|---|
| A | 95 | 90 | 90.8833 | 16.9469 |
| B | 110 | 115 | 116.1833 | 38.2336 |
| C | 140 | 135 | 136.4500 | 12.6025 |

三个参数都增大，三个预测随之上升。A、C 原本偏低，得到改善；B 原本偏高，误差继续扩大。三项误差平方取平均，仍从 25 降到了约 22.59。

## 日志记录的是哪一步

更新参数后，原来的 `loss` 仍保存着更新前的结果：

```python
print(loss.item())        # 25.0
print(loss_after.item())  # 22.594351851851854 左右
```

如果训练日志放在参数更新之后，却打印旧的 `loss`，显示的仍是本次更新前的损失。数值本身没错，但容易被误读为更新后的效果。

下面统一记录“完成多少次更新后的状态”，初始状态记为第 0 步。

## 连续训练 1000 步

将一次更新放进循环。以下代码重新创建参数，可以独立运行：

```python
import torch

X = torch.tensor([[6., 2.], [8., 3.], [10., 3.]], dtype=torch.float64)
y = torch.tensor([95., 110., 140.], dtype=torch.float64)
w = torch.tensor([10., 5.], dtype=torch.float64, requires_grad=True)
b = torch.tensor(20., dtype=torch.float64, requires_grad=True)
lr = 0.005

def report(step):
    with torch.no_grad():
        prediction = X @ w + b
        mse = ((prediction - y) ** 2).mean()
    print(f"{step:4d}  w=[{w[0].item():.4f}, {w[1].item():.4f}]"
          f"  b={b.item():.4f}  loss={mse.item():.4f}")

report(0)
for step in range(1, 1001):
    w.grad = None
    b.grad = None

    pred = X @ w + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    if step in {1, 2, 10, 100, 1000}:
        report(step)
```

PyTorch 2.13.0、CPU 运行结果：

```text
   0  w=[10.0000, 5.0000]   b=20.0000  loss=25.0000
   1  w=[10.1333, 5.0333]   b=20.0167  loss=22.5944
   2  w=[10.1691, 5.0344]   b=20.0216  loss=22.4232
  10  w=[10.2046, 4.9608]   b=20.0311  loss=22.2470
 100  w=[10.4713, 4.1172]   b=20.1313  loss=20.4854
1000  w=[12.3725, -2.2057]  b=21.6926  loss=10.0076
```

每轮开头清空旧梯度，然后用当前参数重新计算。梯度也随参数变化，因此房间权重第一步上升，之后却逐渐下降。1000 步是这里指定的运行次数，并不意味着已经收敛。

每一步都使用全部三条样本，所以本例的一步更新恰好也是一轮完整的数据遍历。数据较多时，通常分成多个小批次，一轮遍历包含多次更新，`step` 和 `epoch` 就不再相同。

## 用优化器替换手写更新

最基本的 SGD 优化器可以执行相同规则。它需要持有待更新的参数，并读取这些参数上的 `.grad`。

下面从初始值重新训练 1000 步，与手写循环比较：

```python
w_sgd = torch.tensor([10., 5.], dtype=torch.float64, requires_grad=True)
b_sgd = torch.tensor(20., dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.SGD([w_sgd, b_sgd], lr=0.005)

for _ in range(1000):
    optimizer.zero_grad(set_to_none=True)
    pred_sgd = X @ w_sgd + b_sgd
    loss_sgd = ((pred_sgd - y) ** 2).mean()
    loss_sgd.backward()
    optimizer.step()

print(torch.allclose(w, w_sgd))  # True
print(torch.allclose(b, b_sgd))  # True
```

这里没有启用动量或权重衰减，SGD 的更新就是 `参数 -= 学习率 × 梯度`。`zero_grad(set_to_none=True)` 对应清空参数的旧梯度，`step()` 负责修改参数；前向计算和反向传播仍由循环中的代码发起。[SGD 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)、[Optimizer.zero_grad 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)

## 损失下降之后

1000 步后，模型在三条训练数据上的误差减小了，但房间权重变成了负数。这不能解释为“房间越多越便宜”：面积和房间数在这三条数据里共同变化，拟合参数并不是因果结论。

事实上，`w=[15, −15]、b=35` 可以让三个预测分别等于 95、110、140，训练损失为零。即使达到这个结果，也只说明三条方程被满足。模型对其他房屋是否准确，仍需独立数据检验。
