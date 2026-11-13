# 模型学会了 到底哪个数字变了

看得懂的 PyTorch · 第 1 讲

给程序四条数据：

| 输入 x | 正确答案 y |
|---|---|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

这些数据满足 `y = 2x + 1`。训练的任务，是从数据中找到系数 2 和常数 1。模型设为：

```python
预测值 = w * 输入 + b
```

`w` 是斜率，`b` 是截距，两者就是全部参数。模型的直线结构由人规定，参数从零开始学习。

## 从零开始预测

以下代码按顺序运行，只需 PyTorch 和 CPU。

```python
import torch

x = torch.tensor([1., 2., 3., 4.])
y = torch.tensor([3., 5., 7., 9.])
w = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)

pred = w * x + b
print(pred.tolist())
```

输出是 `[0.0, 0.0, 0.0, 0.0]`。

`x` 和 `y` 的形状都是 `[4]`；`w` 和 `b` 是标量。同一组参数作用于四条数据，这里使用了 Tensor 的广播机制。

`requires_grad=True` 开启参数的梯度记录，以便计算损失对它的导数。

## 错了多少

把每条预测减去对应答案，得到 `[-3, -5, -7, -9]`。平方后取平均：

```python
loss = ((pred - y) ** 2).mean()
print(loss.item())
```

输出 `41.0`：

```text
(9 + 25 + 49 + 81) / 4 = 41
```

这就是均方误差。平方避免正负误差抵消，也加重了大误差的惩罚。

`.item()` 取出数值用于打印，求梯度仍使用原来的 `loss`。

## 梯度怎样给出方向

先只动 `w`，把它从 0 改成 0.01，保持 `b=0`。四个预测变成 `[0.01, 0.02, 0.03, 0.04]`，损失从 41 降到 40.65075。

`w` 增加 0.01，损失减少约 0.35。两者相除，变化率约为 −35。改动越小，这个变化率越接近当前位置的导数。

PyTorch 使用链式法则，沿着前面的运算直接计算导数：

```python
loss.backward()
print(w.grad.item())  # -35.0
print(b.grad.item())  # -12.0
print(w.item(), b.item())  # 0.0 0.0
```

**`backward()` 算出了梯度，参数仍然是零。**

−35 来自四条数据的贡献：

```text
w 的梯度 = (2/4) × Σ[(预测 − 答案) × 输入]
          = 0.5 × [(-3)×1 + (-5)×2 + (-7)×3 + (-9)×4]
          = -35
```

2 来自平方求导，除以 4 来自取平均。乘以输入，是因为输入越大，`w` 的变化对预测影响越大。

`b` 对每条预测的影响都是一比一，所以它的梯度是：

```text
b 的梯度 = 0.5 × [(-3) + (-5) + (-7) + (-9)] = -12
```

负梯度说明，当前位置适当增大参数可以降低损失；改动过大则未必有效。

求导机制详见 [Automatic Differentiation with torch.autograd（使用 autograd 自动求导）](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

## 更新参数

学习率取 0.05，按“参数减去学习率乘梯度”更新：

```python
with torch.no_grad():
    w -= 0.05 * w.grad
    b -= 0.05 * b.grad
```

于是：

```text
w = 0 - 0.05 × (-35) = 1.75
b = 0 - 0.05 × (-12) = 0.60
```

`torch.no_grad()` 暂停记录这段更新操作，下一轮正常计算时仍可求导。[torch.no_grad 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)

用新参数重新计算：

```python
with torch.no_grad():
    new_pred = w * x + b
    new_loss = ((new_pred - y) ** 2).mean()

print(new_pred.tolist())
print(new_loss.item())
```

| x | 正确答案 | 更新前预测 | 更新后预测 |
|---|---|---|---|
| 1 | 3 | 0 | 2.35 |
| 2 | 5 | 0 | 4.10 |
| 3 | 7 | 0 | 5.85 |
| 4 | 9 | 0 | 7.60 |

损失从 41 降到约 1.12875。旧的 `pred` 不会随参数自动刷新，因此必须重新计算。

## 完整训练

重复上述过程 200 次。以下代码可独立运行：

```python
import torch

x = torch.tensor([1., 2., 3., 4.])
y = torch.tensor([3., 5., 7., 9.])
w = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)

for step in range(200):
    pred = w * x + b
    loss = ((pred - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= 0.05 * w.grad
        b -= 0.05 * b.grad

    w.grad = None
    b.grad = None

with torch.no_grad():
    final_loss = ((w * x + b - y) ** 2).mean()
    answer = w * 5 + b

print(f'w={w.item():.6f}, b={b.item():.6f}')
print(f'loss={final_loss.item():.6f}')
print(f'x=5 时，预测={answer.item():.6f}')
```

PyTorch 2.13.0、CPU 运行结果：

```text
w=2.004774, b=0.985965
loss=0.000033
x=5 时，预测=11.009834
```

PyTorch 默认累加梯度。循环末尾的 `grad = None` 清掉旧值，让下一轮只使用新算出的梯度。

参数已接近 2 和 1，对未参与训练的输入 5，预测约为 11.01。这组人为数据恰好符合直线模型；现实任务中的预测能力，还需独立数据检验。

## 两个对照实验

**只求梯度，不更新参数。** 删掉更新 `w`、`b` 的两行，即使运行 200 次，参数仍为零，损失仍为 41。常规训练代码中的 `optimizer.step()` 承担参数更新的工作。

**把学习率改成 1。** 恢复参数更新，仅训练 5 轮，损失就达到约 **3.73 × 10¹³**。虽然沿着负梯度方向移动，但步子过大，反复越过较好的区域，训练开始发散。

## 从两个参数到矩阵

Transformer 里的 Q、K、V 投影矩阵，也包含需要训练的参数。它们参与前向计算，最终影响损失；反向传播沿着运算关系计算各个参数的梯度，优化器再使用梯度更新参数。
