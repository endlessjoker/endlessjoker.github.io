# 梯度怎样告诉参数往哪边改

看得懂的 PyTorch · 第 4 讲

一套房屋预测 90 万元，实际为 95 万元。预测偏低，增加面积权重、房间权重或偏置，都能把结果抬高。但同样增加 0.01，三个参数产生的效果不同。

梯度衡量的是：其他参数不变时，某个参数的微小变化会让损失改变多少。

## 从一套房屋开始

面积为 6，房间数为 2，沿用前两讲的参数：

```python
import torch

w = torch.tensor([10., 5.], dtype=torch.float64, requires_grad=True)
b = torch.tensor(20., dtype=torch.float64, requires_grad=True)
x = torch.tensor([6., 2.], dtype=torch.float64)

pred = x @ w + b
error = pred - 95.
loss = error.square()

print(pred.item(), error.item(), loss.item())  # 90.0 -5.0 25.0
```

这里只算一条样本，不需要再取平均。使用 `float64` 是为了稍后用微小改动核对梯度。

前向计算的路径是：

```text
面积权重 w[0] ──乘以 6──┐
房间权重 w[1] ──乘以 2──┼─相加─预测 90─减去 95─误差 -5─平方─损失 25
偏置 b ────────────────┘
```

从参数到损失，中间经过了几次运算。链式法则沿着这条路径，将各段的变化率相乘。

## 把三段变化率乘起来

以面积权重为例：

1. 面积权重增加一点，预测改变它的 **6 倍**，因为面积是 6。
2. 预测改变一点，误差也改变同样多，变化率为 **1**。
3. 损失是误差的平方，变化率为 `2 × 误差`，当前位置是 **−10**。

因此：

```text
损失对面积权重的导数 = (-10) × 1 × 6 = -60
损失对房间权重的导数 = (-10) × 1 × 2 = -20
损失对偏置的导数     = (-10) × 1 × 1 = -10
```

这三个导数组成参数梯度。PyTorch 算出的结果相同：

```python
loss.backward()
print(w.grad.tolist())  # [-60.0, -20.0]
print(b.grad.item())    # -10.0
```

−60 表示，在当前位置，面积权重增加一个很小的量 `δ`，损失大约变化 `−60 × δ`。

若只把面积权重从 10 改为 10.01，预测变成 90.06，实际损失为：

```text
(90.06 − 95)² = 24.4036
```

损失减少 0.5964，与梯度估计的 `60 × 0.01 = 0.6` 接近。差距来自损失曲线的弯曲；梯度描述当前位置的变化率。

这里面积权重的梯度幅度较大，是因为面积数值较大。若面积换个计量单位，梯度也会改变，不能据此认定面积比房间数更“重要”。

## 三条样本的意见怎样合并

三套房屋共享同一组参数。第一套和第三套预测偏低，第二套预测偏高，它们对参数的修改方向存在冲突。

```python
X = torch.tensor([[6., 2.], [8., 3.], [10., 3.]], dtype=torch.float64)
y = torch.tensor([95., 110., 140.], dtype=torch.float64)
w = torch.tensor([10., 5.], dtype=torch.float64, requires_grad=True)
b = torch.tensor(20., dtype=torch.float64, requires_grad=True)

pred = X @ w + b
pred.retain_grad()
loss = ((pred - y) ** 2).mean()
loss.backward()

print([round(v, 4) for v in pred.grad.tolist()])
# [-3.3333, 3.3333, -3.3333]
print([round(v, 4) for v in w.grad.tolist()])
# [-26.6667, -6.6667]
print(round(b.grad.item(), 4))
# -3.3333
```

损失取三项平均，因此对每个预测的导数为 `2 × 误差 / 3`。这个信号再乘对应输入，传到共享参数：

| 样本 | 误差 | 对面积权重的梯度贡献 | 对房间权重的梯度贡献 | 对偏置的梯度贡献 |
|---|---|---|---|---|
| A | −5 | −60/3 | −20/3 | −10/3 |
| B | 5 | 80/3 | 30/3 | 10/3 |
| C | −5 | −100/3 | −30/3 | −10/3 |
| 合计 | | **−80/3** | **−20/3** | **−10/3** |

**同一路径上的导数相乘，共享参数从不同路径收到的贡献相加。** 这就是本例反向传播的两条计算规则。

矩阵形式可以直接核对：

```python
with torch.no_grad():
    g = 2 * (pred - y) / y.numel()  # 对三个预测的梯度，[3]
    manual_w = X.T @ g            # [2, 3] @ [3] → [2]
    manual_b = g.sum()            # 三个位置共用一个偏置

print(torch.allclose(manual_w, w.grad))  # True
print(torch.allclose(manual_b, b.grad))  # True
```

`numel()` 返回元素总数，这里是 3。`allclose()` 检查两个 Tensor 的数值是否在容许误差内一致。

## PyTorch 记录了什么

`w`、`b` 是直接创建的可求导 Tensor，在计算图中称为叶子。`pred` 是运算产生的中间结果。执行前向计算时，PyTorch 记录相关运算；调用 `backward()` 后，再沿运算关系使用链式法则计算梯度。[Automatic Differentiation with torch.autograd（使用 autograd 自动求导）](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

默认情况下，叶子的梯度保存在 `.grad` 中，中间结果的梯度不会保留。前面的 `pred.retain_grad()` 专门保留预测位置的梯度，便于观察。删掉这行，`w.grad` 和 `b.grad` 仍能正常算出。

单次反向传播中汇总多条路径的贡献，是链式法则的要求；多次调用 `backward()` 时继续累加到参数的 `.grad`，则是 PyTorch 的默认行为。常规训练在每次参数更新后清空旧梯度，避免把上一轮的结果混入下一轮。

## 不经过反向传播也能核对

梯度可以用有限差分近似验证：只让一个参数分别增加、减少 `eps`，比较两次损失。

```text
导数 ≈ [L(w + eps) − L(w − eps)] / (2 × eps)
```

下面保持其他参数不变，检查面积权重：

```python
eps = 1e-5

with torch.no_grad():
    w_plus = w.clone()
    w_minus = w.clone()
    w_plus[0] += eps
    w_minus[0] -= eps

    loss_plus = ((X @ w_plus + b - y) ** 2).mean()
    loss_minus = ((X @ w_minus + b - y) ** 2).mean()
    numerical = (loss_plus - loss_minus) / (2 * eps)

print(round(numerical.item(), 6))  # -26.666667
print(round(w.grad[0].item(), 6))  # -26.666667
```

有限差分直接比较损失值，适合核对小例子；对每个参数都做一次，需要大量重复计算。自动求导利用已知运算的导数，沿图反向计算，通常无须逐个扰动参数。`eps` 也并非越小越好，过小会使两次损失的差被浮点误差影响。

## 零梯度不等于零误差

假设模型只允许输出一个常数，两条样本的答案分别是 100 和 110。常数取 105 时，两条误差方向相反：

```python
c = torch.tensor(105., requires_grad=True)
targets = torch.tensor([100., 110.])
loss_constant = ((c - targets) ** 2).mean()
loss_constant.backward()

print(loss_constant.item())  # 25.0
print(c.grad.item())         # 0.0
```

两条梯度贡献分别为 5 和 −5，恰好抵消。105 已经是这个常数模型的最佳折中，但每条预测仍差 5。要分别预测正确，需要让模型依据输入产生不同输出，单靠继续更新同一个常数做不到。

在这个简单问题里，零梯度对应最小损失；更复杂的模型中，它也可能出现在鞍点等位置。梯度提供的是当前位置的变化信息，模型能否表达任务所需的关系，还取决于计算结构和输入。
