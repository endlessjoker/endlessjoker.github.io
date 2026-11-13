# 两层网络之间为什么需要激活函数

看得懂的 PyTorch · 第 7 讲

把两个 `Linear` 层接起来，参数更多了，能表示的函数却未必更复杂。如果中间没有激活函数，两层仍能合并成一次线性变换。

一个简单任务就能看出区别：输入 −2、−1、0、1、2，输出它们的绝对值。

## 两层为什么还能合成一层

先看一个输入、两个隐藏单元、一个输出的网络：

```text
第一层：h₁ = 2x + 1
        h₂ = −x + 2

第二层：预测 = 3h₁ + 4h₂ − 5
```

把第一层代入第二层：

```text
预测 = 3(2x + 1) + 4(−x + 2) − 5
     = 6x + 3 − 4x + 8 − 5
     = 2x + 6
```

两层计算最终还是一条直线。用 PyTorch 的矩阵写法核对：

```python
import torch

X = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]])  # [5, 1]
W1 = torch.tensor([[2., -1.]])                       # [1, 2]
b1 = torch.tensor([1., 2.])                         # [2]
W2 = torch.tensor([[3.], [4.]])                      # [2, 1]
b2 = torch.tensor([-5.])                            # [1]

hidden = X @ W1 + b1                                # [5, 2]
pred = hidden @ W2 + b2                             # [5, 1]
print(pred.squeeze(1).tolist())                     # [2, 4, 6, 8, 10]
print(torch.allclose(pred, 2 * X + 6))               # True
```

一般情况下也一样：

```text
H = XW₁ + b₁
Y = HW₂ + b₂
  = X(W₁W₂) + (b₁W₂ + b₂)
```

`W₁W₂` 可以合成一个权重矩阵，`b₁W₂+b₂` 可以合成一个偏置。这里带偏置的变换，严格说叫仿射变换，PyTorch 的 `Linear` 层采用的就是这种形式。

增加隐藏单元、继续叠加 `Linear`，都不会改变上述结论。分层会改变参数组织与训练过程，但不会凭空得到非线性的表达能力。

## 一条直线做不到什么

绝对值的目标是一个 V 形：

| x | −2 | −1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| 目标 | 2 | 1 | 0 | 1 | 2 |

若直线 `ax+b` 同时经过 `(−1,1)` 和 `(1,1)`，必有 `a=0、b=1`。它在 `x=0` 处只能输出 1，无法同时满足目标 0。

对这五个样本，任何直线都无法达到零误差。由于输入和目标左右对称，最优斜率为 0，截距等于目标均值 1.2。最低均方误差是：

```text
[(1.2−2)² + (1.2−1)² + (1.2−0)² + (1.2−1)² + (1.2−2)²] / 5
= (0.64 + 0.04 + 1.44 + 0.04 + 0.64) / 5
= 0.56
```

这次训练的限制来自模型结构。只调整学习率，无法突破这条直线的表达范围。

## ReLU 在哪里改变了计算

ReLU 的规则很简单：正数保留，负数变成零。

```text
ReLU(z) = max(0, z)
```

它逐元素处理 Tensor，不改变形状。[ReLU 官方定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html)

构造两个隐藏单元，分别接收 `x` 和 `−x`，经过 ReLU 后相加：

```text
h₁ = ReLU(x)
h₂ = ReLU(−x)
预测 = h₁ + h₂
```

| x | ReLU(x) | ReLU(−x) | 相加 |
|---|---|---|---|
| −2 | 0 | 2 | 2 |
| −1 | 0 | 1 | 1 |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 |
| 2 | 2 | 0 | 2 |

输入为正时，第一条路径输出 `x`；输入为负时，第二条路径输出 `−x`。两条路径拼出了绝对值。

```python
W1 = torch.tensor([[1., -1.]])  # [1, 2]
W2 = torch.tensor([[1.], [1.]]) # [2, 1]

z = X @ W1                    # [5, 2]，激活前
hidden = torch.relu(z)         # [5, 2]，激活后
pred = hidden @ W2             # [5, 1]
target = X.abs()

print(pred.squeeze(1).tolist())              # [2, 1, 0, 1, 2]
print(((pred - target) ** 2).mean().item())   # 0.0
print((z @ W2).squeeze(1).tolist())           # [0, 0, 0, 0, 0]
```

最后一行跳过 ReLU，`x` 与 `−x` 就会完全抵消。加上 ReLU 后，哪些路径保留输出取决于输入，整条函数因此无法再合并成一条直线。

这里的参数是人为构造的，用于证明网络能表示绝对值，并没有进行训练。

## 用 Linear 层写出同一个网络

`Linear` 的权重按 `[输出数, 输入数]` 保存，因此第一层的权重形状为 `[2,1]`，与上面手写的 `W1` 互为转置。

```python
model = torch.nn.Sequential(
    torch.nn.Linear(1, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(2, 1),
)

with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[1.], [-1.]]))
    model[0].bias.zero_()
    model[2].weight.copy_(torch.tensor([[1., 1.]]))
    model[2].bias.zero_()

print(model(X).detach().squeeze(1).tolist())  # [2, 1, 0, 1, 2]
```

`Sequential` 按顺序把前一层输出交给后一层。ReLU 自身没有可训练参数，训练更新的仍是两个 `Linear` 层的权重和偏置。

网络里有两层带参数的变换；夹在中间的激活函数改变了它们组合起来能表示的函数。

## 梯度怎样穿过 ReLU

ReLU 在正数处的导数为 1，负数处为 0。零点不可导，PyTorch 在这里采用零梯度。

使用刚才的网络，输入 −2，但将这次目标设为 3，制造一个非零损失：

```python
model.zero_grad(set_to_none=True)
sample = torch.tensor([[-2.]])

z = model[0](sample)
z.retain_grad()
hidden = model[1](z)
prediction = model[2](hidden)
loss = (prediction - 3.).square().mean()
loss.backward()

print(z.detach().tolist())           # [[-2.0, 2.0]]
print(hidden.detach().tolist())      # [[0.0, 2.0]]
print(z.grad.tolist())               # [[0.0, -2.0]]
print(model[0].weight.grad.tolist()) # [[0.0], [4.0]]
```

预测为 2，误差为 −1，损失对预测的导数为 −2。第二层两个权重均为 1，因此两条隐藏路径收到的梯度都是 −2。

经过 ReLU 反传时，第一条路径乘以 0，梯度被截断；第二条乘以 1，保留 −2。再乘输入 −2，第一层两个权重的梯度分别为：

```text
第一条：(-2) × 1 × 0 × (-2) = 0
第二条：(-2) × 1 × 1 × (-2) = 4
```

负数被截掉也有代价：如果某个单元对所有训练样本都落在负区间，在普通梯度训练下，它的输入权重和偏置可能一直得不到更新。这是 ReLU 单元失活的问题。

激活函数增加了模型的表达能力，同时改变了梯度流动的方式。网络能够表示某个函数，与训练能否找到合适参数，是两个需要分别检验的问题。
