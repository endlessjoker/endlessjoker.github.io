# 显存不够时 怎样累积梯度

看得懂的 PyTorch · 第 12 讲

想用五条样本算一次更新，但一次只能处理两条，可以先后计算三小批的梯度，最后统一更新。

关键有两点：中途不修改参数，每条样本在最终平均损失里的权重要相同。

## 与普通小批量训练差在哪里

第十讲每批都更新一次：

```text
清梯度 → 第一批反传 → 更新
清梯度 → 第二批反传 → 更新
清梯度 → 第三批反传 → 更新
```

梯度累积则是：

```text
清梯度 → 第一批反传 → 第二批反传 → 第三批反传 → 更新
```

PyTorch 会把新的梯度加到参数已有的 `.grad` 上。平时需要清掉旧值，这次恰好利用这个累加行为。

所有小批次在同一组参数上计算，才有机会还原大批次的梯度。若中间调用 `step()`，计算位置变了，就不再是同一次大批量更新。

## 先算五条数据的标准答案

仍用输入 1 到 5、目标 `y=2x`，初始 `w=0、b=0`，学习率为 0.01。

整体均方误差为 44，梯度为：

```text
w 的梯度 = (2/5)×[−2×1−4×2−6×3−8×4−10×5]
         = (2/5)×(−110) = −44

b 的梯度 = (2/5)×[−2−4−6−8−10] = −12
```

更新后应得到 `w=0.44、b=0.12`。

## 每小批贡献多少梯度

把整体损失拆成三项：

```text
L = (e₁²+e₂²)/5 + (e₃²+e₄²)/5 + e₅²/5
```

每项都除以整个累积窗口的样本数 5，不除以本小批的样本数。按这三项分别反传：

| 输入 | 对 w 的梯度贡献 | 对 b 的梯度贡献 |
|---|---:|---:|
| `[1,2]` | −4 | −2.4 |
| `[3,4]` | −20 | −5.6 |
| `[5]` | −20 | −4 |
| 合计 | −44 | −12 |

例如第一批：

```text
对 w 的贡献 = (2/5)×[−2×1−4×2] = −4
```

梯度对相加运算可分解，因此三次反向传播的结果相加，等于一次对总损失反传。

## 代码核对两种做法

以下代码可以独立运行。先完成一次真正的全批量更新，再从相同初值做梯度累积：

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.arange(1., 6., dtype=torch.float64).reshape(-1, 1)
y = 2 * X


def make_model():
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


full = make_model()
full_optimizer = torch.optim.SGD(full.parameters(), lr=0.01)
full_optimizer.zero_grad(set_to_none=True)
full_loss = (full(X) - y).square().mean()
full_loss.backward()
full_optimizer.step()

accum = make_model()
accum_optimizer = torch.optim.SGD(accum.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X, y), batch_size=2,
                    shuffle=False)
total_elements = y.numel()

accum_optimizer.zero_grad(set_to_none=True)
for xb, yb in loader:
    contribution = (accum(xb) - yb).square().sum() / total_elements
    contribution.backward()
    print(f"grad_w={accum.weight.grad.item():.1f}"
          f"  grad_b={accum.bias.grad.item():.1f}")

accum_optimizer.step()

for name, model in [("full", full), ("accum", accum)]:
    print(f"{name}: w={model.weight.item():.6f}"
          f"  b={model.bias.item():.6f}")

torch.testing.assert_close(full.weight, accum.weight)
torch.testing.assert_close(full.bias, accum.bias)
```

输出：

```text
grad_w=−4.0   grad_b=−2.4
grad_w=−24.0  grad_b=−8.0
grad_w=−44.0  grad_b=−12.0
full: w=0.440000  b=0.120000
accum: w=0.440000  b=0.120000
```

这里打印的是已经累积的梯度，所以第二行是前两批之和。

每小批都单独前向并立即反传，计算图不同，参数相同。默认情况下，反传后用于该次求导的中间缓存可以释放，无需 `retain_graph=True`。若先把多小批的 Tensor 损失全加起来，最后才反传，就会保留多份计算图，削弱节省激活内存的效果。

梯度累积主要减少同时保存的激活，不会消除模型参数、参数梯度和优化器状态本身的内存开销。

## 为什么不能一律把 loss 除以三

若三个小批同样大，可以将各批的平均损失除以 3，再分别反传。但这里批量大小是 2、2、1：

```text
错误组合 = (第一批均值 + 第二批均值 + 第三批均值)/3
```

最后一条样本被赋予了更高权重。各批未加权的均值梯度分别是：

```text
w：[−10, −50, −100]
b：[ −6, −14,  −20]
```

简单除以 3 再相加，得到 `w` 梯度约 −53.3333，`b` 梯度约 −13.3333。更新后变成 `w≈0.533333、b≈0.133333`，已经不是全批量的结果。

如果现有损失函数返回的是批次均值，本例应使用：

```text
加权损失 = 批次均值 × 本批样本数 / 累积窗口总样本数
```

代码选择“平方误差求和，再除以窗口内总元素数”，直接避免尾批权重不一致的问题。

## 多个累积窗口怎样训练

真实数据通常远不止五条，可以每三小批累积一次。最后一个窗口可能只有一小批，也要使用它实际包含的数据量。

下面重新初始化模型。每小批两条，每两小批组成一个累积窗口；五条数据因此分成 `[2,2]` 和 `[1]` 两个窗口：

```python
from itertools import islice

model = make_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
iterator = iter(loader)
updates = 0

while True:
    window = list(islice(iterator, 2))
    if not window:
        break

    window_elements = sum(yb.numel() for xb, yb in window)
    optimizer.zero_grad(set_to_none=True)
    for xb, yb in window:
        loss_part = (model(xb) - yb).square().sum() / window_elements
        loss_part.backward()
    optimizer.step()
    updates += 1
    print(f"update={updates}  elements={window_elements}"
          f"  w={model.weight.item():.6f}  b={model.bias.item():.6f}")
```

PyTorch 2.13.0、CPU 输出：

```text
update=1  elements=4  w=0.300000  b=0.100000
update=2  elements=1  w=1.140000  b=0.268000
```

第一窗口有四条样本，从零参数更新到 `w=0.3、b=0.1`。最后窗口只有 `x=5`：

```text
预测 = 0.3×5+0.1 = 1.6，误差 = 1.6−10 = −8.4
梯度：w 为 2×(−8.4)×5=−84，b 为 2×(−8.4)=−16.8
更新：w=1.14，b=0.268
```

这次共更新两次，当然不等于前面五条样本只更新一次的结果。累积窗口确定了每次更新覆盖哪些样本。

`window` 暂存的是输入数据，不是多份计算图。本例全在 CPU；用于 GPU 时，可以将窗口留在 CPU，每次只搬运当前小批。

## 什么时候不能视为同一个大批次

本例的 Linear 对每条样本独立计算，损失可按样本相加，所以能对齐；浮点运算顺序不同，实际比较仍应允许微小误差。

含 BatchNorm 的网络会在训练时使用当前小批的统计量，分批后的前向结果可能已经不同。对比学习中若损失使用批内其他样本作为负样本，拆批也改变了损失本身。随机层还可能使两次运行无法逐值一致。

梯度累积能组合多个小批的梯度，但不会自动恢复拆批时丢失的样本间计算。
