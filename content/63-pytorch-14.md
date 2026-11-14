# 交叉熵怎样把正确类别推上去

看得懂的 PyTorch · 第 14 讲

模型给猫、狗、兔的概率分别是 `0.665241、0.244728、0.090031`。如果正确答案是狗，损失应该是多少？

对于单个确定类别的标签，交叉熵取正确类别概率的负自然对数：

```text
L = −ln(p正确类别)
```

这里得到 `−ln(0.244728)≈1.407606`。

## 为什么要取负对数

| 正确类别概率 | 损失 −ln(p) |
|---:|---:|
| 0.9 | 0.105361 |
| 0.5 | 0.693147 |
| 0.1 | 2.302585 |
| 0.01 | 4.605170 |

正确类别的概率越低，惩罚越大。把概率从 0.01 提高到 0.1，与从 0.1 提高到 1，都会使损失减少 `ln(10)`。

对独立样本，最大化正确标签的联合概率等价于最大化各概率的乘积。取对数把乘积变成求和，再取负号，就变成最小化各条负对数概率之和。实际训练常再取平均，使数值规模不直接随批量大小增长。

## CrossEntropyLoss 接收 logits

```python
import torch
from torch import nn

logits = torch.tensor([[2., 1., 0.]], dtype=torch.float64,
                      requires_grad=True)
target = torch.tensor([1], dtype=torch.long)  # 狗的编号
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
print(f"loss={loss.item():.6f}")  # 1.407606
```

本例的接口为：

```text
logits：[样本数, 类别数]，浮点数
标签：  [样本数]，整数编号，类型为 torch.long
```

模型输出三个得分，标签只需指出正确的是哪一个，不能把标签误写成 `[[1.]]`。

`CrossEntropyLoss` 接收未归一化的得分。对本例这种整数标签，它等价于先计算 `log_softmax`，再取正确类别对应值的负数；不要在传入前重复做 softmax。接口定义见 [CrossEntropyLoss：输入、目标与损失公式](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

可以这样核对：

```python
manual_loss = -logits.log_softmax(dim=1)[0, target.item()]
print(torch.allclose(loss, manual_loss))  # True
```

这里用 `log_softmax` 直接计算对数概率，避免先得到极小概率再取对数。

## 梯度为什么是“概率减答案”

把正确类别编号记为 t，交叉熵可以写成：

```text
L = −ln[exp(zₜ) / Σexp(zⱼ)]
  = −zₜ + ln Σexp(zⱼ)
```

对某一项得分 zᵢ 求导：

```text
∂L/∂zᵢ = −1{i=t} + exp(zᵢ)/Σexp(zⱼ)
        = pᵢ − 1{i=t}
```

`1{i=t}` 在正确类别处为 1，其他位置为 0。本例正确类别是狗，标签对应的向量为 `[0,1,0]`：

```text
梯度 = [0.665241, 0.244728, 0.090031] − [0,1,0]
     = [0.665241, −0.755272, 0.090031]
```

```python
loss.backward()
print([round(v, 6) for v in logits.grad[0].tolist()])
# [0.665241, -0.755272, 0.090031]
```

若直接对这些得分做梯度下降，狗的得分会升高，其他得分会降低。错误类别中，概率最高的猫收到更大的压低信号。

真实训练更新的是网络参数，得分如何变化还取决于前面的计算。不能把得分梯度直接当成每个权重的梯度。

## 把梯度传到 Linear 的六个参数

用一个独立的小例子算完整更新：两个输入特征 `[2,1]`，两个类别，正确类别编号为 1。把分类层的四个权重和两个偏置全部设为零。

```python
model = nn.Linear(2, 2).double()
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

x = torch.tensor([[2., 1.]], dtype=torch.float64)
y = torch.tensor([1], dtype=torch.long)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad(set_to_none=True)
scores = model(x)
loss = criterion(scores, y)
loss.backward()

print(model.weight.grad.tolist())
print(model.bias.grad.tolist())
```

得分都是 0，概率都是 0.5，因此得分梯度为 `[0.5,−0.5]`。每个权重的梯度还需乘上相应输入：

```text
类别 0：0.5 × [2,1]  = [1,0.5]
类别 1：−0.5 × [2,1] = [−1,−0.5]
偏置梯度：[0.5,−0.5]
```

代码正好打印上述两行权重梯度和偏置梯度。执行更新：

```python
optimizer.step()
with torch.no_grad():
    new_scores = model(x)
    new_probs = new_scores.softmax(dim=1)
    new_loss = criterion(new_scores, y)

print([[round(v, 6) for v in row] for row in new_scores.tolist()])
print([[round(v, 6) for v in row] for row in new_probs.tolist()])
print(f"loss={new_loss.item():.6f}")
```

手算新参数与新得分：

```text
W = [[−0.1,−0.05], [0.1,0.05]]
b = [−0.05,0.05]

类别 0：−0.1×2−0.05×1−0.05 = −0.3
类别 1： 0.1×2+0.05×1+0.05 =  0.3
```

输出：

```text
[[-0.3, 0.3]]
[[0.354344, 0.645656]]
loss=0.437488
```

正确类别概率从 0.5 升到约 0.6457，损失从 `ln(2)≈0.693147` 降到约 0.437488。

## 多条样本怎样合并

默认没有类别权重、忽略标签或标签平滑时，批次交叉熵是各样本损失的平均。若批量大小为 N，每条样本的得分梯度还要除以 N：

```text
第 n 条样本的梯度 = (pₙ − 标签向量ₙ) / N
```

同一层参数接收所有样本的贡献并相加，因此某条样本希望抬高的得分，可能受到其他样本相反方向的影响。

`argmax` 只保留赢家编号，微小改变得分通常不会改变这个整数，无法提供上述连续的训练信号。它适合统计预测类别；计算交叉熵时应保留全部 logits。
