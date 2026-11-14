# 分类模型怎样把分数变成概率

看得懂的 PyTorch · 第 13 讲

回归模型输出一个数，例如价格。分类模型也输出数，但这些数先表示各个类别的得分。

假设一张图片只能属于猫、狗、兔三类，模型输出 `[2,1,0]`。这三个数不是概率：它们没有落在 0 到 1 之间，总和也不是 1。

这种未经概率归一化的输出叫 **logits**。

## 三类需要三个输出

先不处理图片，假设前面的网络已经提取出两个特征 `x=[1,2]`。分类层为 `Linear(2,3)`：两项输入，三项得分。

```python
import torch
from torch import nn

x = torch.tensor([[1., 2.]], dtype=torch.float64)
classifier = nn.Linear(2, 3).double()

with torch.no_grad():
    classifier.weight.copy_(torch.tensor([[2., 0.], [1., 0.], [0., 0.]]))
    classifier.bias.zero_()

logits = classifier(x)
print(logits.detach().tolist())
```

输出 `[[2.0,1.0,0.0]]`。三个类别各有一组权重和一个偏置：

```text
猫：2×1 + 0×2 + 0 = 2
狗：1×1 + 0×2 + 0 = 1
兔：0×1 + 0×2 + 0 = 0
```

这些权重是为了核对计算而手工设置的，尚未训练。形状变化为 `[1,2] → [1,3]`；第一维仍是样本数，最后一维现在是类别数。

## softmax 怎样计算

softmax 先对得分取指数，再除以所有类别的指数之和：

```text
pᵢ = exp(zᵢ) / Σexp(zⱼ)
```

对 `[2,1,0]`：

```text
指数 = [7.3891, 2.7183, 1]
总和 = 11.1073
概率 ≈ [0.665241, 0.244728, 0.090031]
```

对应代码：

```python
probabilities = logits.softmax(dim=1)
print([round(v, 6) for v in probabilities[0].tolist()])
print(probabilities.sum(dim=1).tolist())
```

概率非负，每条样本的三项概率相加为 1。这里的 `dim=1` 指类别维度；不能沿样本维度归一化，把不同图片的得分混在一起。

取指数不仅让值变成正数，还让得分差具有明确含义：

```text
p猫 / p狗 = exp(2) / exp(1) = exp(1) ≈ 2.718
```

因此两类概率之比由它们的得分差决定。

## 为什么所有得分一起加一百，概率不变

将 logits 从 `[2,1,0]` 改成 `[102,101,100]`，各项指数都乘上 `exp(100)`，分子分母中的这个因子会约掉。

```python
shifted = (logits + 100).softmax(dim=1)
print(torch.allclose(probabilities, shifted))  # True
```

这也提供了稳定计算的方法：先减去最大得分，再取指数。

```python
z = logits.detach()
shift = z - z.max(dim=1, keepdim=True).values
exp_values = shift.exp()
manual = exp_values / exp_values.sum(dim=1, keepdim=True)
print(torch.allclose(manual, probabilities))  # True
```

`[2,1,0]` 变成 `[0,−1,−2]`，概率不变，却不必计算很大的正指数。实际代码直接调用 `softmax`，手写过程只用于理解。

但把所有得分**乘以**一个数，效果不同。`[4,2,0]` 的最大类概率约为 0.866813，高于原来的 0.665241。它拉大了差距。

```python
print([round(v, 6) for v in (2 * logits).softmax(dim=1)[0].tolist()])
# [0.866813, 0.11731, 0.015876]
```

## 选类别时不必先算概率

softmax 保持得分的大小顺序。只想选最高分的类别，可以直接用 `argmax`：

```python
names = ["猫", "狗", "兔"]
index = logits.argmax(dim=1).item()
print(index, names[index])  # 0 猫
```

这次类别编号为 0，并不表示“猫的数值等于零”。编号只是名称与输出位置之间的约定，训练和预测时必须保持一致。

最大概率 0.665241 也不能直接解释为“这次有经过验证的 66.5% 正确率”。它是模型根据当前参数分配的概率；是否与真实正确率吻合，需要在独立数据上检查校准情况。

## 一张图可以同时有猫和狗怎么办

上面的 softmax 假设三类互斥，概率需要竞争同一份总量。若一张图可以同时含猫、狗和兔，则应为各标签分别建模，常见做法是每项用 sigmoid，而不是强制三项相加为 1。

二分类也有两种常见接口：

| 任务写法 | 模型输出 | 训练损失 |
|---|---|---|
| 两个互斥类别 | 两个 logits | `CrossEntropyLoss` |
| 一个“是否为正类”的判断 | 一个 logit | `BCEWithLogitsLoss` |

对两个 logits，正类概率可以改写为：

```text
p₁ = exp(z₁) / [exp(z₀)+exp(z₁)]
   = 1 / [1+exp(−(z₁−z₀))]
   = sigmoid(z₁−z₀)
```

两类概率只依赖得分差。两种接口能表达相同形式的概率，但标签形状和损失函数不能随意混用。后面的例子统一采用两个 logits 和整数类别编号。
