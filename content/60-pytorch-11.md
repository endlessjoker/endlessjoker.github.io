# 打乱顺序和最后一个小批次

看得懂的 PyTorch · 第 11 讲

五条数据，每批两条，会剩下一条。它应该丢掉吗？三个批次的损失能直接取平均吗？

这两个选择都会改变训练或统计的含义。

## 数据顺序为什么会影响结果

沿用 `y=2x`，这次输入为 1 到 5，初始参数仍为零，学习率为 0.01。

按顺序分批：

```text
[1,2] → 更新 → [3,4] → 更新 → [5] → 更新
```

倒过来分批：

```text
[5,4] → 更新 → [3,2] → 更新 → [1] → 更新
```

第一种顺序的第一步得到 `w=0.1、b=0.06`。第二种顺序先看到目标 10、8：

```text
w 的梯度 = (−10)×5 + (−8)×4 = −82
b 的梯度 = −10−8 = −18

更新后 w=0.82，b=0.18
```

后续批次从不同的位置继续，结果自然会不同。下面用完全相同的模型和数据，只改变顺序：

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


def one_epoch(order):
    loader = DataLoader(TensorDataset(X[order], y[order]),
                        batch_size=2, shuffle=False)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xb) - yb).square().mean()
        loss.backward()
        optimizer.step()
    return model.weight.item(), model.bias.item()


for name, order in [("forward", [0, 1, 2, 3, 4]),
                    ("reverse", [4, 3, 2, 1, 0])]:
    w, b = one_epoch(order)
    print(f"{name}: w={w:.6f}, b={b:.6f}")
```

输出：

```text
forward: w=1.266220, b=0.330884
reverse: w=0.980404, b=0.251404
```

这里只证明顺序会改变训练轨迹，不能据此判定哪一种顺序长期更好。

## shuffle 打乱的是样本配对

对可以独立取用的训练样本，常见设置是 `shuffle=True`，让每轮重新排列样本，减少固定排序让相似样本连续出现的情况：

```python
generator = torch.Generator().manual_seed(42)
loader = DataLoader(
    TensorDataset(X, y),
    batch_size=2,
    shuffle=True,
    generator=generator,
    num_workers=0,
)

for epoch in range(2):
    batches = [xb.squeeze(1).tolist() for xb, yb in loader]
    print("epoch", epoch + 1, batches)
```

PyTorch 2.13.0、CPU 的本次输出：

```text
epoch 1 [[2.0, 4.0], [3.0, 5.0], [1.0]]
epoch 2 [[1.0, 4.0], [3.0, 2.0], [5.0]]
```

输入和答案按同一个索引取出，不会各自打乱后配错。随机数生成器的状态会向前推进，因此每轮通常得到不同排列。

种子在循环外设置，能帮助复现实验；如果每轮都重新创建同种子的生成器，排列就可能反复重复。种子也不是跨设备、跨版本完全一致的保证。

第九讲的数据划分发生在这之前：先留出验证集和测试集，再打乱训练集。不能为了打乱而把三份数据重新混在一起。

时间序列还要区分“窗口之间”和“窗口内部”。若每个窗口独立训练，窗口之间可以根据任务需要打乱，但窗口内时间顺序应保留；若模型跨批次传递隐状态，则需维护相应顺序，不能直接照搬此设置。

## 最后一个批次只有一条

查看 `drop_last` 的效果：

```python
dataset = TensorDataset(X, y)
for drop in (False, True):
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        drop_last=drop)
    sizes = [len(xb) for xb, yb in loader]
    print("drop_last:", drop, "sizes:", sizes)
```

输出：

```text
drop_last: False sizes: [2, 2, 1]
drop_last: True sizes: [2, 2]
```

默认 `False` 保留最后一批。本例的 `Linear` 能正常处理 `[1,1]` 输入，不需要为了凑整丢掉它。

`True` 会丢弃不满批量的部分。若不打乱，每轮都会漏掉同一条；打乱后，漏掉的样本可能变化，但这一轮仍没用完所有数据。只有训练方法或某些层确实要求相应批量条件时，才应有意识地选择它。

## 三个平均数不能随便再平均

固定模型为 `w=0、b=0`，此时五条数据的平方误差为：

```text
[4, 16, 36, 64, 100]
```

分成大小为 2、2、1 的三个批次，各批平均损失为：

```text
第一批：(4+16)/2 = 10
第二批：(36+64)/2 = 50
第三批：100/1 = 100
```

直接平均这三个数，得到 `(10+50+100)/3≈53.3333`。但五条样本真正的平均误差是：

```text
(4+16+36+64+100)/5 = 44
```

直接平均批次均值，会让最后一条样本独占三分之一的权重，其他样本各占六分之一。正确算法应按批次样本数加权：

```text
(10×2 + 50×2 + 100×1) / 5 = 44
```

对应代码：

```python
model = make_model()
loader = DataLoader(dataset, batch_size=2, shuffle=False)
model.eval()
total_squared_error = 0.0
count = 0
batch_means = []

with torch.no_grad():
    for xb, yb in loader:
        errors = (model(xb) - yb).square()
        total_squared_error += errors.sum().item()
        count += errors.numel()
        batch_means.append(errors.mean().item())

print(f"wrong: {sum(batch_means) / len(batch_means):.6f}")
print(f"right: {total_squared_error / count:.6f}")
```

输出分别为 `53.333333` 和 `44.000000`。

这里按误差元素数计数，对应所有输出元素的均方误差；每条样本只有一个输出，所以也等于按样本平均。变长序列若有填充位置，还需要按有效位置统计，不能把填充值当真实目标。

这段评估全程没有更新参数。如果在训练循环里累计损失，即使加权正确，得到的也是模型沿途表现的汇总；要看本轮结束时的模型表现，应固定最终参数再评估一次。

参数定义见 [torch.utils.data.DataLoader：批量大小、打乱与丢弃尾批](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。
