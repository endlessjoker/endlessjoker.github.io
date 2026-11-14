# 数据分成小批以后 怎样训练

看得懂的 PyTorch · 第 10 讲

四条数据一次算完，只更新一次参数。分成两批，每批算完就更新，则会更新两次，而且第二批看到的已经是新参数。

这就是小批量训练最容易忽略的变化：分批既改变一次处理多少数据，也改变参数更新的节奏。

## 先把四条数据分成两批

任务仍很简单：输入 `1、2、3、4`，目标为 `2、4、6、8`，模型为 `预测=w×x+b`。

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([[1.], [2.], [3.], [4.]], dtype=torch.float64)
y = 2 * X

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=False,
                    num_workers=0)

for xb, yb in loader:
    print(xb.squeeze(1).tolist(), yb.squeeze(1).tolist())
```

输出：

```text
[1.0, 2.0] [2.0, 4.0]
[3.0, 4.0] [6.0, 8.0]
```

`TensorDataset` 按相同索引把输入和答案配成一条样本。`DataLoader` 每次取两条，分别堆成输入和目标 Tensor。

完整输入的形状为 `[4,1]`，每批输入为 `[2,1]`。这里 `num_workers=0`，数据读取在当前进程进行，便于直接运行和排查问题。

## 第一批怎样更新参数

先设 `w=0、b=0`，学习率为 0.01。第一批输入 1、2，目标 2、4：

```text
预测 = [0, 0]
误差 = [−2, −4]
损失 = (4+16)/2 = 10
```

对两条样本取平均，梯度为：

```text
w 的梯度 = (2/2) × [(−2)×1 + (−4)×2] = −10
b 的梯度 = (2/2) × [(−2)+(−4)] = −6
```

更新后：

```text
w = 0 − 0.01×(−10) = 0.1
b = 0 − 0.01×(−6) = 0.06
```

## 第二批已经站在新位置上

第二批输入 3、4，目标 6、8。用刚得到的参数计算：

```text
预测 = [0.1×3+0.06, 0.1×4+0.06] = [0.36, 0.46]
误差 = [−5.64, −7.54]

w 的梯度 = (−5.64)×3 + (−7.54)×4 = −47.08
b 的梯度 = −5.64−7.54 = −13.18
```

再次更新：

```text
w = 0.1  − 0.01×(−47.08) = 0.5708
b = 0.06 − 0.01×(−13.18) = 0.1918
```

如果四条数据一次算完，从零参数出发的梯度却是：

```text
w 的梯度 = (2/4)×[−2×1−4×2−6×3−8×4] = −30
b 的梯度 = (2/4)×[−2−4−6−8] = −10
```

一次更新后得到 `w=0.3、b=0.1`，与分两批的结果不同。原因是更新次数不同，第二批的梯度计算位置也不同。

## 代码里的更新放在哪里

接着运行下面代码，核对刚才的两步：

```python
def make_model():
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


model = make_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()
for step, (xb, yb) in enumerate(loader, start=1):
    optimizer.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = (pred - yb).square().mean()
    loss.backward()
    optimizer.step()
    print(f"step={step}  w={model.weight.item():.4f}"
          f"  b={model.bias.item():.4f}")
```

输出：

```text
step=1  w=0.1000  b=0.0600
step=2  w=0.5708  b=0.1918
```

普通小批量训练中，`zero_grad()` 和 `step()` 都在批次循环内部。每批清掉上一批的梯度，用本批的误差更新一次参数。

如果只在整轮开头清梯度，却仍然每批更新，旧梯度会混入后面的更新。这与上面手算的算法就不同了。

## epoch 和 step 分别数什么

这里完整遍历四条数据叫一个 **epoch**；优化器更新一次叫一个 **step**。

| 批量大小 | 每轮批次数 | 每轮更新次数 |
|---|---:|---:|
| 4 | 1 | 1 |
| 2 | 2 | 2 |
| 1 | 4 | 4 |

因此同样训练 100 轮，不同批量大小的更新次数可能不同。比较实验时，只说“训练轮数相同”还不够。

重新初始化模型，训练 1000 轮：

```python
model = make_model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
steps = 0

for epoch in range(1000):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xb) - yb).square().mean()
        loss.backward()
        optimizer.step()
        steps += 1

model.eval()
with torch.no_grad():
    predictions = model(X)
    final_loss = (predictions - y).square().mean()

print("steps:", steps)
print(f"w={model.weight.item():.6f}  b={model.bias.item():.6f}")
print(f"loss={final_loss.item():.9f}")
```

PyTorch 2.13.0、CPU 运行结果：

```text
steps: 2000
w=1.999602  b=0.001220
loss=0.000000249
```

1000 轮完成了 2000 次更新。最后评估重新用训练结束后的同一组参数计算全部数据，不把训练途中不同参数位置的损失混在一起。

## 为什么需要分批

大网络训练时，需要为反向传播保存中间激活。一次装入的数据越多，这部分内存通常越大；分批可以控制一次计算的规模。

批量过小，设备可能无法充分并行；批量变大，每一步通常覆盖更多样本，但同样遍历一轮时更新次数更少。批量大小还会改变梯度的波动程度，不能只凭“更大”判断效果。

本例只有四条数据，分批的目的只是把更新过程看清楚。真正使用时，应把批量大小、学习率和验证表现放在一起比较。

关于取样、组批和加载参数，参见 [torch.utils.data：PyTorch 数据加载工具](https://docs.pytorch.org/docs/stable/data.html)。
