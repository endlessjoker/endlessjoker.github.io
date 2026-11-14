# 用 PyTorch 把前面的计算接起来

看得懂的 PyTorch · 第 8 讲

第七讲用手工设置的参数，让两层网络算出了绝对值。这次把输出层的两个权重从 1 改成 0.5：预测只有正确答案的一半，再用训练把误差降下来。

输入仍是 `−2、−1、0、1、2`，目标为 `2、1、0、1、2`。以下代码按顺序执行，只需 PyTorch 和 CPU。

## 七个参数放在哪里

```python
import torch
from torch import nn

X = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]],
                 dtype=torch.float64)
y = X.abs()

model = nn.Sequential(
    nn.Linear(1, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
).double()

with torch.no_grad():
    model[0].weight.copy_(torch.tensor([[1.], [-1.]]))
    model[0].bias.zero_()
    model[2].weight.fill_(0.5)
    model[2].bias.zero_()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

两个 `Linear` 层共有七个参数：

| 参数 | 初始值 | 形状 | 数量 |
|---|---|---|---|
| 第一层权重 | `[[1], [-1]]` | `[2, 1]` | 2 |
| 第一层偏置 | `[0, 0]` | `[2]` | 2 |
| 第二层权重 | `[[0.5, 0.5]]` | `[1, 2]` | 2 |
| 第二层偏置 | `[0]` | `[1]` | 1 |

`Linear` 自动把权重和偏置注册为可训练参数，`model.parameters()` 将它们交给优化器。ReLU 没有可训练参数。

`.double()` 将模型参数转为 64 位浮点数，与输入保持一致，方便核对计算。这里特意安排了分别处理正、负输入的初始权重；这是可手算的训练演示，不代表任意初始化都能得到相同结果。

## 一次前向计算经过什么

`Linear` 按 `[输出数, 输入数]` 存储权重，实际计算是 `输入 @ weight.T + bias`。见 [torch.nn.Linear：线性层的定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)。

对输入 `x=−2`：

```text
第一层：z = [1×(−2), (−1)×(−2)] = [−2, 2]
ReLU： h = [0, 2]
第二层：预测 = 0.5×0 + 0.5×2 + 0 = 1
```

五条样本一起计算，形状依次变化：

```text
X [5,1] → Linear [5,2] → ReLU [5,2] → Linear [5,1]
```

第一维一直是样本数。隐藏层的两列，是每条样本的两个中间特征。

```python
pred = model(X)
assert pred.shape == y.shape
loss = criterion(pred, y)

print(pred.detach().squeeze(1).tolist())
print(loss.item())
```

输出：

```text
[1.0, 0.5, 0.0, 0.5, 1.0]
0.5
```

`MSELoss` 默认将所有元素的平方误差取平均。本例每条样本只有一个输出，因此：

```text
误差 = [−1, −0.5, 0, −0.5, −1]
损失 = (1 + 0.25 + 0 + 0.25 + 1) / 5 = 0.5
```

`pred` 和 `y` 都保持 `[5,1]`。若把目标写成 `[5]`，减法会广播成 `[5,5]`，算成预测与各个目标的交叉比较。形状检查能直接拦住这个错误。

## backward 怎样得到七个梯度

先清掉旧梯度，再对刚才的损失反向传播：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()

for name, parameter in model.named_parameters():
    print(name, parameter.grad.tolist())
```

结果如下，省略浮点尾差：

```text
0.weight  [[−0.5], [0.5]]
0.bias    [−0.3, −0.3]
2.weight  [[−1.0, −1.0]]
2.bias    [−1.2]
```

`0` 和 `2` 是两个 `Linear` 在 `Sequential` 中的位置；位置 `1` 是 ReLU。

先算损失对五个预测的导数。均方误差求导后，每条误差乘以 `2/5`：

```text
g = [−0.4, −0.2, 0, −0.2, −0.4]
```

**第二层第一个权重**乘的是 `ReLU(x)`，五条样本的这个特征为 `[0,0,0,1,2]`：

```text
梯度 = Σ(g × 特征)
     = (−0.2)×1 + (−0.4)×2 = −1
```

第二个权重使用 `ReLU(−x)`，由负半轴的两条样本贡献，同样得到 −1。输出偏置对每条预测的影响都是 1，其梯度为 `Σg=−1.2`。

**第一层第一个权重**还要经过输出权重和 ReLU。只有 `x=1、2` 的路径打开：

```text
梯度 = Σ(g × 输出权重 × ReLU导数 × 输入)
     = (−0.2)×0.5×1×1 + (−0.4)×0.5×1×2
     = −0.5
```

第二个隐藏单元在负半轴打开，输入为负，因此它的权重梯度为正：

```text
(−0.4)×0.5×1×(−2) + (−0.2)×0.5×1×(−1) = 0.5
```

第一层偏置的梯度不乘输入，两个都是 `(−0.4−0.2)×0.5=−0.3`。在 `x=0` 处，PyTorch 的 ReLU 使用零导数。

`backward()` 沿着这些乘法、加法关系算完七个梯度，参数尚未改变。

## step 怎样更新整张网络

```python
optimizer.step()
```

这里的 SGD 没有启用动量或权重衰减，执行的仍是 `参数 -= 0.1 × 梯度`。见 [torch.optim.SGD：随机梯度下降优化器](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)。

| 参数 | 更新前 | 梯度 | 更新后 |
|---|---:|---:|---:|
| 第一层权重 1 | 1 | −0.5 | 1.05 |
| 第一层权重 2 | −1 | 0.5 | −1.05 |
| 第一层偏置 1 | 0 | −0.3 | 0.03 |
| 第一层偏置 2 | 0 | −0.3 | 0.03 |
| 第二层权重 1 | 0.5 | −1 | 0.6 |
| 第二层权重 2 | 0.5 | −1 | 0.6 |
| 第二层偏置 | 0 | −1.2 | 0.12 |

全部更新使用同一次反向传播的结果。

再算 `x=−2`：

```text
z = [1.05×(−2)+0.03, (−1.05)×(−2)+0.03]
  = [−2.07, 2.13]
h = [0, 2.13]
预测 = 0.6×2.13 + 0.12 = 1.398
```

```python
with torch.no_grad():
    new_pred = model(X)
    new_loss = criterion(new_pred, y)

print([round(v, 6) for v in new_pred.squeeze(1).tolist()])
print(f"loss={new_loss.item():.7f}")
```

输出：

```text
[1.398, 0.768, 0.156, 0.768, 1.398]
loss=0.1713584
```

整体损失从 0.5 降到了约 0.1714，但 `x=0` 原本预测准确，现在却偏高。两个隐藏偏置和输出偏置都增大了，它们影响所有样本。共享参数的更新仍需兼顾整体误差。

## 把这一步重复起来

已经完成一次更新。下面从第 2 次继续，累计训练 1000 步：

```python
model.train()
for step in range(2, 1001):
    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    if step in {10, 100, 1000}:
        with torch.no_grad():
            current_loss = criterion(model(X), y)
        print(f"step={step:4d}  loss={current_loss.item():.9f}")
```

PyTorch 2.13.0、CPU 运行结果：

```text
step=  10  loss=0.011493952
step= 100  loss=0.000006100
step=1000  loss=0.000000001
```

每次前向计算使用最新参数，并建立本次反向传播所需的计算图。清梯度、算预测、算损失、反传、更新，循环里的五行对应完整训练过程。日志重新计算更新后的损失。

`model.train()` 设置训练模式，本身不会启动训练。这个网络只有 Linear 和 ReLU，训练与评估模式下的前向计算相同；含 Dropout 等层时，模式才会改变行为。

## 换几个没训练过的输入

```python
model.eval()
X_test = torch.tensor([[-1.5], [-0.5], [0.5], [1.5]],
                      dtype=torch.float64)

with torch.no_grad():
    prediction = model(X_test)

print([round(v, 6) for v in prediction.squeeze(1).tolist()])
```

输出约为：

```text
[1.499994, 0.499952, 0.499952, 1.499994]
```

这四个点没有参与训练，预测接近它们的绝对值。这里只检查了训练范围内的几个插值点，不能据此保证任意输入都准确。`eval()` 切换模式，`no_grad()` 关闭这段计算的梯度记录，两者作用不同。

训练后的第一层权重约为 `±1.1985`，第二层权重约为 `0.8344`，没有回到第七讲手工设置的 `±1` 和 `1`。

这并不奇怪：对正数 `c`，`ReLU(cx)=c·ReLU(x)`。忽略很小的偏置，路径的输出取决于两个权重的乘积，这里 `1.1985×0.8344≈1`。不同的内部参数可以实现近似相同的函数。

训练约束的是预测结果，未必能唯一确定每一层该学出什么数字。
