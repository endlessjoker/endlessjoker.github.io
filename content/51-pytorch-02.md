# Tensor 的形状为什么这么重要

看得懂的 PyTorch · 第 2 讲

同样三个预测值、三个答案，均方误差可以算出 25，也可以算出 691.67。数字没变，只是其中一个 Tensor 多了一层括号，PyTorch 就执行了另一种计算，而且不会报错。

形状决定数字怎样配对。先从一张小表开始。

## 一行是一条样本

假设根据面积和房间数预测房价。下面是虚构数据，面积以 10 平方米为单位，价格以万元为单位：

| 房屋 | 面积 | 房间数 | 价格 |
|---|---|---|---|
| A | 6 | 2 | 95 |
| B | 8 | 3 | 110 |
| C | 10 | 3 | 140 |

输入有三条样本，每条两个特征：

```python
import torch

X = torch.tensor([
    [6., 2.],
    [8., 3.],
    [10., 3.],
])
y = torch.tensor([95., 110., 140.])

print(X.shape)  # torch.Size([3, 2])
print(y.shape)  # torch.Size([3])
```

`shape` 按顺序记录每个轴的长度。`X` 的第 0 轴有 3 行，第 1 轴有 2 列；`y` 只有一个轴，装着三个答案。轴从 0 开始编号。

这里约定行表示样本、列表示特征。Tensor 本身只保存数字及其组织方式，不知道哪一列叫“面积”。

```python
print(X[0].tolist())     # [6.0, 2.0]，第一条样本
print(X[:, 0].tolist())  # [6.0, 8.0, 10.0]，所有样本的面积
```

`:` 表示取该轴的全部位置。两个结果都是形状 `[2]` 或 `[3]` 的一维 Tensor，取出一行并不保留原来的二维结构。

## 两个特征怎样合成一个预测

模型设为：

```text
预测价格 = 面积 × 面积权重 + 房间数 × 房间权重 + 偏置
```

为了看清计算，先指定权重为 `[10, 5]`，偏置为 20；这些数尚未经过训练。

```python
w = torch.tensor([10., 5.])
b = torch.tensor(20.)
pred = X @ w + b

print(pred.tolist())  # [90.0, 115.0, 135.0]
print(pred.shape)     # torch.Size([3])
```

`@` 在这里执行矩阵与向量的乘法：每行特征与权重逐项相乘，再加起来。

```text
A：6 × 10 + 2 × 5 + 20 = 90
B：8 × 10 + 3 × 5 + 20 = 115
C：10 × 10 + 3 × 5 + 20 = 135

[3, 2] @ [2] → [3]
```

输入每行有两个特征，因此权重必须有两个元素。三行分别得到一个结果，共三个预测。三套房屋共用同一组权重，模型总共只有两个权重和一个偏置。

`*` 则是逐元素乘法，不会自动求和：

```python
parts = X * w
print(parts.tolist())
# [[60.0, 10.0], [80.0, 15.0], [100.0, 15.0]]

print((parts.sum(dim=1) + b).tolist())
# [90.0, 115.0, 135.0]
```

`sum(dim=1)` 沿第 1 轴求和，也就是把每行的两项相加。若写成 `dim=0`，得到的将是每列跨房屋的总和，含义就变了。

## 一维向量和单列矩阵

`[10., 5.]` 与 `[[10.], [5.]]` 都装着两个数，但形状不同：前者是 `[2]`，后者是 `[2, 1]`。

```python
W = w.unsqueeze(1)
pred_column = X @ W + b

print(W.shape)              # torch.Size([2, 1])
print(pred_column.shape)    # torch.Size([3, 1])
print(pred_column.tolist()) # [[90.0], [115.0], [135.0]]
```

`unsqueeze(1)` 在位置 1 插入一个长度为 1 的轴，原来的两个数没有改变。

这次是矩阵乘法：

```text
[3, 2] @ [2, 1] → [3, 1]
```

中间两个 2 对应需要相乘后求和的特征轴；结果保留左边的 3 行和右边的 1 列。这一列表示每条样本只有一个输出。

**`[3]` 与 `[3, 1]` 数量相同，参与后续运算的方式却不同。**

## 广播怎样把三次比较变成九次

用一维预测计算损失，结果正常：

```python
error = pred - y
print(error.tolist())       # [-5.0, 5.0, -5.0]
print(error.square().mean().item())  # 25.0
```

换成单列预测，答案 `y` 保持不变：

```python
wrong_error = pred_column - y
print(wrong_error.shape)    # torch.Size([3, 3])
print(wrong_error.tolist())
# [[-5.0, -20.0, -50.0],
#  [20.0,   5.0, -25.0],
#  [40.0,  25.0,  -5.0]]

print(wrong_error.square().mean().item())  # 约 691.67
```

PyTorch 对支持广播的运算，从最右侧对齐各轴。对应长度相等可以计算，长度为 1 的轴可以扩展，缺少的前置轴按长度 1 处理。[Broadcasting semantics（广播规则）](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)

这里的对齐过程是：

```text
预测：[3, 1]
答案：   [3] → 按 [1, 3] 对齐
结果：[3, 3]
```

每个预测都减去了全部三个答案：

| | 答案 A：95 | 答案 B：110 | 答案 C：140 |
|---|---|---|---|
| 预测 A：90 | −5 | −20 | −50 |
| 预测 B：115 | 20 | 5 | −25 |
| 预测 C：135 | 40 | 25 | −5 |

只有对角线上的三项是所需误差，另外六项把不同房屋混在了一起。九项平方之和为 6225，取平均得到约 691.67。

这段代码也能求梯度，但它优化的是错误的配对关系。程序成功运行，并不意味着损失符合任务。

修正方法是让预测和答案使用同一种形状：

```python
y_column = y.unsqueeze(1)   # [3] → [3, 1]
assert pred_column.shape == y_column.shape
loss = ((pred_column - y_column) ** 2).mean()
print(loss.item())          # 25.0
```

也可以用 `pred_column.squeeze(1)` 去掉长度为 1 的列轴，恢复成 `[3]` 后与 `y` 相减。这里明确指定轴，避免批次只有一条样本时，无参数的 `squeeze()` 连样本轴也一起删掉。

## 对应到 Linear 层

PyTorch 的 `Linear(2, 1)` 表示每条样本接收两个特征，输出一个数。它将权重存成 `[输出数, 输入数]`，因此这里是 `[1, 2]`，实际计算为 `X @ weight.T + bias`。[torch.nn.Linear 官方说明](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)

```python
layer = torch.nn.Linear(2, 1)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[10., 5.]]))
    layer.bias.fill_(20.)

output = layer(X)
print(output.shape)  # torch.Size([3, 1])
print(output.detach().tolist())  # [[90.0], [115.0], [135.0]]
```

`copy_` 和 `fill_` 用指定数值覆盖初始化参数；`detach()` 在这里仅用于取出不参与求导的结果供打印。二维权重的 `.T` 交换行列，将 `[1, 2]` 转成 `[2, 1]`，正好接上输入的两个特征。

`Linear` 保持每条样本独立计算，不会混合不同房屋的数据。输出中的 3 来自样本数量，1 来自每条样本的输出数量；损失计算必须继续保持这种对应关系。
