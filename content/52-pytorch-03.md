# 预测错了 怎样算出一个损失

看得懂的 PyTorch · 第 3 讲

三套房屋的预测误差分别是 −5、5、−5 万元。如果第三套的误差变成 −50 万元，它应该对训练产生多大影响？

均方误差会把它的单项损失放大到另外两项的 100 倍，绝对误差则放大到 10 倍。损失函数决定了训练对不同错误的重视程度。

## 从误差到一个数

沿用上一讲的虚构房价数据：

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([90., 115., 135.])
target = torch.tensor([95., 110., 140.])
assert pred.shape == target.shape

error = pred - target
print(error.tolist())  # [-5.0, 5.0, -5.0]
```

误差带有方向：负数表示低估，正数表示高估。直接取平均会让两种错误抵消，例如 −5 和 5 的平均是零，预测却都不准确。

均方误差 MSE 先平方，再取平均：

```python
mse = error.square().mean()
print(mse.item())  # (25 + 25 + 25) / 3 = 25

print(F.mse_loss(pred, target).item())  # 25.0
```

平均绝对误差 MAE 则先取绝对值：

```python
mae = error.abs().mean()
print(mae.item())  # (5 + 5 + 5) / 3 = 5

print(F.l1_loss(pred, target).item())  # 5.0
```

`F` 是 PyTorch 函数式接口的常用别名；这里的 `l1_loss` 默认取平均，对应 MAE。两种函数的定义分别见 [MSELoss（均方误差）](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) 和 [L1Loss（绝对误差）](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)。

MAE 为 5，表示平均偏差是 5 万元。MSE 为 25，单位是万元的平方。**不能因为 5 小于 25，就认为 MAE 对应的模型更好**：预测完全相同，只是计分方式不同。

## 一个大误差会占多大分量

保持预测不动，把第三套房屋的目标价格改为 185 万元，制造一个较大的误差：

```python
target_far = torch.tensor([95., 110., 185.])

print(F.mse_loss(pred, target_far, reduction="none").tolist())
# [25.0, 25.0, 2500.0]

print(F.l1_loss(pred, target_far, reduction="none").tolist())
# [5.0, 5.0, 50.0]
```

`reduction="none"` 保留每项损失，便于检查总损失由哪些样本贡献。

| 损失 | 前两项 | 第三项 | 取平均 | 第三项占总和 |
|---|---|---|---|---|
| MSE | 25、25 | 2500 | 850 | 约 98% |
| MAE | 5、5 | 50 | 20 | 约 83% |

平方使大误差迅速占据主导。如果第三套确实是模型尚未学好的房屋，这种惩罚可能有用；如果价格录入错误，训练也会努力追着错误标签跑。损失函数无法自行分辨这两种情况。

## 损失怎样影响更新方向

真正参与参数更新的是梯度。设一项误差为 `e = 预测 − 答案`，共有三项，则损失对每个预测值的导数为：

```text
MSE：2e / 3
MAE：sign(e) / 3
```

`sign(e)` 在正误差处为 1，负误差处为 −1。MAE 在零点不可导，PyTorch 在该点使用零梯度。

让三个预测值直接参与求导，便能看到两种损失给出的信号：

```python
for name, loss_fn in [("MSE", F.mse_loss), ("MAE", F.l1_loss)]:
    p = pred.clone().requires_grad_()
    loss = loss_fn(p, target_far)
    loss.backward()
    print(name, [round(v, 4) for v in p.grad.tolist()])

# MSE [-3.3333, 3.3333, -33.3333]
# MAE [-0.3333, 0.3333, -0.3333]
```

`clone()` 为每次比较创建独立 Tensor，避免混入上一次的梯度。

MSE 下，第三项对预测的梯度幅度是第一项的 10 倍。MAE 下，只要误差非零，梯度幅度就相同；低估 50 万元不会比低估 5 万元产生更大的预测梯度。

这里算的是**损失对预测的梯度**。真实训练还会沿着模型继续求导。例如 `预测 = w × 面积 + b`，传到 `w` 的梯度要再乘面积。因此，MAE 并不保证每条样本对所有参数的影响都一样。

## Smooth L1 怎样折中

MAE 的梯度在零点两侧从 −1 跳到 1；MSE 的梯度会随误差接近零而逐渐减小，却容易受到大误差影响。

Smooth L1 在小误差区域使用二次函数，超过阈值后改为线性增长。设阈值 `beta=10`，单项损失为：

```text
|e| < 10：损失 = e² / 20
|e| ≥ 10：损失 = |e| − 5
```

误差为 5 时，损失为 `25 / 20 = 1.25`；误差为 50 时，损失为 `50 − 5 = 45`。

```python
p = pred.clone().requires_grad_()
items = F.smooth_l1_loss(p, target_far, beta=10., reduction="none")
loss = items.mean()
loss.backward()

print(items.tolist())  # [1.25, 1.25, 45.0]
print(round(loss.item(), 4))  # 15.8333
print([round(v, 4) for v in p.grad.tolist()])
# [-0.1667, 0.1667, -0.3333]
```

对应的单项导数，在小误差区域是 `e / beta`，大误差区域是 `sign(e)`；取平均后再除以 3。第三项误差即使继续扩大，其对预测的梯度幅度也不会超过 `1/3`，而接近正确答案的预测会得到越来越小的修正信号。[SmoothL1Loss 官方定义](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

`beta` 与误差使用相同的单位，这里的 10 表示 10 万元。若目标价格换算成元，阈值也应按比例换算。15.8333、20、850 来自不同损失，不能凭数值大小选出优胜者。

## 平均和求和也会改变梯度

以上函数默认 `reduction="mean"`。改成 `"sum"`，三项损失直接相加：

```python
for mode in ["mean", "sum"]:
    p = pred.clone().requires_grad_()
    loss = F.mse_loss(p, target_far, reduction=mode)
    loss.backward()
    print(mode, loss.item(), [round(v, 4) for v in p.grad.tolist()])

# mean 850.0  [-3.3333, 3.3333, -33.3333]
# sum  2550.0 [-10.0, 10.0, -100.0]
```

求和使损失和梯度都变为三倍。对第一讲那种直接减去“学习率 × 梯度”的更新，保持学习率不变，参数步幅也会变为三倍。

这些损失的 `mean` 对所有元素取平均。如果预测形状是 `[3, 2]`，共有六个输出，就除以 6。批次大小、输出数量和归约方式都应与损失一起检查。

## 比较模型时固定同一把尺子

MSE 强调大误差，MAE 对大误差的增长更温和，Smooth L1 兼顾零点附近的平滑和大误差处的线性增长。选择取决于任务对错误的代价，以及数据中异常值的来源。

用不同损失训练的房价模型，可以在同一份验证集上统一计算 MAE，比较平均偏差，再计算 RMSE 检查大误差：

```python
with torch.no_grad():
    mae = F.l1_loss(pred, target_far)
    rmse = F.mse_loss(pred, target_far).sqrt()

print(round(mae.item(), 2))   # 20.0
print(round(rmse.item(), 2))  # 29.15
```

RMSE 是均方误差的平方根，单位回到万元，但仍对大误差敏感。这里用已有数据演示指标计算；实际比较时，`pred` 和答案应来自未参与参数更新的验证集。训练用什么损失，与最终报告什么指标，可以分别决定。
