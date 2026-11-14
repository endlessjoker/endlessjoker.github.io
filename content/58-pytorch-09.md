# 训练误差很小 为什么换数据就不准了

看得懂的 PyTorch · 第 9 讲

一个模型在三条训练数据上全部答对，另一个三条都有误差。换两个输入，前者却错得更多。

训练损失只衡量模型对训练数据的拟合程度。数据里有噪声时，把每个答案都记准，可能连噪声也一起学进去。

## 故意放进一条不准的数据

假设真实规律为 `y=x`，但训练集在 `x=0` 处发生了一次测量偏差，把 0 记成了 1：

| 输入 x | 真实值 | 训练时提供的答案 |
|---|---:|---:|
| −1 | −1 | −1 |
| 0 | 0 | 1 |
| 1 | 1 | 1 |

训练程序只能看到最后一列，不知道哪条出了错。真实规律是实验设计者设定的，用于事后解释；现实任务通常没有这样一份标准答案。

准备两个模型：

```text
模型 A：预测 = a×x + b
模型 B：预测 = a×x + c×x² + b
```

A 有两个参数，B 有三个。B 多了一个平方特征，可以把直线弯成抛物线。

这次用平方特征控制表达能力，便于手算。第七讲的 ReLU 网络也能表示非线性函数，但“表达能力更强”并不自动保证新数据上的误差更小。

## 两个模型分别学到了什么

对这三条训练数据，模型 A 的最优结果是：

```text
预测 = x + 1/3
```

它无法单独抬高中间那个点，只能整体向上挪一些：

| x | 训练答案 | A 的预测 | 误差 |
|---|---:|---:|---:|
| −1 | −1 | −2/3 | 1/3 |
| 0 | 1 | 1/3 | −2/3 |
| 1 | 1 | 4/3 | 1/3 |

均方误差为：

```text
[(1/3)² + (−2/3)² + (1/3)²] / 3 = 2/9 ≈ 0.2222
```

为什么是这条直线？损失写成：

```text
L = [(-a+b+1)² + (b−1)² + (a+b−1)²] / 3
```

令两个偏导数为零：

```text
∂L/∂a = 4(a−1)/3 = 0  → a=1
∂L/∂b = 2b−2/3 = 0    → b=1/3
```

模型 B 则可以学到：

```text
预测 = x − x² + 1
```

代入 −1、0、1，分别得到 −1、1、1，训练误差为零。它用一段向上拱起的曲线，穿过了那个有测量偏差的点。

## 换两个输入，排名反过来了

拿 `x=−0.5、0.5` 作验证集，答案按真实规律取 −0.5、0.5，不加入测量偏差：

| x | 正确答案 | A 的预测 | B 的预测 |
|---|---:|---:|---:|
| −0.5 | −0.5 | −1/6 | 0.25 |
| 0.5 | 0.5 | 5/6 | 1.25 |

A 的两个误差都是 `1/3`，验证均方误差为 `1/9≈0.1111`。

B 的两个误差都是 `0.75`，验证均方误差为 `0.75²=0.5625`。

**B 把训练误差降到了零，验证误差却是 A 的约 5.06 倍。** 中间那次测量偏差，被它扩展成了附近一整段曲线的隆起。这就是一个过拟合的具体例子。

## 用 PyTorch 实际训练

下面代码从零初始化参数，分别训练两个模型。输入和目标都使用二维 Tensor，避免意外广播。

```python
import torch
from torch import nn

X_train = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64)
y_train = torch.tensor([[-1.], [1.], [1.]], dtype=torch.float64)
X_val = torch.tensor([[-0.5], [0.5]], dtype=torch.float64)
y_val = X_val.clone()


def features(x, degree):
    if degree == 1:
        return x
    return torch.cat([x, x.square()], dim=1)


def fit(degree):
    model = nn.Linear(degree, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs = features(X_train, degree)

    model.train()
    for _ in range(1000):
        optimizer.zero_grad(set_to_none=True)
        pred = model(inputs)
        assert pred.shape == y_train.shape
        loss = (pred - y_train).square().mean()
        loss.backward()
        optimizer.step()
    return model


models = {degree: fit(degree) for degree in (1, 2)}
val_losses = {}

for degree, model in models.items():
    model.eval()
    with torch.no_grad():
        train_pred = model(features(X_train, degree))
        val_pred = model(features(X_val, degree))
        train_loss = (train_pred - y_train).square().mean().item()
        val_loss = (val_pred - y_val).square().mean().item()
    val_losses[degree] = val_loss
    print(f"degree={degree}  train={train_loss:.6f}  val={val_loss:.6f}")
    print("weight:", [round(v, 6) for v in model.weight[0].tolist()],
          "bias:", round(model.bias.item(), 6))
```

PyTorch 2.13.0、CPU 运行结果：

```text
degree=1  train=0.222222  val=0.111111
weight: [1.0] bias: 0.333333
degree=2  train=0.000000  val=0.562500
weight: [1.0, -1.0] bias: 1.0
```

最后的 `0.000000` 是保留六位小数的结果，实际仍有很小的浮点误差。

`features` 对模型 B 把输入展开成两列：

```text
原始输入 [3,1]        新特征 [3,2]
   [−1]               [−1, 1]
   [ 0]       →       [ 0, 0]
   [ 1]               [ 1, 1]
```

`Linear(2,1)` 用两个权重组合这两列，得到 `a×x+c×x²+b`。它对输入特征仍是线性的，但对原始变量 x 已经是二次函数。

两个模型都从零开始，没有复制前面手算的答案。这里是单层回归，零初始化可以正常学习；不能据此把多层网络的所有权重也设为零，否则隐藏单元可能因对称性学不出不同特征。

## 验证集选模型，测试集留到最后

上面的梯度只来自训练集。验证集不参与 `backward()`，但它决定保留哪个模型，因此已经参与了模型选择。

如果再用同一份验证集宣布最终效果，就会忽略“挑选”带来的影响。尝试的结构、学习率和训练轮数越多，越可能碰巧挑中一个适合这份验证集的组合。

预先留出第三组输入作测试集，选定模型后再评估一次：

```python
best_degree = min(val_losses, key=val_losses.get)
best_model = models[best_degree]

X_test = torch.tensor([[-0.75], [-0.25], [0.25], [0.75]],
                      dtype=torch.float64)
y_test = X_test.clone()

best_model.eval()
with torch.no_grad():
    test_pred = best_model(features(X_test, best_degree))
    test_loss = (test_pred - y_test).square().mean()

print("selected degree:", best_degree)
print("test predictions:",
      [round(v, 6) for v in test_pred.squeeze(1).tolist()])
print(f"test loss: {test_loss.item():.6f}")
```

输出：

```text
selected degree: 1
test predictions: [-0.416667, 0.083333, 0.583333, 1.083333]
test loss: 0.111111
```

选中的直线仍整体偏高 `1/3`。它比另一个候选模型更好，却没有恢复真实规律。验证集只能比较候选方案，不能保证其中一定有好答案。

这几组点是刻意构造的教学数据，数量远不足以证明实际任务的泛化能力。真实实验中，三份数据的分工仍然相同：

| 数据 | 用途 |
|---|---|
| 训练集 | 计算梯度，更新参数 |
| 验证集 | 比较结构、学习率、训练轮数等选择 |
| 测试集 | 方案确定后，评估最终表现 |

如果根据测试结果继续改模型，测试集也开始承担验证集的角色，最终评估就需要另留未使用的数据。

## 没有反向传播，也可能泄漏答案

“验证数据没有调用 `backward()`”还不够。例如先用全部数据计算均值和标准差，再划分训练集和验证集，验证数据的信息就进入了预处理。

需要从数据中学习的处理步骤，也应只在训练集上拟合，再原样应用到验证集和测试集。本例的 `x²` 是预先指定的运算，不需要从数据中估计参数。

划分还要符合实际使用方式：同一个人的近似照片分散到两边，可能让模型靠身份线索获得好成绩；预测未来的数据，则应按时间保留后段，并检查特征是否包含当时尚未公开的信息。

训练损失下降后，真正需要检查的是：在没有用于更新参数和挑选方案的数据上，误差是否也足够小。
