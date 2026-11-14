# 把分类训练和评估完整跑一遍

看得懂的 PyTorch · 第 15 讲

每条样本有两个数 `x₁、x₂`。实验规定：`x₁>x₂` 属于类别 0，`x₂>x₁` 属于类别 1，不取相等的点。

这条规则很简单，直接比较大小就能完成任务。这里故意让模型从样本学习，用于看清分类训练的全过程；不把这个玩具实验当成真实分类能力的证明。

## 数据和输出分别是什么形状

准备六条训练数据、四条验证数据。代码不把比较大小的规则写进模型，只提供输入和标签：

```python
import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(1)

X_train = torch.tensor([
    [2., 0.], [3., 1.], [2., 1.],
    [0., 2.], [1., 3.], [1., 2.],
], dtype=torch.float64)
y_train = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

X_val = torch.tensor([
    [1.5, 0.5], [2., 1.5], [0.5, 1.5], [1.5, 2.],
], dtype=torch.float64)
y_val = torch.tensor([0, 0, 1, 1], dtype=torch.long)

model = nn.Linear(2, 2).double()
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=2,
    shuffle=True, generator=torch.Generator().manual_seed(7),
    num_workers=0,
)
```

训练集输入形状为 `[6,2]`，标签为 `[6]`；每批输入 `[2,2]`，输出 logits 也是 `[2,2]`。输入的第二维表示特征数，输出的第二维表示类别数，只是本例恰好都等于 2。

模型共六个参数，两行权重分别负责两个类别的得分，另外各有一个偏置。它只有一层，没有需要打破对称性的隐藏单元，零初始化可以正常训练。

## 第一对样本会怎样推动权重

先不考虑打乱顺序，单独手算 `[2,0]` 和 `[0,2]` 组成的一批，标签分别为 0、1。这只是演算，下面实际代码仍按随机顺序取批。

零参数对应相等得分，两条样本的概率都为 `[0.5,0.5]`。平均交叉熵为 `ln(2)`，对 logits 的梯度需除以批量大小 2：

```text
样本 [2,0]，标签 0：得分梯度 = [−0.25, 0.25]
样本 [0,2]，标签 1：得分梯度 = [ 0.25,−0.25]
```

每条样本的得分梯度乘输入，再相加：

```text
类别 0 的权重梯度：−0.25×[2,0] + 0.25×[0,2] = [−0.5, 0.5]
类别 1 的权重梯度： 0.25×[2,0] − 0.25×[0,2] = [ 0.5,−0.5]
两个偏置的梯度：都为 0
```

学习率为 0.1，更新后两行权重分别为 `[0.05,−0.05]` 和 `[−0.05,0.05]`。

此时类别 0 的得分为 `0.05(x₁−x₂)`，类别 1 的得分相反。模型刚更新一步，已经开始使用两个输入的差异。

## 训练时更新，评估时固定参数

评估函数返回平均交叉熵和准确率：

```python
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = criterion(logits, y).item()
        predicted = logits.argmax(dim=1)
        accuracy = (predicted == y).double().mean().item()
    return loss, accuracy
```

`eval()` 切换运行模式，`no_grad()` 关闭梯度记录；这个 Linear 模型切换模式后运算不变，但保留这样的接口，便于以后加入其他层。

每轮完成三次小批量更新，再用固定参数评估训练集和验证集。保留验证交叉熵最低的参数：

```python
best_val_loss = float("inf")
best_state = None
best_epoch = 0

for epoch in range(1, 101):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    train_loss, _ = evaluate(model, X_train, y_train)
    val_loss, val_accuracy = evaluate(model, X_val, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch

    if epoch in {1, 10, 100}:
        print(f"epoch={epoch:3d}  train={train_loss:.6f}"
              f"  val={val_loss:.6f}  val_acc={val_accuracy:.2%}")

model.load_state_dict(best_state)
print("best epoch:", best_epoch)
```

PyTorch 2.13.0、CPU 的本次输出：

```text
epoch=  1  train=0.509773  val=0.605597  val_acc=100.00%
epoch= 10  train=0.142203  val=0.346613  val_acc=100.00%
epoch=100  train=0.023110  val=0.142852  val_acc=100.00%
best epoch: 100
```

验证准确率第一轮就到 100%，但交叉熵仍持续下降。类别已经选对后，模型还可以给正确类别分配更高概率，因此损失仍有变化。

`state_dict()` 保存参数等状态。这里必须复制一份快照，不能仅保存仍与模型共享底层数据的状态引用，否则继续训练会让“最佳参数”也跟着变。`deepcopy` 在本例中把参数留在内存；没有写入磁盘。

这次最低验证损失恰好出现在最后一轮。代码仍按验证结果选择，而不是预设训练越久一定越好。

## 最后才使用测试集

四条测试数据包括两个差距大的点和两个靠近分界线的点。它们不参与参数更新或轮数选择：

```python
X_test = torch.tensor([
    [4., 1.], [1., 4.], [1.1, 1.], [1., 1.1],
], dtype=torch.float64)
y_test = torch.tensor([0, 1, 0, 1], dtype=torch.long)

model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_probs = test_logits.softmax(dim=1)
    test_pred = test_logits.argmax(dim=1)

for x, probs, predicted in zip(X_test, test_probs, test_pred):
    print(x.tolist(), [round(v, 6) for v in probs.tolist()],
          "class:", predicted.item())

test_loss, test_accuracy = evaluate(model, X_test, y_test)
print(f"test_loss={test_loss:.6f}, test_acc={test_accuracy:.2%}")
```

本次测试输出 `test_loss=0.282439, test_acc=100.00%`。各点的预测概率约为：

| 输入 | 类别 0 概率 | 类别 1 概率 | 预测 |
|---|---:|---:|---:|
| `[4,1]` | 0.999747 | 0.000253 | 0 |
| `[1,4]` | 0.000253 | 0.999747 | 1 |
| `[1.1,1]` | 0.567090 | 0.432910 | 0 |
| `[1,1.1]` | 0.429939 | 0.570061 | 1 |

四条都选对了，但后两条的概率接近五五开。准确率将“略微领先”和“明显领先”都计为一次正确，交叉熵会区分它们。

## 模型到底学出了一条什么线

读取训练后的权重：

```python
print([[round(v, 6) for v in row] for row in model.weight.tolist()])
print([round(v, 6) for v in model.bias.tolist()])
```

得到约：

```text
W = [[ 1.381400, −1.379056],
     [−1.381400,  1.379056]]
b = [−0.005489, 0.005489]
```

选择类别 0，意味着它的得分大于类别 1：

```text
z₀−z₁ ≈ 2.762799×x₁ − 2.758111×x₂ − 0.010978 > 0
```

等号对应一条接近 `x₁=x₂` 的直线。对于 `[1.1,1]`，得分差约为 0.26999；用第十三讲的两类公式：

```text
p₀ = sigmoid(z₀−z₁) ≈ 0.56709
```

靠近分界线，两类得分差小，概率接近 0.5。这是本例模型的计算结果，不代表所有神经网络都会在未知输入处表现得谨慎。

## 把错在哪里数出来

混淆矩阵按“真实类别作为行、预测类别作为列”统计：

```python
confusion = torch.zeros((2, 2), dtype=torch.long)
for actual, predicted in zip(y_test.tolist(), test_pred.tolist()):
    confusion[actual, predicted] += 1
print(confusion.tolist())  # [[2, 0], [0, 2]]
```

对角线是正确分类；左下角表示真实为 1 却判成 0，右上角则相反。

这个玩具测试集每类两条，全部正确；真实任务还应检查各类错误是否集中。假如一百条样本只有五条属于类别 1，永远预测类别 0 就能达到 95% 准确率，却一条类别 1 都找不出来。

因此测试报告至少要同时看损失、准确率和各类错误。六条训练数据、四条测试数据只够验证这套代码和计算，远不足以证明模型在实际数据上可靠。
