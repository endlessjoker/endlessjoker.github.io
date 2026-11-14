# 把完整实验整理成一个小项目

看得懂的 PyTorch · 第 18 讲

最后把前面的步骤放到同一个入口：训练时划分数据、拟合标准化、分批更新、验证选模并保存；预测时只加载产物，接收新输入。

仍使用第十五讲的两特征分类任务。数据很小，适合检查流程，不代表已经解决真实业务问题。

## 保存的不能只有权重

训练集两列的均值都为 1.5，方差都为 `11/12`，标准差约为 0.957427。例如第一列为 `2、3、2、0、1、1`：

```text
均值 = 9/6 = 1.5
方差 = (0.25+2.25+0.25+2.25+0.25+0.25)/6 = 11/12
```

新输入 `[1.1,1]` 应变成：

```text
[(1.1−1.5)/0.957427, (1−1.5)/0.957427]
≈ [−0.417786,−0.522233]
```

预测进程必须获得这两个统计量，以及两列分别是什么意思、输出编号对应什么类别。只保存权重，无法保证后续使用的是同一种输入解释。

本例的产物是一个字典：

| 字段 | 内容 |
|---|---|
| `model` | 验证损失最低时的模型状态 |
| `mean`、`scale` | 仅用训练集计算的标准化统计量 |
| `features` | 固定为 `x1、x2` 的输入顺序 |
| `classes` | 类别编号到名称的映射 |
| `format_version` | 文件格式版本 |
| `best_epoch` | 最佳参数来自哪一轮 |

这是推理产物，不包含优化器状态，不能宣称从保存点精确续训。

## 一个文件，两个命令

将以下代码保存为 `lesson18.py`。代码只依赖 PyTorch 和 Python 标准库，使用 CPU：

```python
import argparse
import copy
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_model():
    model = nn.Linear(2, 2).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(1) == y).double().mean().item()
    return loss, accuracy


def train(output):
    X = torch.tensor([[2., 0.], [3., 1.], [2., 1.],
                      [0., 2.], [1., 3.], [1., 2.]], dtype=torch.float64)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    X_val = torch.tensor([[1.5, .5], [2., 1.5], [.5, 1.5], [1.5, 2.]],
                         dtype=torch.float64)
    y_val = torch.tensor([0, 0, 1, 1])
    X_test = torch.tensor([[4., 1.], [1., 4.], [1.1, 1.], [1., 1.1]],
                          dtype=torch.float64)
    y_test = torch.tensor([0, 1, 0, 1])

    mean = X.mean(0)
    scale = X.std(0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    loader = DataLoader(TensorDataset((X - mean) / scale, y), batch_size=2,
                        shuffle=True, generator=torch.Generator().manual_seed(7))
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, 101):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_loss, _ = evaluate(model, (X_val - mean) / scale, y_val)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    model.load_state_dict(best_state)
    test_loss, test_accuracy = evaluate(model, (X_test - mean) / scale, y_test)
    artifact = {
        "format_version": 1, "model": best_state,
        "mean": mean, "scale": scale,
        "features": ["x1", "x2"], "classes": ["x1_larger", "x2_larger"],
        "best_epoch": best_epoch,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    check = make_model()
    check.load_state_dict(restored["model"])
    with torch.no_grad():
        expected = model((X_test - mean) / scale)
        actual = check((X_test - restored["mean"]) / restored["scale"])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print(f"best_epoch={best_epoch}, test_loss={test_loss:.6f},"
          f" test_acc={test_accuracy:.2%}")
    print("saved:", output)


def predict(artifact_path, values):
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact["format_version"] != 1 or artifact["features"] != ["x1", "x2"]:
        raise ValueError("Unsupported artifact format or feature order")
    X = torch.tensor([values], dtype=torch.float64)
    if not torch.isfinite(X).all():
        raise ValueError("Inputs must be finite")
    model = make_model()
    model.load_state_dict(artifact["model"])
    model.eval()
    with torch.no_grad():
        probabilities = model((X - artifact["mean"]) / artifact["scale"]).softmax(1)[0]
    index = probabilities.argmax().item()
    print("probabilities:", [round(v, 6) for v in probabilities.tolist()])
    print("class:", artifact["classes"][index])


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--artifact", type=Path, default=Path("classifier.pt"))
    parser.add_argument("--x", type=float, nargs=2, metavar=("X1", "X2"))
    args = parser.parse_args()
    if args.mode == "train":
        train(args.artifact)
    else:
        if args.x is None:
            parser.error("predict requires --x X1 X2")
        predict(args.artifact, args.x)


if __name__ == "__main__":
    main()
```

训练和预测分别执行：

```bash
python lesson18.py train --artifact classifier.pt
python lesson18.py predict --artifact classifier.pt --x 1.1 1
```

这里的 `python` 应指向安装了 PyTorch 的环境。训练命令会在指定位置写入产物；已有同名文件会被覆盖，需要保留旧实验时应换一个文件名。

PyTorch 2.13.0、CPU 的本次输出：

```text
best_epoch=100, test_loss=0.281213, test_acc=100.00%
saved: classifier.pt
probabilities: [0.568536, 0.431464]
class: x1_larger
```

与第十五讲的概率略有不同，因为这次增加了标准化，固定步数下的优化轨迹发生变化。模型仍选对同样四条测试样本，但这些点太少，100% 准确率只是一项流程核对结果。

## 哪一步用了哪份数据

训练集同时用于估计标准化统计量和计算参数梯度。验证集只用于选出最低交叉熵的一轮，不参与反向传播，也不重算均值。选定参数后，测试集评估一次。

代码将测试集直接写在文件里，是为了完整展示实验；训练循环和选模条件都不读取其表现。如果观察测试结果后继续修改方案，这份测试集就不再是未参与决策的最终评估数据。

`copy.deepcopy` 固定最佳时刻的参数。最后先加载这份参数，再保存推理产物，避免误把末轮状态当成最佳状态。

## 怎样确认重新加载没有改变结果

训练命令保存后，立即从文件加载一份新模型，用保存的统计量重新变换测试输入，再比较两份 logits：

```text
原模型 + 原统计量 → expected
新模型 + 文件内统计量 → actual
```

代码要求两者逐值一致。这个检查验证文件保存、加载和预处理是否接得上，不验证模型本身是否预测准确。

预测命令则在新的 Python 进程中运行，不依赖训练函数留下的变量。它检查文件格式和特征顺序，拒绝非有限输入，然后输出概率与类别名称。参数文件仍应来自可信来源。

## 同一个模型，换个输入顺序就可能变成另一个问题

命令 `--x 1.1 1` 表示 `x1=1.1、x2=1`。若调用方把两列颠倒，尺寸仍是 `[1,2]`，PyTorch 不会自动知道语义错了。

因此特征顺序、单位和预处理规则应与模型一并约定。这里仅检查了字段顺序；真实项目还应根据数据来源检查单位、缺失值和允许范围，不能以“代码没有报错”代替数据正确。

十八讲到这里形成一个完整实验：Tensor 表示数据，模型把输入变成预测，损失量化错误，梯度和优化器修改参数；验证集帮助选择，测试集评估留出数据，保存的参数和预处理规则共同支撑下一次预测。

换成更深的网络，这条流程仍然成立。需要重新证明的，是新结构和数据能否带来更可靠的结果。
