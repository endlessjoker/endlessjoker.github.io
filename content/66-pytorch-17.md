# 模型怎样保存 加载和继续训练

看得懂的 PyTorch · 第 17 讲

保存权重，可以让模型再次做出相同预测。要从中断处继续训练，还可能需要保存优化器的记忆。

用带动量的 SGD 做两步训练，就能看出区别。

## 优化器记住了什么

普通 SGD 使用当前梯度更新；本例动量 SGD 还保留一个缓冲量 v。动量系数为 0.9，未启用权重衰减、阻尼或 Nesterov：

```text
第一次：v = 当前梯度
后续：  v = 0.9×上次 v + 当前梯度
更新：  参数 -= 学习率×v
```

输入为 1、2，目标为 2、4，模型 `w×x+b` 从零开始，学习率为 0.01。

第一步梯度为 `w:−10、b:−6`，更新到 `w=0.1、b=0.06`。优化器同时记住缓冲量 `−10、−6`。

第二步预测为 0.16、0.26，误差为 −1.84、−3.74：

```text
w 梯度 = −1.84×1−3.74×2 = −9.32
b 梯度 = −1.84−3.74 = −5.58

w 缓冲量 = 0.9×(−10)−9.32 = −18.32
b 缓冲量 = 0.9×(−6)−5.58 = −10.98

新 w = 0.1−0.01×(−18.32) = 0.2832
新 b = 0.06−0.01×(−10.98) = 0.1698
```

若只恢复权重，却创建一个没有历史状态的新优化器，第二步会得到 `w=0.1932、b=0.1158`。起点相同，更新却不同。

## 保存一份检查点，比较三条路线

代码在第一步后保存文件，然后分别继续原模型、完整恢复、只恢复权重。演示使用临时目录，结束后自动清理检查点：

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from torch import nn

X = torch.tensor([[1.], [2.]], dtype=torch.float64)
y = 2 * X


def make_training():
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, optimizer


def train_step(model, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = (model(X) - y).square().mean()
    loss.backward()
    optimizer.step()


model, optimizer = make_training()
train_step(model, optimizer)
print(f"step1: w={model.weight.item():.6f}, b={model.bias.item():.6f}")

with TemporaryDirectory() as directory:
    path = Path(directory) / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": 1,
    }, path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    resumed, resumed_optimizer = make_training()
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])

    weights_only_model, fresh_optimizer = make_training()
    weights_only_model.load_state_dict(checkpoint["model"])

    train_step(model, optimizer)
    train_step(resumed, resumed_optimizer)
    train_step(weights_only_model, fresh_optimizer)

    for name, current in [("continuous", model), ("resumed", resumed),
                          ("weights_only", weights_only_model)]:
        print(f"{name}: w={current.weight.item():.6f},"
              f" b={current.bias.item():.6f}")
    for original, restored in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(original, restored, rtol=0, atol=0)

    resumed.eval()
    with torch.no_grad():
        print("prediction:", resumed(torch.tensor([[3.]], dtype=torch.float64)).item())
```

PyTorch 2.13.0、CPU 输出：

```text
step1: w=0.100000, b=0.060000
continuous: w=0.283200, b=0.169800
resumed: w=0.283200, b=0.169800
weights_only: w=0.193200, b=0.115800
prediction: 1.0194
```

恢复模型与连续训练的参数逐值一致。最后的预测是 `0.2832×3+0.1698=1.0194`，尚远离目标 6，因为这里只训练了两步。

## state_dict 不包含整套程序

`model.state_dict()` 记录参数和已注册的持久缓冲量，不负责重建 Python 模型结构。加载前仍需创建匹配的 `Linear(1,1)`。

`optimizer.state_dict()` 记录优化器状态和参数组配置。本例的动量缓冲量、学习率等在这里恢复。应先用新模型的参数创建优化器，再加载它的状态；不能让优化器仍指向另一个模型。

`map_location="cpu"` 将文件中的 Tensor 加载到 CPU。`weights_only=True` 限制反序列化的对象范围，并不表示只读取字典里名为 model 的一项，本例的优化器 Tensor 状态仍会恢复。只加载来源可信的文件。

保存和加载接口参见 [Saving and Loading Models：PyTorch 模型与检查点保存教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。

## 推理文件与续训文件分别需要什么

| 用途 | 本系列需要保留的内容 |
|---|---|
| 做预测 | 模型参数、结构约定、特征顺序、标准化统计量、类别映射 |
| 继续训练 | 上述内容，加优化器状态和训练进度 |
| 尽量重现原训练轨迹 | 还需对应随机数状态、采样进度及相关训练组件状态 |

本例没有随机层，且固定使用同一批数据，因此模型和优化器状态就足以对齐下一步。若使用打乱的 DataLoader，还需考虑其生成器状态；若在一轮中途停止，还需知道已经取到了哪里。学习率调度器和混合精度缩放器也有各自状态。

保存一个 epoch 编号，并不自动保存这一切。

## 最佳模型和最后模型不要混淆

验证集最好的参数适合用于最终评估；最近一步的完整状态适合恢复训练。这两种检查点可能对应不同时间。

若拿第 20 轮的最佳模型参数，配上第 30 轮的优化器状态，得到的是混合状态。要精确续训，模型、优化器和进度应来自同一个保存时刻。

第十五讲的 `deepcopy(model.state_dict())` 只在内存里保存最佳参数。真正写入磁盘需要 `torch.save`；下次启动则需要 `torch.load` 和 `load_state_dict`，不能依赖上一次进程中的变量仍存在。
