# 四条残差流怎样进同一层注意力

Transformer 与开源大模型 · 第 12 讲

DeepSeek-V4.1-Flash 每个位置在层间带着四条 5120 维向量。某个词进入注意力时，先把四条混成一条，跑一次注意力，出来后再把结果按不同份量加回四条流。为什么要把「怎样混进去」的那组系数提前一个子层算？因为它原本挡在显存读写的关键路径上。[《DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression》§2.4.1](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS1)

## 四条路里还是只跑一次注意力

普通残差是 `x + F(x)`。这里的残差有四行，记作 `X`。`A` 是四个读入系数，`A @ X` 把它们合成一行供注意力或前馈网络用。`C` 用来分配那条新输出，`B` 专门混合旧的四条流。

```text
u = A @ X          [5120]，这次只给子层送一条
y = F(u)           [5120]，F 是一次注意力或一次前馈
X_next = B @ X + C[:, None] * y
```

每个 token 都会从自己的四条流算出这些系数。合起来投影出 `4 + 4 + 16 = 24` 项，权重形状 `[24, 20480]`。两个四维向量分别经过 sigmoid 和两倍 sigmoid。`B` 的 16 个分数则先按行做 softmax，再交替把列和、行和归一化。这就是论文用的 Sinkhorn 迭代。[《mHC: Manifold-Constrained Hyper-Connections》§4.2](https://arxiv.org/html/2512.24880v2) · [发布版 Sinkhorn 内核](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py)

理想状态下，`B` 每行和每列都为 1。行和为 1，每条旧信号的混合结果是四条旧残差的加权平均；列和为 1，每条旧残差被分出去的份量刚好也是 1，所以沿四条流相加，旧信号的总和不变。这个约束管的是绕过 `F` 的那条路，新增的 `C × y` 不受这条求和守恒约束。

## 真实权重做完二十轮，差多少

取 DeepSeek 分词器中的「巴黎」(33870)。第 0 层收到四份相同的词向量，用实际 `hc_attn_fn`、`hc_attn_scale` 和 `hc_attn_base` 生成混合系数。实际 `B` 的第一行约为 `[0.8653, 0.0014, 0.0008, 0.0990]`。按从 0 编号，第 2 行的行和最偏，约 1.0849。列和则已到 1 附近，最大误差约为百万分之一。因为参考内核做固定 20 轮，最后除的是列和，它不会承诺两边都精确等于 1。

下面从[实测记录](/transformer-data/transformer-12.json)拿到真实混合分数，让四条流的某一维临时取四个不同的教学数，观察只经 `B` 混合后的结果。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-12.json").read_text(encoding="utf-8"))["mhc"]
eps = r["eps"]
B = torch.tensor(r["raw_matrix"]).softmax(dim=1) + eps
B = B / (B.sum(dim=0, keepdim=True) + eps)
for _ in range(r["iterations"] - 1):
    B = B / (B.sum(dim=1, keepdim=True) + eps)
    B = B / (B.sum(dim=0, keepdim=True) + eps)

x = torch.tensor([0.5, -0.4, 1.2, -0.3])
y = B @ x
print([round(v, 4) for v in y.tolist()])
print(f"{x.sum():.5f} {y.sum():.5f}")
# [0.4034, -0.2937, 1.1531, -0.2628]
# 1.00000 1.00000
```

## 为什么让前一次混合决定这一次怎么读

旧 mHC 的麻烦在 `A`。每组 `A` 要先看完当前四条流的 20480 项才能算完。等结果出来，还得重新读这四条流，才能把它们混成注意力的输入。即使显卡已经拿到某个维度的数据，也不能马上完成那一维的读入混合，因为权重还没定。

Single-Pass mHC 让本次读入用上一子层已经算好的 `A`。本层依然从当前 `X` 生成自己的 `B`、`C`，同时准备下一个子层的 `A`。发布代码里，本层注意力使用上一次前馈算出的读入系数。本层前馈用的则是本层注意力算出的读入系数。第 0 层注意力没有上一次，于是用 `[1, 0, 0, 0]`。从「巴黎」算出的 `[0.20888, 0.00031, 0.00302, 0.98046]` 留给紧接着的前馈。这些读入系数不做归一化，和 `B` 的要求不同。[发布版 `Block.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py#L970)

以每位置 4 条流、每条 5120 项核算。报告的原多内核方案，每次子层间变换读写 `20 × 5120 = 102,400` 项。即使融合旧 mHC，仍要为当前 `A` 多读一次四条流，合计 71,680 项。改用提前准备的 `A` 后，部署内核做到 `10 × 5120 = 51,200` 项。减半说的是报告所计的混合与预归一化这一段的激活读写，不是整个模型的耗时。[V4.1 技术报告 §2.4.1](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS1)
