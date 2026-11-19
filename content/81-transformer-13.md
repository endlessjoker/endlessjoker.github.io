# Kimi Linear 怎样让同一个头记得有长有短

Transformer 与开源大模型 · 第 13 讲

读到第二个「都是」，Kimi Linear 的同一个注意力头把旧状态的一行缩到原来的 **2.01%**，另一行却保留了 **99.999%**。这是 Kimi Delta Attention（KDA）给线性注意力加的自由度。同一时刻，头里的不同方向可以忘得快慢不同。[《Kimi Linear: An Expressive, Efficient Attention Architecture》§3](https://arxiv.org/html/2510.26692v1#S3)

## 一张矩阵怎样存过往

沿用第 4 讲的记法，每个头维护一张矩阵 `S[128, 128]`。行对应 K 的 128 个分量，列对应 V 的 128 个分量。用当前的 `k @ S` 取出一个 128 维向量，可以把它理解为「按这个键，旧状态会读出什么」。新读到的 `v` 与它有偏差，模型就在当前键的方向写入一笔修正。

Qwen3.5 的 Gated DeltaNet 每个头用一个数字衰减整张 `S`。KDA 每个头算出 128 个数，用 `keep[:, None] * S` 给每一行乘上自己的系数。两者的写入系数 `beta` 都是每头一个数。一个键通常在许多行都有分量，所以一行忘得快，说的是某个方向的旧关联衰减得快。

```python
# 单头。q 和 k 已做 L2 归一化，q 再除以 sqrt(128)
faded = keep[:, None] * S
error = v - k @ faded
S = faded + k[:, None] * (beta * error)[None, :]
out = q @ S
```

这几行的顺序很重要。先衰减旧状态，再用衰减后的状态算误差，最后用新状态读出结果。比如某行的 `keep` 接近 0，它的旧数值几乎被抹掉，但当前的 `k × error` 仍可能往那行写入新内容。

## 128 个保留比例从哪来

Kimi-Linear-48B-A3B-Base 有 32 个 KDA 头。当前 token 的 2304 维输入先被投影到 128 维，再从 128 维投影到 `32 × 128 = 4096` 维。把结果按头拆开，各项加上偏置后，经过 `exp(-exp(A_log) * softplus(z))` 得到保留比例。其中 `A_log` 每头一个，`z` 每行一个。Q、K、V 走另一条路，先线性投影，再各做一次宽度为 4 的因果逐通道卷积和 SiLU。当前位置的投影因此能混入前面三个位置的信息。[《Kimi Linear》§4](https://arxiv.org/html/2510.26692v1#S4) · [Moonshot 发布版 `KimiDeltaAttention`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/3b171c17bfc4ee348599b6781a2ca8715c21c8dc/modeling_kimi.py)

## 把两种遗忘法放在同一个头上

输入「法国的首都是巴黎。德国的首都是」。实验提取首层前四个头的实际权重，在 CPU 用 FP32 从零状态递推到最后一个 token。从四个头中选出换成均匀衰减后变化最大的头，再看它变化最大的一列。以下索引从 0 开始，这是头 1、列 107。

| 行 | 旧状态 | 保留比例 | 衰减后 |
|---|---:|---:|---:|
| 27 | 0.268966 | 0.020132 | 0.005415 |
| 99 | 0.010939 | 0.999988 | 0.010939 |

给同一步做一次干预。旧状态、`q`、`k`、`v` 和 `beta` 都沿用原值，只把 128 个保留比例替换成它们的平均值 0.947058。固定一列时，这个分量的误差和读出都只用这一列的状态，能拿[实测记录](/transformer-data/transformer-13.json)独立复算。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-13.json").read_text(encoding="utf-8"))["kimi"]
old, q, k, keep = [torch.tensor(r[name]) for name in ("old", "query", "key", "keep")]

def read_column(retention):
    faded = old * retention
    error = r["value"] - k @ faded
    return q @ (faded + k * (r["beta"] * error))

print(f"{read_column(keep):.6f}")           # 0.031723
print(f"{read_column(keep.mean()):.6f}")    # 0.027337
```

换成均匀衰减后，`k @ faded` 也跟着变，因此差异会经过下一笔写入。这个实验测的是一次读取怎样响应不同的遗忘方式，没跑完整模型的续写。

每个 KDA 层的 32 张 `128 × 128` 状态表，按 FP32 算共 2 MiB，另有短卷积缓存。上下文加长，状态表的尺寸照旧。代价是越来越多的历史要挤进这批状态里。发布版 27 层中有 20 层 KDA 和 7 层全局 MLA。后者仍留有随长度增长的缓存，让少数层可以直接从历史位置取信息。[《Kimi Linear》§4](https://arxiv.org/html/2510.26692v1#S4) · [48B-A3B-Base 检查点配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/blob/3b171c17bfc4ee348599b6781a2ca8715c21c8dc/config.json)
