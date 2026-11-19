# Qwen3.5 怎样改写已经记住的东西

Transformer 与开源大模型 · 第 4 讲

「法国的首都是巴黎。德国的首都是」。读到最后一个「都是」，Qwen3.5 的首层有个头沿着当前的 K 从旧记忆里读出 **1.542**。这一步想写进去的是 **1.684**。如果直接把新值叠上去，下一次用同一个 K 会读到 **2.448**。模型实际只补了一小步，读数变成 **1.618**。

这一步就是 Gated DeltaNet 的关键。写之前先读，已经记住的部分不用再加一遍。[《Gated Delta Networks: Improving Mamba2 with Delta Rule》§3.1](https://arxiv.org/html/2412.06464#S3.SS1)

## K 用来读旧值，也决定往哪儿写

样章里的完整注意力对每个位置留一对 K、V，新查询要与历史的 K 逐个比。Qwen3.5 的线性注意力层把历史塞进每头一张 `128 × 128` 的矩阵 `S`。向量 `k @ S` 会把 128 行加权相加，读出一条 128 维向量。类似的读法也用在 Q 上，只是 Q 读的是写完这一步之后的状态。

在 [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json) 的首层，每个 1024 维输入先投影为 `3 × 16 × 128 = 6144` 维。每个通道用当前和前三个位置做一次因果卷积，再过 SiLU，分给 Q、K、V。Q、K 做 L2 归一化，Q 还除以 `sqrt(128)`。当前输入再投影出每头的 `a`、`b`。代码用 `α = exp(−exp(A_log) × softplus(a + dt_bias))` 决定旧状态打几折，用 `β = sigmoid(b)` 决定写多少。[Transformers 的首层实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L424-L649)

```text
旧状态先衰减    A = α × S
沿当前 K 读     r = k @ A
算写入的差量    δ = β × (v − r)
更新状态        S = A + outer(k, δ)
沿当前 Q 读     y = q @ S
```

`outer(k, δ)` 是外积，把 `k[i] × δ[j]` 加到矩阵的第 `i` 行、第 `j` 列。同一条 K 再读一次，就会多出 `(k @ k) × δ`。因为 `k @ k` 约等于 1，读数大约向 `v` 靠近了差距的 `β` 倍。若从旧状态读出的已经等于 `v`，写入差量就是零。这个式子也可以从最小化 `½‖k @ A − v‖²` 推出来，`β` 是对临时状态做这一步梯度更新的步长，模型权重没有更新。

## 实算最后一次「都是」

只跑首层，固定最后一个 token，选「是否减去旧读数」差别最大的一维。它在头 2 的第 101 维。头号、位置和维度都从 0 开始。

| 计算 | 头 2，第 101 维 |
|---|---:|
| `α` | 0.998495 |
| `β = sigmoid(0.152507)` | 0.538053 |
| 沿 K 读衰减后的旧状态 `r` | 1.541986 |
| 新输入的 `v` | 1.684128 |
| 写入 `0.538053 × (1.684128 − 1.541986)` | 0.076480 |
| 更新后用同一 K 再读 | 1.618465 |

假如改成 `β × v`，光新写入就会加约 0.906，旧状态那 1.542 还在，于是读数冲到 2.448。实际写入只加约 0.076。再看矩阵的一个格子 `[118, 101]`，该元素衰减后是 `1.026457`，加上 `k[118] × δ[101] ≈ 0.722262 × 0.076480`，得到 `1.081696`。

直接用实测记录里的整组向量复算。`q` 已带归一化与缩放。代码所读的是这一头，不含后续输出门与投影。

保存[实测数据](/transformer-data/transformer-04.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path('transformer-04.json').read_text(encoding='utf-8'))['delta']
q, k, v, alpha, beta = (torch.tensor(r[name]) for name in ('query', 'key', 'value', 'alpha', 'beta'))
S = torch.zeros(k.shape[1], v.shape[1])
j = r['read_dim']
for t in range(len(q)):
    A = alpha[t] * S
    old = k[t] @ A
    S = A + torch.outer(k[t], beta[t] * (v[t] - old))

plain_add = A + torch.outer(k[-1], beta[-1] * v[-1])
print(*(f'{x[j]:.6f}' for x in (old, k[-1] @ S, k[-1] @ plain_add, q[-1] @ S)))
# 1.541986 1.618465 2.448131 0.078017
```

头 2 在这一步保留了旧状态的 99.85%。同一个输入，头 0 的 `α` 约为 0.001016，只留了约千分之一。一次更新怎么影响别的 K，也能直接算。第一个「都是」的 K 和最后一个的 K 点积约为 0.976，因此头 2 沿旧 K 的读数也被这次写入推高了约 `0.976 × 0.07648 ≈ 0.0746`。矩阵中的记忆可以叠到相近的方向上，改一处会带动别处。

## 为什么还留着完整注意力

这个模型的一层 Gated DeltaNet 有 16 个头。每个序列的循环状态共有 `16 × 128 × 128` 个数，FP32 存 1 MiB，另加固定大小的短卷积缓冲。随着前缀从 9 个 token 变成 9,000 个，循环状态的形状不动。逐 token 递推也可以展开成下三角方程，一次求出一段的结果。这两种算法在实测首层上的循环读出最大差约 `2.24 × 10⁻⁸`。把这 9 个 token 拆成前 8 个加最后 1 个，首层注意力投影后的末位置最大差约 `5.96 × 10⁻⁷`。

固定大小是个取舍。历史 K、V 混进了同一张矩阵，已经没有独立的「第 3 个位置」可以点名重读。Qwen3.5 每三层这样的状态层就保留一层能按位置回看的完整注意力。Moonshot 在 [《Kimi Linear: An Expressive, Efficient Attention Architecture》§3、§5.2](https://arxiv.org/html/2510.26692#S3) 走了另一小步：它把一整个头共用的 `α` 拆成沿 K 方向各行不同的衰减率。在论文的消融设置里，三层 KDA 配一层 MLA 的验证困惑度为 5.65，七比一是 5.70。混多少层是要试出来的。
