# Qwen3.5 为什么在注意力输出上加门

Transformer 与开源大模型 · 第 3 讲

Qwen3.5-0.8B-Base 读「法国的首都是巴黎。德国的首都是」。在最后一个「都是」的位置，第一层完整注意力里，头 0 有一维只保留了原结果的 **0.15%**，另一维保留了 **90.63%**。同一个头已经读完同一批 token，为什么还要对输出再选一次？

## q_proj 的 4096 个输出去了哪里

前四层中，第 4 层是第一次完整注意力，也就是从 0 编号的层 3。它收到前面三层处理过的表示，做完层归一化后，每个位置有 1024 维。对于这句 9 个 token 的输入，`q_proj` 的形状变化如下。 [Qwen3.5-0.8B-Base 官方配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json)

```text
[1, 9, 1024]  经 q_proj 变成 [1, 9, 4096]
              按头排列为   [1, 9, 8, 512]
              每头内切开   256 维 Q | 256 维门控
```

平铺的前 256 项是头 0 的 Q，接着 256 项是头 0 的门控，随后才是头 1 的 Q。先按头 reshape，再沿最后一维切分。直接把 4096 从中间切成两个 2048，会切在头 3 与头 4 之间。 [Qwen3.5 的注意力实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L664-L730)

Q 像样章里一样与 K 算分数，在历史位置上做 softmax，再加权读出 V。门控取自当前查询位置的表示，沿 V 的**特征维**乘到这个读出结果上。把所有头拼接后，再过输出投影。

```text
读出的向量 s = Σ(位置权重 × V)
门值 m       = sigmoid(g)
送入输出投影  = s × m        逐元素相乘
```

位置权重的一行加起来为 1。每个特征的 m 可以独立接近 0，头已经分完的注意力仍然可以少写回一些信息。

## 同一个头的两维怎么算

用前四层官方权重跑完整句，在最后一个「都是」的位置截取头 0。下表的 g 和 s 来自同一次前向，通道编号从 0 开始。

| 通道 | 门控投影 g | sigmoid(g) | 读出的 s | 乘门后的结果 |
|---:|---:|---:|---:|---:|
| 92 | −6.470562 | 0.001546 | −0.217632 | −0.000336 |
| 134 | 2.269088 | 0.906284 | 1.133577 | 1.027343 |

第 92 维的门值为 `1 / (1 + exp(6.470562)) ≈ 0.001546`，再乘以 `−0.217632`，只剩约 `−0.000336`。这一位置有 `8 × 256 = 2048` 个门值，其中 338 个小于 0.01。换一段输入，门值由新表示重新计算。

再把结果送进同一张输出矩阵。输出第 0 维使用 `o_proj.weight[0]` 与拼好的 2048 维向量做点积。用实测记录重算。

保存[实测数据](/transformer-data/transformer-03.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch

trace = json.loads(Path("transformer-03.json").read_text())["gated_attention"]
g = torch.tensor(trace["gate_logits"])       # [8, 256]
s = torch.tensor(trace["mixed_values"])      # [8, 256]
w = torch.tensor(trace["o_proj_first_row"])  # [2048]

gate = g.sigmoid()
gated = s * gate
print([round(gated[0, i].item(), 6) for i in (92, 134)])
print(round(w.dot(gated.flatten()).item(), 6))
print(round(w.dot(s.flatten()).item(), 6))
# [-0.000336, 1.027343]
# -0.081206
# 4.694545
```

带门的输出与 Transformers 前向的对应结果一致。最后一行把门全设为 1，固定已经算好的 Q、K、V 和输出矩阵，只观察这一层的变化。每项虽都乘了不大于 1 的数，输出投影里却有正负贡献。两边缩小的幅度不同，最终点积可以变号。

## 为什么门放在加权之后

普通注意力先把每个历史位置的 V 按权重相加，再乘输出矩阵。如果门加在每个 V 上，门取决于当时写入缓存的位置。放在加权之后，门可以由**当前读者**决定，同一份 V 被不同查询读到时，可以写回不同的特征。

这样做要多一组投影权重。本例每个完整注意力层多出 `2048 × 1024 = 2,097,152` 个门控参数。门在当前步用完，历史仍只需为这一层的两个 KV 头保留 K、V。

论文做了一个同量级的参数对照。三组都用 4000 亿 token 训练 15B MoE，后两组各增加约 2.01 亿参数。困惑度 PPL 越低，代表模型给留出数据中正确 token 的平均对数概率越高。 [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free，表 1](https://arxiv.org/html/2505.06708#S3.T1)

| 训练设计 | 平均 PPL | MMLU |
|---|---:|---:|
| 基线，32 个 Q 头 | 6.026 | 58.79 |
| 增至 48 个 Q 头 | 5.953 | 58.45 |
| 在注意力输出加逐元素 sigmoid 门 | 5.761 | 60.82 |

同篇论文还量了模型分给首 token 的平均注意力，从基线的 46.7% 降到带门组的 4.8%。这是重新训练后学到的权重分布。当前层的门是在 softmax **之后**相乘，本次前向中把门设成 1，只会改变写回的向量。 [同篇论文，§4.3 与表 4](https://arxiv.org/html/2505.06708#S4.SS3)

[算例权重的许可证](/transformer-data/license-qwen35.txt)
