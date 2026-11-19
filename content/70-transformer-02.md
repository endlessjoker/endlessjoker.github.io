# RoPE 怎样把距离写进 Q、K 的点积

Transformer 与开源大模型 · 第 2 讲

「法国的首都是巴黎。德国的首都是」里有两个「都是」，在 Qwen3-8B 的首层分别位于 3 和 9。它们从同一行 embedding 出发，投影出来的 K 完全一样。最后一个位置的查询头 1 给这两个位置打分，做完 RoPE 后，两个得分变成了 **2.84851 和 11.20918**。

## 位置 3 和位置 9 实际转成了什么

Qwen3-8B 的每个 Q、K 头有 128 维。实现把第 0 维和第 64 维配成一对，第 1 维和第 65 维配成一对，总共 64 对。每对独立做二维旋转。 [Qwen3 的 RoPE 实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3/modeling_qwen3.py)

```text
原来的坐标是 (a, b)
旋转 α 后是 (a cos α − b sin α, a sin α + b cos α)
```

第 j 对的旋转角是 `位置 × θ^(−2j/128)`。这个检查点的 θ 是 1,000,000。第 0 对每挪一个 token 转 1 弧度，后面的配对转得更慢。同一个位置因此对应一组不同频率的旋转角。 [Qwen3-8B 官方配置](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json)

取首层 KV 组 0 的第 0、64 维。两个「都是」旋转前的坐标都是 `(−4.804094, 5.914950)`。

| 「都是」的位置 | 第 0 对旋转角 | 旋转后第 0 维 | 旋转后第 64 维 |
|---:|---:|---:|---:|
| 3 | 3 弧度 | 3.921299 | −6.533710 |
| 9 | 9 弧度 | 1.939495 | −7.369146 |

第一行的第 0 维就是 `−4.804094 × cos(3) − 5.914950 × sin(3) ≈ 3.921299`。这次旋转发生在归一化后的 K 上。V 保持原样，位置改变的是查询与键的匹配。

## 两个绝对位置怎样变成一个相对距离

设 Q 和 K 在旋转前分别是 q 和 k，位置是 m 和 n。把各配对的旋转合写为矩阵 R。两个向量一起旋转同样的角度，点积保持不变，因此可以从两者的旋转中同时减去 m。

```text
(R(m) q) · (R(n) k) = q · (R(n−m) k)
```

对末位置的查询，m 为 9。两个「都是」对应的 n−m 分别为 −6 和 0。下面的得分用完整 128 维求点积，再除以 √128。

| 键的位置 | 相对位置 n−m | Q、K 都未旋转 | Q、K 按各自位置旋转 |
|---:|---:|---:|---:|
| 3 | −6 | 11.20918 | 2.84851 |
| 9 | 0 | 11.20918 | 11.20918 |

把本次计算的所有位置统一加 5，Q 变成位置 14，两个键变成 8 和 14，相对位置照旧。实测十个得分的最大变化约为 `2.15e-6`，来自浮点计算。在这一步里，q、k 的投影值固定，只有 RoPE 读到的位置编号参加了对照。 [RoFormer: Enhanced Transformer with Rotary Position Embedding，§3.2](https://arxiv.org/abs/2104.09864)

## 解码时把新 token 的位置写回 0 会怎样

假设前 9 个 token 的 K 已经按位置 0 到 8 旋转并缓存。追加第 10 个 token，本应把它的 Q、K 按位置 9 旋转。实验把新位置错设为 0，保持旧缓存原样，并让末位置仍能访问这十个 token。

| 被关注的 token | 正确位置下的权重 | 新位置错设为 0 |
|---|---:|---:|
| 「法国」，位置 0 | 0.00112% | 28.01630% |
| 后一个「首」，位置 8 | 89.79893% | 0.00025% |
| 最后一个「都是」，实际位置 9 | 6.64073% | 69.57967% |

这次查询对自己的得分基本没变，因为最新的 Q、K 一起转到了 0。它与旧键之间的相对角度变了，softmax 重新分配了整行权重。这就是追加单个 token 时，输入长度虽然是 1，位置仍要沿着缓存长度递增的原因。

本讲的记录保存了旋转前的真实 Q、K，可以直接重算这三种情况。

保存[实测数据](/transformer-data/transformer-02.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch

record = json.loads(Path("transformer-02.json").read_text())["rope"]
query = torch.tensor(record["query_before_rotation"])
keys = torch.tensor(record["keys_before_rotation"])
dim = query.numel()
positions = torch.arange(len(keys))
frequency = record["theta"] ** (-torch.arange(0, dim, 2) / dim)

def rotate(x, position):
    phase = position[..., None] * frequency
    angle = torch.cat((phase, phase), dim=-1)
    a, b = x.chunk(2, dim=-1)
    return x * angle.cos() + torch.cat((-b, a), dim=-1) * angle.sin()

def scores(query_pos, key_pos):
    return (rotate(keys, key_pos) @ rotate(query, query_pos)) * dim ** -0.5

normal = scores(positions[-1], positions)
shifted = scores(positions[-1] + 5, positions + 5)
wrong = positions.clone()
wrong[-1] = 0
print((normal - shifted).abs().max().item())
print(round(normal.softmax(0)[8].item() * 100, 5))
print(round(scores(wrong[-1], wrong).softmax(0)[8].item() * 100, 5))
# 约 2.15e-6
# 89.79893
# 0.00025
```

## 有些模型只旋转一部分

Qwen3-8B 旋转整个头。Qwen3.5-0.8B-Base 的完整注意力头有 256 维，实验捕获的正弦和余弦只有 64 维。实现旋转头的前 64 维，再接回原样的 192 维；点积仍用 256 维，缩放仍除以 √256。 [Qwen3.5 的部分旋转实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L615-L628)

DeepSeek-V4.1-Flash 的主注意力头为 512 维，公开推理实现对 Q 和主 KV 向量的**末尾** 64 维做 RoPE。移植时要同时核对整头维度、旋转维度和切片位置。 [DeepSeek-V4.1-Flash 推理配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/config.json) · [官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
