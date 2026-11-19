# Gemma 4 怎样让二十层共用 KV

Transformer 与开源大模型 · 第 18 讲

Gemma 4 E2B 的后 20 层各算自己的 Q，却从前面借 K、V。在 128K 上下文，按 BF16 估算，这个改动把单序列的 KV 从约 **1806 MiB** 压到 **774 MiB**。两层共用同一批 K、V，还能读出不同内容吗？用实际权重算一次。

## 后 20 层去哪里取 K、V

E2B 每五层里有四层滑窗注意力、一层全局注意力。前 15 层各自产生 K、V。之后的层按类型向最后一个同类来源借用，滑窗层都向第 14 层借，全局层都向第 15 层借。第 15 层的 K 已经做了 RoPE，借用层只需旋转自己的 Q。样章的 GQA 让同一层里的多个头共享 K、V。这里省下的是后续层各自再生成、再保存一份 K、V 的开销。[《Gemma 4 Technical Report》§2](https://arxiv.org/html/2607.02770v1#S2) · [Transformers 的 `Gemma4TextAttention`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4/modeling_gemma4.py)

## 用同一份 K、V 算两次

输入仍用「法国的首都是巴黎。德国的首都是」。取这 8 个 token 的实际词向量作固定输入，用第 15 层的权重生成 `K[8,512]` 和 `V[8,512]`。再用第 20、35 层第一个 Q 头的权重各算一次最后一个「都是」的查询。这个小实验固定了各层共同的起始输入，实际模型中各层会收到各自的隐藏状态。

E2B 这里先对 Q、K 做归一化，注意力分数直接用点积，软件中的额外缩放是 1。两次查询读的是同一个 `V[:,431]`，但第 2 个 token「的首」在两次 softmax 中分别拿到 **18.195%** 和 **0.043%**。它对应的 V 分量约为 `−3.30616`，光这一行就会带来不同的读数。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-18.json").read_text(encoding="utf-8"))["shared"]
logits = torch.tensor(r["logits"])     # [2, 8]，各层一行
v = torch.tensor(r["value_column"])    # [8]，两层共用
p = logits.softmax(dim=-1)
print([f"{x:.3%}" for x in p[:, 1]])    # ['18.195%', '0.043%']
print([round(float(x), 5) for x in p @ v])
# [-0.18097, 1.32574]
```

## 省下的 1032 MiB 怎么算

128K 取 131072 个位置，每层只有一个 KV 头。下表按单序列、BF16、滑窗 512 个位置估算缓存张量的容量。全局 KV 每项 512 维，单层 `131072 × 2 × 512 × 2 B = 256 MiB`。滑窗 KV 每项 256 维，单层约 `512 × 2 × 256 × 2 B = 0.5 MiB`。第一个 2 是 K、V 两份，第二个 2 是 BF16 的字节数。[E2B 检查点配置](https://huggingface.co/google/gemma-4-E2B/blob/433813a669639fd7eaa83155828e9000ed357d53/config.json)

| 情况 | 全局缓存 | 滑窗缓存 | 合计 |
|---|---:|---:|---:|
| 假设 35 层各存一份 | 7 × 256 MiB | 28 × 0.5 MiB | 1806 MiB |
| 实际只有前 15 层各存一份 | 3 × 256 MiB | 12 × 0.5 MiB | 774 MiB |

约 **57%** 的节省，主要来自少存四份全局缓存。上层仍执行自己的注意力，仍能给历史分配不同权重。它们各自能读出的内容，受前面两份共享 K、V 的限制。[实测记录](/transformer-data/transformer-18.json)
