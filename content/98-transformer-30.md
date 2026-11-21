# Mistral Large 4 怎样学会转快一点

第 2 讲的 RoPE 按公式规定每对坐标转多快。Mistral Large 4 先用 YaRN 调整频率，再让模型为每一对坐标学一个乘数。它还会随查询所在的位置缩放 Q。一个改旋转角度，一个改 softmax 前分数的大小。

## 32 对坐标，32 个乘数

首层每个 Q、K 有 128 维。前 64 维照常点积，后 64 维两两成对做旋转。先按 YaRN 得到每对的基础频率，再乘上 `exp(alpha)`。官方权重里第 0 对的 `alpha = −0.021118`，基础频率是每格 1 弧度，乘完约为 **0.979103** 弧度。其他对的乘数可以大于 1，每对各自学习。[Mistral Large 4 官方权重和配置](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/tree/af26e7869af601441d5cdf75a1272469ddc3d059) · [Transformers 的 Mistral Large 4 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral_large4/modeling_mistral_large4.py)

## 固定向量，改角度和位置

用首层第 0 个头的实际权重，让 `capital` 查询三个固定键 `Paris、Berlin、London`，坐标分别指定为 0、4095、8191，查询在 8192。这是隔离位置算式用的三键实验。只开 YaRN 时，London 的比例是 **78.96%**。乘入学到的频率后变成 **75.97%**。

再加 SSMax。首层第 0 个头学出的系数把 Q 乘上 `0.980469 + 0.007172 × log(位置 + 1)`。把整组坐标往后移 57344 格，距离相同，旋转点积相同，Q 的乘数却从 **1.04509** 升到 **1.06000**。London 的比例也从 **77.39%** 升到 **77.85%**。

用 [实测记录](/transformer-data/transformer-30.json) 算。每对坐标的普通点积记作 `dot`，交叉项是 `q0*k1 - q1*k0`，乘上对应角度的余弦和正弦就得到旋转后的点积。

```python
import json
from math import log
from pathlib import Path
import torch

d = json.loads(Path("transformer-30.json").read_text())
base, alpha, keys, dot, cross, nope = [
    torch.tensor(d[name], dtype=torch.float64)
    for name in ("yarn_inverse_frequencies", "alpha", "key_positions",
                 "pair_dot", "pair_cross", "nope_dot")
]
learned = base * alpha.exp()
position = d["base_query_position"]
gap = position - keys
for frequency, query_position in [(base, None), (learned, None),
                                  (learned, position), (learned, position + d["offset"])]:
    angle = gap[:, None] * frequency
    rotary = (angle.cos() * dot + angle.sin() * cross).sum(-1)
    logits = (nope + rotary) / d["head_size"]**0.5
    if query_position is not None:
        logits *= d["bias"] + d["slope"] * log(query_position + 1)
    print(f"{logits.softmax(0)[2].item():.2%}")
```

真实模型还穿插 NoPE 层，每四层有一层跳过位置旋转，SSMax 仍作用在查询上。同样的长上下文里，一部分层用距离改变点积，另一部分层仍可按查询所在的位置调节分数。[vLLM《Day 0 support for Mistral Large 4》](https://vllm.ai/blog/2026-10-29-mistral-large-4)
