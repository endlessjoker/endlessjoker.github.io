# Gemma 4 为什么给每一层另备词向量

Transformer 与开源大模型 · 第 14 讲

「法国的首都是巴黎。德国的首都是」里，两个「都是」进 Gemma 4 E2B 时，主词表查的是同一行。第一层还会去另一张表取一份额外的 256 维向量，这份数据也完全相同。可到了往残差流写入的那一步，`product[84]` 却一次是 `+0.00776`，一次是 `−0.24741`。

## 同一个词，35 份小向量

E2B 的主词表每行 1536 维，词向量从模型入口进入残差流。另有一张表，每行是 `35 × 256 = 8960` 维，35 段依次给 35 层使用。拿「都是」的 token ID 15310 查表，第 1 层取第 0–255 项，第 15 层取第 3584–3839 项。实际读取这两段时，余弦相似度约为 0.0372。查的是同一行，段里的数字各自学习。[《Gemma 4 Technical Report》§2、表 1](https://arxiv.org/html/2607.02770v1#S2) · [E2B 检查点配置](https://huggingface.co/google/gemma-4-E2B/blob/433813a669639fd7eaa83155828e9000ed357d53/config.json)

每层拿到的额外输入 `p` 由两部分组成。第一部分来自这张大词表的一段，模型把它乘 `sqrt(256) = 16`。第二部分把入口处的 1536 维词向量投影到 256 维，乘 `1 / sqrt(1536)` 后做 RMSNorm。这两部分相加再除以 `sqrt(2)`。两个「都是」的入口词向量相同，用同一层的投影也就得到相同的 `p`。

## 这一层先看完上下文，再决定写多少

第一层跑完因果注意力和前馈网络便得到残差 `h[1536]`。这时两个位置已经读到了不同的前文，`h` 可以不同。用它算一个 256 维门，再去乘刚查来的 `p`。

```text
gate    = GELU(W_gate @ h)         [256]
product = gate * p                 [256]
delta   = RMSNorm(W_out @ product) [1536]
output  = layer_scalar * (h + delta)
```

门用的是 GELU，能取负值。第一层的两个 `W` 形状分别是 `[256, 1536]`、`[1536, 256]`。下面用 E2B 首层真实的注意力、前馈与门控权重跑两处「都是」，再把差别最大的乘积分量单独取出来。两个位置的 `p[84]` 都是约 `(−2.640625 + 0.158514) / sqrt(2) = −1.755118`，差别来自门。[Transformers 的 `Gemma4TextModel` 与 `Gemma4TextDecoderLayer`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4/modeling_gemma4.py)

| 位置 | 门的输入 | GELU 后 | 乘以同一个 `p[84]` |
|---|---:|---:|---:|
| 第 3 个 token（法国） | −0.008902 | −0.004419 | +0.007756 |
| 第 8 个 token（德国） | +0.237388 | +0.140965 | −0.247410 |

```python
import json
import math
from pathlib import Path
import torch
from torch.nn.functional import gelu

r = json.loads(Path("transformer-14.json").read_text(encoding="utf-8"))["gemma"]
p = (r["identity"] + r["normed_projection"]) / math.sqrt(2)
gate = gelu(torch.tensor(r["gate_input"]), approximate="tanh")
print([round(float(v), 6) for v in gate * p])
# [0.007756, -0.24741]
```

第 11 讲里，DeepSeek 的 Engram 先用相邻 token 决定去表里读哪一行。Gemma 这里两次都读同一行，前文经过注意力，改变的是这一层怎样使用那行数据。

## 这张表有多贵

额外词表有 `262144 × 8960 = 2,348,810,240` 个参数。按 BF16 存满是 **4.375 GiB**。但为一个 token 查它，只需逻辑上读取那一行的 **17.5 KiB**，再跑各层的投影、门控和残差加法。一个宽前馈层会对每个 token 使用整块权重，这张表按 ID 取行，可以增加每个词能携带的参数，而不必把相同数量的参数都塞进矩阵乘法。整张表依然占存储，运行时读表的代价也取决于它放在哪。官方模型卡因此区分了 E2B 的 **2.3B 有效参数**和**含嵌入表的 5.1B 总参数**。[Gemma 4 官方模型卡](https://gemma.ai/models/gemma-4/model-card/)
