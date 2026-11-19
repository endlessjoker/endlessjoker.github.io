# Qwen3.5 复用前缀时还要带上什么

Transformer 与开源大模型 · 第 21 讲

把「行程安排在下个月。第一站是巴黎。第二站是」拆开算，保留前九个 token 的缓存，后面只算「第二」「站」「是」。只带上 K、V，Qwen3.5-0.8B 第四层的输出就会偏离从头重算的结果。这是因为前三层还各自带着一段会跟着输入更新的记忆。

## 第四层保存过去的 K、V，前三层保存过去算出的状态

Qwen3.5-0.8B 的前四层按「三层 Gated DeltaNet 加一层全注意力」排列。第 4 层把前缀的 K、V 留在缓存里，新来的 Query 可以直接读历史位置。前三层则沿着 token 逐步更新状态矩阵。接上新 token 时，还要把因果卷积看到的最近三组投影带过来。卷积核宽 4，当前投影到手后，补上这三组刚好能做一次卷积。[Qwen3.5-0.8B-Base 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json) · [Transformers 的 `Qwen3_5GatedDeltaNet`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

| 前九个 token 算完后 | 实验中的形状 |
|---|---|
| 第 4 层的 K、V | 各 `[9, 2, 256]`，每个 token 有两组 256 维 KV |
| 第 1–3 层的递推状态 | 每层 `[16, 128, 128]`，16 个头各有一张状态表 |
| 第 1–3 层的卷积历史 | 每层 `[1, 6144, 3]`，保留最近三次卷积前的 QKV 投影 |

分段计算时，每层接上自己的旧状态。全注意力的 K、V 沿序列维拼接，新的 Q 按原序列位置旋转，再对新旧 Key 打分。拿掉递推状态或卷积历史，前三层对后缀的输出就会变化，第 4 层收到的查询也跟着变。

## 把「巴黎」换成「伦敦」，能复用几项

两句都被分成 12 个 token，头七个完全相同。第八个是「巴黎」或「伦敦」，余下四个又相同。第九个 token 都是句号，但其第 4 层表征已经经过前三层，受第八个词影响。缓存需要按连续前缀命中。vLLM 给一个缓存块算哈希时，也把前一块的哈希算进去。[vLLM 的《Automatic Prefix Caching》](https://docs.vllm.ai/en/stable/design/prefix_caching/)

下面实跑两句的前四层，比较最后一个 token 的 **1024 维层输出**。差值取 L2 范数。每项的参照都是同一句话从头重算的输出。

| 续算用的缓存 | L2 差值 |
|---|---:|
| 巴黎句，完整复用前 9 项 | 约 0.000001 |
| 巴黎句，前 9 项只带第 4 层 KV | 1.187143 |
| 巴黎句，带递推状态但清空卷积历史 | 0.536753 |
| 伦敦句，只复用与巴黎句相同的前 7 项 | 约 0.000001 |
| 伦敦句，沿用已读过「巴黎」的 9 项状态 | 0.100115 |

```python
import json
from itertools import takewhile
from pathlib import Path
import torch

r = json.loads(Path("transformer-21.json").read_text(encoding="utf-8"))["prefix"]
common = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(r["a_ids"], r["b_ids"])))
print(common)  # 7

for mode, vec in r["reused"].items():
    base = r["full_b"] if mode in ("common_prefix", "stale_prefix") else r["full_a"]
    print(mode, f"{(torch.tensor(base) - torch.tensor(vec)).norm():.6f}")
```

实验在任意 token 边界保存状态，服务端还要看自己在哪些块边界存了可用状态。能共享的计算由**相同的前缀**和**该处已存的缓存**共同决定。[实测记录](/transformer-data/transformer-21.json) · [vLLM 的《Hybrid KV Cache Manager》](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
