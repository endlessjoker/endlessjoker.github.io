# Dory 同一段权重跑两遍，KV 存几份

Dory 1.0 4B 有 16 个输入子层、40 个循环子层和 16 个输出子层。输入块先给出一份表示。每轮开始时，模型把这份表示加回当前状态，归一化，再跑同一组 40 个子层。跑一轮经过 72 个子层，跑四轮经过 192 个。两种深度对应同一个检查点。[《Dory1.0-4B: A Looped Normalized-GPT》](https://arxiv.org/abs/2611.03998)

## 权重相同，第二轮的 K 也会变

用官方 BF16 检查点，取循环块开头的一层滑窗注意力和紧接着的 MLP，在 CPU 上跑两轮。实验把固定的词向量作为循环入口，保留每轮回加输入和 L2 归一化的计算。用同一句的前 11 个 token 预填充，首个 token 两轮产生的 K，相减后的长度是 **1.108710**。

解码第 12 个 token 时，可以让第一轮读前缀的第一轮 KV，第二轮读第二轮 KV。也可以只存前缀最后一轮的 KV，两轮都读这一份。第二种方式中，第一轮读到的历史不同，但当前 token 的 K、V 仍在当前轮现场计算。

用 [实测记录](/transformer-data/transformer-34.json) 复算第一轮首个头的一维读数。两行分数和两行 V 对应这两种缓存策略。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-34.json").read_text())
probe = d["probe_loop_1"]
for mode in ("per_loop", "last_loop"):
    logits = torch.tensor(probe[mode + "_scores"])
    values = torch.tensor(probe[mode + "_values"])
    print(mode, f"{(logits.softmax(0) @ values).item():.8f}")

bytes_per_token = 2 * d["kv_head_count"] * d["head_dim"] * 2
outer_slots = d["other_local_attention"] * d["window"] + d["other_full_attention"] * 32768
loop_slots = d["recurrent_local_attention"] * d["window"] + d["recurrent_full_attention"] * 32768
print([(outer_slots + copies * loop_slots) * bytes_per_token // 2**20 for copies in (4, 1)])
```

逐轮存时读数是 **−0.00913028**，只存最后一轮时是 **−0.02291231**。跑完第二轮，两种策略的输出向量仍相差 **0.192220**（L2）。同一份真实权重下，省掉缓存会改变读取的内容。

## 32K 上下文省下多少

按配置，循环块有 15 层滑窗注意力和 5 层全局注意力，输入、输出块合计还有 12 层滑窗和 4 层全局。每层每个 token 的 BF16 K、V 占 `2 × 4 × 256 × 2 = 4096` 字节。按滑窗 1024、全局 32768 计算，四轮逐轮存是 **3360 MiB**，只留最后一轮是 **1260 MiB**。这是 KV 张量的容量核算，运行时分配另计。[Dory 官方配置](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/ee517dd388a9b348890ae1e00dacc5eb4d0c750d/config.json) · [Dory 论文 §6.1](https://arxiv.org/html/2611.03998v1#S6.SS1)
