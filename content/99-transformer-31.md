# Nemotron 的 Mamba-2 怎样决定忘多少

Nemotron 3.5 Super VL 的语言主干有 40 层 Mamba-2 和 8 层注意力。Mamba-2 每步把旧记忆乘上一个数，再写进当前输入。固定一句「法国的首都是巴黎。德国的首都是」，首层的前两个头在最后一步分别保留 **99.88%** 和 **72.60%**。用真实权重能追到这两个数是怎么来的。[NVIDIA《Nemotron 3.5 Super VL》官方模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16) · [官方配置](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/blob/fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe/config.json)

## 一个步长管两件事

每个头有一张 `64 × 128` 的状态表。64 行对应 64 个输入通道。每行都是 128 维，整张表共享一个衰减系数 `exp(A × Δ)`。`A` 是头自己的负数参数，`Δ` 是当前 token 算出来的正步长。步长一大，旧状态衰减得更多，这一步的写入也会乘上更大的 `Δ`。写入时取当前输入 `x` 和 128 维的 `B` 做外积。读取时，每行跟另一个 128 维向量 `C` 做点积，再加一条直通项 `D * x`。

实验走首层的预填充公式。最后一个 token「都是」到来时，第 0 个头的 `A ≈ −1.20035`，`Δ` 触及 **0.001** 的下限，保留系数为 **0.998800**。第 1 个头的 `A ≈ −0.05057`，`Δ ≈ 6.33290`，保留系数为 **0.725959**。两头共用同一组 `B、C`，有各自的输入、步长和记忆。[《Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality》](https://arxiv.org/abs/2405.21060) · [Transformers 的 NemotronH 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py)

## 留下来的旧状态贡献了多少

用 [实测记录](/transformer-data/transformer-31.json) 复算第 0 个头、第 0 个通道的 128 维状态。输入投影和四格卷积已用官方权重算好。只清空最后一步之前的状态，保留最后一步的输入，让两次读数可比。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-31.json").read_text())
b = torch.tensor(d["B"], dtype=torch.float64)
c = torch.tensor(d["last_C"], dtype=torch.float64)
state = torch.zeros_like(c)
x_values = d["head_0_channel_0_x"]
for x, dt, write_vector in zip(x_values, d["dt"], b):
    retained = torch.exp(state.new_tensor(d["a"][0] * dt[0]))
    incoming = dt[0] * x * write_vector
    state = retained * state + incoming

skip = d["d"][0] * x_values[-1]
print(f"{(state @ c).item():.9f}", f"{(incoming @ c).item():.9f}")
print(f"{(state @ c).item() + skip:.9f}", f"{(incoming @ c).item() + skip:.9f}")
```

记忆支路保留历史时读出 **0.000124074**，清空后为 **−0.000000733**。加上直通项，门控和输出投影前的结果分别是 **−0.011246181** 和 **−0.011370988**。这里读到的是一路累积的状态，序列再长也沿用同样大小的表。按 FP32 核算，一层的 128 个头共需要 **4 MiB** 状态，局部卷积和注意力层的 KV 另计。
