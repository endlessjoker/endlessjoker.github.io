# Olala 的 Mamba-3 为什么在这一步补写上一步

第 31 讲里，Mamba-2 每步先衰减旧状态，再写入当前 token。Olala mini 7A1B 的 Mamba-3 多了一项。读到一个新 token 时，它还会把上一个 token 的写入再加一部分。

## 状态表还是一张

Olala 首层第 0 个头有一张 `64 × 128` 的状态表。每个 token 产生四组写入因子。每组用 64 维向量和 128 维向量做外积，四张结果相加，得到这一步的写入 `U`。读出也分四路，经过门控以后合并。这是 Olala 用的多输入多输出版本，简称 MIMO。

对任意一格，Mamba-3 用三项更新它。旧状态乘保留率 `a`，上一步写入乘 `a * (1 - g) * Δ`，当前写入乘 `g * Δ`。步长 `Δ` 和分配比例 `g` 随输入变。上一步的写入也乘 `a`，因为它跨过了这段时间。推理时只需另外带上上一拍的写入因子，以及用于旋转的累计角度。[《Mamba-3: Improved Sequence Modeling using State Space Principles》](https://arxiv.org/abs/2603.15569) · [Olala 官方实现](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py)

## 旧状态保留，上一次写入单独拿掉

用官方权重跑首层一个头，输入是「法国的首都是巴黎。德国的首都是」。最后一步的 `Δ` 为 **0.335870**，`g` 为 **0.692449**，所以这一步给当前写入的系数是 **0.232573**，给上一拍的系数是 **0.091657**。

状态第 15 行、第 85 列原本是 **2.032554**。上一拍的四路在这一格合计约 **−0.671225**，当前四路合计约 **3.444338**。用 [实测记录](/transformer-data/transformer-32.json) 把三项合起来。

```python
import json
from pathlib import Path

d = json.loads(Path("transformer-32.json").read_text())
a, g, dt = d["decay"], d["trapezoid_gate"], d["delta"]
kept = a * d["pre_update_state"]
previous = a * (1 - g) * dt * sum(d["last_previous_lanes"])
current = g * dt * sum(d["last_current_lanes"])
print(f"{kept:.6f} {previous:.6f} {current:.6f}")
print(f"{kept + previous + current:.6f} {kept + current:.6f}")
```

完整更新得到 **2.543055**。只取消这一步补写的部分，状态就变成 **2.604578**。旧状态仍然带着之前所有写入的结果。这个实验改动的是最后一步的一项，能直接看到它在同一格里抵消了多少新信号。[Olala mini 7A1B 官方模型卡](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B)
