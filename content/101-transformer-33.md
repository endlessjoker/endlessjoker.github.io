# Olala 的注意力权重为什么会是负数

一次 softmax 会给每个历史位置分一个正权重。Olala 的差分注意力做两次 softmax，用一个头的结果减去另一个头的结果。同一位置在两次计算里都拿到正数，合起来却可能为负。

## 两个查询共用同一批值

Olala 第 5 层是第一层差分注意力。每四个 Q 头共用一组 K、V。前三个是信号头，第四个是噪声头。每个头照常做注意力，三个信号头再各减去同一份噪声读数。扣多少，由当前 token 算出的系数 `g` 决定。

K、V 本身也有一处改动。以 V 为例，每个 token 先生成四个系数和四条 128 维向量。每个 KV 组有自己的系数，所有组共用那四条向量，按 `(a0*b0 + a1*b1 + a2*b2 + a3*b3) / 4` 拼回一条 V。这就是张量积注意力（TPA）在 Olala 里生成 K、V 的方式。实现接下来还会混入上一个 token 的 K、V，再把重建后的张量送入注意力。[《Tensor Product Attention Is All You Need》](https://arxiv.org/abs/2501.06425) · [Olala 官方差分 TPA 实现](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/modeling_olala.py)

## 「法国」拿到 −1.05% 的系数

把「法国的首都是巴黎。德国的首都是」的归一化词向量直接送进第 5 层的注意力支路，使用这一层的实际权重。首个 KV 组的信号头 1 给「法国」 **0.777%**，噪声头 3 给 **3.561%**。以 `g = 0.512843` 相减，得到 **−1.049%**。

用 [实测记录](/transformer-data/transformer-33.json) 算。先把四项加起来重建「法国」的 V 坐标，再算同一坐标怎样进入两个头的最终读数。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-33.json").read_text())
v0 = sum(d["raw_value_terms"]) / d["rank"]
signal, noise = torch.tensor(d["scores"]).softmax(dim=-1)
effective = signal - d["gate"] * noise
values = torch.tensor(d["value_coordinate_0"])
print(f"{v0:.9f}", f"{effective[0].item():.3%}")
print(f"{(signal @ values).item():.9f}", f"{(effective @ values).item():.9f}")
```

「法国」的 V 在第 0 维为 **0.000783311**。乘上负系数，这一项会向下拉读数。全部位置加起来，差分前后的读数从 **−0.001616635** 变为 **−0.000918746**。差分后的所有系数加起来是 `1 - g`。把它们当作有正有负的加权系数，就能看懂这个结果。
