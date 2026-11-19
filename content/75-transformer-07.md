# Qwen 怎样给每个 token 挑八个专家

Transformer 与开源大模型 · 第 7 讲

「法国的首都是巴黎。德国的首都是」。两个「首」的 token ID 都是 59975。Qwen3-30B-A3B 的首层却给它们分了不同的活：第一次找了专家 73，第二次把它换成了 44。词表只给出同一行向量，专家是在读过上下文后才选的。

## 专家就是一个小前馈网络

这一层先照常做注意力，把结果加回输入，再过 RMSNorm，得到送进 MoE 的 2048 维向量 `x`。路由器乘一张 `[128, 2048]` 的权重，让 128 个专家各得到一个分数。每个专家的活都是把 `x` 变成另一条 2048 维向量。区别在于各自有一套训练出的权重。[《Qwen3 Technical Report》§2](https://arxiv.org/html/2505.09388#S2) · [Qwen3-30B-A3B 官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json)

每个专家先做两组 `2048 → 768` 投影。一路过 SiLU，再与另一路逐项相乘；最后过 `768 → 2048` 投影。这就是 SwiGLU。路由器先用 128 个分数做 softmax，取最大的八项，再把这八项归一化。专家出了结果，才乘这个权重，然后把八条输出加起来。[Transformers 的 Qwen3 MoE 路由和专家实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)

```text
s = W_router @ x
I = topk(s, 8)
a = softmax(s[I])
expert_i(x) = W_down[i] @ (SiLU(W_gate[i] @ x) * (W_up[i] @ x))
y = Σ a[i] × expert_i(x)
```

两次归一化可以简成对选出来的八个分数做一次 softmax。全体分数的归一化分母在第二次相除时约掉了。专家里面的 `W_gate` 只控制本专家内部哪些特征通过；选择专家的是上面的 `W_router`。

## 真正跑一遍首层

官方分词器把这句切成十个 token。读入检查点的词向量和首层权重，按因果注意力算到路由输入。两次「都是」的八位专家相同，名次和占比发生了变化。

| 位置，从 0 开始 | 从高分到低分的前三名 | 第一名的最终权重 |
|---|---|---:|
| 3，法国后的「都是」 | 59、2、15 | 19.22% |
| 9，德国后的「都是」 | 2、15、59 | 18.99% |

取位置 9。第 2 号专家在 128 项里的概率约为 `0.06708`。入选的八项合计只占 `0.35330`，所以重算后是 `0.06708 / 0.35330 ≈ 0.18987`。八个专家的同一输出维是可以直接相加的。第 2 号专家的第 0 维输出约为 `−0.071246`，以 18.99% 加入，单这一项贡献了约 `−0.013527`。加上另七项，得到 `−0.027843`，随后这条结果还要走残差连接。

这一次只需要八个专家的两组上投影和各一行下投影，就能把最后的点积截在第 0 维。以专家 2 的中间第 0 维为例，两个投影分别给出 `0.457718` 和 `−0.226182`。`SiLU(0.457718) × (−0.226182) ≈ −0.063408`，再乘下投影权重 `−0.009644`，对该输出维贡献约 `0.000611`。中间一共 768 项，全部算完才是负的 `−0.071246`。

用[实测记录](/transformer-data/transformer-07.json)复算路由权重与聚合。

```python
import json
from pathlib import Path
import torch

m = json.loads(Path("transformer-07.json").read_text(encoding="utf-8"))["moe"]
p = torch.tensor(m["logits"][9]).softmax(0)
kept, experts = p.topk(8)
weights = kept / kept.sum()
part = weights * torch.tensor(m["expert_output_coord_0"])

print(experts.tolist())
print(f"{kept.sum():.6f} {weights[0]:.6f}")
print(f"{part[0]:.6f} {part.sum():.6f}")
# [2, 15, 59, 39, 63, 97, 120, 20]
# 0.353301 0.189865
# -0.013527 -0.027843
```

## 为什么 30B 不能当成 3B 装进内存

一个专家有 `3 × 2048 × 768 = 4,718,592` 个参数。该层 128 个加起来约 6.04 亿。一次只运行八个，约 3,775 万个专家参数参与一个 token 的计算，另算路由、注意力和其它部分。官方模型卡给出 305 亿总参数、33 亿激活参数。[Qwen3-30B-A3B 模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B)

激活比例描述单个 token。十个 token 一共发出 80 次专家调用，合起来命中了 **33 个**不同的专家，其中最忙的一个收到了六个 token。服务端常把去同一个专家的 token 凑成一批一起算。如果这些专家分在几张卡，激活还要发过去，再把结果收回来。每 token 的算术量、硬件上要放的权重、整批的传输量，要分开估计。

DeepSeek-V4.1-Flash 在这一步多了一个选择用的偏置。它先算 `sqrt(softplus(logit))`，再用加偏置后的分数选 6 个路由专家。被选中之后，混合权重从没加偏置的分数里取，另有一个始终参与的共享专家。这个偏置能改变流量去向，同时不直接放大被它推上榜的专家输出。[DeepSeek-V4.1-Flash 技术报告 §2.1.1](https://arxiv.org/html/2609.19969v1#S2.SS1.SSS1) · [官方 Gate 与 MoE 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
