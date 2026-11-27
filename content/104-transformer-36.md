# ERNIE 5.5 的两个路由器怎样挑十个专家

第 24 讲的 LatentMoE 把专家入口砍半。百度 ERNIE 5.5 Flash 又改了挑专家的方式。同一个 token 交给两个路由器，各对 512 位专家打分。模型按专家加起两个分数，再统一挑十位。

## 两次 sigmoid，一次前十

两个路由矩阵都是 `512 × 4096`，各自对同一条 4096 维向量做投影。每个专家拿到两个 sigmoid 后的分数。把两分相加得到 `s`，用 `s + bias` 排名。入选后，拿原来的 `s` 在十位专家中归一化，再乘每位专家各自学到的系数。于是，排名用的偏置和最终相加用的权重是两笔账。[《Multi-View Latent MoE: Letting Every Router Shape Expert Computation》](https://arxiv.org/abs/2610.40577) · [百度 vLLM-ERNIE 的 MoE 实现](https://github.com/baidu/vLLM-ERNIE/blob/main/src/vllm_ernie/models/ernie5_5/nvidia/moe.py)

## 第 22 号专家怎么进了前十

取第一层 MoE 的实际权重，给它同一个固定词向量经过 RMSNorm 后的输入。两位路由器给第 22 号专家 **0.640627** 和 **0.720809**。单独留一个路由器、沿用同样的排名偏置时，它分别排 **22** 和 **12**。两个分数相加，就排到 **第 8**。

用 [实测记录](/transformer-data/transformer-36.json) 重算这个名次，再算它实际乘到专家输出上的权重。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-36.json").read_text())
scores = torch.tensor(d["sigmoid_scores"])
bias = torch.tensor(d["bias"])
joint = scores.sum(dim=0)
for score in (scores[0], scores[1], joint):
    rank = (score + bias).argsort(descending=True)
    print((rank == 22).nonzero().item() + 1)

chosen = (joint + bias).topk(d["topk"]).indices
relative = joint[chosen] / joint[chosen].sum()
weight = relative * torch.tensor(d["learned_scale"])[chosen]
print(f"{weight[chosen == 22].item():.6f}")
```

第 22 号专家的原始合分是 **1.361436**，除以十个入选者的合分 **15.666240**，再乘它自己的 **3.389904**，实际系数是 **0.294591**。两个路由器打完分，只派发给名单里的十个专家。算例算到名单和路由权重为止。[ERNIE 5.5 Flash Preview BF16 官方检查点](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview-BF16/tree/33a9925b17506c5ff57accac2b31a6520c951c4e)
