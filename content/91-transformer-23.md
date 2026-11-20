# DFly2 为什么连主模型选好的位置也借

Transformer 与开源大模型 · 第 23 讲

草稿也要看历史。一段提示很长时，DFly2 省掉一轮选址，直接拿主模型选好的位置去读缓存。同一份历史，草稿层依然用自己的 Q 给它打分。[腾讯 Hy4 Flash 的 DFly2 说明](https://huggingface.co/tencent/Hy4-flash#dfly2-for-speculative-decoding)

## 五个草稿层各借哪份

按发布配置，从 0 编号的五个草稿层分别对应主模型的 **25、29、33、37、41** 层。每个草稿层借两样：那一层的 MLA 历史缓存，以及索引器刚选好的 token 地址。vLLM 让草稿层和目标层指到同一块缓存，还保存各层自己的索引快照。草稿块的新位置由草稿层自己先算好 KV，再接在可见历史后面。[发布配置](https://huggingface.co/tencent/Hy4-flash/blob/b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f/config.json) · [vLLM 的缓存共享](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4/nvidia/dfly_target_reuse.py)

每轮配置是 5 个待猜 token，加一个起点，共 6 个槽。对第一个待猜位置，块内能看起点和自己。它们一起与主模型留下的历史竞争权重。[vLLM 的索引拼接](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4/nvidia/dfly_inputs.py)

## 把旧历史折成一个等效项

沿用第 22 讲的 2078 个输入位置，把固定输入送给主模型第 25 层的投影和索引器。它选出 **2046 个历史位置**。再用真实草稿第 0 层的权重，从起点和首个待猜位置生成两条新 KV。这里复算一个头的注意力读出。目标第 25 层和草稿层的输入在实验里固定给定。

历史虽然有 2046 项，在 softmax 加权和里可以精确折成一项。它的分数是旧分数的 `logsumexp`，它的 V 是仅在旧项里做 softmax 后的加权均值。下表的分数都减去了 sink 的分数，加减同一个常数不改变 softmax。

| 项目 | 等效分数 | V 的第 40 项 |
|---|---:|---:|
| 借来的历史 | 7.68793 | 0.108108 |
| 草稿起点 | −1.25587 | −0.222388 |
| 首个待猜位置 | 0.81672 | −0.851148 |
| Sink | 0 | 0 |

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-23.json").read_text(encoding="utf-8"))["dfly"]
z, v = torch.tensor(r["logits"]), torch.tensor(r["values"])
print(f"{z.softmax(0) @ v:.6f}")          # 0.107022
print(f"{z[1:].softmax(0) @ v[1:]:.6f}")   # -0.560768
```

这个头借到历史时读出约 `0.107022`。只留下块内位置和 sink，读出变成 `−0.560768`。历史权重如何分配，由草稿自己的 Query 决定。第 9 讲的 DSpark 在词表分数上加前一词的修正。DFly2 的发布配置还开启了隐藏状态修正，把前一词的词向量与草稿隐藏向量结合，先改隐藏向量，再送进词表投影。[实测记录](/transformer-data/transformer-23.json) · [vLLM 的 DFly2 隐藏状态修正](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4/nvidia/dfly.py)
