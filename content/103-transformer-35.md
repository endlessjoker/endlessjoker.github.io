# WePocketModel 怎样把 1024 个视觉位置装进 512 个 token

同一张图，语言模型既要知道全图有什么，也得看得清角落里的小字。WePocketModel 1.0 2B 给每张大图留 512 个位置，分成 **8 个全图、64 个区域、440 个细节 token**。小图的原始视觉 token 数不超过 512 时，直接保留原序列。这套压缩叫 MGVB。[《WePM: Hierarchical Conflict Reconciliation for Unified MLLMs》](https://arxiv.org/abs/2611.27365) · [WePocketModel 1.0 2B 官方配置](https://huggingface.co/tencent/WePocketModel1.0-2B/blob/68d8975772f76ea053ce9f696e47b185d51936b4/config.json)

## 4 × 4 的区域为什么读了 36 格

直接构造一张 `32 × 32` 的特征网格，模拟视觉编码器合并后的输出。网格里有 1024 条 2048 维向量。区域支路把它切成 64 个 `4 × 4` 小块。位于内部的区域会多看边缘一圈，参与池化的位置就从 16 个变成 36 个。这样一个笔画跨过格线时，两边的区域都有机会读到它。

每个位置的特征和坐标各做一次投影。问题文本取词向量的均值和末项，拼起来投影成一个向量。同一区域的每个位置都加上这份问题向量，再用 `tanh` 和线性层算分。如果去掉 `tanh`，这部分只会给所有候选加同一个分数，softmax 会把它抵消。带上 `tanh`，问题才可能改动候选之间的分差。[腾讯 WePM 的问题向量构造](https://github.com/Tencent/WePocketModel/blob/main/inference/runtime.py) · [官方 MGVB 池化实现](https://huggingface.co/tencent/WePocketModel1.0-2B/blob/68d8975772f76ea053ce9f696e47b185d51936b4/runtime_sources/mgvb.py)

## 把边缘一圈拿掉再算

取左上坐标为 `(12, 12)` 的区域。用官方权重计算，外围 20 个位置一共分到 **55.43%**。下面用 [实测记录](/transformer-data/transformer-35.json) 算池化前后的一个坐标，只改候选范围。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-35.json").read_text())
scores = torch.tensor(d["logits"][0])
core = torch.tensor(d["core_mask"], dtype=torch.bool)
values = torch.tensor(d["native_coordinate"])
full = scores.softmax(0)
inside_only = scores.masked_fill(~core, -torch.inf).softmax(0)
print(f"{full[~core].sum().item():.2%}")
print(f"{(full @ values).item():.6f}", f"{(inside_only @ values).item():.6f}")
```

变化最大的池化坐标是第 1917 维，保留边缘时为 **−0.008329**，只读区域内部时为 **−0.011590**。随后还有一个小网络的残差，以及区域类别和位置向量。这组固定特征下，换问题造成的区域输出变化很小，边缘一圈的影响更明显。
