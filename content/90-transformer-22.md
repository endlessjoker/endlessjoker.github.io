# Hy4 Flash 的 IndexPool 怎样找历史

Transformer 与开源大模型 · 第 22 讲

读到第 2078 个 token，Hy4 Flash 的首层稀疏注意力实际读了 2046 个位置。它先给「四个 token 一组」的索引打分，挑中哪一组，再去读这一组里的 token。[Hy4 Flash 官方模型卡](https://huggingface.co/tencent/Hy4-flash#model-introduction)

## 四个键怎样合成一个

每个 token 先从 3072 维隐藏向量生成一个 128 维索引键。四个键各走两张矩阵，一张出待加权的值，另一张出门控分数。门控还加组内位置偏置，随后沿这四个位置做 softmax。每个通道都可以偏好不同的位置。

用公开检查点的首层权重在 CPU 上以 FP32 复算。实验把同一段 16-token 文本重复到 2078 项。在得分最高的第 518 组，索引键的第 65 维分给四个位置的权重约为 `[0.7196, 0.1139, 0.1320, 0.0344]`，加权结果是 `−0.037900`。这个分量随后和另外 127 维一起做 RMSNorm，再用组首位置做 RoPE。[Transformers 的 IndexPool 实现](https://github.com/huggingface/transformers/pull/50903/files)

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-22.json").read_text(encoding="utf-8"))["index"]
p = torch.tensor(r["gate_logits"]).softmax(0)
print([round(float(w), 4) for w in p])            # [0.7196, 0.1139, 0.132, 0.0344]
print(f"{p @ torch.tensor(r['value_inputs']):.6f}")  # -0.037900
```

## 选中一组，展开成四个地址

索引查询有 16 个 128 维头。每个头对组键做点积，经过 ReLU，再用当前位置给出的系数汇成一个组分数。实验里有 519 个完整旧组，取其中得分最高的 511 组，再保留当前第 519 组。组 `g` 展开成 `4g, 4g+1, 4g+2, 4g+3`。当前组里只有 2076、2077 两个已到达的位置，所以这轮读 `511 × 4 + 2 = 2046` 条。主 MLA 用自己的 Q 给这些 token 的 K 重新打分，才加权读取 V。[vLLM 的组选择与地址展开](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4_flash/nvidia/compressed_indexer.py)

## 一次读多少，和缓存留多少

百万上下文取 1,048,576 个位置。每四个位置合成一条索引时，一个稀疏层最多有 **262,144 条组索引**。主缓存仍按位置保存 MLA 的 512 维潜向量和 64 维位置分支。单层、单序列、按 BF16 每项 2 字节估算，就是 `1048576 × (512+64) × 2 = 1152 MiB`，索引另计。**2048** 约束的是一次注意力从中读取的位置数。42 层里有 12 层做这种稀疏长程读取，另外 30 层只保留 512 个位置的滑窗。[发布配置](https://huggingface.co/tencent/Hy4-flash/blob/b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f/config.json) · [实测记录](/transformer-data/transformer-22.json)

第 6 讲的 DeepSeek 用已经打过分的位置组建候选池，交给后续层重选。这里的分组提前了一步，先合成组键，才开始索引打分。
