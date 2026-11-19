# Qwen3.5 怎样给图像里的位置编号

Transformer 与开源大模型 · 第 16 讲

一张图切成 8 × 8 块，排进语言模型后，左边一格的块在序列里差 1，上面一格却差 8。Qwen3.5 用三轴位置编码保留它们在图上的距离。用真实权重一算，换回一维编号，某个头对这两个邻居的打分顺序会反过来。

## 序列位置和图上坐标，各管一件事

沿用第 15 讲的输入。图片之前有 3 个位置，64 个图像向量从序列索引 3 开始。语言层给每个位置一组三元坐标 `(T, H, W)`。普通文本用 `(p, p, p)`，图像以 3 为起点，按行、列分别加偏移。静态图只有一帧，图像内的 T 都是 3。

| 位置 | 序列索引 | 送进 RoPE 的坐标 |
|---|---:|---|
| 左上角 | 3 | (3, 3, 3) |
| 右一格 | 4 | (3, 3, 4) |
| 下一行首格 | 11 | (3, 4, 3) |
| 右下角 | 66 | (3, 10, 10) |
| 图像结束标记 | 67 | (11, 11, 11) |

64 个图像向量都占序列位置。RoPE 的文本坐标在图片之后从 11 接着数，因为图的行列坐标最远到 10。整段 71 个位置跑完，下一 token 的 RoPE 坐标是 `(15, 15, 15)`。因果遮罩仍按序列先后决定能看哪里。[Qwen3.5 的 `get_rope_index`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

## 三套坐标怎样进同一个头

Qwen3.5-0.8B 的全注意力头宽 256，只旋转前 64 项。第 `j` 项和第 `j+32` 项成对旋转。32 对各用一个频率，按 T、H、W 交错分配，最终是 11 对用 T、11 对用 H、10 对用 W。Qwen2-VL 曾把三个轴各放在一段频带里，Qwen3-VL 改为交错，让每个轴都分到快慢不同的频率。[《Qwen3-VL Technical Report》§2.1](https://arxiv.org/html/2511.21631v1#S2.SS1) · [Qwen3.5-0.8B-Base 的 RoPE 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json)

实跑视觉塔和前三层语言模型，在第 4 层取第 8 个 Q 头。Query 来自 `(3, 4, 4)`，两个 Key 分别来自左边和上边。固定这三个向量，只换位置编号，再除以 `sqrt(256) = 16` 得到下面的注意力分数。

| Key | 三轴位置 | 序列索引重复三次 |
|---|---:|---:|
| 左边 | 8.6833 | 9.0559 |
| 上边 | 11.2571 | 4.1085 |

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-16.json").read_text(encoding="utf-8"))["mrope"]
x = torch.tensor(r["raw_qk"])              # [query, key_left, key_up]
axis = torch.zeros(32, dtype=torch.long)  # T
axis[1:33:3] = 1                           # H
axis[2:30:3] = 2                           # W
freq = 10_000_000 ** (-torch.arange(32) / 32)

def scores(coordinates):
    angle = torch.tensor(coordinates)[:, axis] * freq
    a, b = x[:, :32], x[:, 32:64]
    z = torch.cat((a * angle.cos() - b * angle.sin(),
                   b * angle.cos() + a * angle.sin(), x[:, 64:]), dim=1)
    return (z[1:] @ z[0]) / 16

for key in ("probe_positions", "probe_flat_positions"):
    print([round(float(v), 4) for v in scores(r[key])])
# [8.6833, 11.2571]
# [9.0559, 4.1085]
```

三轴保住了「上面一格」和「左边一格」的区别。每个方向该得多少分，仍由学到的 Q 和 K 决定。这个实验只改变了一次打分的坐标，测的是排序怎样变化。[实测记录](/transformer-data/transformer-16.json)
