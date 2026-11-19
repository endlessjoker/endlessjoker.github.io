# gpt-oss 怎样用 17 字节装下 32 个权重

Transformer 与开源大模型 · 第 20 讲

从 gpt-oss-20b 的一个专家里取出字节 `0x52`。按模型的读法，它表示两个权重，分别是 0.0078125 和 0.0234375。要得到这两个数，还得读同组的一个缩放字节。

## 一字节有两个数，32 个数共用一次缩放

MXFP4 把相邻 32 个权重放一组。每个权重占 4 bit，低三位在 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` 里查幅度，最高位给符号。一个字节先读低四位，再读高四位。每组另外存一个缩放字节 `s`，这一组都乘 `2 ** (s - 127)`。[OpenAI 的 MXFP4 权重解码参考实现](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py)

上面的 `0x52` 先拆出 2，再拆出 5，查到 1 和 3。这组的 `s = 120`，所以 `1 × 2⁻⁷ = 0.0078125`，`3 × 2⁻⁷ = 0.0234375`。同一行的另一组用 `s = 122`，缩放是前一组的四倍。各组选自己的尺度，遇到较大的权重时能抬高范围，细小权重之间的可分辨间隔也随之变大。

## 拆出一个真实专家的一行

沿用第 17 讲的短句，在首层算完注意力和归一化。最后一个 ` is` 的路由选中了 25、26、4、11 号专家，25 号权重最高，约 41.84%。这个专家的 gate/up 合起来有 5760 行，每行 2880 个输入权重，所以每行分 90 组。实际文件把符号和幅度存成 `[5760, 90, 16]` 个字节，缩放另存 `[5760, 90]` 个字节。这里取第 0 行的第 35 组，算它给这一行点积贡献了多少。索引从 0 开始。[《gpt-oss-120b & gpt-oss-20b Model Card》§2.2](https://deploymentsafety.openai.com/gpt-oss/architecture) · [发布版检查点](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee)

```python
import json
from pathlib import Path

r = json.loads(Path("transformer-20.json").read_text(encoding="utf-8"))["mxfp4"]
b = r["blocks"][0]   # gate 第 0 行，第 35 组
lut = (0., .5, 1., 1.5, 2., 3., 4., 6.)
codes = [c for byte in b["packed"] for c in (byte & 15, byte >> 4)]
w = [(-1 if c & 8 else 1) * lut[c & 7] * 2 ** (b["scale"] - 127) for c in codes]
print(w[:4])  # [0.0078125, 0.0234375, 0.0078125, 0.015625]
print(f"{sum(wi * xi for wi, xi in zip(w, b['x'])):.6f}")  # -0.018306
```

把 90 组的点积加起来，再加偏置，gate 的第 0 项约为 `−0.812657`。同法可算出 up 的第 0 项约为 `−1.664626`。两项都落在裁剪范围内，代入发布版的 `gate * sigmoid(1.702 * gate) * (up + 1)`，得到 `0.108295`，随后才进入专家的 down 投影。[OpenAI 的 `swiglu` 实现](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)

一个块用 16 字节放编码、1 字节放尺度，平均每个权重 **4.25 bit**。同样的 32 个数若存为 BF16，要 64 字节。上面这一行的 MXFP4 数据是 `90 × 17 = 1530` 字节，BF16 则是 5760 字节。这笔账只算专家矩阵的权重，检查点还会单独存偏置和注意力张量。[实测记录](/transformer-data/transformer-20.json) · [权重片段许可证](/transformer-data/license-gpt-oss.txt)
