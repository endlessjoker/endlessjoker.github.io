# Gemma 4 Unified 省掉视觉编码器之后

Transformer 与开源大模型 · 第 19 讲

把图片右下角从蓝色改成绿色，左上角什么时候会察觉？在第 15 讲的 Qwen3.5 里，图块进语言模型之前已经互相看过。Gemma 4 12B Unified 把这个过程交给语言层。实跑首层的一个头，左上角的向量读数已经变了。[《Gemma 4 Technical Report》§2.3](https://arxiv.org/html/2607.02770v1#S2.SS3)

## 48 × 48 像素，直接投成一行

模型把相邻的 3 × 3 个 16 × 16 小块排成一个 48 × 48 大块。实验固定 192 × 192 像素，得到 4 × 4 个大块。每块摊开有 `48 × 48 × 3 = 6912` 个 RGB 数值。先做 LayerNorm，再乘 `[3840, 6912]` 的矩阵，把每块投成 3840 维。

```text
x = LayerNorm(Dense(LayerNorm(pixels)))   [3840]
x = LayerNorm(x + pos_x[col] + pos_y[row])
image_vector = W_mm @ RMSNorm(x)          [3840]
```

位置向量按行、列各查一次表，再相加。整个投影逐块执行。所以右下角换色时，交给语言模型的 16 行里，只有最后一行变了。图片仍在原来的 16 个占位位置，换进去的是像素算出的向量。[Transformers 的 `Gemma4UnifiedVisionEmbedder`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py) · [Gemma 4 12B 检查点配置](https://huggingface.co/google/gemma-4-12B/blob/1a2c7729392586f5ecddae8807ee2ef741eabb9a/config.json)

## 第一个图块已经能读到最后一个

12B Unified 的首层是滑窗注意力。同一图像段里的图块在窗口内可以双向读取。实验取左上角的 Query，在第一个头上看它怎样读右下角。两张图的这个 Q 完全相同，区别来自最后一个图块的 K、V。

| 右下角颜色 | 左上角分给右下角的权重 | 左上角读出第 133 项 |
|---|---:|---:|
| 蓝色 | 4.23% | −3.10009 |
| 绿色 | 4.12% | −2.86082 |

权重变化很小，V 的第 133 项却从 `−6.65352` 变成了 `−0.94593`。把遮罩临时改成纯因果，左上角看不到后面的右下角，两张图的读数就同为 `−1.48265`。下面从[实测记录](/transformer-data/transformer-19.json)取首层的分数和一列 V，复算这两个结果。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-19.json").read_text(encoding="utf-8"))["unified"]
causal = torch.tensor(r["visible"]) <= r["probe_query"]
for logits, column in zip(r["logits"], r["value_column"]):
    z, v = torch.tensor(logits), torch.tensor(column)
    actual = z.softmax(0) @ v
    only_earlier = z.masked_fill(~causal, -torch.inf).softmax(0) @ v
    print(f"{actual:.5f}  {only_earlier:.5f}")
# -3.10009  -1.48265
# -2.86082  -1.48265
```

图块投影负责把 RGB 变成语言层能接收的向量，图块之间的关系从语言层开始计算。这次只量到第一个注意力头的读出。发布实现里，同一图像段的双向可见性用在滑窗层，全局层仍沿序列做因果注意力。[Transformers 的图像注意力遮罩构造](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py)
