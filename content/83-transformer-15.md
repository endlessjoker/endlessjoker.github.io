# Qwen3.5 怎样把一张图放进语言模型

Transformer 与开源大模型 · 第 15 讲

Qwen3.5-0.8B-Base 读一张 256 × 256 的图，会在语言模型的输入里占用 64 行。每行都是 1024 维，和普通词向量一样宽。实测只对调图片顶边的两块颜色，64 行全变了。为什么连离改动很远的一块也变了？得从合并前的 256 个小块看起。[Qwen3.5-0.8B-Base 检查点配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json)

## 先把 16 × 16 像素变成一行

图 A 左半红、右半蓝。图 B 把左上角与右上角的 32 × 32 方块对调。两张图的全图 RGB 均值都是 `[127.5, 0, 127.5]`。分块时每边切成 `256 / 16 = 16` 份。处理器把静态图片复制成两帧，凑齐卷积核的时间宽度。每个小块进入模型时就是 `3 × 2 × 16 × 16 = 1536` 个数。

RGB 先除以 255，再减 0.5、除以 0.5。因此纯红块是 `[1, −1, −1]`，纯蓝块是 `[−1, −1, 1]`。卷积核 `[768, 3, 2, 16, 16]` 在每个小块上产出 768 项。拿输出第 0 项来说，这个小块每通道都是常数，可以把第一个卷积核按颜色分别加总，再跟 RGB 做一次三个数的点积。[发布版预处理配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/preprocessor_config.json) · [Transformers 的 `Qwen3_5VisionPatchEmbed`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

红块实测是 `0.170350 − 0.138682 − 0.002904 − 0.040771 ≈ −0.012006`，换成蓝块约为 `−0.346900`。这一步每个块只看自己的像素，交换两个 32 × 32 方块，恰好只改动 `2 × 4 = 8` 行。

## 还没合并，256 个块已经互相看过

起初是 `[256, 768]`。视觉编码器给每个位置加上插值得来的二维位置向量，随后跑 12 层视觉 Transformer。每层把每块的 768 维投影成 Q、K、V，每组又拆为 12 个 64 维的头。Q、K 用二维 RoPE，分别旋转高度和宽度的坐标。单图没有因果遮罩，一个小块的 Query 可以给 256 个 Key 打分，除以 `sqrt(64) = 8` 后做 softmax，再加权读取 V。像素没变的块，经过这一步也能得到不同的向量。[Transformers 的 `Qwen3_5VisionAttention` 与 `Qwen3_5VisionModel`](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

最后，把相邻 2 × 2 的四行各做 LayerNorm，再拼成 3072 维，过一个 `3072 → 3072 → 1024` 的两层 MLP。预处理已经把相邻四块排在一起，所以这里可以直接 `reshape`。

```python
# patches 已按 2 × 2 的邻块排列，形状 [256, 768]
packed = patch_norm(patches).reshape(64, 4 * 768)
image_vectors = fc2(torch.nn.functional.gelu(fc1(packed)))  # [64, 1024]
# ids 为文本、视觉边界和 64 个图像占位 ID 组成的一维序列
inputs = text_embedding(ids)
inputs[ids == 248056] = image_vectors
```

## 64 个同样的 ID，换进去的是 64 个不同的向量

给模型的输入是「图：」加图片，再接「这是什么颜色？」。两个文本片段共 5 个 token，视觉起止标记再占 2 个位置。图像在中间用 ID `248056` 重复 64 次，查词表后把对应的 64 行覆盖掉，最后得到 `[71, 1024]`。两张图的输入 ID 序列完全相同，差异在被写进那 64 行的向量里。

实测第 5 行第 7 列的 32 × 32 方块，它完全没换色。合并后对应图像的第 39 行（索引 38）。在最初的卷积输出里，它对应的四行一模一样。经过全部视觉层和合并器，输出向量的 L2 距离为 **5.30708**。下列代码从[实测记录](/transformer-data/transformer-15.json)重算卷积的第 0 项和这组距离。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-15.json").read_text(encoding="utf-8"))["vision"]
rgb = torch.tensor([[1., -1., -1.], [-1., -1., 1.]])
print((rgb @ torch.tensor(r["rgb_weight_sums"]) + r["patch_bias"]).tolist())
# 约 [-0.012006, -0.346900]
a, b = [torch.tensor(r[name]) for name in ("probe_a", "probe_b")]
print(f"{(a - b).norm():.5f}")  # 5.30708
```

实测到语言模型的输入为止，文本行此刻还完全相同。图像要影响「颜色」的回答，得等后续语言层去读这些位置。按同样的切块和合并尺度，512 × 512 的图会占用 256 行，是本例的四倍。合并前的小块也增至 1024 个，每个视觉注意力头的 Q、K 配对从 `256²` 增到 `1024²`，是十六倍。提高分辨率，视觉编码器和语言模型付出的计算代价不同。
