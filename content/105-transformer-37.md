# MiniMax H3 Image 为什么每一步都换输出头

MiniMax H3 Image 用六次去噪更新生成图像。共享的 DiT 主干每步都跑一次。到了最后一层线性投影，第 1 步用第 1 套权重，第 6 步用第 6 套。发布的 PDD 文件把这六个头放在一起，配好七个噪声档位。[MiniMax H3 Image 官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) · [官方噪声档位](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/8ced60068623def5b74b4d10558117cb61d6482d/pdd_config.json)

## 一条向量怎么变成去噪量

输出头的权重形状是 `[6, 96, 5376]`。其中 96 来自一个 `2 × 2` 的潜空间图块，每格 24 个通道。第 `i` 步用 `W[i] @ h + b[i]` 算这 96 维的速度，再把速度乘 `sigma[i] - sigma[i+1]`，加到当前潜变量。七个档位不等距，第一步只降约 **0.03248**，最后一步降约 **0.23201**。[Diffusers 的 H3 Image 去噪循环](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/minimax_h3_image/denoise.py) · [MiniMax H3 Euler 调度器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_minimax_h3.py)

## 固定主干输出，只换最后一个头

取一条固定的 5376 维诊断向量，把它分别送入官方六个输出头，以隔离最后一层的作用。首步第 0 维的点积是 **−1.331554**，加偏置 **0.058719** 得到速度 **−1.272835**。从 **0.25** 起步，按第一段噪声差更新，这一维变成 **0.208663**。

用 [实测记录](/transformer-data/transformer-37.json) 走完六步。可以并排看每步换头，以及一直用第一个头的结果。

```python
import json
from pathlib import Path

d = json.loads(Path("transformer-37.json").read_text())
start = d["initial_coordinate_0"]
x, frozen = start, start
velocity = [a + b for a, b in zip(d["projected_dots"], d["bias"])]
sigmas = d["sigmas"]
for i, v in enumerate(velocity):
    step = sigmas[i] - sigmas[i + 1]
    x += step * v
    frozen += step * velocity[0]
    print(f"{x:.6f} {frozen:.6f}")
```

末步的读数分别是 **−1.035225** 和 **−1.022835**。两路从第二步开始分开。完整生图时，每一步的主干输入也跟着潜变量更新。这里固定它，可以单独看到每步输出头和噪声步长分别做了什么。
