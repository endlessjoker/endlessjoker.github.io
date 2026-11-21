# LFM3 的七格卷积记得住多远

把句首的 `Paris` 换成 `Berlin`，后面一个 token 都不动。LFM3-1.7B 的首层卷积还能分辨两句话多久？用官方权重算一遍，共同后缀的第六个 token 还有差异，到第七个就完全相同了。

## 七格里放什么

首层先将每个 token 的 2048 维输入投影成三份，记作 `B、C、x`。把 `B * x` 写进窗口，取当前和之前六个位置。每个通道有自己的一组七个卷积权重，点积后再乘上当前位置的 `C`，最后用输出矩阵混合通道。`B` 和 `C` 都是线性投影的结果，可以缩小、放大或翻转信号。

共同后缀是 ` is the capital of a country in Europe. The answer is`。到了第六个 token ` country`，窗口最老的一格仍是城市名。以差值最大的卷积通道 1273 为例，两句在这一格写入 **0.042651** 和 **0.150796**，卷积给它的系数是 **−0.041504**。另外六格一样，当前的 `C` 都是 **0.587797**。这一格就能解释两句在该通道的全部差值。

用 [实测记录](/transformer-data/transformer-29.json) 复算。七行输入按从旧到新的顺序排列。

```python
import json
from pathlib import Path
import torch

data = json.loads(Path("transformer-29.json").read_text())
window = torch.tensor(data["input_windows"])
kernel = torch.tensor(data["kernel"])
for value in (window @ kernel) * data["current_output_gate"]:
    print(f"{value.item():.7f}")
print([round(n, 6) for n in data["operator_l2_differences"][:7]])
```

第六个 token 在通道 1273 的两个结果是 **0.0000928** 和 **−0.0025455**。到了第七个 token ` in`，城市名已经滑出窗口。看整个首层卷积支路的输出，2048 维差向量的长度从 **0.004036** 变成 **0**。逐 token 推进窗口也复算过，在卷积结果上与一次算整句的最大绝对误差约为 **1.5 × 10⁻⁸**。

七格是单层的直接读取范围。下一层还能继续传递前一层留下的信息。LFM3 总共用 13 层这种短卷积、5 层 256-token 滑窗注意力和 6 层全局注意力。近处可以一路推过去，更远的历史也有直接读取的路径。[Liquid AI《Introducing LFM3》](https://www.liquid.ai/blog/lfm3) · [LFM3 官方检查点](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/1da06bec6a819da10b54c67e5c90972b46d81871) · [Transformers 的 LFM3 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm3/modeling_lfm3.py)
