# Kimi K3.1 Pro 的残差怎样回看前面的层

普通残差一路做加法。第 1 层和第 20 层的输出加到一起以后，第 26 层拿到的是一个总和。Kimi K3.1 Pro 按 12 层分块，保留已经算完的块，再让每个子层给这些来源打分。这是月之暗面在 [《Attention Residuals》](https://arxiv.org/abs/2603.15031) 里提出的 Block AttnRes。

## Q 是一行学出来的权重

以第 26 层为例，此时可读四条 7168 维向量。第一条是输入 embedding，接着是两个完整的块，最后是第三块目前的部分和。数据属于**同一个 token**。第 25 讲的 iHC 处理的是同层四条残差流之间的传递。这里的四条来源跨越不同深度。

注意力子层和 MLP 子层各有自己的打分参数。对每一条来源，先用 RMSNorm 得到键 K，再与这一子层专属的 Q 点积。Q 是检查点里一行固定的权重。对四个分数做 softmax，得到四个系数。值 V 直接用归一化之前的向量，因此归一化只影响给它多少权重。

## 两个子层拿同样的输入来算

取第 26 层两组真实的 Q 和 RMSNorm 权重。为了只看打分参数的作用，固定四条来自模型词表的诊断向量，先归一化，再分别指定尺度。真实推理经过注意力子层以后，当前块的部分和还会更新。

设归一化的逐维权重是 `gamma`。先算 `source @ (gamma * q)`，再除以该向量的 RMS，等价于先归一化再点积。注意力子层给第三条来源的分数为 `−2.614076 ÷ 1.884952 ≈ −1.386813`。

用同目录的 [实测记录](/transformer-data/transformer-26.json) 复算。记录保存了点积和输出坐标，可读权重的完整程序见复算方法。

```python
import json
from pathlib import Path
import torch

data = json.loads(Path("transformer-26.json").read_text())
rms = torch.tensor(data["source_rms_with_eps"])
values = torch.tensor(data["source_coordinate_0"])
for name, row in data["branches"].items():
    logits = torch.tensor(row["dot_before_rms"]) / rms
    weights = logits.softmax(dim=0)
    percentages = " ".join(f"{weight.item():.2%}" for weight in weights)
    print(name, percentages, f"{(weights @ values).item():.6f}")
```

注意力子层给当前块 **53.81%**，合成向量的第 0 维为 **0.272892**。MLP 给当前块 **56.78%**，同一维降到 **0.210530**。这一维在当前块是负数，在第一个完整块是正数。两边权重一挪，输出就变了。

每个块压成一个部分和，块内各层的输出只能作为整体被回看。这样在 93 层的 Kimi K3.1 Pro 末尾，一次混合至多要读 embedding 加八个块。模型的 DSA 在 token 之间找历史，AttnRes 在一个 token 的不同深度之间选来源。[《Kimi K3: Open Frontier Intelligence》](https://arxiv.org/abs/2607.24653) · [Kimi K3.1 Pro 官方配置](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) · [vLLM 的 Kimi 实现](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/model.py)
