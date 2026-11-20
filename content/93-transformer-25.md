# Hy4 Flash 的 iHC 怎样让旧残差各走原路

Transformer 与开源大模型 · 第 25 讲

Hy4 Flash 同时带四条 3072 维残差。注意力先把四条合成一条来算，出来后把新结果加回四条。它把旧的四条残差原样送到加法处。这就是 iHC 里的 identity。[Hy4 Flash 官方模型卡](https://huggingface.co/tencent/Hy4-flash#model-introduction) · [vLLM 的 iHC 实现](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4/nvidia/hc.py)

## 少掉的是哪张矩阵

设旧残差是 `X[4,3072]`。先把它摊成 12288 项，投出八个分数，四个用来读入，四个用来写回。读入系数 `a` 负责合成一条输入，子层输出一条 `y`，再按写回系数 `c` 分出去。

```text
u = a @ X
y = Attention(u)          # 也可以是一次前馈
X_next = X + c[:, None] * y
```

第 12 讲的 DeepSeek mHC 则让旧残差先走 `B @ X`，再加同样形式的增量。其中 `B` 是一张 4 × 4 混合矩阵，需做 Sinkhorn 归一化。iHC 对这条旧路径用单位矩阵，所以这 16 项也从门控投影里省掉。

用混元第 0 层 iHC 的实际权重，把句子末尾四个 token 的归一化词向量当作四条固定输入。取第 1131 维，四个旧数约为 `[2.1465, −2.4730, 1.0953, 0.3086]`。给定同一个教学用增量 `y = 0.5`，只切换旧残差走原路还是走第 12 讲实测的 `B`。这样看到的差异来自旧路径本身。[实测记录](/transformer-data/transformer-25.json)

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-25.json").read_text(encoding="utf-8"))["ihc"]
old = torch.tensor(r["old"])
add = torch.tensor(r["outgoing"]) * r["selected_output"]
B = torch.tensor(r["deepseek_mhc"])
print([round(float(v), 4) for v in old + add])       # [2.1666, -2.4405, 1.1232, 0.358]
print([round(float(v), 4) for v in B @ old + add])   # [1.9055, -1.9033, 0.8844, 0.3207]
```

## 一份 Engram 可以写出四种增量

这四条流也能各自决定收下多少词组记忆。末尾「都是」用二元组、三元组各查 8 行，每行 96 维，拼成 1536 维。模型只投影一次共享值，再用四条残差各自算门，另外各有一项逐维缩放。实测共享值的第 2156 项约为 `2.315379`，最后写入四条流的增量约为 `[0.047615, 0.257622, 0.819732, 0.633262]`。在同一份记录里，逐条用 `shared * gate[i] + branch_addition[i]` 就能复算。词组怎样变成哈希地址，沿用第 11 讲的思路。[vLLM 的 Hy4 Flash Engram 实现](https://github.com/vllm-project/vllm/blob/0fda86a35419c9c70dbfada224c838c10fbb62ad/vllm/models/hy_v4_flash/nvidia/engram.py)
