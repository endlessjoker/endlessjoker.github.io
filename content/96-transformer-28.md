# MiMo 怎样把 80 个音频码送进一个位置

文字分词器遇到一个词，通常选出一个或几个词表 ID。MiMo-V2.6 Flash 给一帧音频特征连续选 **20 个** ID。后面的码表每次看到的是前面还没表示出来的部分。这叫残差向量量化，简称 RVQ。

## 第二本码表看到的是什么

音频编码器先输出一条 1024 维的特征。第一本码表有 1024 行，每行也是 1024 维。找距离最近的一行，记下 ID，从原特征里减去这一行。下一本码表在这个差值上再找一次。每级都是两步，`id = ((table - residual)**2).sum(-1).argmin()`，然后 `residual = residual - table[id]`。

从官方权重取齐 20 本码表。诊断输入用真实表行叠加一段小扰动构造，直接从编码器输出的接口开始。第一帧原向量的长度为 **401.54**，第一本选中 **17**，减去这一行后还剩 **234.45**。第二本对这个差值选中 **83**，长度降到 **222.10**。如果给第二本直接看原向量，它会选 **676**。第 20 本结束，这一帧还剩 **87.81**。

## 80 个码先变成四条向量

每帧的 20 个 ID 去查语言模型侧的 20 张 embedding 表。每张取一行，再加起来，得到一条 1024 维向量。这里用的是另一套权重，专门把音频码变成语言模型可处理的特征。

四帧正好查 80 次表，得到 `[4, 1024]`。MiMo 让一个六层小 Transformer 在这四帧内做双向注意力，把输出展平成 4096 维，再经两层投影写到语言模型的一个音频占位符。按官方的 25 帧每秒，时间上平均每 **160 毫秒**对应一个语言模型位置。

代码用 [实测记录](/transformer-data/transformer-28.json) 重算第一帧前三次选码和语言侧的一个坐标。文件中列出各级最接近的三行。完整复算脚本会扫描每一本码表的所有行。

```python
import json
from pathlib import Path
import torch

data = json.loads(Path("transformer-28.json").read_text())
for stage in data["stages"][:3]:
    distance2 = torch.tensor(stage["best_squared_distances"])
    best = distance2.argmin().item()
    print(stage["codebook"] + 1, stage["best_ids"][best],
          f"{distance2[best].sqrt().item():.2f}")

pieces = torch.tensor(data["speech_pieces_at_frame_0_dim_0"])
print(f"{pieces.sum().item():.6f}")
```

第一帧语言侧 embedding 的第 0 维从 `4.5 + 0.625 − 0.412109 − 0.914062 − 1.765625 + …` 用 FP32 加到 **8.095703**。这一步把 20 个离散码合成了一帧向量。四帧保留原来的顺序，交给小 Transformer。[MiMo-V2.6-Flash-MOPD 官方模型卡](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) · [MiMo 音频分词器配置](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/5220c3f513065a0426be5e094571cbed0983386e/audio_tokenizer/config.json) · [MiMo 官方模型实现](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD/blob/5220c3f513065a0426be5e094571cbed0983386e/modeling_mimo_v2.py)
