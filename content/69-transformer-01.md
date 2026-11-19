# 「巴黎」进出模型时读的是同一行吗

Transformer 与开源大模型 · 第 1 讲

在 Qwen3-8B 里，「巴黎」的 token ID 是 106004。它出现在输入时，模型查第 106004 行；它成为下一个 token 的候选时，模型也用第 106004 行。两次用的是不同的矩阵，向量的第 1 维甚至一正一负。

## 同一句话在两个模型里的编号

用两个模型各自的 tokenizer 处理「法国的首都是巴黎。德国的首都是」。Qwen3-8B 切出 10 个 token，Qwen3.5-0.8B-Base 切出 9 个。

| 模型 | 分词结果 | 「巴黎」的 ID |
|---|---|---:|
| Qwen3-8B | 法国 / 的 / 首 / 都是 / 巴黎 / 。 / 德国 / 的 / 首 / 都是 | 106004 |
| Qwen3.5-0.8B-Base | 法 / 国的 / 首 / 都是 / 巴黎 / 。 / 德国的 / 首 / 都是 | 109705 |

ID 是查表用的行号。Qwen3-8B 处理这句话时，先用 10 个 ID 查出 10 行，每行 4096 个数，得到形状为 `[10, 4096]` 的输入。两处「都是」查到同一行，位置差异由后面的注意力处理。

## 读进来和预测出去怎样计算

输入矩阵记为 E，输出矩阵记为 W。「巴黎」出现在输入时，取 `E[106004]`。预测下一个 token 时，取当前最后一个位置经过末层归一化的表示 h，计算 `W @ h`。其中 `W[106004] @ h` 是下一个 token 为「巴黎」的分数，也就是 logit。再对所有候选分数做 softmax，才得到概率。

```text
输入 token 的 ID → E[ID] → 多层计算 → h → W @ h → 词表上的 logits
```

从 Qwen3-8B 的两个官方权重分片取出的数如下。维度从 0 开始，表里各截前四维。

| 参数中的第 106004 行 | 维度 0 | 维度 1 | 维度 2 | 维度 3 |
|---|---:|---:|---:|---:|
| `model.embed_tokens.weight` | 0.015747 | 0.022461 | 0.018921 | 0.022339 |
| `lm_head.weight` | 0.006470 | −0.007721 | 0.035645 | 0.020874 |

完整两行各有 4096 维，余弦相似度约为 **0.01931**。模型内的输入和输出使用同一套 token 编号，各自保存一套权重。 [Qwen3-8B 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) · [官方权重索引](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/model.safetensors.index.json)

做一个受控计算。将 h 的第 0 维增加 1，其余维度保持原值。「巴黎」的 logit 增加 0.006470，「德国」的增加 0.022095。两者的分数差扩大 0.015625，因此 softmax 后「德国」相对「巴黎」的概率比乘上 `exp(0.015625) ≈ 1.01575`。这一步只计算分数的变化，任何原始 h 都适用。

在本讲目录用实测记录重算这两项。

保存[实测数据](/transformer-data/transformer-01.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch
from torch.nn import functional as F

record = json.loads(Path("transformer-01.json").read_text())["embedding"]
embedding = torch.tensor(record["qwen8b_paris_input_row"])
readout = torch.tensor(record["qwen8b_paris_output_row"])
print(round(F.cosine_similarity(embedding, readout, dim=0).item(), 6))

delta_gap = record["qwen8b_germany_output_first"] - readout[0]
print(round(delta_gap.item(), 6), round(delta_gap.exp().item(), 6))
# 0.019310
# 0.015625 1.015748
```

## 为什么有的模型共用一张表

在 Qwen3.5-0.8B-Base 中，输入与输出绑定到检查点里的同一张表 `model.language_model.embed_tokens.weight`。把「巴黎」读入时，取第 109705 行；给「巴黎」打分时，也用这一行和 h 做点积。它的前四维实值约是 `[0.017822, 0.003235, −0.013489, −0.008728]`。 [Qwen3.5-0.8B-Base 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json) · [Transformers 的权重绑定实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1595-L1606)

训练时，假设「法国的首都是」后面应预测「巴黎」。这一步的损失会反馈到输出矩阵的「巴黎」行，让它以后更容易给类似的 h 打高分。输入矩阵的「巴黎」行还没有被这段前缀用到。权重共用时，输出端的这次更新也会改变以后读入「巴黎」时的向量。分开就可以独立调整这两个角色。 [Using the Output Embedding to Improve Language Models，§3](https://arxiv.org/abs/1608.05859)

共用的收益可以直接数参数。下表按模型配置里的行数计算，BF16 每项 2 字节。

| 模型 | 一张表的形状 | 一张表的参数 | 检查点使用的表数 | 这部分 BF16 大小 |
|---|---:|---:|---:|---:|
| Qwen3-8B | 151,936 × 4,096 | 622,329,856 | 2 | 约 2.32 GiB |
| Qwen3.5-0.8B-Base | 248,320 × 1,024 | 254,279,680 | 1 | 485 MiB |

光看 Qwen3-8B，一张表就约 1.16 GiB。共用还能把输入与输出的梯度集中到同一份参数。独立保存则多出一套可训练的坐标。两种设计怎样影响效果，需要在相同规模和训练条件下比较。

[算例权重的许可证](/transformer-data/license-qwen3.txt)
