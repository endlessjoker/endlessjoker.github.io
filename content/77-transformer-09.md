# DeepSeek 的 DSpark 怎样让草稿接得上

Transformer 与开源大模型 · 第 9 讲

前一讲的小模型要写完「 America」，才能动笔写逗号。DeepSeek-V4.1-Flash 的 DSpark 先并行给五个位置打底分，再从左到右做一个轻量修正。前一格到底选了哪个词，会改掉后一格的词表分数。虽然草稿已经写了五个，目标模型有时只值得验两个。

## 五个位置先打底分，再逐个落笔

DSpark 接入主模型第 37、38、39 层的隐藏表示，拼接后从 `3 × 5120` 维投影回 5120 维。草稿器放进一个已确认的起点 token，后面四格放同一个占位 token。三个草稿 Transformer 块通过 128 位置的滑窗读入上下文，一次算出五个位置各自的词表底分。需要逐格等前一词的，只剩下一条低秩修正支路。[《DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression》§2.4.3](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS3) · [发布版配置与实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6/inference)

上一个词 `a` 在 Markov 头的输入表里有一行 256 维向量。拿这条向量和另一个表中候选词 `b` 的行做点积，就得到在 `a` 后面写 `b` 的分数修正。把这项加到草稿底分，然后才选词。落下这一格，再用刚选的词去修正下一格。[《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》§3.1](https://arxiv.org/html/2607.05147v1#S3.SS1)

```text
b 的新分数 = 底分[b] + dot(转移输入表[a], 转移输出表[b])
```

这个词表有 129,280 行。如果存每对词的独立修正，需要 `129280²`，约 167 亿项。分成两张 `[129280, 256]` 的表之后是 6,619 万项。只查上一个词，历史更深的信息仍由三个草稿块的底分带进来。

## 换上一个词，修正差多少

从官方检查点的 `mtp.2.markov_head` 取几行。单独分词时，「法国」「德国」「巴黎」「柏林」都是一个 token。选两个前词、两个后词，把 256 项逐一相乘相加。[发布版权重清单](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/model.safetensors.index.json)

| 上一词 | 给「巴黎」加的分 | 给「柏林」加的分 | 「巴黎」相对「柏林」的分差改变量 |
|---|---:|---:|---:|
| 法国 | −3.0121 | −7.8091 | +4.7970 |
| 德国 | −6.6426 | −5.0566 | −1.5859 |

前词是「法国」时两项都降分，但「柏林」降得更多，故相对分差提高 4.7970。前词换成「德国」，这一支路则把相对分差压低 1.5859。两行展示的是**底分上的增量**。实际草稿的底分随上下文变化，还要和词表里其它词竞争，不能从表里读出最终选词概率。

保存[实测数据](/transformer-data/transformer-09.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch

d = json.loads(Path("transformer-09.json").read_text(encoding="utf-8"))["dspark"]
e = torch.tensor(d["embed_rows"])[:2]
head = torch.tensor(d["readout_rows"])[:2]
bias = e @ head.T
print([[round(x, 4) for x in row] for row in bias.tolist()])
print([round(x, 4) for x in (bias[:, 0] - bias[:, 1]).tolist()])
# [[-3.0121, -7.8091], [-6.6426, -5.0566]]
# [4.797, -1.5859]
```

## 五个草稿都拿去验，会浪费什么

草稿头还给每格一个估计值 `c[j]`，含义是**前 j−1 格已经通过**时，第 j 格能通过的概率。要估第 j 格真的写出去的概率，就要从第一格乘到第 j 格。比如某条请求前三格的 `c` 是 `0.90、0.75、0.40`，第三格真正存活的概率是 `0.90 × 0.75 × 0.40 = 0.27`。拿 0.40 当作第三格的边际收益，会高估它。[《DSpark》§3.2](https://arxiv.org/html/2607.05147v1#S3.SS2)

再给第二条请求 `0.70、0.40、0.20`。两条的存活率分别是 `[0.90, 0.675, 0.27]` 和 `[0.70, 0.28, 0.056]`。如果总共只多验两格，应该先取两条各自的第一格，让这轮期望写出 `2 + 0.90 + 0.70 = 3.60` 个 token；开头的 2 是两条请求各至少写一个。若再多验一格，最划算的是第一条的第二格，期望上升到 4.275。

多验一格会让目标模型的批次变大。假设两格时每秒能跑 84 轮，三格时只能跑 68 轮，系统吞吐量就从 `3.60 × 84 = 302.4` 降到 `4.275 × 68 = 290.7` token/s。这个例子里的概率与轮速是教学数，目的是算清调度器为什么会停在两格。真正部署会事先量出不同批量的轮速，再用校准过的置信度安排每条请求验多长。

验证预算还要注意一个时序。第 j 格是否送去验证，不能看它实际抽中了哪个词再决定。后一格的置信度已经用到了这次抽样，若用后面的置信度回过头来挑这一格，就会改变送去验证的样本分布。论文的调度器在边际吞吐量第一次不再上升时停下，不继续往后找最高点。[《DSpark》附录 A](https://arxiv.org/html/2607.05147v1#A1)

原论文的线上结果来自 V4-Flash 预览版。在测试的匹配系统吞吐量范围内，作者报告每位用户的生成速度比单 token 草稿 MTP-1 高 60% 至 85%。这是他们的服务数据。本讲只用 V4.1-Flash 的真实权重算 Markov 头，公开参考实现提供草稿前向，未接上完整的生产验证调度。[《DSpark》§5.4](https://arxiv.org/html/2607.05147v1#S5.SS4) · [V4.1-Flash 发布版推理代码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)

[算例权重的许可证](/transformer-data/license-dspark.txt)
