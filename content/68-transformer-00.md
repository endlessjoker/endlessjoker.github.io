# Qwen3 的四个注意力头怎样共用一组 K、V

Transformer 与开源大模型 · 样章

给 Qwen3-8B 输入「法国的首都是巴黎。德国的首都是」，读到最后一个「都是」时，首层的查询头 1 给它前面的「首」分了 **89.80%** 的权重，头 0 只给了 **1.48%**。它们用的是同一组 K 和 V。

## 两个头各自读到了什么

用官方 tokenizer 切出 10 个 token。下表记录最后一个位置发起查询时，首层第 0 组中前两个头的注意力。位置和头编号从 0 开始。

| 位置 | Token | ID | 头 0 | 头 1 |
|---:|---|---:|---:|---:|
| 0 | 法国 | 104328 | 1.7037% | 0.0011% |
| 1 | 的 | 9370 | 24.1366% | 0.0967% |
| 2 | 首 | 59975 | 2.3296% | 0.0894% |
| 3 | 都是 | 100132 | 18.0894% | 0.0016% |
| 4 | 巴黎 | 106004 | 0.3800% | 0.0000% |
| 5 | 。 | 1773 | 16.3233% | 0.0000% |
| 6 | 德国 | 104114 | 1.6005% | 0.0012% |
| 7 | 的 | 9370 | 28.2472% | 3.3704% |
| 8 | 首 | 59975 | 1.4772% | 89.7989% |
| 9 | 都是 | 100132 | 5.7124% | 6.6407% |

Qwen3-8B 的隐藏维度为 4096。PyTorch 按 `[输出维度, 输入维度]` 保存权重，`q_proj` 是 `[4096, 4096]`，`k_proj` 和 `v_proj` 都是 `[1024, 4096]`。把输出每 128 个数分成一头，就得到 32 个 Q 头和 8 个 KV 头。Q 头 0 到 3 读 KV 头 0，Q 头 4 到 7 读 KV 头 1，依次类推。这个分法叫 GQA。 [Qwen3-8B 官方配置](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) · [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

## 把 89.80% 算出来

根据 token ID 查 embedding 表。查出的向量先过首层 RMSNorm，再分别乘三个权重矩阵的转置，得到 Q、K、V。模型按头归一化 Q、K，再用 RoPE 根据位置旋转这两种向量。 [Qwen3Attention 实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3/modeling_qwen3.py#L185-L249)

位置 9 的头 1 查询，与位置 8 的键，各取前四维如下。实际点积要乘完 128 项再相加。

```text
Q[头 1, 位置 9] = [-5.752970, 0.243228, 3.410489, 2.325048, ...]
K[组 0, 位置 8] = [-4.438119, 3.605790, 1.693468, 0.519630, ...]

点积 ≈ 156.282303
得分 = 点积 / √128 ≈ 13.813535
```

这是该头在这一行的最高分。做 softmax 时先从所有得分里减去这个最大值，位置 8 的分子就是 1。十个指数相加约为 1.113599，所以该位置的权重是 `1 / 1.113599 ≈ 0.897989`。

接着用同一行权重乘 V。位置 8 的 V 第一维为 `-0.060301`，它贡献约 `-0.054150`。十个位置的贡献加起来，头 1 的输出第一维为 **-0.053414**，头 0 则为 **-0.014254**。两头使用相同的 V，改变的是各位置的混合比例。

位置 3 和 9 都是 token 100132。首层两处的 V 一样，K 经过 RoPE 后却不同。头 1 给这两个位置的权重分别是 0.0016% 和 6.6407%。

将[实测记录](/transformer-data/transformer-00.json)放到当前目录，就可以用 PyTorch 重算上面这一行：

```python
import json
from pathlib import Path
import torch

trace = json.loads(Path("transformer-00.json").read_text())["qwen8b"]
q = torch.tensor(trace["last_query"][1], dtype=torch.float32)
k = torch.tensor(trace["keys"], dtype=torch.float32)
v = torch.tensor(trace["values"], dtype=torch.float32)

scores = (k @ q) / q.numel() ** 0.5
weights = scores.softmax(dim=0)
output = weights @ v
print(round(weights[8].item(), 6), round(output[0].item(), 6))
# 0.897989 -0.053414
```

这次实跑到 Qwen3-8B 首层第 0 组的输出投影 `W_o` 之前。用同一组权重对照 Transformers 的 Qwen3Attention，注意力权重和输出均在浮点误差内一致。

## Qwen3.5 怎样少存一些历史

Qwen3.5-0.8B-Base 每四层有三层 Gated DeltaNet 和一层完整注意力。同一句话在它的词表里切成 9 个 token，其中「德国的」合在一个 token 里。用官方权重跑前四层，先读前 8 个 token，再追加最后一个。 [Qwen3.5-0.8B-Base 官方配置](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/blob/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/config.json)

| 层（从 0 起） | 保存的张量 | 读完前 8 个 token | 追加第 9 个后 |
|---|---|---|---|
| 0–2，每层 | 卷积状态 | `[1, 6144, 4]` | `[1, 6144, 4]` |
| 0–2，每层 | 递推状态 | `[1, 16, 128, 128]` | `[1, 16, 128, 128]` |
| 3 | K 和 V，各一个 | `[1, 2, 8, 256]` | `[1, 2, 9, 256]` |

第四层仍是 GQA，8 个 Q 头共用 2 个 KV 头，每头 256 维。前三层更新状态，大小保持固定。以第 0 层为例，递推状态的最大元素变化为 `0.20510`。一次计算 9 个位置，与缓存 8 个再追加 1 个，末位置表示的最大绝对差约为 `3.1e-05`。

第四层的 `q_proj.weight` 是 `[4096, 1024]`，但实际查询只有 `8 × 256 = 2048` 维。每头先切成 256 维 Q 和 256 维门控，注意力算完后，再乘以门控的 sigmoid。缓存里的 K 和 V 各是 `2 × 256` 维。 [Qwen3.5 Gated Attention 实现](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L664-L730)

同样存 32,768 个 token 的 BF16 KV，批量为 1 时，Qwen3-8B 的 36 层需要 `36 × 2 × 8 × 128 × 32768 × 2` 字节，即 **4.5 GiB**。Qwen3.6-27B 的 64 层只有 16 层保存这类 KV，每层 4 个 KV 头、每头 256 维，算下来是 **2 GiB**。这里算的是随序列长度增长的张量，固定大小的状态另计。 [Qwen3.6-27B 官方配置](https://huggingface.co/Qwen/Qwen3.6-27B/blob/8f33b98937c6031bb49e713dee3e8e185aafd26f/config.json)

## DeepSeek-V4.1-Flash 的 890 字节怎么算

DeepSeek 的 CSA2 把共享扩展到了层之间，后续的层能读前面某层生成的**全局** KV。40 层主干只有编号 2、8、14、20 的层生成全局缓存条目。前三个来源每 2 个 token 存一条，层 20 每个 token 存一条。于是平均每个输入 token 对应 `3 × ½ + 1 = 2.5` 条。 [DeepSeek-V4.1-Flash 推理配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/config.json) · [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression，§2.3、§4.2.1](https://arxiv.org/html/2609.19969v1#S2.SS3)

配置里有 8 个 `index_source_layers`。其中四层生成索引 K，另外四层读已有的索引 K，再为自己的查询重选 Top-K。代码用 `Indexer.owns_k` 区分这两种情况。 [DeepSeek-V4.1-Flash 的 Indexer 实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py#L488)

每条里有两份数据。主 KV 是供注意力读取的 512 维向量。索引 K 有 128 维，用于先筛选需要读的条目。按报告使用的 FP4 格式，两个数占一字节，另外每一小组要记一个 1 字节的缩放因子。

| 每条全局缓存 | FP4 数值 | 缩放因子 | 合计 |
|---|---:|---:|---:|
| 主 KV，512 维，每 16 维一组 | 256 B | 32 B | 288 B |
| 索引 K，128 维，每 32 维一组 | 64 B | 4 B | 68 B |

```text
平均每 token 的全局缓存 = 2.5 × (288 + 68) = 890 B
32,768 个 token          = 29,163,520 B ≈ 27.81 MiB
```

这是用配置和报告重算的全局缓存容量，模型还保留各层自己的滑动窗口 KV。这个数对应随上下文变长的全局部分。 [DeepSeek-V4.1-Flash 技术报告，§2.4.4](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4) · [公开推理实现中的 KV 分组和量化](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)

实测环境为 PyTorch 2.13.0、Transformers 5.20.2、CPU，官方权重转为 float32 计算。上方的实测记录只保留复算这行注意力所需的中间量。
