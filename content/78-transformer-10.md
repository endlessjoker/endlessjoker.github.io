# DeepSeek 为什么让长提示少跑二十层

Transformer 与开源大模型 · 第 10 讲

一段 32,768 个 token 的提示交给 DeepSeek-V4.1-Flash。第一步要把历史放进缓存，让模型接着写。但在论文的部署方案里，前面三万多个位置用不着一路跑到底。全部位置通过前 20 层，只有最后 128 个位置再走后 20 层。省下来的那几十万次层计算，来自一次改动：解码器以后要查的全局 KV，可以提前从编码器最后的结果做出来。[《DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression》§2.2](https://arxiv.org/html/2609.19969v1#S2.SS2)

## 编码器怎样把历史交给解码器

层号从 0 开始。输入到第 19 层处理完后，每个位置都有 5120 维表示。第 20 层从这份表示投影出全局 KV，每个位置保留一条 512 维记录，再生成索引键。缓存来源只有它，后面的 19 个解码层都读它。第 6 讲讲的是每一层怎样从其中挑历史，这里要问的是：历史本身由谁算出来？[发布配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/config.json) · [发布权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/model.safetensors.index.json)

```text
每个历史位置 → 因果编码器 20 层 → 第 19 层输出
                                      ↓
                               第 20 层做全局 KV
                                      ↓
                         解码层 20–39 按需查询同一份历史
```

这里的编码器依然戴着因果掩码。读到「巴黎」时，它只知道自己和左边的词，后面接不接「德国」都不会改变已经算好的那份「巴黎」。T5 的编码器会双向读取整条源文本，解码器另外生成目标序列。DeepSeek 的前后两半都走在同一条续写序列上，新生成的每个 token 仍要通过整条主干。这个共享历史的思路来自 [《You Only Cache Once: Decoder-Decoder Architectures for Language Models》](https://arxiv.org/abs/2405.05254)。T5 的原始对照见[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683)。

## 最后 128 个位置为什么还要算

全局 KV 解决了长历史。有一份缓存仍各算各的：**滑窗 KV**。解码层需要从本层的输入表示生成近邻的键和值，无法直接拿编码器输出替代。论文做法是在预填充末尾，取最后 128 个位置的编码结果再跑解码器，同时把滑窗起点截在这 128 个位置内。全局分支仍然能按因果和稀疏规则看更早的缓存。[技术报告 §3.2.2](https://arxiv.org/html/2609.19969v1#S3.SS2.SSS2)

截掉更早的滑窗，会算出略不同的局部状态。用一个只为解释依赖的两层例子。两层都取当前和前两项的均值，第 3 个位置有一个 1，其它全为 0。第一层第 5 个位置看到这个 1，得到 `1/3`；第二层第 7 个位置看到第 5 个位置，得到 `1/9`。如果只从最后三项，即第 5 个位置开始重跑，第 3 个位置就再也传不进来了。

```python
import torch

def window_mean(x):
    return torch.stack([x[max(0, i - 2):i + 1].mean() for i in range(len(x))])

x = torch.zeros(8)
x[3] = 1
for seq in (x, x[-3:]):
    print(f"{window_mean(window_mean(seq))[-1]:.6f}")
# 0.111111
# 0.000000

for n in (128, 512, 4096, 32768):
    full = 40 * n
    replay = 20 * n + 20 * min(n, 128)
    print(n, full, replay, f"{replay/full:.2%}")
# 128 5120 5120 100.00%
# 512 20480 12800 62.50%
# 4096 163840 84480 51.56%
# 32768 1310720 657920 50.20%
```

下半段的数字统计「一个 token 经过一层」的次数。32,768 个位置里，完整跑 40 层要 1,310,720 次，按论文方案是 657,920 次。只有一百来个位置时，几乎省不到这一笔。表里没计索引、投影或内存访问的实际耗时。发布的参考 `Transformer.forward` 仍遍历全部主干层，批量计数来自报告的部署路径，不是拿参考脚本跑出的加速。报告也明确把滑窗回放当近似，并在后训练中模拟了这种截断。[发布版推理代码](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
