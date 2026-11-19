# DeepSeek 的 MLA 为什么不用展开 K、V

Transformer 与开源大模型 · 第 5 讲

DeepSeek-V2-Lite 的首层有 16 个注意力头，各头仍能读出不同的 V。推理时，每个位置却可以只存一条 **512 维内容向量**和一条 **64 维位置键**。做法是把原来作用在每个旧 K 上的矩阵，挪到这一步的 Q 上。

下面用 V2-Lite 的真实首层权重计算。它保留了 [《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》§2.1](https://arxiv.org/html/2405.04434#S2.SS1) 提出的 MLA。V2-Lite 的头数较少，Q 投影也直接从输入生成，适合把每一项看清楚。[V2-Lite 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json)

## 所有头读同一份内容，各有自己的读法

这次同一句「法国的首都是巴黎。德国的首都是」分成 8 个 token。「法国」和「德国」各是一个 token。首层收到每位置 2048 维输入，按下面的形状投影。表中矩阵按 PyTorch 的 `[输出, 输入]` 排列。[V2-Lite 的 `DeepseekV2Attention`](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py)

| 权重 | 形状 | 输出 |
|---|---:|---|
| `q_proj` | `[3072, 2048]` | 16 头，每头 128 维内容 Q 和 64 维位置 Q |
| `kv_a_proj_with_mqa` | `[576, 2048]` | 512 维内容 `c`，经 RMSNorm 后缓存；另有 64 维共享位置 K，转过 RoPE 后缓存 |
| `kv_b_proj` | `[4096, 512]` | 16 头，每头从 `c` 展出 128 维内容 K 和 128 维 V |

其中一个头展开 K 的权重记为 `Wk`，形状 `[128, 512]`。如果把向量都写成列向量，`Wk @ c` 就是这个头的 K。在一次解码里，同一个新 Q 会拿去和许多旧 K 做点积。括号可以移一下。

```text
先展开旧 K   score = qᵀ @ (Wk @ c)
先变换新 Q   score = (Wkᵀ @ q)ᵀ @ c
```

左式给每个历史位置展开一次 K，或者把展开的 K 一直存着。右式把当前查询变成 512 维，用它去扫缓存中的 `c`。矩阵随头不同，所以就算大家读的是同一份 `c`，分数仍然不同。这和 GQA 里的几个头直接用同一份 K、V 有区别。

位置项要另算。若把 RoPE 直接插到内容 K 上，`qᵀ @ R(旧位置) @ Wk @ c` 中间就多了一个对每个旧位置都不同的旋转。那样没法一次把 `Wk` 吸到 Q 侧。MLA 让 128 维内容分支保持原样，额外存 64 维已旋转的位置 K。两个点积相加，再乘缩放因子做 softmax。

## 用真实 Q 和缓存算一次

取最后一个「都是」的头 0，回看第 3 个位置的「巴黎」，位置从 0 起算。内容分数约为 `−7.1833`，位置分数约为 `2.2605`，两项相加再乘模型含 YaRN 修正的缩放因子 `0.1147214`，该位置的 logit 约为 `−0.56475`。把它和另外 7 个 logit 一起做 softmax，「巴黎」的权重约 **3.0064%**。

这次还有两个「都是」。首层的内容 K 相同，它们与当前 Q 的内容分数都是 `−0.9953`。位置分数却分别是 `3.8283` 与 `8.2031`，最后一个「都是」拿到的注意力权重是前一个的约 **1.65 倍**。这一步的位置区分来自那条单独缓存的 64 维分支。

V 的展开也能推后。先按注意力权重把多个 `c` 加起来，再乘这一头的 `Wv`，结果和先把每个 `c` 变成 V、再加权相同。下面用 `Wv` 第一行运算后的两个中间量复算。一路取每个位置展开后的第 0 维，另一路取先加权再投影时的乘积，每 32 项汇总一组。

保存[实测数据](/transformer-data/transformer-05.json)到代码所在目录，即可复算。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path('transformer-05.json').read_text(encoding='utf-8'))['mla']
def tensor(key):
    return torch.tensor(r[key])

c = tensor('latents')
score = c @ tensor('latent_query')
torch.testing.assert_close(score, tensor('expanded_content_scores'),
                           atol=2e-5, rtol=1e-5)
p = ((score + tensor('pos_keys') @ tensor('pos_query')) * r['scale']).softmax(0)
torch.testing.assert_close(p @ c, tensor('mix_latent_moment'))

print(f'{p[3].item():.6f}')                       # 0.030064
print(f'{p @ tensor("expanded_value_coord_0"):.6f}')  # -0.002738
print(f'{tensor("late_projection_groups").sum():.6f}') # -0.002738
```

完整的首层实验还比较了两条路径的 16 个头和输出投影。CPU 上用 FP32 计算，投影后的最大绝对差 `8.34 × 10⁻⁷`。这里只验证同一层同一组权重可以换计算顺序，未比较整模型速度。

## 缓存是 36 MiB，还是 320 MiB

按单序列、单层、32,768 个已缓存位置、BF16 计算。若每头都存展开的 192 维 K 和 128 维 V，一个位置需 `16 × (192 + 128) × 2 = 10,240` 字节，整层为 **320 MiB**。若只存归一化后的 `c` 和共享位置 K，一个位置需 `(512 + 64) × 2 = 1,152` 字节，整层为 **36 MiB**。就算展开版本只存一份共享位置 K，也要 260 MiB。

最早的 V2-Lite 发布代码里有先展开再写缓存的路径。DeepSeek 的 [V3 官方推理实现](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) 把两条路线写成了 `naive` 和 `absorb`。真正切到 `absorb`，才会少存展开的 K、V。代价是打内容分数时每头要做 512 维点积，原先那部分只有 128 维。节省的是历史读写，至于单步有多快，还要看读缓存和算点积哪边更紧。
