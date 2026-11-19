# gpt-oss 为什么让注意力把权重分给零

Transformer 与开源大模型 · 第 17 讲

softmax 通常把全部权重分给可见 token。gpt-oss-20b 让每个头多一个选项，读到的是零向量。实测首层同一个头，短提示把约 **24%** 分给这个选项，长提示里升到约 **92%**。这个选项的分数始终是 4.0625。

## 分母里多出一项

先照常算出可见 token 的分数 `z = K @ q / sqrt(d)`。每个头另有一个训练出来的标量 `s`，叫 attention sink。把 `s` 接在 `z` 后一起做 softmax，再丢掉最后一项，剩余权重乘上 V。等价于给 sink 配一行全零的 V。[《gpt-oss-120b & gpt-oss-20b Model Card》§2.2](https://deploymentsafety.openai.com/gpt-oss/architecture) · [OpenAI 的 PyTorch 参考实现 `sdpa`](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)

```text
Z      = sum(exp(z))
p_sink = exp(s) / (exp(s) + Z)
y      = (1 - p_sink) * (softmax(z) @ V)
```

所以对同一组分数，真实 token 之间的轻重比例保持不变。sink 调的是整个头此刻输出多少。比如不加 sink 时，一个分量是 `−3.01981`，sink 分走 `23.88%` 后，它就变成 `−3.01981 × 0.76124 ≈ −2.29879`。

## 同一个「is」，为什么输出差这么多

短句是「The capital of France is Paris. The capital of Germany is」。长句在中间插入 14 遍「The map shows rivers, hills, and roads.」。最后都是同一个 ` is` token。用 gpt-oss-20b 的实际词向量、首层 QKV、YaRN 和 sink 参数算到注意力输出，挑出变化明显的第 51 号头（从 0 起算）。[gpt-oss-20b 检查点配置](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json)

| 提示 | 首层可见 token | 真实分数的 `log(Z)` | 分给 sink | 输出第 41 项 |
|---|---:|---:|---:|---:|
| 短句，12 个 token | 12 | 5.22198 | 23.88% | −2.29879 |
| 长句，152 个 token | 128 | 1.68288 | 91.53% | −0.26416 |

首层滑窗让长句只看到索引 24–151。参与竞争的位置虽多，分数合起来却更低，`s` 因此分到更多权重。用[实测记录](/transformer-data/transformer-17.json)重算一列 V 就能看到结果。

```python
import json
from pathlib import Path
import torch

r = json.loads(Path("transformer-17.json").read_text(encoding="utf-8"))["sink"]
for prompt in ("short", "long"):
    row = r[prompt]
    z = torch.tensor(row["logits"])
    v = torch.tensor(row["value_column"])
    p = torch.cat((z, z.new_tensor([r["sink_logit"]]))).softmax(0)
    print(f"{p[-1]:.2%}, {p[:-1] @ v:.5f}")
# 23.88%, -2.29879
# 91.53%, -0.26416
```

滑窗先定这个头能读哪些位置，sink 再定这些位置合起来能贡献多少。不同头的输出还要汇进输出投影，表里的比例只描述这一个头在该位置的计算。
