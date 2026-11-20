# Hy4 Flash 的 LatentMoE 把专家入口砍半

Transformer 与开源大模型 · 第 24 讲

Hy4 Flash 仍用完整的 3072 维向量挑专家。挑完后，16 个路由专家拿到的是同一份 **1536 维**输入。它们的结果先在窄空间相加，再做一次投影回到 3072 维。另一个共享专家继续用完整输入。[Hy4 Flash 官方模型卡](https://huggingface.co/tencent/Hy4-flash#model-introduction) · [Transformers 的 LatentMoE 实现](https://github.com/huggingface/transformers/pull/50903/files)

## 第 75 号专家贡献了多少

沿用「法国的首都是巴黎。德国的首都是」。把末尾「都是」的真实词向量归一化，固定为第 1 层 MoE 的输入，层号从 0 开始。用发布权重算出 512 个路由分数，经 sigmoid 后加选择偏置，排进前 16 的有 75、67、395、107 等。选完后从原 sigmoid 分数取权重，归一后再乘配置里的 `3.998`。75 号专家这次的加权系数约为 `0.250759`。[发布配置](https://huggingface.co/tencent/Hy4-flash/blob/b2498eba6a1a3f9bbcab4190bc1313a3c7010c2f/config.json)

实际取 75 号专家的 gate、up 和 down 矩阵。第 404 个中间分量的 gate 是 `4.911687`，up 是 `−4.327164`，裁剪后做 `SiLU(gate) × up` 得 `−21.098392`。把全部 1536 个中间分量送入 down 投影，再经过共用的上投影。这位专家单独对模型输出第 958 维的贡献是 `5.106425 × 0.250759 ≈ 1.280483`，此处还在共享专家和末尾归一化之前。

之所以能算一位专家的贡献，是因为共用上投影是线性的。先给每位专家投回 3072 维再加，相当于先在 1536 维相加再投一次。推理采用后者。

## 砍半以后，还有两张共用矩阵

保持专家中间宽度 1536，一次激活 16 个路由专家。每个专家的 gate、up、down 各有一张矩阵。以一个权重对应一次乘加估算，完整宽度是 `3 × 3072 × 1536 × 16`。窄版的专家矩阵少一半，再加上 `3072 → 1536` 和 `1536 → 3072` 两张共用矩阵。

```python
import json
from pathlib import Path

r = json.loads(Path("transformer-24.json").read_text(encoding="utf-8"))["latent"]
d, l, f, k = (r[n] for n in ("input_width", "latent_width", "intermediate_width", "top_k"))
full = 3 * d * f * k
latent = 3 * l * f * k + 2 * d * l
print(full, latent, f"{1 - latent / full:.2%}")       # 226492416 122683392 45.83%
print(f"{r['unweighted'] * r['router_weights'][0]:.6f}")   # 1.280483
```

**45.83%** 是这个对照里路由分支的矩阵乘加节省。路由打分、共享专家和注意力各有自己的开销。[实测记录](/transformer-data/transformer-24.json)
