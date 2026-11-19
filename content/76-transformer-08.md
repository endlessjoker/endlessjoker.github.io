# 投机解码猜错一个，为什么后面也要丢

Transformer 与开源大模型 · 第 8 讲

给 Pythia-14M 一段提示 `The United States of`，它起草了四个 token：` America`、`,`、` and`、` the`。把草稿交给 Pythia-31M，前两个都对齐了，第三个它想写的是 ` the`。第四个位置上，两个模型虽然都填了 ` the`，也必须把草稿的第四个丢掉。因为目标模型给第四个位置打分时，假设前一个字是 ` and`。

这里用 EleutherAI 发布的两个完整检查点做 CPU 实验，共用一套分词器，模型内部仍是自回归解码。31M 担任目标模型，只负责判定这次续写该是什么。[《Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling》](https://arxiv.org/abs/2304.01373) · [14M 检查点](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac) · [31M 检查点](https://huggingface.co/EleutherAI/pythia-31m/tree/e556ace21b489575e94e9d50b6dad2fcc7419679)

## 一次前向，拿到五行分数

每个输入位置的输出都在给**下一个** token 打分。请求进来，缓存提示词的前三个 token，把最后一个 ` of` 留作这轮验证的起点。草稿已经有了，目标模型可以一口气吃进下面五项。因果掩码让第 i 行只看得到这一行及之前的输入。五行并排算，每行仍在预测自己的下一个 token。[《Fast Inference from Transformers via Speculative Decoding》§2](https://arxiv.org/abs/2211.17192)

| 目标模型本行读进的 token | 草稿猜的下一个 | 目标模型分数最高的下一个 |
|---|---|---|
| ` of` | ` America` | ` America` |
| ` America` | `,` | `,` |
| `,` | ` and` | ` the` |
| ` and` | ` the` | ` the` |
| ` the` | 到此没有草稿 | ` United` |

这里的行是同一条序列的不同位置。第三行出错后，第四行起的输入都建立在错误前缀上，连最后那一行的 ` United` 也不能沿用。确认写出去的是 ` America, the`。新前缀共七个 token，裁到六个已确认位置的 KV。刚纠正出来的 ` the` 还没喂给目标模型，留作下一轮第一项；草稿末尾碰巧也有一个 ` the`，它在不同的位置，前面还多了错误的 ` and`。

下一轮草稿的 ` United`、` States`、` of`、` America` 四项全对，目标模型最后一行还能直接送出一个逗号。这时一次验证就写出 **五个** token。

## 拒绝以后从哪里接着写

下面读[实际解码记录](/transformer-data/transformer-08.json)，只重放贪心验收规则。两套模型已经各自实跑过，记录里保存的是 token ID。目标模型分数最高的词与草稿逐项比较，直到第一次不同为止。全对时，`checks` 比草稿长一项，取出来的自然就是额外那项。

```python
import json
from pathlib import Path

s = json.loads(Path("transformer-08.json").read_text(encoding="utf-8"))["spec"]
example = s["prompts"]["The United States of"]
replayed = []
for step in example["rounds"]:
    draft, checks = step["draft"], step["target_argmax"]
    j = next((i for i, (d, t) in enumerate(zip(draft, checks)) if d != t), len(draft))
    prefix = draft[:j] + [checks[j]]
    replayed.extend(prefix)
    if len(replayed) == len(prefix):
        print([s["scored_piece"][str(i)] for i in prefix])
assert replayed == example["output_tokens"]
print(len(replayed), len(example["rounds"]))
# [' America', ',', ' the']
# 16 5
```

对九条提示，分块验证和目标模型逐步贪心生成的 token 序列都一致。这条提示的 16 个输出 token 从 16 次目标解码前向减到 5 次，但每轮最多打五行分，合计打了 **25 行**，另有草稿模型的工作。换成 `One, two, three, four,`，同样写 16 个 token，需要 11 轮、打 50 行分。前缀预填不算在这些回合里。这些是调用和计算位置的计数，没有拿它们换算本机速度，也没有用小模型的续写评估内容正确性。

## 用采样时，还能保住原来的概率吗

贪心看两边的第一名。要随机抽样，目标分布记为 `p`，草稿分布记为 `q`。若草稿已经从 `q` 抽中 token `v`，以 `min(1, p[v]/q[v])` 的概率接受。拒绝时从 `(p − q).clamp_min(0)` 归一化后抽。不经校正直接从 `p` 重抽，会再次抽到原本已经被草稿覆盖过的概率质量。[《Fast Inference from Transformers via Speculative Decoding》§2.3、附录 A](https://arxiv.org/abs/2211.17192)

还用第一轮的第三步，前缀已经写到 `The United States of America,`。真实 softmax 给 ` and` 的目标概率是 4.8348%，草稿概率是 11.2734%。在草稿确实抽中它的条件下，接受率为 `0.048348 / 0.112734 ≈ 42.89%`。乘回草稿抽中它的概率，留下的质量恰好是 4.8348%。对每个词，接受路径贡献 `min(p,q)`，拒绝路径补上正差，两项加起来就是 `p`。这个等式保证的是同一条件分布；换了一条采样路径，随机种子相同也可以抽出不同的文本。
