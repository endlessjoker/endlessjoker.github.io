# MiniMax M3.1 Flash 怎样凭一个词挑中整段

一段话里只有一个 token 特别相关，其余 127 个都很普通，整段还要不要读？MiniMax M3.1 Flash 先给每个 token 打分，把块内最高分当作整块的分数。只要有一个位置分数够高，整块的 K、V 就会被读进来。[《MiniMax Sparse Attention》](https://arxiv.org/abs/2606.13392)

## 一组查询头共用一张清单

首层有 64 个查询头、4 组 KV，每组 16 个查询头。索引支路也投影出 4 个查询头，每组一个，历史每个位置只存一条 128 维索引键。索引用自己的一套 Q、K，做归一化和 RoPE 后，按 `Q @ K.T / √128` 打分。每块 128 个位置取一个最大值。

当前块占一个名额，剩下 15 个留给最高分的历史块。第 22 讲的 IndexPool 先把四个键压成一个，再跟查询打分。MiniMax 把这两步倒过来，代价是索引支路要给各个历史位置打分，收益是单个特别强的匹配不会被同块其他位置稀释。入选以后，主注意力用自己的 Q、K、V 重新计算。

## 首层读到了哪几段

用官方首层的 MXFP8 权重，按每 32 个权重共用一个缩放值还原成 FP32。把 19 条短句各重复到 128 个 token，再接上 6 个 token 的「现在回答：德国的首都是」。固定这份输入，只看最后一个 token。

第 0 块「法国的首都是巴黎」在第 0 组得 **2.1247**，落选；在第 1 组得 **3.2859**，入选。每组最后读取 `15 × 128 + 6 = 1926` 个位置，输入共有 2438 个位置。第 0 和第 1 个查询头共用第 0 组的清单，各自算 softmax。

这段代码用 [实测记录](/transformer-data/transformer-27.json) 复放计算。每块的 `log_z` 是块内主注意力分数的 `logsumexp`，`mean_v` 是块内加权后的 V。对入选块的 `log_z` 再做 softmax，就能得到每块在总输出里的比重。

```python
import json
from pathlib import Path
import torch

data = json.loads(Path("transformer-27.json").read_text())
groups = []
for row in data["groups"]:
    past = torch.tensor(row["block_maxima"][:-1])
    local = len(past)
    picked = past.topk(data["block_budget"] - 1).indices.tolist() + [local]
    groups.append(sorted(picked))
    print(row["group"], sorted(set(range(local)) - set(picked)))

for head in data["heads"]:
    log_z = torch.tensor(head["block_log_partitions"])
    mean_v = torch.tensor(head["block_value_means"])
    kept = groups[head["group"]]
    print(head["head"], f"{(log_z[kept].softmax(0) @ mean_v[kept]).item():.6f}",
          f"{(log_z.softmax(0) @ mean_v).item():.6f}")
```

第 0 个头的第 0 维用稀疏清单算出 **0.004528**，同样的 Q、K、V 读完整输入会得到 **−0.005528**。这个差值只描述固定输入下的首层、单头、单坐标。每头最多对 2048 个位置做主注意力，比例再小，索引仍需要扫描历史。没在这一次选中的块也可能在下次被读到。[MiniMax M3.1 Flash 官方配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/ce7f4df87d2e05262f5775f373dac4772f134041/config.json)
