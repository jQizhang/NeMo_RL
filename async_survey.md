公开资料里有，但同时满足四个条件（single controller + 异步 GRPO + 数学任务 + 公开 staleness 消融/曲线）的只有 3 个。下表的数字来自网页摘要，没有逐个打开 PDF 核对图表。

## 完全符合的公开结果

| 工作                                                                                      | 框架 / 控制器                                    | 模型与数据                                                        | 测到的最大 stale                                                                      | 结果                                                                                                                                           | 曲线                               |
| ----------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [verl fully_async_policy](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | verl，Ray single controller                      | Qwen2.5-Math-7B 和 Qwen3-30B-A3B；DAPO-Math-17k 训练，AIME24 评测 | `staleness_threshold`（样本比例，不是版本差）测了 0 / 0.1 / 0.3 / 0.5，多轮工具测了 1 | 0.1-0.5 都和同步持平：AIME acc max 0.35 左右，提速 1.7-2.7x                                                                                    | ✅ wandb（hou-zg-meituan）         |
| [AReaL](https://arxiv.org/html/2505.24298v5)（NeurIPS'25）                                | 一个中心 Rollout Controller，基于 ReaLHF         | R1-Distill-Qwen-1.5B，DeepScaleR 数据                             | 用版本差 η 计，测了 0 / 1 / 2 / 4 / 8 / 16 / ∞                                        | 加 decoupled PPO 时 η≤4 基本无损：AIME24 42.2 vs oracle 42.0。η=16 掉到 38.7，∞ 掉到 36.9。不加 decoupled 时 η=4 就崩到 23.3。默认数学任务 η=8 | ✅ 论文图 5（学习曲线和吞吐）      |
| [ROLL Flash](https://arxiv.org/html/2510.11345v1)（阿里）                                 | 一个 AsyncController（是否 Ray driver 文中没写） | Qwen3-8B-Base / Think，DAPO-Math-18K                              | `async_ratio` 测了 0 / 1 / 2 / 4 / 8                                                  | 各个 ratio 的 Pass@1 "相当"，推荐用 2。还比较了 TIS / CISPO / TOPR / decoupled PPO                                                             | 论文图 4（文字里没有给出具体数值） |

## 部分相关

- **[Stale but Stable / SAT](https://jyyang26.github.io/stable_async_analysis/)**：Qwen3-30B-A3B-Base，DAPO+Dolci 数学，SGLang + Megatron，参数命名是 slime 风格，但没写用的哪个框架。
  - 比较了 lag 1 和 lag 8 两档。纯 GRPO 在 lag 8 下崩了：AIME24 从 31.25 掉到 30.17。SAT-GSPO+R3 在 lag 8 下是 34.79（lag 1 是 35.83）。
  - 有逐步训练曲线，单 seed。
- **[Applied Compute 博客](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)**：Qwen3-8B GRPO + DAPO math，研究的是怎样预测和控制 staleness（最大测到 4）。没有给 reward 或精度随 staleness 变化的曲线，作者把这一点列为后续工作。
- **[Relax](https://arxiv.org/html/2604.11554v2)**（基于 slime，`--max-staleness`）：Qwen3-4B 和 Qwen3-Omni-30B，staleness=2，提速 1.76-2.0x，reward 收敛到同一水平。论文没有做 staleness 扫描。
- **[GAC](https://arxiv.org/pdf/2603.01501)**：在 DeepScaleR 上测到 staleness 16 和 32（Qwen3 1.7B/4B/8B），stale 程度最大的公开数学实验。不过它的重点是新算法，不是框架。
- **[PipelineRL](https://arxiv.org/pdf/2509.19128)**：有 `max_lag` 背压，但只分析了 ESS（on-policy 程度），没有按 lag 扫描 AIME。

## 关于 slime / miles

- **slime**：官方没有发布数学任务的 staleness 实验。
- **miles**：[v0.1 版](https://www.lmsys.org/blog/2026-08-18-miles-v0-1/)（2026-08）已经加了 `--fully-async` 和 `--max-weight-staleness`（[文档](https://miles.radixark.com/docs/user-guide/fully-async)），staleness 按 group 内最老的权重版本计算。
  - 本地 `miles/` 的 commit 是 2026-04 的，比这个版本旧，所以上一轮我说"miles 没有 staleness 控制"只对本地这份代码成立。
  - 它的公开异步结果只有 GLM-5.2 744B 的 terminal 编码任务（平均落后 1.7 步），没有数学任务。

## 结论

- 想直接复用：verl `fully_async_policy` 的 DAPO-7B 配方，配置、wandb 和 staleness 消融都有，但只测到 0.5。
- 想知道 stale 推到多大还能不掉点：参考 AReaL 表 2，加 decoupled PPO / TIS 这类修正时版本差 ≤4 基本无损，8 开始轻微下降。目前公开数学实验里 stale 测得最深的是 GAC 的 32。

Sources:
- [verl fully async docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- [AReaL (arXiv 2505.24298)](https://arxiv.org/html/2505.24298v5)
- [ROLL Flash (arXiv 2510.11345)](https://arxiv.org/html/2510.11345v1)
- [Stale but Stable / SAT](https://jyyang26.github.io/stable_async_analysis/)
- [Applied Compute: staleness in fully async RL](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)
- [Relax (arXiv 2604.11554)](https://arxiv.org/html/2604.11554v2)
- [GAC (arXiv 2603.01501)](https://arxiv.org/pdf/2603.01501)
- [PipelineRL (arXiv 2509.19128)](https://arxiv.org/pdf/2509.19128)
- [Miles v0.1 blog](https://www.lmsys.org/blog/2026-08-18-miles-v0-1/)
- [Miles fully async docs](https://miles.radixark.com/docs/user-guide/fully-async)
- [HF blog: Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape)