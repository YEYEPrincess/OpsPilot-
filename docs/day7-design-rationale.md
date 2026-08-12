# Day 7：检索基线评测设计说明

## 1. 今天要解决的工程问题

前几天已经完成了文档解析、Chunk、Embedding、Qdrant 和基础 RAG。现在不能只凭“问一个问题看起来能回答”判断检索系统好不好，因为生成模型可能会把错误的上下文说得很像正确答案。

Day 7 先把生成模型拿开，只评估检索器：给定一个问题，系统能不能把标注的正确证据 Chunk 找回来，以及正确 Chunk 排得够不够靠前。这一步叫检索基线（retrieval baseline）。它是后面比较 BGE/E5、BM25、混合检索和 Reranker 的参照物。

本日新增文件：

- `evaluation/retrieval_metrics.py`：只包含 Recall@k、Hit Rate@k、MRR@k 公式，便于独立测试；
- `scripts/evaluate_retrieval_baseline.py`：读取评测集、调用 Embedding 和 Qdrant、计算指标、输出错误分析与复现实验配置；
- `tests/test_retrieval_metrics.py`：用小例子验证指标公式；
- `data/eval/day7_retrieval_baseline.json`：section-aware Chunk 的基线结果；
- `data/eval/day7_retrieval_errors.jsonl`：section 基线在 Top-5 的未召回和晚排序案例；
- `data/eval/day7_experiment_config.json`：section 实验的代码、数据、模型、索引和环境指纹；
- `data/eval/day7_fixed_retrieval_baseline.json`：fixed-length Chunk 对照结果；
- `data/eval/day7_fixed_retrieval_errors.jsonl`：fixed 对照的错误案例；
- `data/eval/day7_fixed_experiment_config.json`：fixed 对照实验配置和 Chunk ID 映射信息。

## 2. 系统整体架构

```mermaid
flowchart LR
    A[公开技术文档] --> B[解析与清洗]
    B --> C[Section / Fixed Chunk JSONL]
    C --> D[Embedding Provider]
    D --> E[Qdrant 向量索引]
    F[Day 6 评测问题] --> G[问题 Embedding]
    G --> E
    E --> H[Top-k Chunk IDs]
    I[Gold Evidence 标注] --> J[指标计算]
    H --> J
    J --> K[Recall@k / Hit Rate@k / MRR@k]
    H --> L[未召回与错误排序分析]
    M[代码、模型、数据、索引指纹] --> N[可复现实验配置]
```

对一条问题来说，输入和输出如下：

| 环节 | 输入 | 输出 |
|---|---|---|
| 问题编码 | `question` 字符串 | 384 维查询向量 |
| 向量检索 | 查询向量、Qdrant collection、`top_k` | 按相似度排序的 Chunk ID 列表 |
| 评测 | 检索 ID、`gold_evidence` ID | 单题 Recall、Hit、MRR |
| 聚合 | 全部可回答问题的单题结果 | 总体指标、按 split 指标 |
| 错误分析 | Top-5 与 gold 的差异 | `no_gold_in_top_k` 或 `late_gold_rank` |

这里不调用生成模型。检索指标先回答“证据有没有被送到生成模型面前”，生成质量放到后续端到端评测。

## 3. 为什么只评估 answerable 问题

Day 6 有 60 条记录，其中 42 条是 `answerable`，12 条是 `clarification_required`，6 条是 `unanswerable`。后两类题目按设计没有完整的 gold evidence：

- `clarification_required` 需要先问用户缺少的环境信息；
- `unanswerable` 是知识库没有依据或超出范围，正确行为是拒答。

如果把空 evidence 的题目也放进 Recall 计算，任何检索结果都会被错误地当成“召回成功”；如果把它们排除在分母外，指标才表示“对于确实有证据的问题，检索器找回证据的能力”。因此脚本报告 `records_total: 60`，但 `records_evaluated: 42`，并在配置中记录排除的 18 条及原因。

这不是丢弃风险题。风险题应该在后续单独评估“拒答准确率”和“澄清率”，不能混用检索 Recall。

## 4. 三个指标的概念与公式

令某题的 gold evidence 集合为 (G)，检索结果前 (k) 个唯一 Chunk 为 (R_k)。

### 4.1 Recall@k

```text
Recall@k = |G ∩ R_k| / |G|
```

它问的是：标准证据中有多少比例被找回？如果一题有两个 gold Chunk，Top-5 找到一个，则 Recall@5 = 1/2 = 0.5；两个都找到才是 1.0。

Recall 适合衡量“证据覆盖率”，特别适合复杂问题。但它不关心正确 Chunk 排第几：只要在 Top-k 内，排第 1 和排第 10 对 Recall 都一样。

### 4.2 Hit Rate@k

```text
Hit Rate@k = 1  （Top-k 至少命中一个 gold Chunk）
              0  （一个也没命中）
```

对所有问题取平均，就是 Top-k 至少找到一条可用证据的比例。它适合回答“生成模型有没有机会看到正确方向”，但不能表示多证据问题是否全部覆盖。

### 4.3 MRR@k

设第一个相关 Chunk 的排名为 (rank)：

```text
MRR@k = 1 / rank       rank <= k
        0              Top-k 没有相关 Chunk
```

第 1 名是 1.0，第 2 名是 0.5，第 5 名是 0.2。MRR 强调排序质量：如果正确证据经常排在很后面，生成模型在上下文长度受限时仍可能看不到它。

### 4.4 指标之间为什么不能互相替代

一个系统可能 Recall 高但 MRR 低：它最终找到了全部证据，但正确证据总在第 9、10 名。也可能 Hit Rate 高但 Recall 低：每题都找到一条相关 Chunk，却遗漏了复杂问题所需的第二条证据。本项目同时报告三个指标，避免用单一数字掩盖问题。

## 5. 代码模块如何工作

### 5.1 `evaluation/retrieval_metrics.py`

这个文件不依赖 Qdrant。`recall_at_k`、`hit_rate_at_k` 和 `reciprocal_rank_at_k` 只接受 gold ID、检索 ID 和 k，因此可以用单元测试验证数学定义。

代码首先去重检索 ID，并保留原始排名。去重很重要：同一个 Chunk 被重复返回不能算成两条证据，也不能让排名被重复项人为推高。

### 5.2 `scripts/evaluate_retrieval_baseline.py`

脚本执行以下流程：

1. 解析 `--top-k 1,3,5,10`，并确定最大检索深度 10；
2. 读取 Day 6 JSONL 和 Chunk JSONL；
3. 筛选 `answerability == answerable`；
4. 使用与建库完全相同的 `hash-v1`、384 维 Embedding；
5. 调用 Qdrant 的纯向量查询；
6. 明确传入 `query_text=""`，关闭 Qdrant 适配器中的轻量词法 tie-breaker，隔离 dense baseline；
7. 对每题计算多个 k 的 Recall、Hit Rate、MRR；
8. 在 `analysis_k=5` 下保存未命中或晚排序案例；
9. 保存按 split 的汇总、运行环境、Git commit、输入文件 SHA-256 和 Qdrant storage 指纹。

这里使用“同一批查询一次取 Top-10，再从前 10 个截取 Top-1/3/5”的方式，而不是为每个 k 重复查询。这样更快，也保证不同 k 的比较来自同一次排序。

### 5.3 为什么 baseline 不启用词法重排

`QdrantVectorStore.search` 支持把向量分数和词法重叠融合，但那已经是一个混合检索变体。Day 7 需要先知道“当前 Embedding + 向量索引”本身的能力，所以用纯向量 baseline。

可以在后续实验中单独报告：

- dense-only：纯向量相似度；
- lexical tie-break：向量候选上加入词法重排；
- BM25 + dense：两个召回器融合；
- reranker：取较大候选集后使用 Cross-Encoder 重排。

如果一开始就把所有技巧混在一起，就无法判断性能提升到底来自 Embedding、Chunk、词法匹配还是重排模型。

## 6. 如何运行

先进入项目目录并使用项目虚拟环境：

```powershell
cd D:\Documents\大模型项目\opspilot
.\.venv\Scripts\python.exe scripts\validate_eval_set.py `
  --eval data\eval\day6_eval.jsonl `
  --chunks data\processed\chunks_section.jsonl `
  --min-count 50
```

运行 section-aware baseline：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_baseline.py `
  --chunks data\processed\chunks_section.jsonl `
  --qdrant-path data\index\qdrant `
  --collection opspilot_chunks_v1 `
  --top-k 1,3,5,10 `
  --analysis-k 5
```

运行 fixed-length 对照时，必须同时指定查询 Chunk 文件和 gold Chunk 文件：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval_baseline.py `
  --chunks data\processed\chunks_fixed.jsonl `
  --gold-chunks data\processed\chunks_section.jsonl `
  --qdrant-path data\index\qdrant_day7_fixed `
  --collection opspilot_chunks_fixed_v1 `
  --top-k 1,3,5,10 `
  --analysis-k 5 `
  --results data\eval\day7_fixed_retrieval_baseline.json `
  --errors data\eval\day7_fixed_retrieval_errors.jsonl `
  --config data\eval\day7_fixed_experiment_config.json
```

`--gold-chunks` 是为了解决 Chunk ID 命名空间问题：Day 6 的证据标注基于 section Chunk，而 fixed 索引使用 fixed Chunk。脚本按 `document_id + exact section_path` 将 section gold 映射到 fixed Chunk；如果直接比较两个不同 ID，结果会被错误地算成全 miss。

## 7. 本次 section baseline 结果

### 7.1 总体指标

| k | Recall@k | Hit Rate@k | MRR@k |
|---:|---:|---:|---:|
| 1 | 0.0952 | 0.0952 | 0.0952 |
| 3 | 0.1627 | 0.1905 | 0.1389 |
| 5 | 0.1865 | 0.2143 | 0.1437 |
| 10 | 0.2341 | 0.2857 | 0.1511 |

随着 k 从 1 增加到 10，三个指标都上升，说明正确证据有一部分在较靠后的位置；但 Top-10 仍只有 28.57% 的问题至少命中一条 gold evidence，说明当前 `hash-v1` 只能作为工程基线，不能当作生产级语义 Embedding。

### 7.2 按 split 观察

开发集 Top-10 Hit Rate 为 0.36，验证集为 0.3333，测试集为 0。测试集的零结果不是“测试集无意义”，而是提醒我们：当前 hash baseline 对测试集的表达方式和具体技术细节泛化较差。最终报告必须保留测试集结果，不能因为结果不好就调整 gold 标注或反复针对测试题调参。

### 7.3 固定长度 Chunk 对照

在完成 section-to-fixed gold 映射后，fixed 对照结果如下：

| k | Recall@k | Hit Rate@k | MRR@k |
|---:|---:|---:|---:|
| 1 | 0.0608 | 0.2381 | 0.2381 |
| 3 | 0.1227 | 0.2857 | 0.2540 |
| 5 | 0.1396 | 0.3095 | 0.2599 |
| 10 | 0.2362 | 0.4762 | 0.2830 |

fixed 在 Hit Rate 和 MRR 上较高，表示更容易在前几名找到至少一个覆盖同一 section 的 Chunk；但因为一个 section gold 可能映射到多个 fixed Chunk，Recall 的分母和 section 实验不完全同构，所以不能简单宣布 fixed 全面优于 section。这个结果更合理的结论是：Chunk 粒度改变了“相关证据集合”的定义，必须先统一标注粒度，再做严格 A/B 比较。

## 8. 错误分析

`data/eval/day7_retrieval_errors.jsonl` 记录 Top-5 没有命中 gold 的问题，分为两类：

- `no_gold_in_top_k`：Top-5 完全没有标准证据；
- `late_gold_rank`：标准证据存在，但排名在 Top-5 之后。

section baseline 在 Top-5 上共有 33 个错误，其中 30 个是完全未召回，3 个是晚排序。典型现象是：问题属于 Docker，但返回的是多个 Docker 相关的相邻章节；或者问题属于 PyTorch，但返回了同一产品的其他 API 章节。产品标签正确不等于证据 Chunk 正确，这说明只看 domain/product 命中会高估真实检索质量。

例如 `d6-009` 的 gold 是 `pytorch-001:section:0052`，Top-5 返回了多个 PyTorch Chunk，但 gold 在第 10 位，属于 `late_gold_rank`。这类问题优先考虑增加候选 Top-k 或使用 Reranker；而 `d6-005` 的 Top-5 都是 Docker 章节但没有目标 `docker-002:section:0001`，说明同产品内部的细粒度区分仍然失败，更需要更好的 Embedding、BM25 关键词或更合理的 Chunk。

### 8.1 错误分析后的动作优先级

1. 先保留 Top-10 候选，观察上下文预算和 Hit Rate 的收益；
2. 对技术标识符、命令名和版本号加入 BM25 或词法召回；
3. 使用 BGE/E5 等语义 Embedding 做同一 Chunk 集合的对照；
4. 对 Top-20/Top-50 候选使用 Cross-Encoder Reranker；
5. 重新检查“问题应该引用哪一个 Chunk”的 gold 标注粒度；
6. 不在测试集上反复调参，把参数选择限制在 development/validation。

## 9. Top-k 和 Chunk 参数为什么这样调

### 9.1 Top-k

k 越大，召回正确证据的机会通常越大，但会带来三个代价：

- 传给生成模型的上下文更多，增加 token 成本和延迟；
- 混入无关 Chunk，增加模型被噪声干扰的风险；
- 需要更强的重排，否则正确证据可能被淹没。

因此先测 1、3、5、10，观察“召回收益—上下文成本”曲线，而不是凭经验直接选择 20。当前 section 结果从 Top-5 到 Top-10 Hit Rate 从 0.2143 提升到 0.2857，说明扩大候选有收益，但仍需要 reranker 解决排序问题。

### 9.2 Chunk 策略

section-aware Chunk 保留标题和主题边界，相关性更容易解释，gold 标注也更直接；缺点是章节大小不均，某些主题可能过长或过短。fixed-length Chunk 让长度更均匀，适合控制上下文预算；缺点是可能跨越主题边界，且 gold evidence 需要映射。

本次不直接选择“看起来指标最高”的策略，因为 section 和 fixed 的 evidence ID 粒度不同。生产决策应在统一 gold 标注、相同评测问题和相同 Embedding 下比较，并同时观察答案事实一致性、延迟和 token 成本。

## 10. 实验复现：怎样保证结果可信

“同一代码、模型、数据和索引”不能只靠 README 里的文字描述。`day7_experiment_config.json` 记录了：

- Git commit：代码版本；
- Python、操作系统和 `qdrant-client` 版本；
- Embedding provider、模型名和维度；
- Qdrant collection、路径和 retrieval mode；
- eval JSONL、Chunk JSONL、vector manifest、Qdrant storage 的字节数和 SHA-256；
- top-k、analysis-k、Chunk 数量和字符长度统计；
- fixed 实验的 section-to-fixed gold mapping 规则。

SHA-256 是文件内容指纹。只要文件发生一位变化，指纹通常就会变化；它不能防止文件被替换，但能让实验比较时发现输入不一致。

本地 Qdrant storage 被 `.gitignore` 忽略，因此真正的团队或生产环境还应把索引作为版本化构建产物保存到对象存储或模型仓库，并记录下载地址、权限和校验值。仅提交脚本而不保存索引快照，无法保证别人拿到完全相同的向量库。`day6_eval.json` 与用于运行的 JSONL 也必须保持同一份 UTF-8 内容；如果只更新其中一个文件，指标变化可能来自评测输入变化而不是检索器变化。因此本次运行前重新由正确的 JSON 数组生成 JSONL，并把最终 JSONL 的 SHA-256 写入实验配置。

建议的复现流程是：

1. checkout 记录的 Git commit；
2. 使用锁定的 Python 和依赖版本；
3. 校验 eval、Chunk 和索引 SHA-256；
4. 确认 Embedding provider、dimension、距离指标一致；
5. 用完全相同的 top-k 和 Chunk mapping 运行脚本；
6. 比较生成的 JSON 结果，而不是只比较终端截图。

## 11. 方法选择与替代方案

### 11.1 为什么先做离线 Recall，而不是直接看最终答案

最终答案同时受检索、Prompt、生成模型、温度和解析器影响。若答案错了，无法判断是“没找到证据”还是“找到了但模型没用好”。先做检索基线可以把问题拆开，降低调试复杂度。

### 11.2 为什么不用人工逐题打分替代指标

人工判断事实正确性很重要，但成本高、标准容易漂移、难以快速比较 1/3/5/10 多个 k。自动指标提供稳定回归信号；人工复核保留在错误分析和后续端到端评测中。二者是互补关系，不是互斥替代。

### 11.3 为什么不直接使用 BM25 或 Reranker

BM25 对命令、库名、版本号很强，但不擅长同义表达；Reranker 相关性通常更好，但要先取较大的候选集，并增加模型延迟和 GPU/API 成本。先固定 dense-only baseline，后面才能量化每个增强模块的边际收益。

## 12. 单元测试与验收

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check evaluation\retrieval_metrics.py scripts\evaluate_retrieval_baseline.py tests\test_retrieval_metrics.py
```

本次结果：

```text
...............                                                          [100%]
15 passed in 3.20s
All checks passed!
```

15 个测试全部通过，新增的 3 个测试分别验证多 gold Recall、首个相关结果的 Hit/MRR，以及空 gold 不应被当成命中。测试保证公式实现没有明显回归，但不证明 embedding 的语义质量；语义质量由本日指标和错误案例体现。

## 13. 今日结论

Day 7 的主要成果不是追求一个漂亮的数字，而是建立了一个可解释、可复现的检索测量闭环：

```text
固定评测集
  -> 固定 Embedding / Qdrant / Chunk
  -> Top-k 检索
  -> Recall + Hit Rate + MRR
  -> 错误案例分类
  -> 记录代码、数据、模型、索引指纹
  -> 为下一轮 Embedding / BM25 / Reranker 优化提供基线
```

当前 `hash-v1` 的结果说明系统已经能运行，但语义检索能力有限；下一步应该优先在相同评测集和相同指标下比较真正的语义 Embedding，并单独测量词法召回和重排的收益。
