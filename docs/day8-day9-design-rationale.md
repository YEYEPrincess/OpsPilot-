# Day 8–9：BM25、混合检索与 Reranker 学习指导

> 使用方式：本文件是“你自己实现、运行和记录”的实验手册。文中的代码不会自动出现在项目中；请按照步骤逐个创建文件，每完成一步就运行相应检查。不要一次性复制全部代码后再排错。

## 1. 两天的总体目标

Day 7 已经得到纯向量检索基线：问题经过 `hash-v1` 编码后，Qdrant 按余弦相似度返回 Top-k。这个基线证明检索链路能够运行，也暴露了两个问题：

1. `hash-v1` 对中文问题和英文技术文档的语义对应能力弱；
2. 只靠向量分数，精确命令、错误码、类名和版本号可能排得不够靠前。

Day 8 增加 BM25 关键词检索，并用 RRF 把 Dense 与 BM25 的结果融合。Day 9 再用 Cross-Encoder Reranker 对少量候选逐对打分。

```text
                         ┌─ Dense / Qdrant ─┐
用户问题 ──预处理────────┤                   ├─ RRF融合 ─ Top-20候选
                         └─ BM25关键词检索 ──┘                  │
                                                              ▼
                                              Cross-Encoder Reranker
                                                              │
                                                              ▼
                                                        最终Top-5证据
```

这三层承担不同职责：

| 层级 | 首要目标 | 特点 |
|---|---|---|
| Dense / BM25 召回 | 尽量不要漏掉正确证据 | 快，可以搜索全部 Chunk |
| RRF 融合 | 合并不同检索器的优势 | 不需要让两种原始分数处于同一量纲 |
| Reranker 精排 | 把最相关证据排到前面 | 更准但更慢，只处理少量候选 |

## 2. 实验纪律：一次只改变一个变量

比较 Dense、BM25、Hybrid、Hybrid + Reranker 时，必须固定：

- 同一份 `day6_eval.jsonl`；
- 同一份 `chunks_section.jsonl`；
- 同一套 gold evidence；
- 同一组 answerable 问题；
- 同样的 Recall@k、Hit Rate@k、MRR@k 定义；
- 同一台机器或至少记录设备差异。

否则指标变化可能来自数据、Chunk 或机器，而不是新检索方法。

## 3. 先安装依赖
Day 8 安装 BM25；Day 9 安装 Cross-Encoder 支持：
为什么使用 `uv add`：它会同时更新 `pyproject.toml` 和 `uv.lock`。`pyproject.toml` 表达允许的依赖范围，`uv.lock` 记录这次实际解析到的精确版本，有利于复现实验。

为什么 Day 8 选择 `rank-bm25`：当前只有约一千个 Chunk，它的接口简单，适合理解 BM25 原理。没有选择 Elasticsearch/OpenSearch，是因为后者需要额外服务、JVM、索引配置和运维，四小时内会把重点从检索原理转移到基础设施。生产数据达到几十万或百万 Chunk、需要持久化倒排索引和并发过滤时，再考虑 Elasticsearch/OpenSearch。

---

# Day 8：BM25 与混合检索

## 4. BM25 到底在算什么

BM25 是词法检索算法。它不把文本编码成神经网络向量，而是根据“查询词是否出现在文档中”计算相关性。简化公式为：

```text
BM25(q, d) = 对查询中每个词t求和：

IDF(t) × TF(t,d)的饱和函数 × 文档长度修正
```

三个核心思想：

1. **IDF**：越少见的词越重要。`docker` 可能出现在大量文档中，而 `CUDA_VISIBLE_DEVICES` 很少出现，后者区分力更强；
2. **词频饱和**：一个词从出现 0 次变成 1 次很重要，从 50 次变成 51 次并不会继续大幅加分；
3. **长度归一化**：长文档自然包含更多词，不能仅因为更长就获得不公平优势。

BM25 擅长：

- `torch.cuda.is_available`、`docker logs` 等命令；
- `HTTP 429`、`CUDA OOM` 等错误标识；
- `v0.6.3`、`CUDA 12.4` 等版本；
- 类名、配置键和参数名。

BM25 不擅长：

- 同义表达，例如“显存不足”和“GPU out of memory”；
- 中英文跨语言匹配；
- 拼写变化和没有共同词的改写。

因此它适合补充 Dense Retrieval，而不是完全替代 Dense Retrieval。

## 5. 第一步：实现 BM25 索引模块（约 45 分钟）
文件：retrieval/bm25_store.py
为什么标题和正文一起索引：用户可能问“Qdrant collection 的向量维度”，关键词只出现在章节标题。只索引正文会损失这类信号。

为什么不使用 pickle：pickle 加载时可以执行任意 Python 对象，不适合加载来源不可信的索引文件。JSON gzip 更透明、安全、可检查。代价是加载时需要重新构造 BM25 统计量；对一千个 Chunk 可以接受。

## 6. 第二步：建立 BM25 索引（约 15 分钟）
文件：
scripts/build_bm25_index.py
验收：`documents` 应与 section Chunk 数量一致；manifest 中保存输入和索引 SHA-256，这样能确认之后的实验使用同一份数据。

## 7. 第三步：实现单独的关键词查询（约 30 分钟）
文件：scripts/query_bm25.py
这一步是在做 BM25 的“单问题冒烟测试”：确认索引能正常加载、关键词能被正确切分，并观察关键词检索返回的 Top-5 Chunk 是否合理。这里没有使用embedding、Qdrant向量相似度、大模型和reranker。实际流程:
输入问题
  ↓
使用与建库相同的 tokenizer 分词
  ↓
BM25 计算问题与全部 Chunk 的关键词相关性
  ↓
按 BM25 分数降序排列
  ↓
输出 Top-5 Chunk

运行两个问题对比：

```powershell
.\.venv\Scripts\python.exe scripts\query_bm25.py `
  "torch.cuda.is_available" --top-k 5

.\.venv\Scripts\python.exe scripts\query_bm25.py `
  "显存不够怎么办" --top-k 5
```
第一条结果输出举例：
{
  "score": 8.713856469726062,  #BM25 相关性分数。分数越高，说明该 Chunk 与查询词的词法匹配越强。
  "chunk_id": "pytorch-001:section:0053",
  "product": "pytorch",
  "section_path": [
    "CUDA semantics #",
    "Best practices #",
    "Device-agnostic code #"#表示 Chunk 在原文档中的章节层级。
它帮助我们判断结果是否只是“关键词相同”，还是章节主题也真正相关。
  ],
  "text_preview": "..." #这是 Chunk 正文的前 300 个字符，方便在终端快速检查结果。
}

结果分析：精确关键词召回成功，但细粒度排序质量一般。产品级召回正确：5 条全部属于 PyTorch；
API 相关页面进入 Top-5；
直接定义页面进入了 Top-5；
最直接的答案只排第 5；
部分结果是导航页和网站样板内容；
BM25 擅长找“出现这个词的页面”，不一定能判断“哪个页面最能回答问题”。

第二条输出结果及分析：
返回：
[]
这不是程序报错，而是表示：
BM25 没有找到分数大于 0 的 Chunk。
中文问题没有召回结果，因为知识库技术文档主要是英文，中英文没有词法交集因此返回[]。直接证明了：
BM25 擅长精确关键词，但不理解同义表达，也不理解中英文之间的语义对应。

它不知道：
显存不够
≈ GPU out of memory
≈ CUDA OOM

共同分析两组问题探究BM25的特点和问题：
查询问题	                   BM25结果	         说明
torch.cuda.is_available	 返回5条PyTorch结果	精确技术标识符召回能力强
显存不够怎么办	              []	        中文和英文之间没有共同关键词


因此，BM25 不能单独作为 OpsPilot 的最终检索器。
合理架构是：
中文用户问题
  ├─ Dense Retrieval：负责语义、同义表达和跨语言
  └─ BM25：负责命令、错误码、版本号和精确关键词
              ↓
            RRF融合
              ↓
         Cross-Encoder重排

不过当前的 Dense Retriever 是 hash-v1，它同样不擅长真正的跨语言语义。后续使用 BGE-M3 或多语言 E5 后，Dense 路径才能更有效地补救中文查询。
总结：BM25 单问题查询实验

本实验使用 query_bm25.py 对已建立的 BM25 索引进行关键词检索冒烟测试。查询过程不使用 Embedding、Qdrant 向量相似度或生成模型，而是将问题按照与索引一致的 tokenizer 进行分词，然后计算问题与全部 Chunk 的 BM25 相关性分数，并返回分数最高的 Top-5 Chunk。

对于精确技术标识符 torch.cuda.is_available，BM25 返回的五条结果全部来自 PyTorch 文档，说明技术 API 被 tokenizer 正确保留，BM25 索引和关键词查询流程能够正常工作。其中 API 定义相关 Chunk 进入了 Top-5，但只排在第 5 位；部分排名更高的结果只是包含相同 API 名称的最佳实践、导航或资源页面。这说明 BM25 擅长寻找包含相同关键词的文档，但不能充分理解用户希望获得的是 API 定义、使用方法还是副作用，细粒度排序仍有改进空间。

对于中文查询“显存不够怎么办”，BM25 返回空列表。知识库文档主要使用英文表达，例如 “CUDA out of memory” 或 “GPU memory”，中文查询 token 与英文文档不存在直接词法交集，因此所有 BM25 分数为 0。该结果不是程序失败，而是 BM25 不具备同义表达和跨语言语义理解能力的体现。

两组查询共同说明：BM25 适合召回命令、API、错误码、配置项和版本号等精确技术标识符，但不适合单独处理中文自然语言与英文技术文档之间的语义匹配。本项目后续需要将 BM25 与 Dense Retrieval 结合，通过 RRF 合并两路候选，再使用 Cross-Encoder Reranker 改善细粒度排序。

## 8. 第四步：实现 RRF 融合（约 45 分钟）

### 8.1 为什么选择 RRF，不先选择加权分数融合

Dense 余弦分数可能在 `0.1～0.8`，BM25 分数可能在 `0～20`。直接相加没有意义。加权融合必须先做 min-max、z-score 或其他校准，而且校准结果会随查询改变。

RRF（Reciprocal Rank Fusion互逆排序融合）只使用排名：

```text
RRF分数(chunk) = Σ weight / (rrf_k + rank)
```

如果一个 Chunk 在 Dense 排第 2、BM25 排第 5：

```text
1 / (60 + 2) + 1 / (60 + 5)
```
它不要求两种分数在同一量纲，所以适合作为第一版混合检索。`rrf_k=60` 是平滑常数：值越大，前几名与后几名的差距越缓和。不要把它与检索 `top_k` 混淆。

文件：
retrieval/hybrid_search.py

优点：实现简单、对分数尺度不敏感、容易解释。缺点：丢弃了原始分数的置信度；两个检索器中“第 1 名比第 2 名强很多”的信息不会被利用。后续数据足够时，可以在 validation 集上校准加权融合，但不能在 test 集反复调权重。

## 9. 第五步：比较 Dense、BM25 与 Hybrid（约 60 分钟）

文件`scripts/evaluate_day8_hybrid.py`。
该文件在同一评测集上比较 Dense、BM25 和 RRF Hybrid。
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day8_hybrid.py `
  --candidate-k 20 --rrf-k 60
```

分析时填写：

| 方法 | Recall@5 | Hit Rate@5 | MRR@5 | P50延迟 | P95延迟 |
| Dense | 0.1865 | 0.2143 | 0.1437 | 2.919ms | 3.359ms |
| BM25 |0.1984 | 0.2143| 0.1397 | 3.795 ms | 6.136ms |
| Hybrid RRF | 0.2341 | 0.2619 | 0.1754 | 6.937ms | 9.814ms |

不要预设 Hybrid 一定最好。中文问题与英文文档没有共同词时，BM25 可能贡献很少；这本身就是有效实验结论。
分析：
在 Top-5 指标上，Hybrid RRF 是三种方案中效果最好的：
找回的标准证据更多；
至少命中一条正确证据的问题更多；
正确证据的平均排名更靠前。
但它的实际顺序调用延迟也是最高的。
这说明 Hybrid RRF 带来了比较明确的质量提升，代价是同时调用 Dense 和 BM25 后产生的额外延迟。
BM25 的 Recall@5 相比Dense略高，说明它找回了更多 gold Chunk，特别可能来自：
命令名称；
API 名称；
错误码；
配置项；
版本号；
产品名称。
Hybrid：让dense和BM25实现优势互补：
Dense：
负责模糊相似、部分语义关系
BM25：
负责API、命令、版本号、错误码等精确匹配

根据当前结果，建议：
第一阶段召回：
Dense Top-20 + BM25 Top-20

第二阶段融合：
RRF，rrf_k=60

第三阶段：
Reranker将Hybrid Top-20重排为最终Top-5

Dense Retrieval的延迟最低，但当前系统使用的hash-v1不具备充分的语义和跨语言理解能力，因此Top-5召回质量有限。BM25的Recall@5略高于Dense，说明精确命令、API名称、错误码和版本号等词法信号能够补充部分证据；但BM25的Hit Rate@5与Dense相同，MRR@5略低，表明它虽然能够找到包含相同关键词的Chunk，却不一定能将最能回答问题的Chunk排在前面。

Hybrid RRF在三个Top-5质量指标上均取得最佳结果。相比Dense，Recall@5从0.1865提高到0.2341，相对提升约25.5%；Hit Rate@5从0.2143提高到0.2619，相对提升约22.2%；MRR@5从0.1437提高到0.1754，相对提升约22.1%。42条问题中，Dense和BM25分别约有9条问题在Top-5至少命中一条gold evidence，而Hybrid约有11条，说明融合增加了约2条问题的有效命中。

在Top-10上，BM25的Recall和Hit Rate分别达到0.3294和0.3810，高于Hybrid的0.3056和0.3571；但Hybrid的MRR@10达到0.1881，高于BM25的0.1600。这说明BM25在较大的候选集合中覆盖了更多精确关键词证据，而RRF更有利于把两路检索器共同认可的结果提升到前面。由于RAG最终只能将少量Chunk送入生成模型，Hybrid在Top-5和MRR上的优势具有实际价值；同时可以保留Hybrid Top-20供后续Cross-Encoder Reranker进一步精排。

延迟方面，Dense的P50和P95分别为2.919 ms和3.350 ms，BM25为3.795 ms和6.136 ms。当前Hybrid采用顺序调用，P50为6.937 ms，P95为9.814 ms。根据平均延迟估算，RRF融合本身只消耗约0.066 ms，主要开销来自Dense与BM25两路检索。若未来真正并行调用，理论P50约为3.858 ms、P95约为6.208 ms；该数值目前只是估计，不能作为实际并行延迟报告。（P代表Percentile,即百分位数。假设你对42个问题分别测量检索时间，得到42个延迟值，然后从小到大排列P50 表示：50%的请求延迟不超过这个值，另外50%的请求比它慢。P95尾延迟：95%的请求延迟不超过这个值，最慢的5%请求超过它。）

综合质量和延迟，本项目暂时选择Dense Top-20与BM25 Top-20并行召回、使用RRF（rrf_k=60）融合，然后由Reranker将候选重排为最终Top-5。后续需要在validation集上比较不同RRF参数和候选数量，并单独测量真实并行调用的端到端延迟。



## 10. Day 8 测试与验收
 `tests/test_hybrid_retrieval.py`：
这个文件是混合检索模块的单元测试，主要检查：
tokenizer是否保留技术标识符；
RRF是否正确融合Dense和BM25排名；
两路检索都找到的Chunk是否会获得更高的融合分数。
运行：

```powershell
.\.venv\Scripts\python.exe -m ruff check `
  retrieval\bm25_store.py retrieval\hybrid_search.py `
  scripts\build_bm25_index.py scripts\query_bm25.py `
  scripts\evaluate_day8_hybrid.py tests\test_hybrid_retrieval.py

.\.venv\Scripts\python.exe -m pytest -q
```
结果：Found 3 errors.
[*] 3 fixable with the `--fix` option.
..................                                                                                               [100%]
18 passed in 2.48s
---

# Day 9：Reranker 重排序

## 11. Reranker 与 Embedding 的根本区别

Dense Retriever 通常是 Bi-Encoder：问题和文档分别编码，文档向量可以预先存入 Qdrant。查询时只需编码问题并做近邻搜索，所以速度快。

Cross-Encoder Reranker 把 `(问题, 候选Chunk)` 同时送进 Transformer。问题 token 可以直接关注文档 token，相关性判断更细，但每个候选都需要一次前向计算，不能预先缓存成单独文档向量。

```text
Bi-Encoder：encode(query) 与 encode(document) → 余弦相似度

Cross-Encoder：[query tokens; document tokens] → Transformer → relevance score
```

因此典型流程是“先召回，再精排”：检索器从全部 Chunk 中取 Top-20/50，Reranker 只处理这些候选。官方 Sentence Transformers 文档也采用 Retrieve & Re-Rank 两阶段结构。

## 12. 候选数量为什么选 20，最终为什么选 5

本地学习默认：

```text
Hybrid candidate_k = 20
Reranker final_k = 5
batch_size = GPU 8 / CPU 2
max_length = 512 tokens
```

候选太少：正确证据没有进入候选，Reranker 再强也无法创造新证据。候选太多：延迟和显存近似随 `(query, document)` 对数增加。

最关键的上限关系：

```text
Reranker后的Hit@5 ≤ 候选集合的Hit@20上限
``
Reranker主要改善 Top-5 的 MRR 和 Hit Rate；它不应该改变候选 Top-20 的 Recall，因为只是重新排序同一批候选。

## 13. 模型选择

推荐主实验：

```text
BAAI/bge-reranker-v2-m3
```

选择理由：项目问题主要是中文、文档主要是英文，需要多语言相关性判断；模型卡将它定位为 multilingual reranker。它约 0.6B 参数，质量和部署成本都高于小型 MiniLM。

资源不足时的替代：

- `BAAI/bge-reranker-base`：中英文、相对轻量；
- `cross-encoder/ms-marco-MiniLM-L6-v2`：速度快，适合验证代码，但主要针对英文检索，不适合作为本项目最终多语言结论。

为什么不直接使用大语言模型 Prompt 打分：LLM 重排更灵活，但延迟、成本、输出稳定性和批处理难度更高。Day 9 先用输出单个 relevance score 的专用 Cross-Encoder，实验变量更清晰。

## 14. 第一步：实现 Reranker（约 60 分钟）

retrieval/reranker.py
为什么使用 raw logits 而不强制 sigmoid：重排序只关心相对顺序，sigmoid 是单调函数，不会改变排名。需要把分数展示为概率时才考虑归一化，但 reranker score 通常不应直接解释成真实概率。

## 15. 第二步：先做单问题冒烟测试
 `scripts/query_reranker.py` 
第一次运行需要下载模型，不能把下载耗时算进稳定推理延迟。先加载模型并做一次 warm-up，再正式计时。

## 16. 第三步：运行有无 Reranker 的严格对照
Day 8 的 `day8_retrieval_comparison.json` 已保存每个问题的 Hybrid 候选 ID。Day 9 应固定这批候选，避免重新检索导致候选变化。请创建：

scripts/evaluate_day9_reranker.py

运行：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day9_reranker.py `
  --candidate-k 20 --final-k 5
```

如果 CPU 太慢，先用小模型验证代码：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_day9_reranker.py `
  --model cross-encoder/ms-marco-MiniLM-L6-v2 `
  --candidate-k 10 --final-k 5 --batch-size 2
```

小模型结果只能作为代码冒烟测试，不能代替多语言主实验结论。

## 17. Day 9 测试

向 `tests/test_hybrid_retrieval.py` 增加不下载模型的排序测试

不在单元测试中下载真实模型，因为这会让测试依赖网络、耗时长并产生缓存。真实模型加载放在冒烟测试和对照实验中。

## 18. 如何分析质量—延迟权衡

填写：

| 方法 | Candidate k | Final k | Hit@5 | MRR@5 | P50 | P95/Max | 设备 |
|---|---:|---:|---:|---:|---:|---:|---|
| Hybrid | 20 | 5 | 待填写 | 待填写 | 待填写 | 待填写 | CPU/GPU |
| Hybrid + Reranker | 20 | 5 | 待填写 | 待填写 | 待填写 | 待填写 | CPU/GPU |

如果 MRR 明显提高而 Recall@20 不变，这是正常结果：Reranker 改善排序，没有扩大候选集合。如果 candidate Recall@20 本身很低，应先改进 BM25/Dense 召回，而不是继续堆更大的 Reranker。

推荐做一个小型参数实验：

```text
candidate_k ∈ {10, 20, 50}
batch_size  ∈ {2, 4, 8}
final_k = 5
```

只用 development 调试、validation 选参数，test 最后报告一次。不要根据 test 反复调整候选数。

---

# 部署思考

## 19. Dense 与 BM25 是否并行调用

线上建议并行，因为两者互不依赖：

```text
请求到达
 ├─ 分支A：Query Embedding → Qdrant
 └─ 分支B：Tokenize → BM25
             ↓
         两路完成后RRF
```

顺序延迟约为：

```text
T_dense + T_bm25 + T_fusion
```

并行延迟约为：

```text
max(T_dense, T_bm25) + T_fusion
```

但 Day 8 先写顺序版本，因为更容易调试、复现实验。确认正确后再用线程池或异步任务实现真实并行；文档中的 `hybrid_parallel_estimate` 只是估计，不应冒充实测并行延迟。

## 20. 一个检索器失败时怎么办

推荐 fail-soft：

| 状态 | 行为 |
|---|---|
| Dense成功、BM25成功 | RRF融合 |
| Dense失败、BM25成功 | 降级为BM25 |
| Dense成功、BM25失败 | 降级为Dense |
| 两者都失败 | 返回503，不调用生成模型猜答案 |

响应和日志应包含：

```json
{
  "retrieval_mode": "bm25_fallback",
  "degraded": true,
  "failed_component": "dense",
  "request_id": "..."
}
```

降级不等于悄悄忽略错误。需要记录失败率、超时率和降级率，否则线上质量下降很难追踪。

## 21. Reranker 是否会成为瓶颈

很可能。每个请求有 20 个候选，就需要构造 20 个 `(query, passage)` 对。并发 10 个请求时瞬间可能产生 200 对。

部署措施：

- 模型启动时加载并 warm-up，不要每次请求重新加载；
- 使用有上限的队列和 semaphore 控制并发；
- 将不同请求的 pair 合并成动态 batch；
- 设置最大 candidate_k 和最大输入长度；
- 记录排队时间、模型推理时间和端到端时间；
- GPU OOM 或超时时回退到 RRF 顺序；
- 队列已满时限流，而不是无限堆积造成雪崩。

Reranker 降级原则：它只负责改善排序，因此失败时使用 Hybrid 原排序仍能提供服务；不能因为可选精排失败就让整个问答系统完全不可用。


### Day 9（4小时）

| 时间 | 任务 | 交付 |
|---|---|---|
| 0:00–0:30 | 选择 candidate_k=20、final_k=5 | 参数假设 |
| 0:30–1:30 | 实现 CrossEncoderReranker | Reranker 模块 |
| 1:30–2:30 | 冒烟、warm-up、调整 batch | 可运行推理配置 |
| 2:30–3:30 | 有无 Reranker A/B | 指标与延迟 |
| 3:30–4:00 | 质量—延迟分析 | 参数选择说明 |

## 23. 最终验收清单

Day 8：

- [ ] BM25 索引数量与 Chunk 数量一致；
- [ ] 精确命令查询能返回合理文档；
- [ ] RRF 单元测试通过；
- [ ] Dense、BM25、Hybrid 使用同一评测集；
- [ ] 同时记录质量指标与延迟；
- [ ] 不把并行延迟估计写成实测值。

Day 9：

- [ ] Reranker 只重排固定候选；
- [ ] 正式计时前完成 warm-up；
- [ ] 记录模型名、设备、batch、candidate_k、final_k；
- [ ] 比较 Hybrid 与 Hybrid + Reranker；
- [ ] 验证 candidate Recall 没有被重排改变；
- [ ] 记录失败时回退策略。

## 24. 完成后提交

先检查，不要使用 `git add .`：

```powershell
git status
```

Day 8 建议提交：

```powershell
git add pyproject.toml uv.lock
git add retrieval\bm25_store.py retrieval\hybrid_search.py
git add scripts\build_bm25_index.py scripts\query_bm25.py
git add scripts\evaluate_day8_hybrid.py tests\test_hybrid_retrieval.py
git add data\manifest\bm25_index.json
git add data\eval\day8_retrieval_comparison.json
git add docs\day8-day9-design-rationale.md
git commit -m "feat(retrieval): add bm25 and hybrid search"
git push origin main
```

Day 9 建议提交：

```powershell
git add pyproject.toml uv.lock
git add retrieval\reranker.py scripts\evaluate_day9_reranker.py
git add tests\test_hybrid_retrieval.py
git add data\eval\day9_reranker_comparison.json
git add docs\day8-day9-design-rationale.md
git commit -m "feat(retrieval): add cross-encoder reranking"
git push origin main
```

本地 BM25/Qdrant 索引通常体积较大且在 `.gitignore` 中；仓库提交构建脚本、manifest 和校验值即可。团队部署时应把索引产物放入版本化对象存储。

## 25. 参考资料

- Rank-BM25 官方仓库：https://github.com/dorianbrown/rank_bm25
- Sentence Transformers Retrieve & Re-Rank：https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- BAAI BGE Reranker 模型卡：https://huggingface.co/BAAI/bge-reranker-v2-m3
